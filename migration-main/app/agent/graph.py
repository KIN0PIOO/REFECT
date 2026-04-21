import time
import os
import re
from typing import Literal
from langgraph.graph import StateGraph, END
from app.core.logger import logger
from app.core.exceptions import (
    LLMBaseError, LLMAuthenticationError, LLMTokenLimitError, LLMInvalidRequestError,
    DBSqlError, VerificationFailError, BatchAbortError
)
from app.agent.llm_client import generate_sqls
from app.agent.executor import execute_migration, drop_table_if_exists
from app.agent.verifier import execute_verification
from app.domain.mapping.repository import update_job_status, check_dependencies, is_first_job_for_target
from app.domain.history.repository import log_generated_sql, log_business_history
from app.core.db import fetch_table_ddl
from app.agent.state import MigrationState

LLM_MAX_RETRY = 2
BIZ_MAX_ATTEMPTS = 3

def _extract_table_names(fr_table: str) -> list:
    """FR_TABLE 표현식에서 실제 테이블명만 추출합니다."""
    parts = re.split(
        r'\b(?:(?:LEFT|RIGHT|FULL|INNER|CROSS)\s+(?:OUTER\s+)?)?JOIN\b',
        fr_table, flags=re.IGNORECASE
    )
    tables = []
    for part in parts:
        part = re.split(r'\bON\b', part, flags=re.IGNORECASE)[0].strip()
        tokens = part.split()
        if tokens and tokens[0].upper() not in ('SELECT', 'WITH', 'FROM', '('):
            tables.append(tokens[0])
    return tables

# Nodes
def fetch_ddl_node(state: MigrationState) -> dict:
    job = state["next_sql_info"]
    source_ddl = {}
    for tbl_name in _extract_table_names(job.fr_table):
        rows = fetch_table_ddl(tbl_name)
        if rows:
            source_ddl[tbl_name] = rows
            logger.info(f"[Graph:DDL] {tbl_name} 컬럼 {len(rows)}개 조회 완료")
    return {"source_ddl": source_ddl if source_ddl else None}

def check_dependency_node(state: MigrationState) -> dict:
    job = state["next_sql_info"]
    dep_status = check_dependencies(job.map_id, job.to_table, job.priority)
    
    if dep_status != "READY":
        logger.warning(f"[Graph:DEP] map_id={job.map_id} | 선행 작업 미통과 ({dep_status}). 작업을 SKIP 합니다.")
        return {"status": "SKIP", "error_type": "DEPENDENCY_FAIL", "last_error": f"선행 작업 상태: {dep_status}"}
    
    return {"error_type": None}

def generate_sql_node(state: MigrationState) -> dict:
    job = state["next_sql_info"]
    job.retry_count = state["db_attempts"] - 1
    
    llm_retry = state.get("llm_retry_count", 0)
    attempt_msg = f"{state['db_attempts']}"
    if llm_retry > 0:
        attempt_msg += f" (LLM Retry {llm_retry}/{LLM_MAX_RETRY})"
    
    logger.info(f"[Graph:LLM] Attempt {attempt_msg} | SQL 생성 요청")
    try:
        is_append = not is_first_job_for_target(job.map_id, job.to_table, job.priority)
        ddl_sql, migration_sql, v_sql = generate_sqls(
            job, 
            state["last_error"], 
            state["last_sql"], 
            state["source_ddl"],
            is_append=is_append
        )
        
        # DB 기록
        log_generated_sql(job.map_id, migration_sql, v_sql)
        
        return {
            "last_sql": migration_sql,
            "current_ddl_sql": ddl_sql,
            "current_migration_sql": migration_sql,
            "current_v_sql": v_sql,
            "error_type": None
        }
    except (LLMAuthenticationError, LLMTokenLimitError, LLMInvalidRequestError) as e:
        logger.error(f"[Graph:LLM_FATAL] {str(e)}")
        raise BatchAbortError(f"LLM 치명적 에러: {str(e)}") from e
    except LLMBaseError as e:
        return {"error_type": "LLM_RETRY", "last_error": str(e)}

def execute_sql_node(state: MigrationState) -> dict:
    job = state["next_sql_info"]
    to_table = job.to_table
    
    try:
        # 해당 타겟 테이블에 대해 내가 첫 번째 작업인지 확인
        if is_first_job_for_target(job.map_id, to_table, job.priority):
            logger.info(f"[Graph:EXEC] map_id={job.map_id} | 최초 작업 감지: 클린업(DROP) 및 테이블 생성(DDL) 진행")
            drop_table_if_exists(to_table)
            
            if state.get("current_ddl_sql"):
                execute_migration(state["current_ddl_sql"])
        else:
            logger.info(f"[Graph:EXEC] map_id={job.map_id} | 누적(Append) 모드 감지: DROP 및 DDL 단계 생략")
            
        # 데이터 이관(DML) 실행
        execute_migration(state["current_migration_sql"])
        return {"status": "EXECUTED", "error_type": None}
    except DBSqlError as e:
        logger.error(f"[Graph:EXEC_FAIL] {str(e)}")
        return {"error_type": "BIZ_RETRY", "last_error": str(e)}

def verify_sql_node(state: MigrationState) -> dict:
    v_sql = state.get("current_v_sql")
    if not v_sql:
        return {"status": "PASS"}
        
    try:
        logger.info(f"[Graph:VERIFY] 데이터 정합성 검증 시작")
        is_valid, v_msg = execute_verification(v_sql)
        if not is_valid:
            return {"error_type": "BIZ_RETRY", "last_error": f"데이터 불일치: {v_msg}"}
        return {"status": "PASS", "error_type": None}
    except (VerificationFailError, DBSqlError) as e:
        return {"error_type": "BIZ_RETRY", "last_error": str(e)}

def finalize_node(state: MigrationState) -> dict:
    job = state["next_sql_info"]
    elapsed = int(time.time() - state["job_start_time"])
    mig_kind = os.getenv("MIG_KIND", "DB_MIG")
    
    if state["status"] == "PASS":
        update_job_status(job.map_id, "PASS", elapsed, state["db_attempts"])
        log_business_history(job.map_id, "INFO", "INFO", "VERIFY", "PASS", "Migration Success", state["db_attempts"], mig_kind)
        logger.info(f"[Graph:FINISH] map_id={job.map_id} | >>> 성공 <<<")
        return {"elapsed_time": elapsed, "status": "PASS"}
    elif state["status"] == "SKIP":
        update_job_status(job.map_id, "SKIP", elapsed, state["db_attempts"])
        log_business_history(job.map_id, "JOB_SKIP", "WARN", "DEP_CHECK", "SKIP", state["last_error"], state["db_attempts"], mig_kind)
        logger.warning(f"[Graph:FINISH] map_id={job.map_id} | >>> SKIP (의존성 실패) <<<")
        return {"elapsed_time": elapsed, "status": "SKIP"}
    else:
        update_job_status(job.map_id, "FAIL", elapsed, state["db_attempts"])
        log_business_history(job.map_id, "JOB_FAIL", "ERROR", "FINAL", "FAIL", "Max Attempts Reached", state["db_attempts"], mig_kind)
        logger.error(f"[Graph:FINISH] map_id={job.map_id} | >>> 실패 <<<")
        return {"elapsed_time": elapsed, "status": "FAIL"}

# Routing Logic
def should_continue(state: MigrationState) -> Literal["generate", "finalize", "verify", "execute", "llm_retry_wait"]:
    error_type = state.get("error_type")
    
    if state.get("status") in ("PASS", "SKIP"):
        return "finalize"
        
    if error_type == "DEPENDENCY_FAIL":
        return "finalize"

    if error_type == "LLM_RETRY":
        last_err = state.get("last_error", "").lower()
        # 429 에러(Quota Exceeded) 또는 할당량 초과 메시지가 포함된 경우 즉시 배치 중단
        if "429" in last_err or "quota" in last_err or "limit" in last_err:
            logger.critical(f"[Graph:LLM_FATAL] 할당량 초과 또는 인프라 에러 감지. 배치를 즉시 중단합니다: {state['last_error']}")
            raise BatchAbortError(f"LLM 인프라 에러(할당량 초과 등): {state['last_error']}")

        if state["llm_retry_count"] < LLM_MAX_RETRY:
            return "llm_retry_wait"
        else:
            raise BatchAbortError(f"LLM 재시도 초과: {state['last_error']}")

    if error_type == "BIZ_RETRY":
        if state["db_attempts"] < state["max_attempts"]:
            return "generate"
        else:
            return "finalize"
            
    # 에러가 없고 현재 상태에 따라 다음 단계 진행
    if state.get("status") == "EXECUTED":
        return "verify"
    
    # SQL이 아직 생성되지 않은 상태(check_dependency 이후)라면 generate로 진행
    if not state.get("current_migration_sql"):
        return "generate"
    
    return "execute"

def llm_retry_wait_node(state: MigrationState) -> dict:
    time.sleep(1)
    return {"llm_retry_count": state["llm_retry_count"] + 1}

def biz_retry_prepare_node(state: MigrationState) -> dict:
    # 비즈니스 에러 발생 시 시도 횟수 증가 및 로그 기록
    job = state["next_sql_info"]
    mig_kind = os.getenv("MIG_KIND", "DB_MIG")
    step_name = "SQL_EXEC" if "DBSqlError" in state["last_error"] else "VERIFY"
    
    log_business_history(job.map_id, "ROW_ERROR", "WARN", step_name, "FAIL", state["last_error"], state["db_attempts"], mig_kind)
    time.sleep(1)
    return {"db_attempts": state["db_attempts"] + 1, "error_type": None}

# Graph Construction
workflow = StateGraph(MigrationState)

workflow.add_node("fetch_ddl", fetch_ddl_node)
workflow.add_node("check_dependency", check_dependency_node)
workflow.add_node("generate", generate_sql_node)
workflow.add_node("execute", execute_sql_node)
workflow.add_node("verify", verify_sql_node)
workflow.add_node("finalize", finalize_node)
workflow.add_node("llm_retry_wait", llm_retry_wait_node)
workflow.add_node("biz_retry_prepare", biz_retry_prepare_node)

workflow.set_entry_point("fetch_ddl")
workflow.add_edge("fetch_ddl", "check_dependency")

workflow.add_conditional_edges(
    "check_dependency",
    should_continue,
    {
        "generate": "generate",
        "finalize": "finalize",
        "execute": "generate" # 라우터가 execute를 반환하더라도 generate로 리다이렉트
    }
)

workflow.add_conditional_edges(
    "generate",
    should_continue,
    {
        "execute": "execute",
        "llm_retry_wait": "llm_retry_wait",
        "finalize": "finalize"
    }
)

workflow.add_edge("llm_retry_wait", "generate")

workflow.add_conditional_edges(
    "execute",
    should_continue,
    {
        "verify": "verify",
        "generate": "biz_retry_prepare",
        "finalize": "finalize"
    }
)

workflow.add_conditional_edges(
    "verify",
    should_continue,
    {
        "finalize": "finalize",
        "generate": "biz_retry_prepare"
    }
)

workflow.add_edge("biz_retry_prepare", "generate")
workflow.add_edge("finalize", END)

migration_graph = workflow.compile()
