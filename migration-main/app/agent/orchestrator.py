import time
import os
from app.core.logger import logger
from app.agent.graph import migration_graph, BIZ_MAX_ATTEMPTS
from app.domain.mapping.repository import increment_batch_count
from app.core.exceptions import BatchAbortError

class MigrationOrchestrator:
    def __init__(self):
        self.mig_kind = os.getenv("MIG_KIND", "DB_MIG")

    def process_job(self, NEXT_SQL_INFO):
        logger.info(f"\n==========================================")
        logger.info(f"[JOB_START] 대상 작업(map_id={NEXT_SQL_INFO.map_id}) 프로세스 시작 (LangGraph)")
        
        # 0. BATCH_COUNT 증가 (작업 시작 기록)
        increment_batch_count(NEXT_SQL_INFO.map_id)
        
        # 1. 초기 상태 정의
        initial_state = {
            "next_sql_info": NEXT_SQL_INFO,
            "source_ddl": None,
            "last_error": None,
            "last_sql": None,
            "db_attempts": 1,
            "max_attempts": BIZ_MAX_ATTEMPTS,
            "llm_retry_count": 0,
            "current_ddl_sql": None,
            "current_migration_sql": None,
            "current_v_sql": None,
            "error_type": None,
            "status": "RUNNING",
            "elapsed_time": 0,
            "job_start_time": time.time()
        }

        # 4. 그래프 실행 (상태 기반 오케스트레이션)
        try:
            logger.info(f"[Orchestrator] map_id={NEXT_SQL_INFO.map_id} | 시작")
            final_state = migration_graph.invoke(initial_state)
            
            # 최종 요약 로그
            status = final_state.get("status", "UNKNOWN")
            elapsed = final_state.get("elapsed_time", 0)
            logger.info(f"[JOB_DONE] map_id={NEXT_SQL_INFO.map_id} | 최종 상태: {status} | 소요시간: {elapsed}초")
            
        except BatchAbortError as abort_err:
            # 치명적인 배치 중단 오류는 상위 스케줄러로 전파하여 전체 프로세스 중단
            logger.critical(f"[Orchestrator] map_id={NEXT_SQL_INFO.map_id} | 치명적 배치 중단 요청 접수: {abort_err}")
            raise abort_err
        except Exception as e:
            # 그래프 실행 중 예상치 못한 치명적 크래시 발생 시
            logger.error(f"[Orchestrator] map_id={NEXT_SQL_INFO.map_id} | 치명적 크래시 발생: {str(e)}", exc_info=True)
            
            # 크래시가 나더라도 해당 작업에 갇히지 않도록 FAIL 처리 시도 (USE_YN='N')
            from app.domain.mapping.repository import update_job_status
            update_job_status(NEXT_SQL_INFO.map_id, "FAIL", 0, 0)
            
            logger.warning(f"[Orchestrator] map_id={NEXT_SQL_INFO.map_id} | 크래시로 인한 강제 FAIL 처리 완료. 다음 작업으로 넘어갑니다.")
            # 에러를 더 이상 raise 하지 않고, 이 작업만 실패로 마감하여 스케줄러 루프 유지
