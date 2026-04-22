import os
import json
import openai
from openai import OpenAI
from app.core.exceptions import (
    LLMConnectionError, LLMAuthenticationError, LLMTokenLimitError,
    LLMRateLimitError, LLMInvalidRequestError, LLMServerError
)
from app.core.logger import logger
from dotenv import load_dotenv

# .env 로드 (루트 경로 기준)
env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ".env"
)
load_dotenv(env_path)


def get_client():
    """OpenAI 호환 모델 클라이언트를 반환합니다."""
    api_key = os.getenv("OPEN_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")

    if not api_key:
        error_msg = f"API Key(OPEN_API_KEY)가 설정되지 않았습니다. (Path: {env_path})"
        logger.error(f"[LLM] {error_msg}")
        raise LLMAuthenticationError(error_msg)

    # OpenAI 호환 클라이언트 초기화 (base_url이 있으면 사내 Gateway 등으로 연결)
    return OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None
    )


def _format_ddl_info(ddl_rows: list) -> str:
    """컬럼 메타데이터 튜플 리스트를 프롬프트용 텍스트로 변환합니다."""
    if not ddl_rows:
        return "  (조회된 컬럼 정보 없음)"
    lines = []
    for col_name, data_type, data_length, data_precision, data_scale, nullable in ddl_rows:
        if data_type == "NUMBER":
            if data_precision is not None and data_scale not in (None, 0):
                type_str = f"NUMBER({data_precision},{data_scale})"
            elif data_precision is not None:
                type_str = f"NUMBER({data_precision})"
            else:
                type_str = "NUMBER"
        elif data_type in ("VARCHAR2", "CHAR", "NVARCHAR2", "NCHAR") and data_length:
            type_str = f"{data_type}({data_length})"
        else:
            type_str = data_type
        null_str = "NULL" if nullable == "Y" else "NOT NULL"
        lines.append(f"  {col_name:<30} {type_str:<25} {null_str}")
    return "\n".join(lines)


def generate_sqls(NEXT_SQL_INFO, last_error=None, last_sql=None, source_ddl=None, is_append=False):
    """
    OpenAI 호환 API를 호출하여 Oracle 21c 마이그레이션 SQL들을 생성합니다.
    (DDL, Migration, Verification 분리)

    Args:
        NEXT_SQL_INFO: 매핑 규칙 객체
        last_error: 이전 실행 실패 에러 메시지 (재시도 시)
        last_sql: 이전 실행 실패 SQL (재시도 시)
        source_ddl: fetch_table_ddl()로 조회한 소스 테이블 컬럼 메타데이터 (읽기 전용 조회 결과)
    """
    client = get_client()
    model_name = os.getenv("LLM_MODEL") or "gpt-4o-mini"

    from_table = NEXT_SQL_INFO.fr_table
    to_table = NEXT_SQL_INFO.to_table

    # 컬럼 매핑 정보 정리
    details = NEXT_SQL_INFO.details
    mapping_info = "\n".join([f"  - {d.fr_col} -> {d.to_col}" for d in details])

    # 소스 테이블 DDL 정보 (실제 컬럼 타입/길이/제약조건)
    # source_ddl은 {table_name: [rows]} 형태의 dict
    ddl_info_block = ""
    if source_ddl and isinstance(source_ddl, dict):
        table_blocks = []
        for tbl_name, rows in source_ddl.items():
            formatted = _format_ddl_info(rows)
            table_blocks.append(
                f"    테이블: {tbl_name}\n"
                f"    {'컬럼명':<30} {'데이터타입':<25} {'NULL여부'}\n"
                f"    {'-'*70}\n"
                f"{formatted}"
            )
        ddl_info_block = f"""
    [소스 테이블 실제 DDL 정보] (ALL_TAB_COLUMNS 읽기 전용 조회 결과)
{chr(10).join(table_blocks)}

    ※ 위 DDL 정보를 타겟 테이블 생성 시 반드시 참고하여 정확한 타입을 사용하십시오.
"""

    # verification_sql 지시문: 일반 모드 vs Append 모드 분기
    if is_append:
        verification_instruction = f"""
    3. verification_sql: (※ Append 모드 — 선행 job 데이터가 타겟에 이미 존재함)
       - **[핵심 주의]** 타겟 전체 COUNT(*)를 소스와 비교하면 선행 job이 INSERT한 행까지 포함되어 항상 불일치합니다. 절대 타겟 전체 COUNT를 사용하지 마십시오.
       - **[구조 필수]** 반드시 UNION ALL로 아래 두 가지 검증 항목을 포함하십시오:
         (1) 이번 job이 이관한 행 개수 검증:
             `SELECT ABS(S.CNT - T.CNT) AS DIFF
              FROM (SELECT COUNT(*) CNT FROM {from_table}) S,
                   (SELECT COUNT(*) CNT FROM {to_table} T
                    WHERE EXISTS (SELECT 1 FROM {from_table} SRC WHERE T.{{타겟_키_컬럼}} = SRC.{{소스_키_컬럼}})) T`
         (2) 매핑된 각 컬럼별 NOT NULL 개수 검증: 소스 COUNT(컬럼) vs 타겟에서 EXISTS 필터링 후 COUNT(컬럼) 형식으로 UNION ALL 하십시오.
       - **[타입 안전]** 비교 시 데이터 타입이 다르면 반드시 `CAST` 또는 `TO_NUMBER`를 사용하십시오.
       - **[단일 출력]** 오직 'DIFF' 컬럼 하나만 출력해야 합니다.
       - **[합격 기준]** 모든 UNION ALL 행의 DIFF가 전부 0이어야 검증 통과입니다."""
    else:
        verification_instruction = f"""
    3. verification_sql:
       - **[구조 필수]** 반드시 UNION ALL로 아래 두 가지 검증 항목을 모두 포함하십시오:
         (1) 행 개수 검증: `SELECT ABS(S.CNT - T.CNT) AS DIFF FROM (SELECT COUNT(*) CNT FROM {from_table}) S, (SELECT COUNT(*) CNT FROM {to_table}) T`
         (2) 매핑된 각 컬럼별 NOT NULL 개수 검증: 매핑된 컬럼마다 `SELECT ABS(S.CNT - T.CNT) AS DIFF FROM (SELECT COUNT(소스컬럼) CNT FROM {from_table}) S, (SELECT COUNT(타겟컬럼) CNT FROM {to_table}) T` 형식으로 UNION ALL 하십시오.
         - 소스 컬럼이 표현식(CASE WHEN 등)인 경우, 해당 표현식을 COUNT(CASE WHEN ... THEN 1 END) 형태로 변환하십시오.
       - **[타입 안전]** 비교 시 데이터 타입이 다르면 반드시 `CAST` 또는 `TO_NUMBER`를 사용하십시오.
       - **[단일 출력]** 오직 'DIFF' 컬럼 하나만 출력해야 합니다.
       - **[합격 기준]** 모든 UNION ALL 행의 DIFF가 전부 0이어야 검증 통과입니다."""

    # 프롬프트 구성 (Oracle 전문 마이그레이션 전략가 페르소나 적용)
    prompt = f"""
    당신은 Oracle 데이터 마이그레이션 전문가이자 SQL 튜닝 전략가입니다.
    제시된 매핑 규칙과 소스 테이블의 실제 DDL 정보를 기반으로
    (1) 타겟 테이블 생성 DDL, (2) 데이터 이관 DML, (3) 정합성 검증 SQL을 JSON 형식으로 생성하십시오.

    [핵심 원칙 - 절대 준수]
    1. **환각 방지 (Zero Hallucination)**:
       - **[매핑 규칙]** 및 **[소스 테이블 실제 DDL 정보]**에 명시되지 않은 컬럼은 절대 사용하지 마십시오.
       - 예를 들어, 'SALARY'가 매핑 규칙에 없다면 설령 소스 테이블에 있더라도 절대 사용하지 마십시오.
       - 임의로 'CURRENT_SALARY', 'TOTAL_AMT' 등 그럴싸한 컬럼명을 날조하는 행위는 시스템을 파괴하는 치명적인 오류입니다.

    2. **데이터 타입 정합성**:
       - 숫자(`NUMBER`)와 문자열(`VARCHAR2`)을 비교할 때는 반드시 명시적 타입 변환(`TO_NUMBER`, `TO_DATE`)을 사용하십시오.
       - 특히 `verification_sql`에서 소스의 문자열 값을 타겟의 숫자 컬럼과 비교할 때 `ORA-01722` 에러를 방지하도록 설계하십시오.

    3. **Oracle 11.2 XE 환경 제약**:
       - 12c 이상 전용 기능(LATERAL, STANDARD_HASH, FETCH FIRST 등)은 절대 사용하지 마십시오.
       - 해시 비교가 필요하면 `ORA_HASH(column)`를 활용하십시오.

{ddl_info_block}
    [매핑 규칙]
    - 소스 테이블: {from_table}
    - 타겟 테이블: {to_table}
    - 컬럼 매핑 정보:
{mapping_info}

    [상세 요구사항]
    1. ddl_sql:
       - 타겟 테이블('{to_table}') 생성 'CREATE TABLE' 문장만 작성하십시오.
       - DDL 정보의 컬럼 타입을 엄격히 따르십시오. 길이나 정밀도를 임의로 줄이지 마십시오.

    2. migration_sql:
       - 'INSERT INTO {to_table} (컬럼...) SELECT (표현식...) FROM {from_table}' 형식을 따르십시오.
       - 소스 테이블명이 {from_table}이면 alias를 사용하여 가독성을 높이십시오 (예: FROM {from_table} src).
       - 소스에 파생 표현식(파생 컬럼)이 많은 경우, 핵심 로직은 내부 서브쿼리(src_base)에서 처리하고 외부 SELECT에서는 매핑만 수행하십시오.
{verification_instruction}

    4. 공통:
       - 출력은 반드시 JSON 형태여야 하며, SQL 내부에 불필요한 주석을 넣지 마십시오.
    """

    # [추가] 인간 전문가가 직접 수정한 정답 SQL이 있다면 프롬프트에 반영
    if NEXT_SQL_INFO.correct_sql:
        logger.info(f"[LLM] map_id={NEXT_SQL_INFO.map_id} | 인간 전문가의 정답 SQL을 프롬프트에 반영합니다.")
        prompt += f"\n\n[인간 전문가가 검증한 정답 SQL 예시]\n{NEXT_SQL_INFO.correct_sql}\n"
        prompt += "- 위 예시의 패턴을 참고하여 ddl_sql, migration_sql, verification_sql로 나누어 생성하십시오.\n"

    if last_error:
        prompt += f"""
        
        [이전 실행 실패 피드백]
        - 실패한 SQL: {last_sql}
        - 발생한 에러: {last_error}
        - 작업: 위 에러를 분석하여 올바르게 수정한 쿼리들을 다시 생성하십시오.
        """

    if is_append:
        prompt += f"""

[참고: 누적(Append) 모드 — migration_sql 지침]
- 타겟 테이블 '{to_table}'이 이미 존재하며 다른 작업(선행 job)이 INSERT한 데이터가 있습니다.
- 'ddl_sql'은 빈 문자열로 두어도 됩니다.
- 기존 데이터를 보존하면서 이번 소스 데이터만 추가하는 INSERT 문을 작성하십시오.
"""

    try:
        #logger.debug(f"[LLM_PROMPT] map_id={NEXT_SQL_INFO.map_id}\n{'='*60}\n{prompt}\n{'='*60}")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates Oracle SQL in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        ddl_sql = result.get("ddl_sql", "")
        migration_sql = result.get("migration_sql", "")
        verification_sql = result.get("verification_sql", "")

        # (Post-processing) 리스트 형태일 경우 문자열로 병합
        def merge_list(val):
            if isinstance(val, list):
                return "\n/\n".join(val)
            return val

        logger.info(f"[LLM] SQL 생성 완료 (Model: {model_name})")
        return (
            merge_list(ddl_sql),
            merge_list(migration_sql),
            merge_list(verification_sql)
        )

    except openai.AuthenticationError as e:
        logger.error(f"[LLM] 인증 실패 (API Key 확인 필요): {e}")
        raise LLMAuthenticationError(f"API Key 인증 실패: {str(e)}")
    except openai.RateLimitError as e:
        logger.error(f"[LLM] 호출 한도 초과 (429): {e}")
        raise LLMRateLimitError(f"API 호출 한도 초과: {str(e)}")
    except openai.BadRequestError as e:
        logger.error(f"[LLM] 잘못된 요청 (400): {e}")
        raise LLMInvalidRequestError(f"잘못된 요청 형식: {str(e)}")
    except openai.APIStatusError as e:
        if e.status_code >= 500:
            logger.error(f"[LLM] 서버 에러 ({e.status_code}): {e}")
            raise LLMServerError(f"LLM 서버 에러: {str(e)}")
        logger.error(f"[LLM] API 에러 ({e.status_code}): {e}")
        raise LLMConnectionError(f"LLM API 에러: {str(e)}")
    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        logger.error(f"[LLM] 연결/타임아웃 에러: {e}")
        raise LLMConnectionError(f"LLM 연결 실패: {str(e)}")
    except Exception as e:
        logger.error(f"[LLM] 예상치 못한 에러: {e}")
        raise LLMConnectionError(f"LLM 호출 중 알 수 없는 에러: {str(e)}")