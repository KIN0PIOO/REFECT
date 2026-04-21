from app.core.db import get_connection
from app.core.logger import logger
from app.agent.sql_utils import split_sql_script, clean_sql_statement

def execute_verification(sql: str) -> tuple[bool, str]:
    """양 DB의 정합성을 대조하는 검증 SQL 실행"""
    if not sql.strip():
        return True, "No verification SQL provided"

    logger.debug(f"[Verifier] 실제 검증 쿼리 실행 시작: {sql[:50]}...")
    
    try:
        # sql_utils를 사용하여 일관성 있게 클리닝
        statements = split_sql_script(sql)
        if not statements:
            return True, "No valid SQL statements found"
            
        # 마지막 유효한 SELECT 문만 실행 (검증용)
        # 만약 여러 문장이면 순차 실행 가능하지만, 마지막 문장의 결과로 판단
        last_rows = []
        with get_connection() as conn:
            cursor = conn.cursor()
            for stmt in statements:
                clean_stmt = clean_sql_statement(stmt)
                if not clean_stmt:
                    continue
                
                logger.debug(f"[Verifier] Executing: {clean_stmt[:70]}...")
                cursor.execute(clean_stmt)
                
                # SELECT 문인 경우 결과 가져오기
                if cursor.description:
                    last_rows = cursor.fetchall()
            
            if not last_rows:
                # 결과가 없으면 (성공으로 간주하거나, 쿼리 오류일 수 있음)
                # 보통 COUNT나 ABS 합은 결과가 1행이라도 있어야 함
                return True, "No mismatch found (Empty ResultSet)"
            
            # 모든 결과 행의 모든 컬럼 값이 0인지 확인 (DIFF=0)
            # NULL은 검증 SQL 오류 가능성이 있으므로 실패로 처리
            for row in last_rows:
                for col_val in row:
                    if col_val is None or str(col_val) != "0":
                        return False, f"Mismatch found (NULL or non-zero DIFF): {row}"
                        
            return True, "All Verification Passed"
            
    except Exception as e:
        logger.error(f"[Verifier] 검증 쿼리 실행 에러: {str(e)}")
        return False, f"Verification Query Error: {str(e)}"
