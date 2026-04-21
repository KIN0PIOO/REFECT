import oracledb
from app.core.db import get_connection
from app.core.logger import logger
from app.core.exceptions import DBSqlError
from app.agent.sql_utils import split_sql_script, clean_sql_statement

def drop_table_if_exists(table_name: str):
    """테이블이 존재하면 삭제합니다. (Clean Retry를 위함)"""
    logger.debug(f"[Executor] 테이블 삭제 시도: {table_name}")
    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            # Oracle에서 안전하게 테이블을 삭제하는 PL/SQL 블록
            sql = f"""
            BEGIN
                EXECUTE IMMEDIATE 'DROP TABLE {table_name} CASCADE CONSTRAINTS PURGE';
            EXCEPTION
                WHEN OTHERS THEN
                    IF SQLCODE != -942 THEN
                        RAISE;
                    END IF;
            END;
            """
            cursor.execute(sql)
            conn.commit()
            logger.info(f"[Executor] 테이블 삭제 완료(또는 미존재): {table_name}")
    except Exception as e:
        logger.warning(f"[Executor] 테이블 삭제 중 경고 (무시 가능): {str(e)}")

def execute_migration(sql_script: str):
    """생성된 SQL 스크립트를 Oracle DB 엔진에 실행"""
    if not sql_script.strip():
        logger.debug("[Executor] 실행할 SQL 스크립트가 비어있습니다.")
        return

    try:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # sql_utils를 사용하여 일관성 있게 분할
            statements = split_sql_script(sql_script)
            
            for stmt in statements:
                clean_stmt = clean_sql_statement(stmt)
                if not clean_stmt:
                    continue

                # PL/SQL 블록(BEGIN/DECLARE)인지 일반 SQL인지 구분
                is_plsql = clean_stmt.upper().startswith(('BEGIN', 'DECLARE'))
                
                logger.info(f"[Executor] Executing {'PL/SQL' if is_plsql else 'SQL'}: {clean_stmt[:70]}...")
                
                try:
                    # PL/SQL은 끝에 세미콜론과 개행이 필요할 수 있음
                    exec_stmt = clean_stmt if not is_plsql else clean_stmt + "\n"
                    cursor.execute(exec_stmt)
                except oracledb.DatabaseError as e:
                    # ORA-00955: 이미 존재하는 객체는 경고 후 진행 (DDL 단계에서 DROP 하므로 발생 빈도 낮아짐)
                    if "ORA-00955" in str(e):
                        logger.warning(f"[Executor] 객체가 이미 존재하여 건너뜁니다.")
                        continue
                    raise e
            
            conn.commit()
            logger.info(f"[Executor] All commands executed and committed successfully.")
            
    except Exception as e:
        logger.error(f"[Executor] SQL 실행 실패: {str(e)}")
        raise DBSqlError(f"Oracle 쿼리 실행 에러: {str(e)}")
