import oracledb
import os
from app.core.logger import logger
from dotenv import load_dotenv

# .env 로드는 시도하되, 실패해도 아래 기본값이 적용되도록 함
load_dotenv()

# CLOB/BLOB 데이터를 조회 시 자동으로 문자열/바이트로 변환 (DPI-1010 방지)
oracledb.defaults.fetch_lobs = False

# Oracle DB 접속 정보
DB_USER = os.getenv("DB_USER") or "scott"
DB_PASS = os.getenv("DB_PASS") or "tiger"
DB_HOST = os.getenv("DB_HOST") or "localhost"
DB_PORT = os.getenv("DB_PORT") or "1521"
DB_SID = os.getenv("DB_SID") or "xe"

# Oracle Client Path (설정 시 Thick 모드 활성화)
ORACLE_CLIENT_PATH = os.getenv("ORACLE_CLIENT_PATH")

def fetch_table_ddl(table_name: str) -> list:
    """소스 테이블의 컬럼 메타데이터를 읽기 전용으로 조회합니다 (DB 변경 없음).

    Returns:
        list of tuples: (COLUMN_NAME, DATA_TYPE, DATA_LENGTH, DATA_PRECISION, DATA_SCALE, NULLABLE)
    """
    if "." in table_name:
        owner, tbl = table_name.upper().split(".", 1)
    else:
        owner, tbl = None, table_name.upper()

    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                if owner:
                    cursor.execute(
                        "SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, DATA_PRECISION, DATA_SCALE, NULLABLE "
                        "FROM ALL_TAB_COLUMNS "
                        "WHERE OWNER = :1 AND TABLE_NAME = :2 "
                        "ORDER BY COLUMN_ID",
                        [owner, tbl]
                    )
                else:
                    cursor.execute(
                        "SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, DATA_PRECISION, DATA_SCALE, NULLABLE "
                        "FROM USER_TAB_COLUMNS "
                        "WHERE TABLE_NAME = :1 "
                        "ORDER BY COLUMN_ID",
                        [tbl]
                    )
                return cursor.fetchall()
    except Exception as e:
        logger.warning(f"[DB] 소스 테이블 DDL 조회 실패 ({table_name}): {e}")
        return []


def get_connection():
    """Oracle DB에 접속하여 Connection 객체를 반환합니다 (Thin/Thick 동적 전환)."""
    try:
        # 1. Thick 모드 활성화 여부 결정
        if ORACLE_CLIENT_PATH and os.path.exists(ORACLE_CLIENT_PATH):
            try:
                oracledb.init_oracle_client(lib_dir=ORACLE_CLIENT_PATH)
                logger.debug(f"[DB] Oracle Thick Mode 활성화 (Path: {ORACLE_CLIENT_PATH})")
            except oracledb.ProgrammingError:
                pass # 이미 초기화된 경우
            mode_str = "Thick Mode"
        else:
            logger.debug("[DB] Oracle Thin Mode 접속 시도 (No Client Path set)")
            mode_str = "Thin Mode"
            
        # 2. 접속 시도
        # DB_HOST에이미 '/'가 포함(Easy Connect)되어 있거나 전체 DSN 주소인 경우 그대로 사용
        if "/" in DB_HOST or "(" in DB_HOST:
            dsn = DB_HOST
        else:
            dsn = f"{DB_HOST}:{DB_PORT}/{DB_SID}"
            
        connection = oracledb.connect(
            user=DB_USER,
            password=DB_PASS,
            dsn=dsn
        )
        
        # 3. 세션 날짜 형식 표준화 (ORA-01847 등 방지)
        with connection.cursor() as cursor:
            cursor.execute("ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD HH24:MI:SS'")
            cursor.execute("ALTER SESSION SET NLS_TIMESTAMP_FORMAT = 'YYYY-MM-DD HH24:MI:SS.FF'")
            
        return connection
    except Exception as e:
        logger.error(f"[DB] Oracle 접속 중 에러 발생 ({mode_str}, USER: {DB_USER}): {e}")
        raise e
