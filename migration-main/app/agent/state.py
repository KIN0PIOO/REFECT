from typing import TypedDict, Optional, Dict, Any

class MigrationState(TypedDict):
    """
    마이그레이션 에이전트의 상태를 관리하는 객체.
    모든 노드(Node)는 이 상태를 읽고 업데이트하여 흐름을 제어합니다.
    """
    # 현재 마이그레이션 대상 작업 정보 (MappingRule 객체)
    next_sql_info: Any
    
    # 소스 테이블 DDL 정보
    source_ddl: Optional[Dict[str, Any]]
    
    # 재시도 및 에러 컨텍스트
    last_error: Optional[str]
    last_sql: Optional[str]
    
    # 카운터
    db_attempts: int
    max_attempts: int
    llm_retry_count: int
    
    # 생성된 SQL 임시 보관
    current_ddl_sql: Optional[str]
    current_migration_sql: Optional[str]
    current_v_sql: Optional[str]
    error_type: Optional[str]

    # 결과 및 시간
    status: str  # "PASS", "FAIL", "SKIP", "RUNNING"
    elapsed_time: int
    job_start_time: float
