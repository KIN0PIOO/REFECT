class AgentBaseException(Exception):
    """모든 에이전트 예외의 상위 예외"""
    pass

class BatchAbortError(AgentBaseException):
    """배치 전체 흐름을 즉시 중단해야 할 때 발생하는 예외"""
    pass

class LLMBaseError(AgentBaseException):
    """LLM API 호출 관련 모든 에러의 상위 예외"""
    pass

class LLMRateLimitError(LLMBaseError):
    """LLM API Rate Limit 초과 등 일시적 호출 제한 시 발생하는 에러"""
    pass

class LLMConnectionError(LLMBaseError):
    """네트워크 연결, 타임아웃 등 연결 문제가 발생했을 때 에러"""
    pass

class LLMAuthenticationError(LLMBaseError):
    """API 키 오류 등 인증 문제가 발생했을 때 에러"""
    pass

class LLMTokenLimitError(LLMBaseError):
    """프롬프트 최대 토큰수 초과 시 에러"""
    pass

class LLMInvalidRequestError(LLMBaseError):
    """잘못된 포맷 요청 등 4xx 에러"""
    pass

class LLMServerError(LLMBaseError):
    """서버 측 5xx 에러"""
    pass

class DBSqlError(AgentBaseException):
    """생성된 SQL 실행 중 문법/런타임 DB 오류 발생 시 에러 (재시도용)"""
    pass

class VerificationFailError(AgentBaseException):
    """검증(Verification) 실패 시 에러 (데이터 불일치 등)"""
    pass
