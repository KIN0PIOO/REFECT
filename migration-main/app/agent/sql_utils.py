import re

def split_sql_script(script: str) -> list[str]:
    """
    SQL 스크립트를 '/' (슬래시) 또는 ';' (세미콜론) 기준으로 분할합니다.
    Oracle 특화 처리를 위해 PL/SQL 블록(BEGIN/DECLARE)은 하나로 묶어 반환합니다.
    """
    if not script:
        return []

    # 1. '/' (슬래시) 단독 라인 기준으로 크게 분할
    parts = re.split(r'^\s*/\s*$', script, flags=re.MULTILINE)
    
    statements = []
    for part in parts:
        clean_part = part.strip()
        if not clean_part:
            continue
            
        # 주석 제거하고 실질적인 시작 단어 확인
        content_only = re.sub(r'--.*$', '', clean_part, flags=re.MULTILINE)
        content_only = re.sub(r'/\*.*?\*/', '', content_only, flags=re.DOTALL).strip()
        
        # PL/SQL 블록인 경우 하나로 유지
        if re.match(r'^(BEGIN|DECLARE)', content_only, re.IGNORECASE):
            statements.append(clean_part)
        else:
            # 일반 SQL은 세미콜론으로 쪼개기
            sub_stmts = [s.strip() for s in clean_part.split(';') if s.strip()]
            statements.extend(sub_stmts)
            
    return statements

def clean_sql_statement(stmt: str) -> str:
    """
    개별 SQL 문장에서 불필요한 공백, 트레일링 세미콜론 등을 제거합니다.
    (Oracle cursor.execute() 시 세미콜론이 있으면 에러 발생 가능)
    """
    if not stmt:
        return ""
        
    cleaned = stmt.strip()
    # 끝에 세미콜론이나 슬래시가 있으면 제거
    cleaned = re.sub(r'[;/]\s*$', '', cleaned)
    return cleaned.strip()
