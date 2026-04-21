from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

@dataclass
class MappingDetail:
    """DB에 저장된 메타데이터(NEXT_MIG_INFO_DTL)를 표현하는 객체"""
    map_dtl: int
    map_id: int
    fr_col: str
    to_col: str

@dataclass
class MappingRule:
    """DB에 저장된 메타데이터(NEXT_MIG_INFO)를 표현하는 객체"""
    map_id: int
    map_type: str
    fr_table: str
    to_table: str
    use_yn: str
    target_yn: str        # 작업대상
    priority: int          # 기존 exe_order
    mig_sql: Optional[str] = None
    verify_sql: Optional[str] = None
    status: Optional[str] = None
    correct_sql: Optional[str] = None
    user_edited: Optional[str] = None
    batch_cnt: int = 0
    elapsed_seconds: int = 0
    retry_count: int = 0
    created_at: Optional[datetime] = None
    upd_ts: Optional[datetime] = None
    details: List[MappingDetail] = field(default_factory=list)

    @property
    def from_columns(self) -> str:
        """이관 대상의 from_column을 콤마(,)로 구분한 문자열 반환 (이름순 정렬)"""
        return ", ".join(d.fr_col for d in sorted(self.details, key=lambda x: x.fr_col))

    @property
    def to_columns(self) -> str:
        """이관 대상의 to_column을 콤마(,)로 구분한 문자열 반환 (from_column 정렬 기준 순서 유지)"""
        return ", ".join(d.to_col for d in sorted(self.details, key=lambda x: x.fr_col))

