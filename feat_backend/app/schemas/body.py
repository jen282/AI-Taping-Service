from pydantic import BaseModel
from typing import Optional

class BodyMatchResponse(BaseModel):
    session_id: str
    status: str
    model_id: str
    base_glb_url: str
    match_type: str
    metrics: dict  # shape_score 등 상세 지표
    guide_video_url: Optional[str] = None # RAG 결과 반영
    artifacts: Optional[dict] = None     # 디버그 이미지 경로 등