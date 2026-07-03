from pydantic import BaseModel
from typing import List

class TapingRequest(BaseModel):
    session_id: str

class TapingOption(BaseModel):
    rank: int
    registry_key: str
    technique_code: str
    body_part: str
    region: str
    laterality: str
    tape_type: str
    asset_id: str
    guide_video_url: str 
    combined_glb_url: str
    instruction: str

class TapingResponse(BaseModel):
    session_id: str
    status: str
    analysis: str
    options: List[TapingOption]