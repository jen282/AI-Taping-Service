from pydantic import BaseModel, Field

class SymptomRequest(BaseModel):
    body_part: str = Field(..., example="knee")
    situation: str = Field(..., example="after_exercise")
    raw_text: str = Field(..., example="달리기 후 무릎 바깥쪽이 아파요")

class SymptomResponse(BaseModel):
    session_id: str
    status: str
    message: str