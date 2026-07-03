from fastapi import APIRouter, HTTPException
import uuid
from datetime import datetime, timezone
from app.schemas.symptoms import SymptomRequest, SymptomResponse

# 방금 만든 db 매니저를 불러옵니다!
from app.services.db_manager import db 

symptom_router = APIRouter()

@symptom_router.post("/analyze", response_model=SymptomResponse)
async def analyze_symptoms(request: SymptomRequest):
    try:
        new_session_id = f"sess_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"
        
        # 1. Cosmos DB에 넣을 세션 데이터(JSON) 구조 만들기
        session_document = {
            "id": new_session_id,            # 필수 키
            "session_id": new_session_id,    # 파티션 키
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "SCENE_1_COMPLETED",
            "symptom_info": {
                "body_part": request.body_part,
                "situation": request.situation,
                "raw_text": request.raw_text
            },
            # 추후 Scene 2, 3에서 채워질 빈 공간
            "physical_info": {},
            "body_match_result": {}
        }

        # 2. 진짜로 DB에 밀어 넣기!
        db.create_session(session_document)
        print(f"[LOG] DB 저장 완료. Session: {new_session_id}")
        
        return SymptomResponse(
            session_id=new_session_id,
            status="SCENE_1_COMPLETED",
            message="증상 데이터 수신 및 DB 저장이 완료되었습니다."
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 저장 중 오류 발생: {str(e)}")