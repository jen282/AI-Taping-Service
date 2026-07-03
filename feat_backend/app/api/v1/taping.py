from fastapi import APIRouter, HTTPException
from app.schemas.taping import TapingRequest, TapingResponse, TapingOption

taping_router = APIRouter()

@taping_router.post("/recommend", response_model=TapingResponse)
async def recommend_taping(request: TapingRequest):
    try:
        print(f"[LOG] 테이핑 추천 요청 수신. Session: {request.session_id}")
        
        # TODO: Cosmos DB에서 세션 조회 -> Azure AI Search(RAG) -> GPT-4o 추론

        # 테이핑 레지스트리(CSV)와 완벽하게 싱크를 맞춘 Mock 데이터 반환
        mock_option = TapingOption(
            rank=1,
            registry_key="REG_KNEE_001",
            technique_code="generic_knee_y",
            body_part="knee_generic",
            region="knee_lateral",
            laterality="right",
            tape_type="Y-strip",
            asset_id="knee_y_strip_01",
            guide_video_url="https://[YOUR_STORAGE].blob.core.windows.net/videos/y_strip_guide.mp4",
            combined_glb_url="https://[YOUR_STORAGE].blob.core.windows.net/models/M0234_knee_generic_right_generic_knee_y.glb",
            instruction="장경인대를 따라 Y자 형태로 가볍게 텐션을 주어 부착합니다."
        )

        return TapingResponse(
            session_id=request.session_id,
            status="SCENE_6_COMPLETED",
            analysis="입력하신 증상과 체형을 바탕으로 장경인대 이완에 최적화된 테이핑을 제안합니다.",
            options=[mock_option]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))