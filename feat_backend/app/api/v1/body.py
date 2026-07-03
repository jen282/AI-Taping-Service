from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
import uuid
from app.schemas.body import BodyMatchResponse
from app.services.cv_module import body_analyzer
from app.services.db_manager import db
from app.services.storage_manager import upload_file_to_blob 

body_router = APIRouter()

@body_router.post("/match", response_model=BodyMatchResponse)
async def match_body(
    session_id: str = Form(...),
    image: UploadFile = File(...),
    height_cm: Optional[float] = Form(None),
    weight_kg: Optional[float] = Form(None),
    gender: Optional[str] = Form(None)
):
    try:
        actual_gender = gender if gender else "male"
        print(f"[LOG] 체형 분석 요청 수신. Session: {session_id}")

        # 1. 파일 업로드 (Blob Storage - user-uploads 컨테이너 사용 권장)
        # 파일명을 유니크하게 생성하여 저장
        file_extension = image.filename.split('.')[-1]
        blob_name = f"{session_id}_{uuid.uuid4().hex}.{file_extension}"
        
        # 실제 업로드 로직 실행
        blob_url = await upload_file_to_blob(image, blob_name)
        print(f"[LOG] 이미지 업로드 완료: {blob_url}")

        # 2. CV 모듈 분석 (Blob URL을 전달)
        analysis_result = await body_analyzer.analyze_image(
            blob_url=blob_url, 
            height=height_cm, 
            weight=weight_kg
        )

        # 3. DB 세션 업데이트 (PATCH)
        # 분석 결과와 모델 정보를 Sessions 컨테이너에 저장
        session_update_data = {
            "session_id": session_id,
            "status": "SCENE_3_COMPLETED",
            "body_match_result": {
                "model_id": "M0234", # 추후 분석 결과에 따라 동적 변경
                "metrics": analysis_result["metrics"],
                "shape_score": analysis_result["shape_score"]
            }
        }
        
        # db.update_item 로직이 있다면 사용, 없으면 create_item으로 대체 가능
        # db.session_container.upsert_item(session_update_data) 

        return BodyMatchResponse(
            session_id=session_id,
            status="SCENE_3_COMPLETED",
            model_id="M0234",
            glb_url="https://[YOUR_STORAGE].blob.core.windows.net/models/M0234_base.glb",
            match_type="cv_analyzed",
            metrics=analysis_result["metrics"]
        )

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"체형 분석 실패: {str(e)}")