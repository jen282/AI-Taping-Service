from fastapi import FastAPI
from app.api.v1.symptoms import symptom_router
from app.api.v1.body import body_router       # 새로 추가
from app.api.v1.taping import taping_router   # 새로 추가

app = FastAPI(title="AI Taping Guide API", version="1.0.0")

# 기존 증상 라우터
app.include_router(symptom_router, prefix="/api/v1/symptoms", tags=["Symptoms"])
# 새로운 라우터 2개 조립!
app.include_router(body_router, prefix="/api/v1/body", tags=["Body Match"])
app.include_router(taping_router, prefix="/api/v1/taping", tags=["Taping"])

@app.get("/")
def read_root():
    return {"message": "AI Taping Guide Backend is running!"}