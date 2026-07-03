class BodyAnalyzer:
    def __init__(self):
        # MediaPipe 모델 로드 (초기화)
        print("[LOG] MediaPipe 모델 로딩 완료")

    async def analyze_image(self, blob_url: str, height: float, weight: float):
        """
        Blob Storage에서 사진을 받아 분석 결과를 반환
        """
        # TODO: blob_url을 통해 이미지 다운로드
        # TODO: MediaPipe로 관절 포인트 추출 (landmark detection)
        # TODO: 추출된 좌표로 체형 등급 계산

        # 테스트용 Mock 결과 반환
        return {
            "shape_score": 0.95,
            "metrics": {"shoulder_width": 40.5, "hip_width": 38.2}
        }

# 외부에서 가져다 쓸 수 있도록 인스턴스 생성
body_analyzer = BodyAnalyzer()