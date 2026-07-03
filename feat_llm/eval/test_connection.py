"""
RAGAS + Azure OpenAI 연결 확인 스크립트

실행 방법:
  cd feat_llm
  python eval/test_connection.py

확인 항목:
  1. .env 환경 변수 로드
  2. Azure OpenAI LLM 응답
  3. Azure OpenAI 임베딩
  4. RAGAS LLM/Embedding 래핑
  5. RAGAS faithfulness 메트릭 계산 (더미 데이터)
"""

import sys
import io
from pathlib import Path

# Windows cp949 환경에서 유니코드 출력 보장
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# config.py가 있는 eval/ 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config import get_ragas_llm, get_ragas_embeddings, RESULTS_DIR, SCRIPTS_DIR
sys.path.insert(0, str(SCRIPTS_DIR))
from resource_config import setup_global_llm_and_embedding
from ragas.metrics import faithfulness
from ragas import evaluate
from datasets import Dataset


def check_env():
    """1단계: 환경 변수 확인"""
    import os
    required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    print("\n[1/4] 환경 변수 확인")
    all_ok = True
    for key in required:
        val = os.getenv(key)
        if val:
            print(f"  ✓ {key} = {val[:30]}...")
        else:
            print(f"  ✗ {key} 없음 → .env 파일을 확인하세요")
            all_ok = False
    return all_ok


def check_llm(ragas_llm):
    """2단계: Azure OpenAI LLM 응답 확인"""
    print("\n[2/4] Azure OpenAI LLM 연결 확인")
    try:
        response = ragas_llm.llm.complete("Say 'connection ok' in one word.")
        print(f"  ✓ LLM 응답: {response.text.strip()}")
        return True
    except Exception as e:
        print(f"  ✗ LLM 연결 실패: {e}")
        return False


def check_embeddings(ragas_embeddings):
    """3단계: Azure OpenAI 임베딩 확인"""
    print("\n[3/4] Azure OpenAI 임베딩 연결 확인")
    try:
        vector = ragas_embeddings.embeddings.get_text_embedding("test")
        print(f"  ✓ 임베딩 차원: {len(vector)}")
        return True
    except Exception as e:
        print(f"  ✗ 임베딩 연결 실패: {e}")
        return False


def check_ragas_metric(ragas_llm, ragas_embeddings):
    """4단계: RAGAS faithfulness 메트릭 계산 확인 (더미 데이터)"""
    print("\n[4/4] RAGAS faithfulness 메트릭 계산 확인")
    try:
        # 더미 데이터: 무릎 테이핑 관련 예시
        dummy_data = Dataset.from_dict({
            "question": [
                "무릎 바깥쪽 통증에 사용하는 테이핑 기법은?"
            ],
            "answer": [
                "IT 밴드 마찰 증후군에는 I-스트립을 사용하여 슬개골 외측에서 대퇴부 방향으로 붙입니다."
            ],
            "contexts": [[
                "Iliotibial band syndrome is treated with an I-strip applied laterally from the patella "
                "toward the thigh. Apply with 25-50% stretch tension."
            ]],
        })

        faithfulness.llm = ragas_llm
        faithfulness.embeddings = ragas_embeddings

        result = evaluate(dataset=dummy_data, metrics=[faithfulness])
        score = result["faithfulness"]
        # RAGAS 버전에 따라 float 또는 list 반환
        if isinstance(score, list):
            score = sum(score) / len(score)
        print(f"  OK faithfulness 점수: {score:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ RAGAS 메트릭 계산 실패: {e}")
        return False


def main():
    print("=" * 50)
    print("RAGAS + Azure OpenAI 연결 확인")
    print("=" * 50)

    results = {}

    results["env"] = check_env()
    if not results["env"]:
        print("\n환경 변수 설정 후 다시 실행하세요.")
        sys.exit(1)

    setup_global_llm_and_embedding()  # 한 번만 초기화
    ragas_llm        = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

    results["llm"]        = check_llm(ragas_llm)
    results["embeddings"] = check_embeddings(ragas_embeddings)
    results["ragas"]      = check_ragas_metric(ragas_llm, ragas_embeddings)

    print("\n" + "=" * 50)
    print("결과 요약")
    print("=" * 50)
    labels = {
        "env":        "환경 변수",
        "llm":        "Azure LLM",
        "embeddings": "Azure 임베딩",
        "ragas":      "RAGAS 메트릭",
    }
    all_pass = all(results.values())
    for key, ok in results.items():
        mark = "✓" if ok else "✗"
        print(f"  {mark} {labels[key]}")

    print()
    if all_pass:
        print("모든 연결 확인 완료. 평가 파이프라인 구현을 시작할 수 있습니다.")
    else:
        print("일부 항목이 실패했습니다. 위 오류 메시지를 확인하세요.")


if __name__ == "__main__":
    main()
