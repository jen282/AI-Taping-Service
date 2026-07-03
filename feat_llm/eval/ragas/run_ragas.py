"""
RAGAS 평가 실행 스크립트

흐름:
  1. generated.json 로드
  2. 각 한국어 질문을 실제 RAG 파이프라인에 투입
     - 한국어 → 영어 번역
     - Azure AI Search 검색 (실제 컨텍스트 수집)
     - LLM 응답 생성
  3. RAGAS 메트릭 계산
     - faithfulness       : 응답이 검색 문서에 근거했는가
     - answer_relevancy   : 응답이 질문에 관련 있는가
     - context_precision  : 검색된 문서 중 관련 있는 비율
     - context_recall     : 정답에 필요한 문서가 검색됐는가
  4. 결과를 eval/results/ragas/ 에 저장

실행 방법:
  cd feat_llm
  python eval/ragas/run_ragas.py
"""

import sys
import io
import json
from datetime import datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EVAL_DIR    = Path(__file__).parent.parent
SCRIPTS_DIR = EVAL_DIR.parent / "scripts"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from dotenv import load_dotenv
load_dotenv(dotenv_path=EVAL_DIR.parent.parent / ".env")

from config import get_ragas_llm, get_ragas_embeddings, DATASETS_DIR, RESULTS_DIR
from resource_config import setup_global_llm_and_embedding
from llm import TapingRAGSystem

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

DATASET_PATH = DATASETS_DIR / "synthetic" / "generated.json"
INDEX_NAME   = "taping-guide-index"


def load_dataset():
    """generated.json 로드."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        records = json.load(f)
    print(f"[1/4] 데이터셋 로드: {len(records)}개 Q&A")
    return records


def run_rag_pipeline(records, rag: TapingRAGSystem):
    """각 질문에 대해 실제 RAG 파이프라인 실행 후 결과 수집."""
    print(f"\n[2/4] RAG 파이프라인 실행 ({len(records)}개 질문)")

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, record in enumerate(records):
        q_ko = record["question"]
        gt   = record.get("ground_truth", "")
        print(f"\n  [{i+1}/{len(records)}] {q_ko}")

        try:
            # 1) 영어 번역
            e_query = rag.translate_to_english(q_ko)

            # 2) 검색 (실제 컨텍스트 수집)
            retriever = rag._build_retriever()
            nodes     = retriever.retrieve(e_query)
            ctx_texts = [node.get_content() for node in nodes]

            # 3) LLM 응답 생성 → text 필드(자연어)만 평가에 사용
            response_dict = rag.get_structured_response(q_ko)
            answer_text   = response_dict.get("text", "")

            questions.append(q_ko)
            answers.append(answer_text)
            contexts.append(ctx_texts)
            ground_truths.append(gt)
            print(f"       컨텍스트 {len(ctx_texts)}개 검색 완료")

        except Exception as e:
            print(f"       [ERROR] 건너뜀: {e}")

    return questions, answers, contexts, ground_truths


def run_evaluation(questions, answers, contexts, ground_truths, ragas_llm, ragas_embeddings):
    """RAGAS 메트릭 계산."""
    print(f"\n[3/4] RAGAS 메트릭 계산 ({len(questions)}개)")

    dataset = Dataset.from_dict({
        "question":      questions,
        "answer":        answers,
        "contexts":      contexts,
        "ground_truth":  ground_truths,
    })

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm        = ragas_llm
        m.embeddings = ragas_embeddings

    result = evaluate(dataset=dataset, metrics=metrics)
    return result


def save_results(result, questions, answers):
    """결과를 JSON으로 저장."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = RESULTS_DIR / f"{timestamp}.json"

    df          = result.to_pandas()
    metric_cols = list(df.select_dtypes(include="number").columns)

    # 전체 평균 점수
    scores = {col: float(df[col].mean()) for col in metric_cols}

    # 질문별 개별 점수
    details = []
    for i, (q, a) in enumerate(zip(questions, answers)):
        row = {"question": q, "answer": a}
        if i < len(df):
            for col in metric_cols:
                row[col] = float(df.loc[i, col]) if not df.loc[i, col] != df.loc[i, col] else None
        details.append(row)

    output = {
        "timestamp": timestamp,
        "scores":    scores,
        "details":   details,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[4/4] 결과 저장: {output_path}")
    return scores


def main():
    print("=" * 55)
    print("RAGAS 평가 실행")
    print("=" * 55)

    setup_global_llm_and_embedding()
    ragas_llm        = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

    records = load_dataset()
    rag     = TapingRAGSystem(index_name=INDEX_NAME)

    questions, answers, contexts, ground_truths = run_rag_pipeline(records, rag)
    result = run_evaluation(questions, answers, contexts, ground_truths, ragas_llm, ragas_embeddings)
    scores = save_results(result, questions, answers)

    print("\n" + "=" * 55)
    print("평가 결과")
    print("=" * 55)
    for metric, score in scores.items():
        bar = "#" * int(score * 20)
        print(f"  {metric:<22} {score:.4f}  [{bar:<20}]")
    print()


if __name__ == "__main__":
    main()
