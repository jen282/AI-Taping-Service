"""
RAGAS TestsetGenerator로 골든 데이터셋 자동 생성

흐름:
  1. processed_md/ 에서 실제 테이핑 챕터 문서 로드
  2. TestsetGenerator로 영어 Q&A 생성
  3. 생성된 질문을 GPT-4.1로 한국어 번역
  4. eval/datasets/synthetic/generated.json 저장

실행 방법:
  cd feat_llm
  python eval/ragas/generate_dataset.py
"""

import sys
import io
import json
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# eval/, scripts/ 경로 추가
EVAL_DIR    = Path(__file__).parent.parent
SCRIPTS_DIR = EVAL_DIR.parent / "scripts"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from config import get_ragas_llm, get_ragas_embeddings, DATASETS_DIR
from resource_config import setup_global_llm_and_embedding
from llama_index.core import Settings, SimpleDirectoryReader
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona

# ch03 (무릎 챕터)만 사용
CHAPTER_PATTERNS = [
    "book1_gibbons_ch03_knee.md",
]

PROCESSED_MD_DIR = EVAL_DIR.parent / "data" / "processed_md"
OUTPUT_PATH      = DATASETS_DIR / "synthetic" / "generated.json"


def load_chapter_docs():
    """평가에 사용할 챕터 문서만 로드."""
    files = []
    for pattern in CHAPTER_PATTERNS:
        files.extend(PROCESSED_MD_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(f"문서를 찾을 수 없습니다: {PROCESSED_MD_DIR}")

    print(f"[1/4] 문서 로드: {len(files)}개 챕터")
    for f in sorted(files):
        print(f"       - {f.name}")

    docs = SimpleDirectoryReader(input_files=[str(f) for f in sorted(files)]).load_data()
    print(f"       총 {len(docs)}개 노드 로드 완료")
    return docs


def generate_testset(docs, ragas_llm, ragas_embeddings, testset_size: int = 20):
    """RAGAS TestsetGenerator로 영어 Q&A 생성."""
    print(f"\n[2/4] TestsetGenerator 실행 (목표: {testset_size}개)")

    # 의학 비전문가 일반 사용자 페르소나 설정
    # → 어려운 의학 용어 없이 일상적인 언어로 질문 생성 유도
    persona = Persona(
        name="general_user",
        role_description=(
            "A person with no medical background describing their physical discomfort "
            "in very simple, casual language — like they are texting a friend. "
            "Express symptoms as short, everyday phrases. "
            "Examples of the style to follow: "
            "'My outer knee hurts', "
            "'My knee feels stiff and sore', "
            "'The front of my knee aches', "
            "'My knee feels unstable', "
            "'I want to prevent knee injury'. "
            "Never use medical terminology. Keep questions under 10 words."
        ),
    )

    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
        persona_list=[persona],
    )
    testset = generator.generate_with_llamaindex_docs(
        docs,
        testset_size=testset_size,
    )
    df = testset.to_pandas()
    print(f"       생성 완료: {len(df)}개")
    return df


def translate_questions(df, llm):
    """생성된 영어 질문을 한국어로 번역."""
    print(f"\n[3/4] 질문 한국어 번역 ({len(df)}개)")
    korean_questions = []

    korean_ground_truths = []

    for i, row in df.iterrows():
        # 질문 번역
        q_prompt = (
            "Translate the following English symptom phrase into natural, casual Korean "
            "as if a person is describing their discomfort in everyday speech. "
            "Keep it short and conversational. "
            "Output only the Korean translation, nothing else.\n\n"
            f"English: {row['user_input']}\nKorean:"
        )
        korean_q = llm.complete(q_prompt).text.strip()
        korean_questions.append(korean_q)

        # ground_truth 번역
        gt = row.get("reference", "")
        if gt:
            gt_prompt = (
                "You are a medical translator specializing in kinesiology taping. "
                "Translate the following English text into natural Korean. "
                "Output only the Korean translation, nothing else.\n\n"
                f"English: {gt}\nKorean:"
            )
            korean_gt = llm.complete(gt_prompt).text.strip()
        else:
            korean_gt = ""
        korean_ground_truths.append(korean_gt)

        print(f"  [{i+1}/{len(df)}] {row['user_input'][:60]}")
        print(f"         Q  -> {korean_q}")
        print(f"         GT -> {korean_gt[:60]}")

    df["question_ko"]    = korean_questions
    df["ground_truth_ko"] = korean_ground_truths
    return df


def save_dataset(df):
    """평가용 JSON 형식으로 저장."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in df.iterrows():
        records.append({
            "question":     row["question_ko"],          # 한국어 질문 (RAG 입력)
            "question_en":  row["user_input"],            # 영어 원문 (참고용)
            "ground_truth": row.get("ground_truth_ko", ""),  # 한국어 정답
            "ground_truth_en": row.get("reference", ""), # 영어 원문 (참고용)
            "contexts":     row.get("reference_contexts", []),
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n[4/4] 저장 완료: {OUTPUT_PATH}")
    print(f"       총 {len(records)}개 Q&A")
    return records


def main():
    print("=" * 55)
    print("RAGAS 합성 데이터셋 생성")
    print("=" * 55)

    setup_global_llm_and_embedding()
    ragas_llm        = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

    docs = load_chapter_docs()
    df   = generate_testset(docs, ragas_llm, ragas_embeddings, testset_size=10)
    df   = translate_questions(df, Settings.llm)
    save_dataset(df)

    print("\n완료. 다음 단계: python eval/ragas/run_ragas.py")


if __name__ == "__main__":
    main()
