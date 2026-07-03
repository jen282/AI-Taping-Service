import json
import sys
from typing import Any, Dict

sys.stdout.reconfigure(encoding='utf-8')

from resource_config import get_azure_openai_client

STRUCTURE_SYMPTOM_PROMPT = """\
You are a kinesiology taping assistant that converts a user's symptom selection into a structured clinical description.

Given the following input fields:
- body_part: the body part (PoC: "knee" only)
- situation: when the symptom occurs (before_exercise | during_exercise | after_exercise | daily | other)
- symptom_type: type of symptom (stiffness | anterior_pain | lateral_pain | instability | prevention | custom)
- user_text: free-text description, non-null only when symptom_type is "custom"

Produce a JSON object with a single key "structured_symptom" containing:
- area (str): anatomical sub-region (e.g. "lateral_knee", "anterior_knee", "patella", "medial_knee", "general_knee")
- keywords (list[str]): 2–5 relevant clinical/kinesiology terms in English (e.g. muscle names, syndrome names, taping techniques)
- summary (str): one-sentence Korean summary of the likely issue and taping rationale (25–40 chars)

Rules:
- Output ONLY valid JSON. No markdown, no extra keys.
- If symptom_type is "custom", incorporate user_text into area, keywords, and summary.
- If symptom_type is "prevention", keywords should focus on prophylactic taping.
- Keywords must be in English; summary must be in Korean.

Symptom-to-area mapping guide (use as reference, not strict rule):
  lateral_pain    → lateral_knee  (IT band, fibular collateral ligament)
  anterior_pain   → anterior_knee (patella, quadriceps, patellar tendon)
  stiffness       → general_knee  (joint capsule, hamstring, calf)
  instability     → general_knee  (MCL, LCL, ACL support)
  prevention      → general_knee  (prophylactic support)

Output schema:
{
  "structured_symptom": {
    "area": "<anatomical sub-region>",
    "keywords": ["<term1>", "<term2>", ...],
    "summary": "<Korean sentence>"
  }
}
"""


class SymptomStructurer:
    def __init__(self):
        self._client = get_azure_openai_client()

    def structure(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        입력 증상 선택값을 구조화된 임상 정보로 변환.

        input 기대 키:
            body_part    (str): "knee"
            situation    (str): before_exercise | during_exercise | after_exercise | daily | other
            symptom_type (str): stiffness | anterior_pain | lateral_pain | instability | prevention | custom
            user_text    (str | None): custom일 때만 non-null
        """
        user_message = json.dumps(input, ensure_ascii=False)

        response = self._client.chat.completions.create(
            model="gpt-4.1",
            response_format={"type": "json_object"},
            temperature=0.1,
            messages=[
                {"role": "system", "content": STRUCTURE_SYMPTOM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        raw = response.choices[0].message.content.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM JSON 파싱 실패: {e}\n원문: {raw}")

        if "structured_symptom" not in result:
            raise ValueError(f"응답에 'structured_symptom' 키 없음: {result}")

        return result


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path

    load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

    structurer = SymptomStructurer()

    # symptom_type 케이스별 테스트
    test_cases = [
        # 1) lateral_pain — 기본 케이스
        {
            "body_part": "knee",
            "situation": "after_exercise",
            "symptom_type": "lateral_pain",
            "user_text": None,
        },
        # 2) anterior_pain
        {
            "body_part": "knee",
            "situation": "during_exercise",
            "symptom_type": "anterior_pain",
            "user_text": None,
        },
        # 3) stiffness
        {
            "body_part": "knee",
            "situation": "daily",
            "symptom_type": "stiffness",
            "user_text": None,
        },
        # 4) instability
        {
            "body_part": "knee",
            "situation": "before_exercise",
            "symptom_type": "instability",
            "user_text": None,
        },
        # 5) prevention
        {
            "body_part": "knee",
            "situation": "before_exercise",
            "symptom_type": "prevention",
            "user_text": None,
        },
        # 6) custom — user_text non-null
        {
            "body_part": "knee",
            "situation": "during_exercise",
            "symptom_type": "custom",
            "user_text": "무릎 안쪽이 접질릴 것 같은 느낌이 들어요",
        },
    ]

    PASS = 0
    FAIL = 0

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"[테스트 {i}] symptom_type={case['symptom_type']}")
        print(f"입력: {json.dumps(case, ensure_ascii=False)}")
        try:
            result = structurer.structure(case)
            ss = result["structured_symptom"]
            assert isinstance(ss.get("area"), str) and ss["area"], "area 누락"
            assert isinstance(ss.get("keywords"), list) and 2 <= len(ss["keywords"]) <= 5, "keywords 범위 오류"
            assert isinstance(ss.get("summary"), str) and ss["summary"], "summary 누락"
            print(f"출력: {json.dumps(result, ensure_ascii=False, indent=2)}")
            print(f"[PASS]")
            PASS += 1
        except Exception as e:
            print(f"[FAIL] {e}")
            FAIL += 1

    print(f"\n{'='*50}")
    print(f"결과: {PASS} 통과 / {FAIL} 실패")
