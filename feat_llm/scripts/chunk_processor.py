import re
from pathlib import Path
from typing import List, Dict

def extract_metadata_from_filename(file_path: Path) -> Dict[str, str]:
    """파일명에서 기본 메타데이터 추출 (예: book1_gibbons_ch03_knee.md)"""
    parts = file_path.stem.split('_')
    return {
        "source": f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else file_path.stem,
        "body_part": parts[-1] if len(parts) > 0 else "unknown",
        "chapter": parts[2] if len(parts) >= 3 else "unknown"
    }

def extract_stretch_range(text: str) -> List[int]:
    """텍스트 내에서 텐션(%) 수치를 찾아 [min, max] 형태로 반환.
    % 기호 앞에 붙은 1~3자리 숫자만 대상으로 하며, 값은 0~150으로 클램핑.
    """
    # '%' 앞의 1~3자리 숫자만 추출 (ISBN 등 긴 숫자 제외)
    numbers = re.findall(r'\b(\d{1,3})\s*%', text)
    if not numbers:
        return [0, 0]
    nums = [max(0, min(150, int(n))) for n in numbers]
    return [min(nums), max(nums)]

_TAPE_PATTERNS: List[tuple] = [
    # (label, pattern) — small "I" strip은 반드시 "I" strip보다 앞에 위치
    ("X-strip",       r'["\u201c\u2018]X["\u201d\u2019][\s-]?(?:strip|shape|shaped?)?'
                      r'|\bx[\s-]?(?:strip|shaped?)\b'),
    ("Big-Daddy",     r'\bbig[\s-]?daddy\b'),
    ("Y-strip",       r'["\u201c\u2018]Y["\u201d\u2019][\s-]?strip'
                      r'|\by[\s-]?(?:strip|shaped?|tape|cut)\b'),
    ("Small-I-strip", r'small\s+["\u201c\u2018]I["\u201d\u2019][\s-]?strip'
                      r'|\bsmall[\s-]?i[\s-]?strip\b'),
    ("I-strip",       r'["\u201c\u2018]I["\u201d\u2019][\s-]?strip'
                      r'|\bi[\s-]?strip\b|\bsingle[\s-]?strip\b|\bstraight[\s-]?strip\b'),
]


def extract_tape_types(text: str) -> List[str]:
    """텍스트에 등장하는 모든 테이프 타입을 순서대로(중복 없이) 반환.

    본문 표기 기준:
      "X" shape       → X-strip
      Big Daddy       → Big-Daddy
      "Y" strip       → Y-strip
      small "I" strip → Small-I-strip
      "I" strip       → I-strip
    """
    found = []
    for label, pattern in _TAPE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE) and label not in found:
            found.append(label)
    return found if found else ["unknown"]

