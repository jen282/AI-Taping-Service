"""
parse_elements.py — STEP 1: element-type 파싱 (txt 기반)

Input:  data/raw_text/*.txt  (fetch_raw_data.py로 다운로드한 파일)
Output: data/processed/<book>/<stem>_elements.json
        book1_gibbons / book2_kim 은 파일명 prefix로 자동 결정

Usage:
    python parse_elements.py
"""

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# 패턴 상수
# ---------------------------------------------------------------------------

RE_CHAPTER_TITLE = re.compile(r"^C\s+H\s+A\s+P\s+T\s+E\s+R\s+\d+", re.IGNORECASE)
RE_FIGURE_CAPTION = re.compile(r"^Figure\s+\d+\.\d+", re.IGNORECASE)
RE_STEP = re.compile(r"^(\d+)\.\t\s*(.*)|^(\d+)\.\s{2,}(.*)|^(\d+)\.\s*$")
RE_NOISE = re.compile(r"^[SFab]$|^---+$|^#")  # S/F markers, markdown hr, markdown headings

# label 판별용: 동사 패턴
RE_VERB = re.compile(
    r"\b(is|are|was|were|has|have|had|do|does|did"
    r"|apply|ask|place|use|attach|wrap|ensure|allow|cross"
    r"|repeat|start|finish|secure|position|flex|extend"
    r"|stretch|pull|hold|bring|keep|make|take|move|turn"
    r"|applied|asked|placed|used|attached|wrapped|secured"
    r"|positioned|flexed|extended|stretched|pulled)\b",
    re.IGNORECASE,
)

# params 추출용
RE_TAPE_TYPE = re.compile(r'"([YIXx])"\s*strip|Big Daddy', re.IGNORECASE)
RE_STRETCH = re.compile(r"(\d+)[–\-](\d+)%|(\d+)%|no[\s-]stretch", re.IGNORECASE)
RE_POSITION = re.compile(
    r"long sitting|on (?:their|his|her) side|prone position|supine"
    r"|standing|side[- ]lying|knee (?:straight|bent|flexed|at \d+°)",
    re.IGNORECASE,
)
RE_ANCHOR = re.compile(
    r"(?:starting )?from (.+?) (?:finishing at|to) (.+?)(?:\.|,|$)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 줄 단위 판별 함수
# ---------------------------------------------------------------------------

def _is_chapter_title(line: str) -> bool:
    return bool(RE_CHAPTER_TITLE.match(line.strip()))


def _is_figure_caption(line: str) -> bool:
    return bool(RE_FIGURE_CAPTION.match(line.strip()))


def _is_noise(line: str) -> bool:
    return bool(RE_NOISE.match(line.strip()))


def _is_all_caps(line: str) -> bool:
    """section_header 판별: 소문자 없고 알파벳 포함, 2자 이상, 괄호 약어 제외"""
    s = line.strip()
    if s.startswith("("):  # (PFPS) 같은 괄호 약어는 헤더가 아님
        return False
    return bool(s) and s == s.upper() and any(c.isalpha() for c in s) and len(s) >= 2


def _is_step_start(line: str) -> bool:
    return bool(RE_STEP.match(line))


def _is_subsection(lines: list[str]) -> bool:
    """
    단일 줄 블록이 subsection인지 판별.
    조건: 1줄, 2~5단어, 첫 글자 대문자 + 소문자 포함, 마침표 없음, 대문자만은 아님.
    """
    if len(lines) != 1:
        return False
    s = lines[0].strip()
    if s.endswith(".") or s == s.upper():
        return False
    words = s.split()
    if not (2 <= len(words) <= 5):
        return False
    return s[0].isupper() and any(c.islower() for c in s)


def _classify_text_block(lines: list[str]) -> str:
    """여러 줄 텍스트 블록을 subsection / label / narrative 로 분류"""
    if _is_subsection(lines):
        return "subsection"

    # label: ≤4줄, 줄당 평균 25자 이하, 동사 없음
    if len(lines) <= 4:
        avg_len = sum(len(l.strip()) for l in lines) / len(lines)
        text = " ".join(lines)
        if avg_len <= 25 and not RE_VERB.search(text):
            return "label"

    return "narrative"


# ---------------------------------------------------------------------------
# params 추출
# ---------------------------------------------------------------------------

def _extract_params(content: str) -> dict:
    params: dict = {}

    tape_m = RE_TAPE_TYPE.search(content)
    if tape_m:
        params["tape_type"] = (
            "Big Daddy" if tape_m.group(0).lower().startswith("big")
            else f'{tape_m.group(1).upper()}-strip'
        )

    stretch_m = RE_STRETCH.search(content)
    if stretch_m:
        if stretch_m.group(0).lower().startswith("no"):
            params["stretch_pct"] = "0"
        elif stretch_m.group(1):
            params["stretch_pct"] = f'{stretch_m.group(1)}~{stretch_m.group(2)}'
        else:
            params["stretch_pct"] = stretch_m.group(3)

    pos_m = RE_POSITION.search(content)
    if pos_m:
        params["position"] = pos_m.group(0)

    anchor_m = RE_ANCHOR.search(content)
    if anchor_m:
        params["anchor"] = f'{anchor_m.group(1).strip()} → {anchor_m.group(2).strip()}'

    return params


# ---------------------------------------------------------------------------
# 핵심 파서
# ---------------------------------------------------------------------------

def _book_name(txt_path: str) -> str:
    """파일명 prefix에서 book 식별자 추출 (book1_gibbons / book2_kim)"""
    stem = Path(txt_path).stem
    if stem.startswith("book2"):
        return "book2_kim"
    return "book1_gibbons"


def parse_txt(txt_path: str) -> list[dict]:
    """
    txt 파일을 줄 단위로 파싱하여 element dict 리스트 반환.

    처리 우선순위 (위에서 아래로):
      noise → chapter_title → figure_caption → section_header → step → 나머지(subsection/label/narrative)

    멀티라인 처리:
      - section_header: 연속 대문자 줄을 하나로 합침
      - step: 다음 step 시작 / section_header / figure_caption 전까지 누적
      - 나머지: 빈 줄로 단락 구분 후 블록 단위로 타입 결정
    """
    text = Path(txt_path).read_text(encoding="utf-8", errors="replace")
    raw_lines = text.splitlines()

    elements: list[dict] = []
    current_section = ""
    current_technique = ""
    pending: list[str] = []  # narrative/label/subsection 후보 누적

    book = _book_name(txt_path)

    def make_elem(el_type: str, content: str, **extra) -> dict:
        base = {
            "type": el_type,
            "content": content,
            "section": current_section,
            "technique": current_technique,
            "source": book,
        }
        base.update(extra)
        return base

    def flush_pending() -> None:
        if not pending:
            return
        block_type = _classify_text_block(pending)
        content = " ".join(l.strip() for l in pending if l.strip())
        if content:
            elements.append(make_elem(block_type, content))
        pending.clear()

    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        stripped = line.strip()

        # 빈 줄 → 현재 pending 블록 확정
        if not stripped:
            flush_pending()
            i += 1
            continue

        # noise (S, F, a, b 단독 줄)
        if _is_noise(stripped):
            i += 1
            continue

        # chapter_title
        if _is_chapter_title(stripped):
            flush_pending()
            i += 1
            continue

        # figure_caption: Figure N.N 으로 시작 + 이어지는 줄 누적
        if _is_figure_caption(stripped):
            flush_pending()
            caption_lines = [stripped]
            i += 1
            while i < len(raw_lines):
                ns = raw_lines[i].strip()
                if (not ns or _is_noise(ns) or _is_figure_caption(ns)
                        or _is_step_start(raw_lines[i]) or _is_all_caps(ns)):
                    break
                caption_lines.append(ns)
                i += 1
            elements.append(make_elem("figure_caption", " ".join(caption_lines)))
            continue

        # section_header: 연속 대문자 줄 누적
        if _is_all_caps(stripped):
            flush_pending()
            header_lines = [stripped]
            i += 1
            while i < len(raw_lines):
                ns = raw_lines[i].strip()
                if _is_all_caps(ns):
                    header_lines.append(ns)
                    i += 1
                else:
                    break
            header_text = " ".join(header_lines)
            current_section = header_text
            current_technique = header_text
            elements.append(make_elem("section_header", header_text))
            continue

        # step: 숫자.\t / 숫자.  (공백 2개+) / 숫자. (단독 줄, book2) 로 시작
        step_m = RE_STEP.match(line)
        if step_m:
            flush_pending()
            step_num = int(step_m.group(1) or step_m.group(3) or step_m.group(5))
            first_line = (step_m.group(2) or step_m.group(4) or "").strip()
            step_lines = [first_line] if first_line else []
            i += 1
            while i < len(raw_lines):
                nl = raw_lines[i]
                ns = nl.strip()
                if (_is_step_start(nl) or _is_all_caps(ns)
                        or _is_figure_caption(ns) or _is_noise(ns)):
                    break
                if ns:
                    step_lines.append(ns)
                i += 1
            content = " ".join(step_lines).strip()
            elements.append(make_elem(
                "step", content,
                step_number=step_num,
                params=_extract_params(content),
            ))
            continue

        # 나머지: pending에 누적 (빈 줄 기준으로 나중에 블록 분류)
        pending.append(stripped)
        i += 1

    flush_pending()

    return elements


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir   = Path(__file__).parent
    feat_llm_dir = script_dir.parent
    raw_dir       = feat_llm_dir / "data" / "raw_text"
    processed_dir = feat_llm_dir / "data" / "processed_element"

    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"{raw_dir} 에 .txt 파일이 없습니다. fetch_raw_data.py를 먼저 실행하세요."
        )

    for txt_path in txt_files:
        print(f"[parse_elements] {txt_path.name}")
        elements = parse_txt(str(txt_path))

        book = _book_name(str(txt_path))
        out_dir = processed_dir / book
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{txt_path.stem}_elements.json"
        out_file.write_text(
            json.dumps(elements, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        type_counts: dict[str, int] = {}
        for e in elements:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
        print(f"  total {len(elements)}: {type_counts}")
        if "step" not in type_counts:
            print("  [WARN] no steps found - may not be a chapter file")
        print(f"  -> {out_file}")

    print("[parse_elements] Done.")


if __name__ == "__main__":
    main()
