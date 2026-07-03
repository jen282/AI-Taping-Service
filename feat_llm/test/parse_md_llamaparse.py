import os
import sys
from dotenv import load_dotenv
from llama_parse import LlamaParse

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

def parse_document_to_md(input_pdf_path, output_md_path):
  print(f"[진행 중] LlamaParse로 파싱 시작: {input_pdf_path}")
  # 기획 문서에서 검증된 시스템 프롬프트
  system_prompt = """
  이 문서를 파싱하여 모든 내용을 원문 영어 그대로 추출하세요. 절대 내용을 요약하거나 삭제하지 마세요.

  ### 문서 구조 규칙:
  문서의 전체 흐름을 그대로 보존해야 합니다. 파싱된 콘텐츠를 다음 형식으로 분류하세요:

  1. 서술 및 설명 (Narratives & Explanations):
    모든 배경 정보, 원인, 해부학적 설명 및 일반 텍스트를 표준 마크다운(Markdown) 문단으로 유지하세요.

  2. 그림 캡션 (Figure Captions - 매우 중요):
    이미지나 그림 설명을 절대 삭제하지 마세요. 그림 참조 또는 캡션(예: "Figure 3.1...", "Fig 1...")이 보이면 반드시 다음과 같이 인용구(blockquote) 형식으로 작성하세요:
    > **Figure [번호]**: [캡션 내용 원문 그대로]

  3. 테이핑 방법 및 단계 (Taping Methods & Steps):
    테이핑 기법에 대한 실제 단계별 적용 지침이 나올 때만, 해당 단계들을 다음 헤더를 가진 마크다운 표로 변환하세요: [Step, Application Path, Position, Stretch %].
    하나의 기법에 대한 초기 세로 스트립과 후속 교차/X 스트립 단계를 모두 하나의 표로 통합하세요.

  ### 포맷 규칙:
  - 제목: 주요 섹션에는 "## [대상 부위 / 질환명]"을 사용하세요.
  - 전문 용어: 모든 전문 의학 및 해부학 용어는 원문에 나타난 그대로 보존하세요.
  - 환각 방지: 본문에 명시적으로 언급된 stretch % 값만 사용하세요. "no stretch"라고 언급된 경우 "0%"로 기록하세요.

  ### 최종 목표:
  원본 컨텍스트를 단 한 방울도 잃지 않고 모든 서술 내용, 모든 그림 인용구, 그리고 깔끔하게 표로 정리된 절차 단계가 포함된 포괄적인 마크다운 문서를 생성하는 것입니다.
    """

  parser = LlamaParse(
    result_type="markdown",
    system_prompt=system_prompt,
    language='en'
  )

  documents = parser.load_data(input_pdf_path)

  if not documents:
    print('[에러] 파싱할 문서가 없습니다.')
    return
  
  md_content = documents[0].text
  
  with open(output_md_path, 'w', encoding='utf-8') as f:
    f.write(md_content)
  
  print(f"[완료] 마크다운 저장됨: {output_md_path}")

def process_books(input_dir, output_dir):
  supported_extensions = ['.pdf', '.epub']

  for filename in os.listdir(input_dir):
    format = os.path.splitext(filename)[1].lower()
    print(f"\n[진행 중] 파일 처리 시작: {filename}")
    input_path = os.path.join(input_dir, filename)
    output_filename = os.path.splitext(filename)[0] + '.md'
    output_path = os.path.join(output_dir, output_filename)

    parse_document_to_md(input_path, output_path)


if __name__ == "__main__":
  # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  # INPUT_DIR = os.path.join(BASE_DIR, "data", "raw_text")
  # OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed_text")

  # 테스트용 
  BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  INPUT_DIR = os.path.join(BASE_DIR, "test", "data")
  OUTPUT_DIR = os.path.join(BASE_DIR, "test", "data")

  print("시작")
  process_books(INPUT_DIR, OUTPUT_DIR)
  print("끝")