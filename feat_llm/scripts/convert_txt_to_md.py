import os
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv
from scripts.resource_config import get_azure_openai_client

# .env 파일 로드 (scripts 폴더 기준으로 상위 폴더의 상위 폴더에 있다고 가정)
load_dotenv(Path(__file__).parent.parent.parent / ".env")

def convert_txt_to_md_azure(client, input_txt_path, output_md_path):
    """
    Azure OpenAI를 사용하여 TXT 파일을 구조화된 마크다운으로 변환합니다.
    """
    print(f"\n[진행 중] 파일 읽기: {input_txt_path.name}")
    
    with open(input_txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    system_prompt = """
    You are a medical documentation expert. Your task is to convert raw text into a structured Markdown document.
    Keep the original English text exactly as it is. Do NOT summarize or omit any content.

    ### STRUCTURE RULES:
    1. Narratives: Keep all background, anatomy, and explanations as standard paragraphs.
    2. Figure Captions: Format any figure references as blockquotes:
       > **Figure [Number]**: [Caption text]
    3. Taping Procedures: ONLY for step-by-step instructions, use a Markdown Table:
    | Step | Application Path | Position | Stretch % |
    Combine all steps of one technique into a single table.

    ### FORMATTING:
    - Headers: Use "## [Target Area / Condition]" for major sections.
    - Stretch %: Use explicitly stated values; if "no stretch", use "0%".
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Structure this text:\n\n{raw_text}"}
            ],
            temperature=0,
        )
        
        md_result = response.choices[0].message.content

        # 결과 저장
        os.makedirs(output_md_path.parent, exist_ok=True)
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_result)
            
        print(f"[완료] 저장됨: {output_md_path.name}")

    except Exception as e:
        print(f"[에러 발생] {input_txt_path.name}: {e}")

def main():
    # 1. 경로 설정 (이미지 구조 기준)
    SCRIPTS_DIR = Path(__file__).parent
    BASE_DIR = SCRIPTS_DIR.parent
    INPUT_DIR = BASE_DIR / "data" / "raw_text"
    OUTPUT_DIR = BASE_DIR / "data" / "processed_md"

    # 2. Azure OpenAI 클라이언트 초기화 (루프 밖에서 한 번만 수행)
    client = get_azure_openai_client()

    # 3. 모든 TXT 파일 처리
    txt_files = list(INPUT_DIR.glob("*.txt"))
    
    if not txt_files:
        print(f"처리할 파일이 없습니다. 경로를 확인하세요: {INPUT_DIR}")
        return

    print(f"총 {len(txt_files)}개의 파일을 변환합니다...")

    for input_path in txt_files:
        # 출력 파일명을 .md로 변경
        output_path = OUTPUT_DIR / input_path.with_suffix(".md").name
        convert_txt_to_md_azure(client, input_path, output_path)

if __name__ == "__main__":
    main()