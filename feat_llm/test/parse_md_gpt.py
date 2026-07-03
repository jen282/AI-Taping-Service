import os
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")

def convert_txt_to_md_azure(input_txt_path, output_md_path):
    """
    Azure OpenAI를 사용하여 TXT 파일을 구조화된 마크다운으로 변환합니다.
    """
    print(f"\n[진행 중] 파일 읽기: {input_txt_path}")
    
    # 1. 원본 텍스트 로드
    with open(input_txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 2. Azure OpenAI 클라이언트 초기화
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") # 예: https://your-resource.openai.azure.com/
    )

    # 3. 시스템 프롬프트 (구조 보존 및 표 생성 로직)
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

    print("[진행 중] Azure OpenAI 가공 중... (GPT-4o)")

    # 4. API 호출
    try:
        response = client.chat.completions.create(
            model="gpt-4.1", # Azure Portal에서 설정한 Deployment Name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Structure this text:\n\n{raw_text}"}
            ],
            temperature=0, # 일관성을 위해 무작위성 제거
        )
        
        md_result = response.choices[0].message.content

        # 5. 결과 저장
        os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(md_result)
            
        print(f"[완료] 저장된 파일: {output_md_path}")

    except Exception as e:
        print(f"[에러 발생] {e}")

if __name__ == "__main__":
    # 경로 설정 (사용자 환경에 맞게 수정)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    INPUT_DIR = os.path.join(BASE_DIR, "test", "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "test", "data")
    # 폴더 내 모든 TXT 파일 처리
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".txt"):
            input_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename.replace(".txt", ".md"))
            convert_txt_to_md_azure(input_path, output_path)