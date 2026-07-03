import os
import re
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
from typing import List
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore, IndexManagement
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from resource_config import get_search_client, setup_global_llm_and_embedding

def extract_metadata_from_filename(file_path: Path):
    """파일명에서 기본 메타데이터 추출 (예: book1_gibbons_ch03_knee.md)"""
    parts = file_path.stem.split('_')
    return {
        "source": f"{parts[0]}_{parts[1]}",
        "body_part": parts[-1],  # 마지막 단어를 신체 부위로 간주
        "chapter": parts[2]
    }

def extract_stretch_range(text: str) -> List[int]:
    """텍스트 내에서 텐션(%) 수치를 찾아 [min, max] 형태로 반환"""
    numbers = re.findall(r'(\d+)', text)
    if not numbers:
        return [0, 0]
    nums = [int(n) for n in numbers]
    return [min(nums), max(nums)]

def extract_tape_type(text: str) -> str:
    """텍스트 내에서 테이프 형태를 파싱하여 정규화된 tape_type 반환.

    반환 가능한 값: "Y-strip" | "I-strip" | "X-strip" | "Big-Daddy" | "unknown"

    우선순위: 더 구체적인 패턴(X, Big-Daddy)을 먼저 검사하여 오탐 방지.
    """
    t = text.lower()

    # X-strip: "x-strip", "x strip", "x shaped", "x-shape"
    if re.search(r'\bx[\s-]?strip\b|\bx[\s-]?shaped?\b', t):
        return "X-strip"

    # Big-Daddy / fan strip: "big daddy", "big-daddy", "fan strip", "fan-strip", "wide strip"
    if re.search(r'\bbig[\s-]?daddy\b|\bfan[\s-]?strip\b|\bwide[\s-]?strip\b', t):
        return "Big-Daddy"

    # Y-strip: "y-strip", "y strip", "y shaped", "y-tape", "y-cut"
    if re.search(r'\by[\s-]?strip\b|\by[\s-]?shaped?\b|\by[\s-]?tape\b|\by[\s-]?cut\b', t):
        return "Y-strip"

    # I-strip: "i-strip", "i strip", "single strip", "straight strip"
    if re.search(r'\bi[\s-]?strip\b|\bsingle[\s-]?strip\b|\bstraight[\s-]?strip\b', t):
        return "I-strip"

    return "unknown"

def process_and_index():
    # 1. 경로 설정
    BASE_DIR = Path(__file__).parent.parent
    INPUT_DIR = BASE_DIR / "data" / "processed_md"

    # 2. 모델 설정
    setup_global_llm_and_embedding()
    search_client = get_search_client()
    index_name = "taping-guide-index"  

    # 인덱스를 완전히 비우고 싶다면 아래 주석을 해제하고 실행하세요.
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential
    client = SearchClient(os.getenv("AZURE_AI_SEARCH_ENDPOINT"), index_name, AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_KEY")))
    results = client.search(search_text="*", select="chunk_id")
    ids_to_delete = [{"chunk_id": result["chunk_id"]} for result in results]
    if ids_to_delete: client.delete_documents(documents=ids_to_delete)

    # 3. Azure AI Search 설정
    vector_store = AzureAISearchVectorStore(
        search_or_index_client=search_client,
        index_name=index_name,
        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
        # 1. 필수 키 매핑 (Azure Portal 실제 이름 기준)
        id_field_key="chunk_id",
        chunk_field_key="chunk",
        embedding_field_key="text_vector",
        doc_id_field_key="parent_id",
        metadata_string_field_key="metadata", # ★ 다 밖으로 빼더라도 이건 LlamaIndex 복원용으로 반드시 남겨둬야 합니다!
        
        # 2. 모든 메타데이터를 개별 필드로 밖으로 빼기
        filterable_metadata_field_keys=[
            "source", "body_part", "technique_code", "condition", "region",
            "min_stretch", "max_stretch", "tape_type"
        ],
        metadata_column_container={
            "source": "source",
            "body_part": "body_part",
            "technique_code": "technique_code",
            "region": "region",
            "condition": "condition",
            "min_stretch": "min_stretch",
            "max_stretch": "max_stretch",
            "tape_type": "tape_type",
            "contraindication": "contraindication",
            # "condition_aliases": "condition_aliases" # (주의) 리스트 데이터는 아래 설명 참고
    }
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    all_docs = []

    # 4. MD 파일 읽기 및 메타데이터 주입
    for md_file in INPUT_DIR.glob("*.md"):
        base_meta = extract_metadata_from_filename(md_file)
        
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        # 질환별(##)로 섹션 분리 (간이 파싱)
        sections = content.split('## ')[1:]
        for section in sections:
            lines = section.split('\n')
            condition_name = lines[0].strip()
            body_text = '\n'.join(lines[1:])
            
            # 수치 데이터 자동 추출
            stretch_range = extract_stretch_range(body_text)

            # tape_type 파싱 (테이프 형태 자동 감지)
            tape_type = extract_tape_type(body_text)

            # JSON 형태의 메타데이터 정의
            metadata = {
                **base_meta,
                "id": f"{base_meta['source']}_{condition_name.lower().replace(' ', '_')}",
                "condition": condition_name,
                "region": "lateral" if "lateral" in body_text.lower() else "medial",
                "min_stretch": stretch_range[0],
                "max_stretch": stretch_range[1],
                "tape_type": tape_type,
                "technique_code": f"KT_{base_meta['body_part'].upper()}_001", # 규칙 기반 생성
                "test": body_text[:200].replace('\n', ' ') # 검색 최적화용 샘플 텍스트
            }
            # 고유 ID 생성 (이미 metadata['id']에 로직이 있으므로 이를 활용)
            unique_id = metadata['id']

            # [디버깅 코드] 생성된 메타데이터 확인
            print(f"\n[DEBUG] 매핑된 메타데이터 (ID: {metadata['id']}):")
            for k, v in metadata.items():
                print(f"  - {k}: {v}")

            
            all_docs.append(Document(text=section, metadata=metadata, doc_id=unique_id))

    # 5. 인덱싱 실행 (Parent-Child 노드 파싱은 이 단계에서 내부적으로 수행 가능)
    # 여기서는 단순화를 위해 Document 단위 인덱싱을 수행합니다.
    index = VectorStoreIndex.from_documents(
        all_docs, 
        storage_context=storage_context,
        show_progress=True
    )
    print("\n[완료] Azure AI Search로 모든 데이터 업로드 완료.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    process_and_index()