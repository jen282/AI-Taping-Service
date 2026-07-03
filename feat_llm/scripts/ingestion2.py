import os
import sys
import json
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore, IndexManagement
from resource_config import get_search_client, setup_global_llm_and_embedding
from chunk_processor import extract_metadata_from_filename, extract_stretch_range, extract_tape_types

DOCSTORE_PATH = Path(__file__).parent.parent / "data" / "docstore"
METADATA_PATH = Path(__file__).parent.parent / "data" / "metadata"


def _extract_technique_code(metadata: dict, body_part: str) -> str:
    """헤더 정보로부터 technique_code 생성.

    1. 'Taping Procedure: XXX' 형식 헤더 → KT_{body_part}_XXX
    2. 'Taping Procedure' (콜론 없음) → KT_{body_part}_{상위 헤더명}
    3. 'Taping Procedure' 헤더 없음 → KT_{body_part}
    """
    header_keys = ["Header 1", "Header 2", "Header 3", "Header 4", "Header 5"]
    bp = body_part.upper()

    for i, key in enumerate(header_keys):
        val = metadata.get(key, "")
        if not val.startswith("Taping Procedure"):
            continue
        if ":" in val:
            procedure_name = val.split(":", 1)[1].strip()
            return f"KT_{bp}_{procedure_name}"
        else:
            # 콜론 없음 → 상위 헤더에서 이름 가져오기
            for parent_key in reversed(header_keys[:i]):
                parent_val = metadata.get(parent_key, "")
                if parent_val:
                    return f"KT_{bp}_{parent_val}"
            return f"KT_{bp}"

    return f"KT_{bp}"


def _enrich_node_metadata(node, base_meta: dict) -> None:
    """노드 텍스트에서 tape_type·stretch 등을 추출해 metadata에 주입."""
    text = node.get_content()
    stretch_range = extract_stretch_range(text)
    tape_types = extract_tape_types(text)

    # MarkdownNodeParser는 헤더 정보를 "Header 1", "Header 2" 등의 키로 metadata에 저장
    condition = (
        node.metadata.get("Header 2")
        or node.metadata.get("Header 1")
        or ""
    )

    body_part = base_meta.get("body_part", "unknown")
    technique_code = _extract_technique_code(node.metadata, body_part)

    node.metadata.update({
        **base_meta,
        "condition": condition,
        "tape_type": ", ".join(tape_types) if tape_types else "",
        "min_stretch": str(stretch_range[0]),
        "max_stretch": str(stretch_range[1]),
        "region": "lateral" if "lateral" in text.lower() else "medial",
        "technique_code": technique_code,
    })


def process_and_index():
    BASE_DIR = Path(__file__).parent.parent
    INPUT_DIR = BASE_DIR / "data" / "processed_md"

    setup_global_llm_and_embedding()
    search_client = get_search_client()
    index_name = "taping-guide-index"

    vector_store = AzureAISearchVectorStore(
        search_or_index_client=search_client,
        index_name=index_name,
        index_management=IndexManagement.NO_VALIDATION,
        id_field_key="chunk_id",
        chunk_field_key="chunk",
        embedding_field_key="text_vector",
        doc_id_field_key="parent_id",
        metadata_string_field_key="metadata",
        filterable_metadata_field_keys=[
            "source", "body_part", "technique_code", "condition", "region",
            "tape_type", "min_stretch", "max_stretch",
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
        },
    )

    # 1. .md 파일 전체를 Document로 로드 (수동 섹션 분리 없음)
    all_docs = []
    for md_file in sorted(INPUT_DIR.glob("*.md")):
        base_meta = extract_metadata_from_filename(md_file)
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        doc_id = f"{base_meta['source']}_{base_meta['chapter']}"
        all_docs.append(Document(text=content, metadata=base_meta, doc_id=doc_id))

    print(f"[INFO] 로드된 문서: {len(all_docs)}개")

    # 2. MarkdownNodeParser로 헤더 기반 청킹 → 계층적 parent-child 구조 생성
    #    H1(#) → H2(##) → H3(###) 순으로 parent-child 관계가 자동 설정됨
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(all_docs, show_progress=True)
    print(f"[INFO] 파싱된 전체 노드 수: {len(nodes)}")

    # 3. 각 노드에 도메인 메타데이터(tape_type, stretch, condition 등) 주입
    for node in nodes:
        _enrich_node_metadata(node, node.metadata.copy())

    # 4. 노드 메타데이터를 로컬 metadata 폴더에 저장 (node_id 기준 파일명)
    METADATA_PATH.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        file_path = METADATA_PATH / f"{node.node_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "node_id": node.node_id,
                "metadata": node.metadata,
                "content": node.get_content(),
            }, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 메타데이터 저장 완료: {METADATA_PATH} ({len(nodes)}개)")

    # 6. 모든 노드(parent 포함)를 로컬 docstore에 저장
    #    AutoMergingRetriever가 검색 시 parent 노드를 복원하는 데 사용됨
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    DOCSTORE_PATH.mkdir(parents=True, exist_ok=True)
    persist_path = str(DOCSTORE_PATH / "docstore.json")
    docstore.persist(persist_path)
    print(f"[INFO] Docstore 저장 완료: {persist_path}")

    # 7. leaf 노드만 Azure AI Search 벡터 인덱스에 업로드
    leaf_nodes = get_leaf_nodes(nodes)
    print(f"[INFO] leaf 노드: {len(leaf_nodes)}개 / 전체: {len(nodes)}개")

    for node in leaf_nodes:
        print(f"  [LEAF] {node.node_id[:16]}... "
              f"condition={node.metadata.get('condition', '')} "
              f"tape={node.metadata.get('tape_type', [])}")

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=docstore,
    )
    VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    print("\n[완료] Azure AI Search에 leaf 노드 업로드 완료.")
    print("[INFO] 검색 시 AutoMergingRetriever가 leaf → parent 노드로 자동 병합합니다.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    process_and_index()
