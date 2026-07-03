"""
ingestion3.py
- taping-guide-index-3 인덱스 생성
- chunk_processor 자동 추출 + ch03 수동 매핑까지 모두 적용
- 스키마 변경: region -> body_region, pain_area(Collection) 추가

실행: python feat_llm/scripts/ingestion3.py
"""
import os
import sys
import json
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore, IndexManagement

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchField, SearchFieldDataType

from resource_config import get_search_client, setup_global_llm_and_embedding
from chunk_processor import extract_metadata_from_filename, extract_stretch_range, extract_tape_types

INDEX_NAME    = "taping-guide-index-3"
DOCSTORE_PATH = Path(__file__).parent.parent / "data" / "docstore3"
METADATA_PATH = Path(__file__).parent.parent / "data" / "metadata3"
CH03_MAPPING  = Path(__file__).parent.parent / "data" / "final_chunks_ch03" / "_all_chunks.json"


def _extract_technique_code(metadata: dict, body_part: str) -> str:
    header_keys = ["Header 1", "Header 2", "Header 3", "Header 4", "Header 5"]
    bp = body_part.upper()
    for i, key in enumerate(header_keys):
        val = metadata.get(key, "")
        if not val.startswith("Taping Procedure"):
            continue
        if ":" in val:
            return f"KT_{bp}_{val.split(':', 1)[1].strip()}"
        for parent_key in reversed(header_keys[:i]):
            parent_val = metadata.get(parent_key, "")
            if parent_val:
                return f"KT_{bp}_{parent_val}"
        return f"KT_{bp}"
    return f"KT_{bp}"


def _enrich_node_metadata(node, base_meta: dict) -> None:
    text = node.get_content()
    stretch_range = extract_stretch_range(text)
    tape_types = extract_tape_types(text)
    condition = node.metadata.get("Header 2") or node.metadata.get("Header 1") or ""
    body_part = base_meta.get("body_part", "unknown")
    technique_code = _extract_technique_code(node.metadata, body_part)
    node.metadata.update({
        **base_meta,
        "condition": condition,
        "tape_type": ", ".join(tape_types) if tape_types else "",
        "min_stretch": str(stretch_range[0]),
        "max_stretch": str(stretch_range[1]),
        "body_region": "lateral" if "lateral" in text.lower() else "medial",
        "technique_code": technique_code,
    })


def add_pain_area_field(endpoint: str, key: str, index_name: str) -> None:
    """LlamaIndex가 만든 인덱스에 pain_area(Collection) 필드를 추가."""
    idx_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    index = idx_client.get_index(index_name)
    if any(f.name == "pain_area" for f in index.fields):
        print("  - pain_area 필드 이미 존재")
        return
    index.fields.append(SearchField(
        name="pain_area",
        type=SearchFieldDataType.Collection(SearchFieldDataType.String),
        filterable=True,
    ))
    idx_client.create_or_update_index(index)
    print("  + pain_area (Collection(Edm.String)) 추가")


def apply_ch03_manual_mapping(endpoint: str, key: str, index_name: str) -> None:
    """final_chunks_ch03/_all_chunks.json을 읽어 chunk 텍스트 매칭으로 매핑 적용."""
    if not CH03_MAPPING.exists():
        print(f"  [WARN] 매핑 파일 없음: {CH03_MAPPING}")
        return

    with open(CH03_MAPPING, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    saved_chunks = mapping_data["chunks"]
    print(f"  매핑 소스: {len(saved_chunks)}개 청크")

    doc_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))
    new_docs = list(doc_client.search(
        search_text="*",
        filter="body_part eq 'knee' and source eq 'book1_gibbons'",
        select="chunk_id,chunk,metadata",  # metadata 문자열 필드도 함께 조회
        top=100,
    ))
    print(f"  index-3 ch03 청크: {len(new_docs)}개")

    def key_of(text: str) -> str:
        return (text or "").strip()[:100]

    new_lookup      = {key_of(d["chunk"]): d["chunk_id"]          for d in new_docs}
    metadata_lookup = {d["chunk_id"]:      d.get("metadata", "{}") for d in new_docs}

    updates = []
    matched, unmatched = 0, []
    for sc in saved_chunks:
        match_key = key_of(sc.get("chunk", ""))
        new_id = new_lookup.get(match_key)
        if not new_id:
            unmatched.append(sc.get("chunk_id"))
            continue
        matched += 1

        # 기존 metadata JSON 문자열을 파싱해서 수동 매핑값으로 덮어쓰기
        # → LlamaIndex가 n.metadata를 metadata 문자열에서 읽으므로 함께 패치 필요
        try:
            meta = json.loads(metadata_lookup.get(new_id, "{}"))
        except json.JSONDecodeError:
            meta = {}
        meta["technique_code"] = sc.get("technique_code")
        meta["condition"]      = sc.get("condition")
        meta["body_region"]    = sc.get("body_region")

        updates.append({
            "chunk_id":      new_id,
            "technique_code": sc.get("technique_code"),
            "condition":      sc.get("condition"),
            "body_region":    sc.get("body_region"),
            "pain_area":      sc.get("pain_area") or [],
            "metadata":       json.dumps(meta, ensure_ascii=False),  # 문자열 필드도 패치
        })

    print(f"  매칭 성공: {matched}/{len(saved_chunks)}개")
    if unmatched:
        print(f"  매칭 실패 chunk_id (기존): {unmatched}")

    if updates:
        results = doc_client.merge_or_upload_documents(documents=updates)
        ok = sum(1 for r in results if r.succeeded)
        ng = sum(1 for r in results if not r.succeeded)
        print(f"  업데이트 적용: 성공 {ok} / 실패 {ng}")


def process_and_index():
    BASE_DIR  = Path(__file__).parent.parent
    INPUT_DIR = BASE_DIR / "data" / "processed_md"

    setup_global_llm_and_embedding()
    search_client = get_search_client()

    vector_store = AzureAISearchVectorStore(
        search_or_index_client=search_client,
        index_name=INDEX_NAME,
        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
        id_field_key="chunk_id",
        chunk_field_key="chunk",
        embedding_field_key="text_vector",
        doc_id_field_key="parent_id",
        metadata_string_field_key="metadata",
        filterable_metadata_field_keys=[
            "source", "body_part", "technique_code", "condition", "body_region",
            "tape_type", "min_stretch", "max_stretch",
        ],
        metadata_column_container={
            "source": "source",
            "body_part": "body_part",
            "technique_code": "technique_code",
            "body_region": "body_region",
            "condition": "condition",
            "min_stretch": "min_stretch",
            "max_stretch": "max_stretch",
            "tape_type": "tape_type",
            "contraindication": "contraindication",
        },
    )

    # 1. 마크다운 문서 로드
    all_docs = []
    for md_file in sorted(INPUT_DIR.glob("*.md")):
        base_meta = extract_metadata_from_filename(md_file)
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        doc_id = f"{base_meta['source']}_{base_meta['chapter']}"
        all_docs.append(Document(text=content, metadata=base_meta, doc_id=doc_id))
    print(f"[INFO] 로드된 문서: {len(all_docs)}개")

    # 2. 청킹
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(all_docs, show_progress=True)
    print(f"[INFO] 파싱된 전체 노드 수: {len(nodes)}")

    # 3. 메타데이터 enrich
    for node in nodes:
        _enrich_node_metadata(node, node.metadata.copy())

    # 4. 메타데이터 로컬 저장
    METADATA_PATH.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        with open(METADATA_PATH / f"{node.node_id}.json", "w", encoding="utf-8") as f:
            json.dump({
                "node_id": node.node_id,
                "metadata": node.metadata,
                "content": node.get_content(),
            }, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 메타데이터 저장 완료: {METADATA_PATH}")

    # 5. docstore 저장
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    DOCSTORE_PATH.mkdir(parents=True, exist_ok=True)
    docstore.persist(str(DOCSTORE_PATH / "docstore.json"))
    print(f"[INFO] Docstore 저장 완료: {DOCSTORE_PATH}")

    # 6. leaf 노드만 Azure Search 업로드 (인덱스 자동 생성)
    leaf_nodes = get_leaf_nodes(nodes)
    print(f"[INFO] leaf 노드: {len(leaf_nodes)}개")

    storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
    VectorStoreIndex(leaf_nodes, storage_context=storage_context, show_progress=True)
    print(f"\n[완료] {INDEX_NAME} 업로드 완료.")

    # 7. pain_area 필드 추가
    print("\n[추가 작업 1] pain_area 필드 스키마 추가...")
    endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    key      = os.getenv("AZURE_AI_SEARCH_KEY")
    add_pain_area_field(endpoint, key, INDEX_NAME)

    # 8. ch03 수동 매핑 적용
    print("\n[추가 작업 2] ch03 수동 매핑 적용...")
    apply_ch03_manual_mapping(endpoint, key, INDEX_NAME)

    print("\n[전체 완료]")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--patch-only":
        # metadata 문자열 패치만 단독 실행 (전체 re-ingestion 불필요 시)
        # 실행: python ingestion3.py --patch-only
        print("[패치 모드] ch03 수동 매핑 (metadata 문자열 포함) 재적용...")
        endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        key      = os.getenv("AZURE_AI_SEARCH_KEY")
        apply_ch03_manual_mapping(endpoint, key, INDEX_NAME)
        print("[완료]")
    else:
        process_and_index()
