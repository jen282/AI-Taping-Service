import os
import sys
from pathlib import Path
from dotenv import load_dotenv
sys.stdout.reconfigure(encoding='utf-8')

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore, MetadataIndexFieldType
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from resource_config import setup_global_llm_and_embedding

DOCSTORE_PATH = Path(__file__).parent.parent / "data" / "docstore" / "docstore.json"


def test_search_quality(index_name: str, query: str, filter_body_part: str = None):
    """
    AutoMergingRetriever 기반 검색 품질 평가.
    leaf 노드가 충분히 검색되면 parent 노드로 자동 병합되어 더 넓은 컨텍스트를 반환합니다.
    """
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
    setup_global_llm_and_embedding()

    print(f"\n[1] '{index_name}' 인덱스에 연결 중...")

    search_client = SearchIndexClient(
        endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_KEY")),
    )

    vector_store = AzureAISearchVectorStore(
        search_or_index_client=search_client,
        index_name=index_name,
        id_field_key="chunk_id",
        chunk_field_key="chunk",
        embedding_field_key="text_vector",
        doc_id_field_key="parent_id",
        metadata_string_field_key="metadata",
        filterable_metadata_field_keys={
            "source": ("source", MetadataIndexFieldType.STRING),
            "body_part": ("body_part", MetadataIndexFieldType.STRING),
            "technique_code": ("technique_code", MetadataIndexFieldType.STRING),
            "condition": ("condition", MetadataIndexFieldType.STRING),
            "region": ("region", MetadataIndexFieldType.STRING),
            "min_stretch": ("min_stretch", MetadataIndexFieldType.INT32),
            "max_stretch": ("max_stretch", MetadataIndexFieldType.INT32),
        },
        metadata_column_container={
            "source": "source",
            "body_part": "body_part",
            "technique_code": "technique_code",
            "region": "region",
            "condition": "condition",
            "min_stretch": "min_stretch",
            "max_stretch": "max_stretch",
            "contraindication": "contraindication",
        },
    )

    # 로컬 docstore 로드 (ingestion2.py가 저장한 parent 노드 복원용)
    if not DOCSTORE_PATH.exists():
        print(f"[WARN] Docstore 없음: {DOCSTORE_PATH}")
        print("[WARN] ingestion2.py를 먼저 실행해 docstore를 생성하세요.")
        docstore = None
    else:
        docstore = SimpleDocumentStore.from_persist_path(str(DOCSTORE_PATH))
        print(f"[INFO] Docstore 로드 완료: {len(docstore.docs)}개 노드")

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    filters = None
    if filter_body_part:
        filters = MetadataFilters(
            filters=[ExactMatchFilter(key="body_part", value=filter_body_part.strip().lower())]
        )
        print(f"[2] 필터 적용됨: body_part == '{filter_body_part}'")

    print(f"[3] 질문 검색 중: '{query}'\n")

    # leaf 노드 6개를 검색한 뒤, 같은 parent를 공유하는 노드가 일정 비율 이상이면 parent로 병합
    base_retriever = index.as_retriever(similarity_top_k=6, filters=filters)

    if docstore:
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore,
        )
        retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)
    else:
        retriever = base_retriever

    nodes = retriever.retrieve(query)

    print("=" * 50)
    print(f"검색 결과 (AutoMerging 적용, {len(nodes)}개)")
    print("=" * 50)

    if not nodes:
        print("검색된 데이터가 없습니다.")
        return

    for i, node in enumerate(nodes, 1):
        meta = node.metadata
        score = getattr(node, "score", None)
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"\n[{i}순위] Score: {score_str}")
        print(f"  부위/질환: {meta.get('body_part')} / {meta.get('condition')}")
        print(f"  tape_type: {meta.get('tape_type', [])}  stretch: {meta.get('min_stretch')}~{meta.get('max_stretch')}%")
        print(f"  원문 미리보기:\n   {node}...")


if __name__ == "__main__":
    TARGET_INDEX = "taping-guide-index"

    TEST_QUERY = "Lateral knee pain"
    test_search_quality(TARGET_INDEX, TEST_QUERY)
    test_search_quality(TARGET_INDEX, TEST_QUERY, filter_body_part="knee")
