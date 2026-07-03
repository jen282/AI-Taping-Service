import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.stdout.reconfigure(encoding='utf-8')

from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate, Settings
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from resource_config import setup_global_llm_and_embedding

DOCSTORE_PATH = Path(__file__).parent.parent / "data" / "docstore" / "docstore.json"

#추천 옵션 생성용 시스템 프롬프트 (llm.complete()로 직접 호출)
RECOMMENDATION_PROMPT = """\
SYSTEM:
You are a kinesiology taping recommendation assistant.
You will receive (1) a structured symptom, and (2) retrieved textbook passages.
Your job is to recommend taping options and explain WHY each option helps.

=== CRITICAL CONSTRAINTS ===
- You MUST only recommend taping_ids from the VALID SET provided below.
- Do NOT invent techniques outside this list.
- If no technique in the valid set matches, return an empty options array.
- All "why" fields must be grounded in the retrieved passages.
  If a passage does not support a claim, do not make the claim.
- Output ONLY valid JSON. No explanation, no markdown.

=== VALID TAPING IDs FOR THIS REQUEST ===
{valid_taping_ids}

=== RETRIEVED TEXTBOOK PASSAGES ===
{rag_chunks}

=== STRUCTURED SYMPTOM ===
{structured_symptom}

Output schema:
{{
  "options": [
    {{
      "taping_id": "generic_knee_y",
      "name_ko": "Y자 테이핑",
      "tape_type": "Y-strip",
      "stretch_pct_range": [100, 100],
      "why": "IT 밴드 외측 긴장 완화를 위해 Y자 스트립을 무릎 외측에 적용해요. 교본에 따르면 이 기법은 외측 대퇴골 과부의 마찰을 줄여줍니다.",
      "disclaimer": "이 추천은 교본 자료에 기반하며, 의료 진단을 대체하지 않아요."
    }}
  ],
  "coaching_text": "무릎 외측 통증에는 IT 밴드 긴장 완화 테이핑이 도움될 수 있어요. 아래 옵션 중 하나를 선택해 주세요."
}}

=== SAFETY RULES ===
- If structured_symptom contains "acute": true → return {{"options": [], "redirect": "hospital"}}
- Never mention body parts outside the structured_symptom body_part.
- If textbook passages are insufficient, set why to: "교본 자료가 제한적이에요. 전문가 확인을 권해요."
"""

# LLM은 자연어 지침(text)만 생성
TEXT_PROMPT = PromptTemplate(
    "당신은 키네시올로지 테이핑 전문가입니다. 아래 제공된 [Context] 정보를 바탕으로 질문에 한글로 답하세요.\n"
    "전체 테이핑 단계(Step 1, 2...), 준비 자세(Position), 테이프 텐션(Stretch %)을 포함한 상세 줄글 가이드를 작성하세요.\n"
    "문단 구분은 '\\n'을 사용하세요. 오직 가이드 텍스트만 출력하세요.\n\n"
    "[Context]\n"
    "{context_str}\n\n"
    "질문: {query_str}\n"
    "답변:"
)

class TapingRAGSystem:
    def __init__(self, index_name: str):
        setup_global_llm_and_embedding()
        self.index_name = index_name
        self.llm = Settings.llm
        self._setup_index()

    def _setup_index(self):
        """Azure AI Search 연결 및 로컬 docstore 로드."""
        search_client = SearchIndexClient(
            endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_KEY")),
        )
        self.vector_store = AzureAISearchVectorStore(
            search_or_index_client=search_client,
            index_name=self.index_name,
            id_field_key="chunk_id",
            chunk_field_key="chunk",
            embedding_field_key="text_vector",
            doc_id_field_key="parent_id",
            metadata_string_field_key="metadata",
            filterable_metadata_field_keys=[
                "source", "body_part", "technique_code", "condition", "region",
                "tape_type", "min_stretch", "max_stretch",
            ],
        )
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)

        if DOCSTORE_PATH.exists():
            self.docstore = SimpleDocumentStore.from_persist_path(str(DOCSTORE_PATH))
            print(f"[INFO] Docstore 로드 완료: {len(self.docstore.docs)}개 노드")
        else:
            self.docstore = None
            print("[WARN] Docstore 없음. ingestion2.py를 먼저 실행하세요.")

    def _build_retriever(self, filters=None):
        """AutoMergingRetriever 생성. docstore가 없으면 기본 retriever 반환."""
        base_retriever = self.index.as_retriever(similarity_top_k=6, filters=filters)

        if self.docstore is None:
            return base_retriever

        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.docstore,
        )
        return AutoMergingRetriever(base_retriever, storage_context, verbose=True)

    def translate_to_english(self, k_query: str) -> str:
        """한국어 질문을 영어 의학 용어로 번역."""
        prompt = (
            "You are a medical translator specializing in kinesiology taping. "
            "Translate the user's Korean question into English medical terms for searching professional documents.\n\n"
            f"Korean: {k_query}\nEnglish:"
        )
        response = self.llm.complete(prompt)
        translated = response.text.strip()
        print(f"[DEBUG] 번역된 쿼리: {translated}")
        return translated

    def _extract_metadata(self, nodes: list) -> Dict[str, Any]:
        """검색된 노드 중 가장 관련도 높은 노드의 메타데이터 반환."""
        if not nodes:
            return {}
        meta = nodes[0].metadata
        return {
            "chunk_id":       nodes[0].node_id,
            "source":         meta.get("source"),
            "body_part":      meta.get("body_part"),
            "technique_code": meta.get("technique_code"),
            "condition":      meta.get("condition"),
            "region":         meta.get("region"),
            "tape_type":      [t.strip() for t in meta.get("tape_type", "").split(",") if t.strip()],
            "stretch_pct_range": [
                int(meta.get("min_stretch", 0)),
                int(meta.get("max_stretch", 0)),
            ],
        }

    def get_recommendation(
        self,
        structured_symptom: Dict[str, Any],
        valid_taping_ids: List[str],
        filter_body_part: str = None,
    ) -> Dict[str, Any]:
        """추천 옵션 JSON 반환 (RECOMMENDATION_PROMPT 사용)."""
        e_query = self.translate_to_english(structured_symptom.get("condition", ""))

        filters = None
        if filter_body_part:
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="body_part", value=filter_body_part.strip().lower())]
            )

        retriever = self._build_retriever(filters=filters)
        nodes = retriever.retrieve(e_query)
        rag_chunks = "\n\n".join(n.get_content() for n in nodes)

        prompt = RECOMMENDATION_PROMPT.format(
            valid_taping_ids=json.dumps(valid_taping_ids, ensure_ascii=False),
            rag_chunks=rag_chunks,
            structured_symptom=json.dumps(structured_symptom, ensure_ascii=False),
        )

        print("[DEBUG] 추천 옵션 생성 중...")
        response = self.llm.complete(prompt)
        return json.loads(response.text.strip())

    def get_structured_response(self, k_query: str, filter_body_part: str = None) -> Dict[str, Any]:
        """번역 → 검색 → 메타데이터 추출 + LLM 자연어 지침 생성 → 합쳐서 반환."""
        e_query = self.translate_to_english(k_query)

        filters = None
        if filter_body_part:
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="body_part", value=filter_body_part.strip().lower())]
            )

        retriever = self._build_retriever(filters=filters)

        # 1) 검색 → 메타데이터 추출
        nodes    = retriever.retrieve(e_query)
        metadata = self._extract_metadata(nodes)

        # 2) LLM은 자연어 지침(text)만 생성
        from llama_index.core.query_engine import RetrieverQueryEngine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_mode="compact",
        )
        query_engine.update_prompts({"response_synthesizer:text_qa_template": TEXT_PROMPT})

        print("[DEBUG] AI가 자연어 지침을 생성하는 중...")
        text = str(query_engine.query(e_query)).strip()

        # 3) FastAPI 응답 구성: 메타데이터 + LLM 텍스트
        return {**metadata, "text": text}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    rag = TapingRAGSystem(index_name="taping-guide-index")

    test_question = "무릎 바깥쪽 통증 테이핑 방법 알려줘"
    print(f"\n[질문] {test_question}")

    # get_structured_response
    result = rag.get_structured_response(test_question, filter_body_part="knee")

    import json
    print("\n" + "=" * 50)
    print("최종 응답")
    print("=" * 50)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # --- get_recommendation 테스트 ---
    print("\n" + "=" * 50)
    print("추천 옵션 테스트")
    print("=" * 50)

    test_symptom = {
        "condition": "무릎 바깥쪽 통증",
        "body_part": "knee",
        "acute": False,
    }
    test_valid_ids = ["generic_knee_y", "generic_knee_i", "generic_itband_x"]

    print(f"[증상] {json.dumps(test_symptom, ensure_ascii=False)}")
    print(f"[허용 ID] {test_valid_ids}")

    rec_result = rag.get_recommendation(
        structured_symptom=test_symptom,
        valid_taping_ids=test_valid_ids,
        filter_body_part="knee",
    )
    print(json.dumps(rec_result, indent=2, ensure_ascii=False))
