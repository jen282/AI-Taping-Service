import os
import sys
import json
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import IndexNode
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore

# 출력 인코딩 및 환경 설정
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# Azure OpenAI 설정 (기존과 동일)
Settings.llm = AzureOpenAI(engine="gpt-4.1", api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), api_version="2024-02-15-preview")
Settings.embed_model = AzureOpenAIEmbedding(model="text-embedding-3-small", deployment_name="text-embedding-3-small", api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), api_version="2024-02-15-preview")

def load_json_hierarchical(json_path):
    """
    JSON의 'technique'을 부모로, 각 'element'를 자식으로 하는 계층적 노드를 생성합니다.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 테크닉별로 데이터 그룹화
    techniques = {}
    for item in data:
        t_name = item.get("technique", "Unknown")
        if t_name not in techniques:
            techniques[t_name] = []
        techniques[t_name].append(item)
    
    all_nodes = []
    for t_name, elements in techniques.items():
        # 1. 부모 노드 생성 (해당 테크닉의 모든 텍스트 통합)
        full_text = "\n".join([e['content'] for e in elements])
        parent_node = Document(text=full_text, metadata={"technique": t_name, "type": "parent"})
        
        # 2. 자식 노드 생성 (각 개별 요소)
        for element in elements:
            child_node = IndexNode(
                text=element['content'], 
                index_id=f"child_{t_name}_{elements.index(element)}",
                metadata={"type": element['type'], "technique": t_name},
                obj=parent_node # 자식이 검색되면 부모를 반환하도록 설정
            )
            all_nodes.append(child_node)
            
    return all_nodes

def compare_final_responses(user_query):
    data_dir = r"feat_llm\test\data"
    
    # --- 방법 A: Markdown ---
    md_docs = SimpleDirectoryReader(input_dir=data_dir, required_exts=[".md"]).load_data()
    md_nodes = MarkdownNodeParser().get_nodes_from_documents(md_docs)
    md_index = VectorStoreIndex(md_nodes)
    md_engine = md_index.as_query_engine(similarity_top_k=2)
    
    # --- 방법 B: Plain Text (Hierarchical) ---
    txt_docs = SimpleDirectoryReader(input_dir=data_dir, required_exts=[".txt"]).load_data()
    h_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 256])
    txt_nodes = h_parser.get_nodes_from_documents(txt_docs)
    txt_index = VectorStoreIndex(get_leaf_nodes(txt_nodes))
    txt_engine = txt_index.as_query_engine(similarity_top_k=2)

    # --- 방법 C: JSON Parent-Child ---
    json_path = os.path.join(data_dir, "book1_gibbons_ch03_knee_elements.json")
    json_nodes = load_json_hierarchical(json_path)
    json_index = VectorStoreIndex(json_nodes)
    json_engine = json_index.as_query_engine(similarity_top_k=3)

    print("\n" + "="*80)
    print(f"🤔 질문: {user_query}")
    print("="*80)

    for label, engine in [("Markdown", md_engine), ("Plain TXT", txt_engine), ("JSON Parent-Child", json_engine)]:
        print(f"\n[방법: {label}]")
        response = engine.query(user_query)
        print(f"💬 GPT 답변:\n{response}")
        print(f"📍 참고 노드 수: {len(response.source_nodes)}")
        print("-" * 40)

if __name__ == "__main__":
    query = "Lateral knee pain taping steps and specific stretch percentages for each step"
    compare_final_responses(query)