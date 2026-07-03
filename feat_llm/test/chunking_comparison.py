import os
import sys
import json
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import MarkdownNodeParser, HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import IndexNode, Document
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

# 인코딩 설정 (이모지 및 한글 깨짐 방지)
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# Azure OpenAI 설정 (사용자 환경에 맞게 키/엔드포인트 확인 필요)
Settings.llm = AzureOpenAI(
    engine="gpt-4.1", 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_version="2024-02-15-preview"
)
Settings.embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-small", 
    deployment_name="text-embedding-3-small", 
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
    api_version="2024-02-15-preview"
)

def load_json_hierarchical_nodes(json_path):
    """
    JSON 데이터를 기법(technique)별로 묶어 부모-자식 관계 노드로 변환합니다.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 기법별 그룹화
    tech_map = {}
    for item in data:
        tech = item.get("technique", "Unknown")
        if tech not in tech_map: tech_map[tech] = []
        tech_map[tech].append(item)
    
    all_nodes = []
    for tech, items in tech_map.items():
        # 부모 노드: 해당 기법의 모든 텍스트 결합 (문맥 보존용)
        full_context = "\n".join([i['content'] for i in items])
        parent_doc = Document(text=full_context, metadata={"technique": tech})
        
        # 자식 노드: 개별 요소 (Step, Narrative 등)
        for idx, item in enumerate(items):
            # IndexNode를 사용하여 자식이 검색되면 부모 객체를 가리키도록 설정
            node = IndexNode(
                text=item['content'], 
                index_id=f"json_{tech}_{idx}",
                metadata={"type": item['type'], "technique": tech},
                obj=parent_doc 
            )
            all_nodes.append(node)
    return all_nodes

def compare_chunking_results(query):
    data_dir = r"feat_llm\test\data"
    
    # 1. Markdown 파싱 (헤더 기준)
    md_docs = SimpleDirectoryReader(input_dir=data_dir, required_exts=[".md"]).load_data()
    md_nodes = MarkdownNodeParser().get_nodes_from_documents(md_docs)
    md_index = VectorStoreIndex(md_nodes)
    
    # 2. Plain Text 파싱 (계층적: 1024 -> 256)
    txt_docs = SimpleDirectoryReader(input_dir=data_dir, required_exts=[".txt"]).load_data()
    h_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[1024, 256])
    txt_all_nodes = h_parser.get_nodes_from_documents(txt_docs)
    txt_index = VectorStoreIndex(get_leaf_nodes(txt_all_nodes))

    # 3. JSON 파싱 (요소별 자식 노드)
    json_path = os.path.join(data_dir, "book1_gibbons_ch03_knee_elements.json")
    json_nodes = load_json_hierarchical_nodes(json_path)
    json_index = VectorStoreIndex(json_nodes)

    print(f"\n" + "="*70)
    print(f"🔍 질문: {query}")
    print("="*70)

    strategies = [
        ("실험 A: MarkdownNodeParser", md_index),
        ("실험 B: HierarchicalNodeParser (TXT)", txt_index),
        ("실험 C: JSON Parent-Child", json_index)
    ]

    for name, index in strategies:
        print(f"\n[{name}]")
        retriever = index.as_retriever(similarity_top_k=2)
        results = retriever.retrieve(query)
        
        for i, res in enumerate(results):
            print(f"\n📌 Node {i+1} (Score: {res.score:.4f})")
            # JSON 방식의 경우 자식이 아닌 연결된 부모(Full Context)의 일부를 보여줌으로써 문맥 보존 확인
            content = res.node.get_content()
            print(f"내용 요약: {content[:500]}...")
            if hasattr(res.node, 'metadata'):
                print(f"메타데이터: {res.node.metadata}")
        print("-" * 70)

if __name__ == "__main__":
    test_query = "lateral pain taping steps and stretch percentage"
    compare_chunking_results(test_query)