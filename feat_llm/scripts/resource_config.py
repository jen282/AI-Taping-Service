import os
from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import Settings
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from openai import AzureOpenAI as AzureOpenAIClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

def setup_global_llm_and_embedding():
    """LlamaIndex 전역에 적용될 LLM과 임베딩 모델을 설정합니다."""
    
    # 1. 임베딩 모델 설정 (Azure OpenAI)
    embed_model = AzureOpenAIEmbedding(
        model="text-embedding-3-small",       # 사용할 모델 이름
        deployment_name="text-embedding-3-small", # Azure Portal 배포 이름
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-02-01",
    )
    
    # 2. 답변 생성용 LLM 설정 
    llm = AzureOpenAI(
        model="gpt-4.1",
        deployment_name="gpt-4.1",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-02-01",
        temperature=0.1
    )
    
    # 3. LlamaIndex 글로벌 설정에 주입
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    print("[시스템] 전역 임베딩 모델 및 LLM 설정 완료.")

def get_azure_openai_client() ->  AzureOpenAIClient:
    """Raw openai.AzureOpenAI 클라이언트 반환 (chat.completions용)"""
    return AzureOpenAIClient(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

def get_search_client():
    """Azure AI Search 클라이언트를 전역으로 생성하여 반환합니다."""
    endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_AI_SEARCH_KEY")
    
    if not endpoint or not key:
        raise ValueError("환경 변수에 AZURE_AI_SEARCH_ENDPOINT와 AZURE_AI_SEARCH_KEY가 설정되지 않았습니다.")
        
    client = SearchIndexClient(
        endpoint=endpoint, 
        credential=AzureKeyCredential(key)
    )
    
    print("[시스템] Azure AI Search 클라이언트 연결 준비 완료.")
    return client