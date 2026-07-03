import sys
from pathlib import Path

from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper

# scripts/resource_config.py import 경로 추가
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from resource_config import setup_global_llm_and_embedding
from llama_index.core import Settings

EVAL_DIR     = Path(__file__).parent
DATASETS_DIR = EVAL_DIR / "datasets"
RESULTS_DIR  = EVAL_DIR / "results" / "ragas"


def get_ragas_llm() -> LlamaIndexLLMWrapper:
    """scripts/resource_config.py의 전역 LLM을 RAGAS용으로 래핑."""
    setup_global_llm_and_embedding()
    return LlamaIndexLLMWrapper(Settings.llm)


def get_ragas_embeddings() -> LlamaIndexEmbeddingsWrapper:
    """scripts/resource_config.py의 전역 임베딩을 RAGAS용으로 래핑."""
    setup_global_llm_and_embedding()
    return LlamaIndexEmbeddingsWrapper(Settings.embed_model)
