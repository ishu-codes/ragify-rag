from os import getenv

from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = getenv("EMBED_MODEL", "qllama/bge-small-en-v1.5:latest")
# RERANKER_MODEL = getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_MODEL = getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANKER_BACKEND = getenv("RERANKER_BACKEND", "transformers")
# RERANKER_MODEL = getenv("RERANKER_MODEL", "Felladrin/gguf-Q8_0-bge-reranker-v2-m3")

# Classification
CLASSIFICATION_MODEL = getenv("CLASSIFICATION_MODEL", "meta/llama-3.2-1b-instruct")
CLASSIFICATION_URL = getenv("CLASSIFICATION_URL", "")
CLASSIFICATION_API_KEY = getenv("CLASSIFICATION_API_KEY", "")

# Generation LLM
LLM_MODEL = getenv("LLM_MODEL", "qwen/qwen3-next-80b-a3b-thinking")
LLM_URL = getenv("LLM_URL", "")
LLM_API_KEY = getenv("LLM_API_KEY", "")
# Some OpenAI-compatible providers don't support structured output
# (response_format). Enable only when the configured model supports it.
LLM_STRUCTURED_OUTPUT = getenv("LLM_STRUCTURED_OUTPUT", "false").lower() in (
    "1",
    "true",
    "yes",
)

# VectorDB
VECTORDB_URL = getenv("VECTORDB_URL", "http://localhost:6333/")
COLLECTION_NAME = getenv("COLLECTION_NAME", "documents")
VECTOR_SIZE = int(getenv("VECTOR_SIZE", "384"))
MAX_TOKENS = int(getenv("MAX_TOKENS", "400"))
OVERLAP = int(getenv("OVERLAP", "50"))
