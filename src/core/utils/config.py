from os import getenv

from dotenv import load_dotenv

load_dotenv()

# Embedding backend: "transformers" (default, runs in-process) or "ollama"
# (sidecar server serving a GGUF build).
EMBED_BACKEND = getenv("EMBED_BACKEND", "transformers")
# transformers backend: the upstream Hugging Face repo.
EMBED_MODEL_HF = getenv("EMBED_MODEL_HF", "BAAI/bge-small-en-v1.5")
# ollama backend only: the Ollama repack of the same model.
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
VECTORDB_API_KEY = getenv("VECTORDB_API_KEY", "")
COLLECTION_NAME = getenv("COLLECTION_NAME", "documents")
VECTOR_SIZE = int(getenv("VECTOR_SIZE", "384"))
MAX_TOKENS = int(getenv("MAX_TOKENS", "400"))
OVERLAP = int(getenv("OVERLAP", "50"))

# Parallelism / batching for the CPU-bound ingest path (semantic chunking and
# embedding). Chunking a section and embedding a batch are independent of every
# other one, so both fan out across threads.
INGEST_WORKERS = max(1, int(getenv("INGEST_WORKERS", "4")))
EMBED_BATCH_SIZE = max(1, int(getenv("EMBED_BATCH_SIZE", "64")))

# Where Grobid artifacts (pdf / xml / json / chunks) are persisted for
# debugging, relative to the rag working directory.
ARTIFACTS_ROOT = getenv("ARTIFACTS_ROOT", "source/workspace")
