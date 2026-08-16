# ragify-rag

RAG engine for Ragify, exposed as a standalone gRPC server. It handles document ingestion (chunking, markdown conversion, PDF parsing via Grobid), embeddings (Ollama), vector storage (Qdrant) and the LangGraph query pipeline (routing, retrieval, evaluation, generation).

## Stack

- Python 3.12, gRPC (`grpcio`)
- LangChain, LangGraph, LangChain Qdrant/Ollama
- Qdrant client, Ollama client, Tavily (web search)
- Grobid client + grobid2json (PDF parsing)

## Layout

```text
src/
  core/
    classification/   query routing
    evaluation/       retrieval evaluator
    generation/       LangGraph graph, agent, prompts, state
    ingestion/        chunking, markdown conversion, PDF ingestion
    pipeline/         end-to-end pipeline
    reranking/        rerankers
    retrieval/        embedder, retriever, Qdrant vector store
    utils/            config, logger
  grpc/
    ragify.proto      shared contract
    server.py         gRPC server (three services)
```

## gRPC services

| Service | Methods | Backing modules |
| --- | --- | --- |
| `VectorStoreService` | `CreateCollection`, `DeleteCollection`, `InsertDocuments` | `src.core.retrieval` |
| `IngestionService` | `ProcessSection`, `ConvertToMarkdown`, `ProcessDocument`, `IngestPdf` | `src.core.ingestion` |
| `RagService` | `Query` | `src.core.generation` |

The proto lives at `src/grpc/ragify.proto`; a copy of the generated stubs is kept in `api-python/src/ragify_client/protos/`. Regenerate both with `make grpc-gen` from the repo root.

## Run

```bash
uv sync
.venv/bin/python -m src.grpc
```

or `make ragify-server` from the repo root. Default bind: `0.0.0.0:50051` (`RAGIFY_GRPC_HOST` / `RAGIFY_GRPC_PORT`).

### Prerequisites

- Qdrant running at `VECTORDB_URL` (default `http://localhost:6333/`)
- Ollama running with the embedding model (`EMBED_MODEL`, default `qllama/bge-small-en-v1.5:latest`)
- Grobid at `:8070` for PDF ingestion
- An OpenAI-compatible LLM endpoint (`LLM_URL` / `LLM_MODEL` / `LLM_API_KEY`) and, optionally, classification + Tavily keys

All config is read from the root `.env` via `src/core/utils/config.py`.

## Document ingestion flow

```mermaid
sequenceDiagram
    participant API as FastAPI
    participant RAG as rag gRPC server
    participant OLL as Ollama
    participant QD as Qdrant
    participant GB as Grobid

    API->>RAG: IngestionService.ProcessDocument (gRPC)
    alt PDF
        RAG->>GB: parse PDF to XML / Markdown
    end
    RAG->>RAG: convert to markdown + chunk
    RAG->>OLL: embed chunks
    RAG->>QD: insert points into workspace collection
    RAG-->>API: chunk count
```

## Query flow

```mermaid
sequenceDiagram
    participant API as FastAPI
    participant RAG as rag gRPC server
    participant OLL as Ollama
    participant QD as Qdrant
    participant LLM as LLM provider

    API->>RAG: RagService.Query (gRPC)
    RAG->>LLM: classify route (index / general / search)
    RAG->>OLL: embed query
    RAG->>QD: retrieve context
    RAG->>LLM: evaluate retrieval / generate answer
    RAG-->>API: answer
```

## Configuration

Key variables (see repo root `.env.example`): `VECTORDB_URL`, `EMBED_MODEL`, `VECTOR_SIZE`, `MAX_TOKENS`, `OVERLAP`, `LLM_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_STRUCTURED_OUTPUT`, `CLASSIFICATION_URL`, `CLASSIFICATION_MODEL`, `CLASSIFICATION_API_KEY`, `TAVILY_API_KEY`, `RAGIFY_GRPC_HOST`, `RAGIFY_GRPC_PORT`, `RAGIFY_GRPC_MAX_WORKERS`.

## Development

```bash
uv pip install -e .      # installs the ragify-server console script
python -m grpc_tools.protoc -I src/grpc ...   # or make grpc-gen
```
