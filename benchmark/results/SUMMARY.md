# Ragify RAG Benchmark

- Generated: 2026-08-20T12:58:53.487483+00:00
- Queries: 45
- Top-K: 5
- Reranker: BAAI/bge-reranker-v2-m3 (transformers, initial_k=25)
- Embedding model: bge-small-en-v1.5 (Ollama)
- Generation/judge LLM: configured production LLM

| Metric | Baseline | With Reranker | Delta |
| --- | ---: | ---: | ---: |
| Doc Recall@5 | 0.8667 | 0.7778 | -0.0889 |
| MRR@5 | 0.7878 | 0.7667 | -0.0211 |
| NDCG@5 | 0.8069 | 0.7696 | -0.0373 |
| Chunk Relevance@5 | 0.8711 | 0.9333 | +0.0622 |
| Answer Fidelity | 0.6476 | 0.6713 | +0.0237 |
| Faithfulness (LLM judge) | 1.0000 | 0.9778 | -0.0222 |
| Avg Retrieval (ms) | 99.2721 | 112.0848 | +12.8126 |
| Avg Rerank (ms) | 0.0000 | 47152.0556 | +47152.0556 |
| Avg Latency (ms) | 99.2746 | 47264.1466 | +47164.8719 |
| p95 Latency (ms) | 140.0038 | 67207.0221 | +67067.0183 |
