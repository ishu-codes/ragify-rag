from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResult:
    query: str
    retrieved_docs: list[str]
    retrieved_chunks: list[str]
    latency_ms: float


@dataclass
class EvaluationResult:
    query_id: str
    query: str
    doc_recall: float
    chunk_relevance: float
    generation_score: float | None
    latency_ms: float
    retrieval_latency_ms: float | None = None
    rerank_latency_ms: float | None = None
    retrieved_docs: list[str] = None
    answer: str | None = None


@dataclass
class BenchmarkResult:
    experiment_name: str
    doc_recall_at_k: float
    chunk_relevance_at_k: float
    generation_score: float | None
    avg_latency_ms: float
    p95_latency_ms: float
    avg_retrieval_latency_ms: float | None = None
    avg_rerank_latency_ms: float | None = None
    total_queries: int = 0
    results: list["EvaluationResult"] = None


def compute_recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in relevant if doc in retrieved_k)
    return hits / len(relevant)


def compute_mrr(retrieved: list[str], relevant: list[str]) -> float:
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def compute_ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], 1):
        if doc in relevant:
            dcg += 1.0 / (i if i <= 10 else 10)
    idcg = sum(
        1.0 / (i if i <= 10 else 10) for i in range(1, min(len(relevant), k) + 1)
    )
    return dcg / idcg if idcg > 0 else 0.0


def compute_chunk_relevance(chunks: list[str], expected_points: list[str]) -> float:
    if not chunks or not expected_points:
        return 0.0
    relevant = 0
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(point.lower() in chunk_lower for point in expected_points):
            relevant += 1
    return relevant / len(chunks)


def compute_answer_fidelity(answer: str, context: str) -> float:
    answer_lower = answer.lower()
    context_lower = context.lower()
    answer_words = set(answer_lower.split())
    context_words = set(context_lower.split())
    if not answer_words:
        return 0.0
    matching = len(answer_words & context_words)
    return matching / len(answer_words)
