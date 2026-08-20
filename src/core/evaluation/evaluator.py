import time
from dataclasses import field
from typing import Any, Callable

from src.core.evaluation.metrics import (
    BenchmarkResult,
    EvaluationResult,
    RetrievalResult,
    compute_answer_fidelity,
    compute_chunk_relevance,
    compute_mrr,
    compute_ndcg_at_k,
    compute_recall_at_k,
)


class RetrievalEngine:
    def __init__(
        self,
        embed_fn: Callable,
        vector_store_fn: Callable,
        reranker: Callable | None = None,
    ):
        self.embed_fn = embed_fn
        self.vector_store_fn = vector_store_fn
        self.reranker = reranker

    def retrieve(
        self, query: str, k: int = 5, initial_k: int | None = None
    ) -> tuple[list[Any], float, float]:
        start_total = time.perf_counter()

        q_vec = self.embed_fn(query)
        vector_store = self.vector_store_fn()

        retrieval_k = initial_k if initial_k is not None and self.reranker else k
        raw_results = vector_store.similarity_search_with_score(
            query=query, k=retrieval_k
        )

        chunks = []
        for item in raw_results:
            if isinstance(item, tuple) and len(item) == 2:
                chunks.append(item[0])
            else:
                chunks.append(item)

        retrieval_latency_ms = (time.perf_counter() - start_total) * 1000

        rerank_latency_ms = 0.0
        if self.reranker:
            start_rerank = time.perf_counter()
            chunks = self.reranker(query, chunks, top_k=k)
            rerank_latency_ms = (time.perf_counter() - start_rerank) * 1000

        total_latency_ms = (time.perf_counter() - start_total) * 1000

        return chunks[:k], total_latency_ms, rerank_latency_ms, retrieval_latency_ms


class RAGEvaluator:
    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        llm_generate: Callable | None = None,
        llm_judge: Callable | None = None,
    ):
        self.retrieval_engine = retrieval_engine
        self.llm_generate = llm_generate
        self.llm_judge = llm_judge

    def _extract_doc_ids(self, chunks: list[Any]) -> list[str]:
        """Map retrieved chunks back to their source document id.

        Different ingestion paths store the source under different metadata
        keys (source / paper_id / doc_id / ...), so look them all up.
        """
        id_keys = (
            "source",
            "paper_id",
            "doc_id",
            "document_id",
            "file_name",
            "filename",
            "file",
        )
        ids: list[str] = []
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) or {}
            doc_id = next((metadata[k] for k in id_keys if metadata.get(k)), None)
            ids.append(str(doc_id) if doc_id is not None else "unknown")
        return ids

    def _extract_chunk_texts(self, chunks: list[Any]) -> list[str]:
        return [getattr(c, "page_content", str(c)) for c in chunks]

    def evaluate_query(
        self,
        query: str,
        relevant_docs: list[str],
        expected_points: list[str],
        k: int = 5,
        initial_k: int | None = None,
    ) -> EvaluationResult:
        chunks, latency_ms, rerank_latency_ms, retrieval_latency_ms = self.retrieval_engine.retrieve(
            query, k=k, initial_k=initial_k
        )

        retrieved_doc_ids = self._extract_doc_ids(chunks)
        retrieved_texts = self._extract_chunk_texts(chunks)

        doc_recall = compute_recall_at_k(retrieved_doc_ids, relevant_docs, k)
        chunk_relevance = compute_chunk_relevance(retrieved_texts, expected_points)
        mrr = compute_mrr(retrieved_doc_ids, relevant_docs)
        ndcg = compute_ndcg_at_k(retrieved_doc_ids, relevant_docs, k)
        context = "\n\n".join(retrieved_texts)

        generation_score = None
        answer = None
        faithful = None

        if self.llm_generate:
            try:
                if callable(self.llm_generate):
                    response = self.llm_generate(query, context)
                else:
                    llm_client = getattr(self.llm_generate, "client", self.llm_generate)
                    response = llm_client.invoke(query, context)
                answer = (
                    response.content if hasattr(response, "content") else str(response)
                )
                generation_score = compute_answer_fidelity(answer, context)
            except Exception:
                pass

        if self.llm_judge and answer:
            try:
                verdict = self.llm_judge(query, context, answer)
                faithful = bool(verdict) if verdict is not None else None
            except Exception:
                faithful = None

        return EvaluationResult(
            query_id="",
            query=query,
            doc_recall=doc_recall,
            chunk_relevance=chunk_relevance,
            generation_score=generation_score,
            latency_ms=latency_ms,
            # retrieval_latency_ms=latency_ms - rerank_latency_ms,
            retrieval_latency_ms=retrieval_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            retrieved_docs=retrieved_doc_ids,
            answer=answer,
            mrr=mrr,
            ndcg_at_k=ndcg,
            faithful=faithful,
        )

    def run_benchmark(
        self,
        queries: list[dict[str, Any]],
        k: int = 5,
        experiment_name: str = "benchmark",
        initial_k: int | None = None,
    ) -> BenchmarkResult:
        results: list[EvaluationResult] = []
        doc_recalls = []
        chunk_relevances = []
        generation_scores = []
        latencies = []
        retrieval_latencies = []
        rerank_latencies = []
        mrrs = []
        ndcgs = []
        faithfuls = []

        for q in queries:
            result = self.evaluate_query(
                query=q["query"],
                relevant_docs=q.get("relevant_docs", []),
                expected_points=q.get("expected_points", []),
                k=k,
                initial_k=initial_k,
            )
            result.query_id = q.get("id", "unknown")
            results.append(result)

            doc_recalls.append(result.doc_recall)
            chunk_relevances.append(result.chunk_relevance)
            latencies.append(result.latency_ms)
            if result.retrieval_latency_ms is not None:
                retrieval_latencies.append(result.retrieval_latency_ms)
            if result.rerank_latency_ms is not None:
                rerank_latencies.append(result.rerank_latency_ms)

            mrrs.append(result.mrr)
            ndcgs.append(result.ndcg_at_k)
            if result.faithful is not None:
                faithfuls.append(result.faithful)

            if result.generation_score is not None:
                generation_scores.append(result.generation_score)

        import statistics

        return BenchmarkResult(
            experiment_name=experiment_name,
            doc_recall_at_k=statistics.mean(doc_recalls) if doc_recalls else 0.0,
            chunk_relevance_at_k=statistics.mean(chunk_relevances)
            if chunk_relevances
            else 0.0,
            generation_score=statistics.mean(generation_scores)
            if generation_scores
            else None,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
            p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)]
            if latencies
            else 0.0,
            avg_retrieval_latency_ms=statistics.mean(retrieval_latencies)
            if retrieval_latencies
            else None,
            avg_rerank_latency_ms=statistics.mean(rerank_latencies)
            if rerank_latencies
            else None,
            total_queries=len(queries),
            results=results,
            mrr_at_k=statistics.mean(mrrs) if mrrs else 0.0,
            ndcg_at_k=statistics.mean(ndcgs) if ndcgs else 0.0,
            faithfulness_rate=statistics.mean(faithfuls) if faithfuls else None,
        )
