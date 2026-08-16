import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apps.benchmarking.list import DOCS
from src.core.evaluation import RAGEvaluator, RetrievalEngine
from src.core.ingestion.grobid_ingestion import GrobidIngestor
from src.core.retrieval import embeddings, vector_store_manager
from src.utils.json import load_json_file
from src.utils.threads import run_in_threads
from src.utils.yaml import load_config

logging.getLogger("httpx").setLevel(logging.WARNING)

COLLECTION_NAME = "benchmark"
DOCS_DIR = "./data/papers"


class Benchmark:
    def __init__(
        self, collection_name: str = COLLECTION_NAME, docs_dir: str = DOCS_DIR
    ) -> None:
        self._collection_name = collection_name
        self._docs_dir = docs_dir
        os.makedirs(docs_dir, exist_ok=True)
        self._queries = load_json_file("./data/queries.json")

    # Download docs
    def download_docs(self) -> None:
        run_in_threads(self._fetch_docs, DOCS)

    def _fetch_docs(self, filename: str) -> None:
        try:
            with requests.get(
                f"https://arxiv.org/pdf/{filename}.pdf", stream=True, timeout=30
            ) as r:
                r.raise_for_status()

                with open(f"{self._docs_dir}/{filename}.pdf", "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

        except Exception as e:
            print(f"[FAIL] {filename} -> {e}")

    # Ingest docs
    def ingest_docs(self) -> None:
        ingestor = GrobidIngestor(self._collection_name, self._docs_dir)
        ingestor.ingest()

    # Run benchmarks
    def execute(self) -> None:
        baseline_result = self._run_baseline()
        rerank_result = self._run_with_reranker()

        print("\nCOMPARISON")
        print("=" * 60)
        print(f"{'Metric':<20} {'Baseline':<12} {'With Reranker':<12} {'Delta':<12}")
        print("-" * 60)

        metrics = [
            ("Doc Recall@K", "doc_recall_at_k"),
            ("Chunk Relevance@K", "chunk_relevance_at_k"),
            ("Avg Retrieval (ms)", "avg_retrieval_latency_ms"),
            ("Avg Rerank (ms)", "avg_rerank_latency_ms"),
            ("Avg Latency (ms)", "avg_latency_ms"),
        ]

        for name, key in metrics:
            base = baseline_result.get(key, 0) or 0
            rerank = rerank_result.get(key, 0) or 0
            if base == 0 and rerank == 0:
                continue
            delta = rerank - base
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            print(f"{name:<20} {base:<12.4f} {rerank:<12.4f} {delta_str:<12}")

    def _run_baseline(self) -> dict[str, Any]:
        config = load_config("./config/baseline.yaml")

        engine = RetrievalEngine(
            embed_fn=embeddings.encode,
            vector_store_fn=self._get_vector_store,
            reranker=None,
        )

        evaluator = RAGEvaluator(
            retrieval_engine=engine,
            llm_generate=None,
        )

        k = config.get("top_k", 5)
        experiment_name = config.get("experiment_name", "BASELINE")

        print(f"\nRunning BASELINE: {experiment_name} (without reranker)...")
        print("=" * 60)
        result = evaluator.run_benchmark(
            self._queries, k=k, experiment_name=experiment_name
        )

        result = {
            "experiment_name": result.experiment_name,
            "doc_recall_at_k": result.doc_recall_at_k,
            "chunk_relevance_at_k": result.chunk_relevance_at_k,
            "generation_score": result.generation_score,
            "avg_latency_ms": result.avg_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
            "avg_retrieval_latency_ms": result.avg_retrieval_latency_ms,
            "avg_rerank_latency_ms": result.avg_rerank_latency_ms,
            "total_queries": result.total_queries,
        }
        print(f"Results: {json.dumps(result, indent=2)}")
        return result

    def _run_with_reranker(self) -> dict[str, Any]:
        from src.core.reranking.reranker import get_reranker

        config = load_config("./config/rerank.yaml")

        initial_k = config.get("initial_k", 25)
        reranker_instance = get_reranker()

        engine = RetrievalEngine(
            embed_fn=embeddings.encode,
            vector_store_fn=self._get_vector_store,
            reranker=reranker_instance.rerank,
        )

        evaluator = RAGEvaluator(
            retrieval_engine=engine,
            llm_generate=None,
        )

        k = config.get("top_k", 5)
        experiment_name = config.get("experiment_name", "RERANK")

        print(f"\nRunning {experiment_name} (initial_k={initial_k}, top_k={k})...")
        print("=" * 60)
        result = evaluator.run_benchmark(
            self._queries, k=k, experiment_name=experiment_name, initial_k=initial_k
        )

        result = {
            "experiment_name": result.experiment_name,
            "doc_recall_at_k": result.doc_recall_at_k,
            "chunk_relevance_at_k": result.chunk_relevance_at_k,
            "generation_score": result.generation_score,
            "avg_latency_ms": result.avg_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
            "avg_retrieval_latency_ms": result.avg_retrieval_latency_ms,
            "avg_rerank_latency_ms": result.avg_rerank_latency_ms,
            "total_queries": result.total_queries,
        }
        print(f"Results: {json.dumps(result, indent=2)}")
        return result

    def _get_vector_store(self):
        return vector_store_manager.get_or_create(self._collection_name)


def main():
    benchmark = Benchmark()

    # ToDo: Add download checker to avoid redownloading files if they already exists
    # print("=" * 60)
    # print("DOWNLOADING DOCS")
    # print("=" * 60)
    # benchmark.download_docs()

    # print("=" * 60)
    # print("INGESTING PAPERS WITH GROBID + CONTEXT-AWARE CHUNKING")
    # print("=" * 60)
    # benchmark.ingest_docs()

    print("\n" + "=" * 60)
    print("RUNNING BENCHMARK")
    print("=" * 60)
    benchmark.execute()


if __name__ == "__main__":
    main()
