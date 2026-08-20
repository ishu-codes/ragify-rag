import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# Allow importing both the benchmark-local modules and the rag engine:
#   benchmark dir -> from list import DOCS
#   rag dir       -> from src.core... import ...
_BENCH_DIR = Path(__file__).resolve().parent
_RAG_DIR = _BENCH_DIR.parent
sys.path.insert(0, str(_BENCH_DIR))
sys.path.insert(0, str(_RAG_DIR))

from list import DOCS
from src.core.evaluation import RAGEvaluator, RetrievalEngine
from src.core.ingestion.grobid_ingestion import GrobidIngestor
from src.core.retrieval import embeddings, vector_store_manager
from src.utils.json import load_json_file
from src.utils.threads import run_in_threads
from src.utils.yaml import load_config

logging.getLogger("httpx").setLevel(logging.WARNING)

COLLECTION_NAME = "benchmark"
DOCS_DIR = "./data/papers"
RESULTS_DIR = _BENCH_DIR / "results"


def _make_llm_generate():
    """Return generate(query, context) -> answer using the configured LLM."""
    from langchain_core.prompts import PromptTemplate

    from src.core.generation.llm import llm

    template = PromptTemplate(
        template=(
            "Answer the user's question using ONLY the retrieved context below. "
            "Be concise and factual. If the context does not contain enough "
            "information to answer, say so explicitly and do not invent facts.\n\n"
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:"
        ),
        input_variables=["question", "context"],
    )
    chain = template | llm.client

    def generate(query: str, context: str) -> str:
        response = chain.invoke({"question": query, "context": context})
        return response.content if hasattr(response, "content") else str(response)

    return generate


def _make_llm_judge():
    """Return judge(query, context, answer) -> bool using the configured LLM."""
    from langchain_core.prompts import PromptTemplate

    from src.core.generation.llm import llm
    from src.core.generation.prompts import prompts

    chain = (
        PromptTemplate(
            template=prompts.verify_prompt,
            input_variables=["question", "context", "final_answer"],
        )
        | llm.client
    )

    def judge(query: str, context: str, answer: str) -> bool | None:
        response = chain.invoke(
            {"question": query, "context": context, "final_answer": answer}
        )
        text = response.content if hasattr(response, "content") else str(response)
        match = re.search(r'"faithful"\s*:\s*(true|false)', text.lower())
        return match.group(1) == "true" if match else None

    return judge


class Benchmark:
    def __init__(
        self, collection_name: str = COLLECTION_NAME, docs_dir: str = DOCS_DIR
    ) -> None:
        self._collection_name = collection_name
        self._docs_dir = docs_dir
        os.makedirs(docs_dir, exist_ok=True)
        self._queries = load_json_file("./data/queries.json")
        RESULTS_DIR.mkdir(exist_ok=True)

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

        self._save_experiment(baseline_result)
        self._save_experiment(rerank_result)
        self._write_latest(baseline_result, rerank_result)
        self._print_comparison(baseline_result, rerank_result)
        self._write_summary(baseline_result, rerank_result)

    def _run_baseline(self) -> dict[str, Any]:
        config = load_config("./config/baseline.yaml")

        engine = RetrievalEngine(
            embed_fn=embeddings.encode,
            vector_store_fn=self._get_vector_store,
            reranker=None,
        )

        evaluator = RAGEvaluator(
            retrieval_engine=engine,
            llm_generate=(
                _make_llm_generate() if config.get("use_llm_generate", True) else None
            ),
            llm_judge=(
                _make_llm_judge() if config.get("use_llm_judge", False) else None
            ),
        )

        k = config.get("top_k", 5)
        experiment_name = config.get("experiment_name", "BASELINE")

        print(f"\nRunning BASELINE: {experiment_name} (without reranker)...")
        print("=" * 60)
        result = evaluator.run_benchmark(
            self._queries, k=k, experiment_name=experiment_name
        )

        result_dict = self._result_to_dict(result)
        result_dict["top_k"] = k
        print(f"Results: {json.dumps(result_dict, indent=2)}")
        return result_dict

    def _run_with_reranker(self) -> dict[str, Any]:
        from src.core.reranking.reranker import get_reranker

        config = load_config("./config/rerank.yaml")

        initial_k = config.get("initial_k", 25)
        reranker_instance = get_reranker(
            backend=config.get("reranker_backend", "transformers"),
            model=config.get("reranker"),
        )

        engine = RetrievalEngine(
            embed_fn=embeddings.encode,
            vector_store_fn=self._get_vector_store,
            reranker=reranker_instance.rerank,
        )

        evaluator = RAGEvaluator(
            retrieval_engine=engine,
            llm_generate=(
                _make_llm_generate() if config.get("use_llm_generate", True) else None
            ),
            llm_judge=(
                _make_llm_judge() if config.get("use_llm_judge", False) else None
            ),
        )

        k = config.get("top_k", 5)
        experiment_name = config.get("experiment_name", "RERANK")

        print(f"\nRunning {experiment_name} (initial_k={initial_k}, top_k={k})...")
        print("=" * 60)
        result = evaluator.run_benchmark(
            self._queries, k=k, experiment_name=experiment_name, initial_k=initial_k
        )

        result_dict = self._result_to_dict(result)
        result_dict["top_k"] = k
        result_dict["reranker"] = config.get("reranker")
        result_dict["reranker_backend"] = config.get("reranker_backend")
        result_dict["initial_k"] = initial_k
        print(f"Results: {json.dumps(result_dict, indent=2)}")
        return result_dict

    def _get_vector_store(self):
        return vector_store_manager.get_or_create(self._collection_name)

    @staticmethod
    def _result_to_dict(result) -> dict[str, Any]:
        """Flatten a BenchmarkResult into a JSON-serializable dict."""
        return {
            "experiment_name": result.experiment_name,
            "doc_recall_at_k": result.doc_recall_at_k,
            "chunk_relevance_at_k": result.chunk_relevance_at_k,
            "mrr_at_k": result.mrr_at_k,
            "ndcg_at_k": result.ndcg_at_k,
            "generation_score": result.generation_score,
            "faithfulness_rate": result.faithfulness_rate,
            "avg_latency_ms": result.avg_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
            "avg_retrieval_latency_ms": result.avg_retrieval_latency_ms,
            "avg_rerank_latency_ms": result.avg_rerank_latency_ms,
            "total_queries": result.total_queries,
            "results": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "doc_recall": r.doc_recall,
                    "chunk_relevance": r.chunk_relevance,
                    "mrr": r.mrr,
                    "ndcg_at_k": r.ndcg_at_k,
                    "generation_score": r.generation_score,
                    "faithful": r.faithful,
                    "latency_ms": r.latency_ms,
                    "retrieved_docs": r.retrieved_docs,
                    "answer": r.answer,
                }
                for r in (result.results or [])
            ],
        }

    @staticmethod
    def _save_experiment(result: dict[str, Any]) -> None:
        """Persist one experiment's full results as JSON."""
        name = result["experiment_name"].lower().replace(" ", "_")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = RESULTS_DIR / f"{name}_{timestamp}.json"
        path.write_text(json.dumps(result, indent=2))
        print(f"\nSaved results -> {path}")

    def _write_latest(
        self, baseline: dict[str, Any], rerank: dict[str, Any]
    ) -> None:
        """Maintain a pointer file to the newest pair of results."""
        path = RESULTS_DIR / "latest.json"
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "queries": baseline.get("total_queries"),
            "top_k": baseline.get("top_k", 5),
            "experiments": {
                baseline["experiment_name"]: {
                    "doc_recall_at_k": baseline["doc_recall_at_k"],
                    "chunk_relevance_at_k": baseline["chunk_relevance_at_k"],
                    "mrr_at_k": baseline["mrr_at_k"],
                    "ndcg_at_k": baseline["ndcg_at_k"],
                    "generation_score": baseline["generation_score"],
                    "faithfulness_rate": baseline["faithfulness_rate"],
                    "avg_latency_ms": baseline["avg_latency_ms"],
                    "p95_latency_ms": baseline["p95_latency_ms"],
                },
                rerank["experiment_name"]: {
                    "doc_recall_at_k": rerank["doc_recall_at_k"],
                    "chunk_relevance_at_k": rerank["chunk_relevance_at_k"],
                    "mrr_at_k": rerank["mrr_at_k"],
                    "ndcg_at_k": rerank["ndcg_at_k"],
                    "generation_score": rerank["generation_score"],
                    "faithfulness_rate": rerank["faithfulness_rate"],
                    "avg_latency_ms": rerank["avg_latency_ms"],
                    "p95_latency_ms": rerank["p95_latency_ms"],
                    "reranker": rerank.get("reranker"),
                },
            },
        }
        path.write_text(json.dumps(data, indent=2))
        print(f"Saved pointer -> {path}")

    @staticmethod
    def _print_comparison(baseline: dict[str, Any], rerank: dict[str, Any]) -> None:
        print("\nCOMPARISON")
        print("=" * 64)
        print(f"{'Metric':<26} {'Baseline':<12} {'With Reranker':<14} {'Delta':<12}")
        print("-" * 64)

        metrics = [
            ("Doc Recall@K", "doc_recall_at_k"),
            ("MRR@K", "mrr_at_k"),
            ("NDCG@K", "ndcg_at_k"),
            ("Chunk Relevance@K", "chunk_relevance_at_k"),
            ("Answer Fidelity", "generation_score"),
            ("Faithfulness (LLM judge)", "faithfulness_rate"),
            ("Avg Retrieval (ms)", "avg_retrieval_latency_ms"),
            ("Avg Rerank (ms)", "avg_rerank_latency_ms"),
            ("Avg Latency (ms)", "avg_latency_ms"),
            ("p95 Latency (ms)", "p95_latency_ms"),
        ]

        for name, key in metrics:
            base = baseline.get(key, 0) or 0
            rer = rerank.get(key, 0) or 0
            if base == 0 and rer == 0:
                continue
            delta = rer - base
            delta_str = f"{delta:+.4f}"
            print(f"{name:<26} {base:<12.4f} {rer:<14.4f} {delta_str:<12}")

    def _write_summary(
        self, baseline: dict[str, Any], rerank: dict[str, Any]
    ) -> None:
        """Write a human-readable comparison into results/SUMMARY.md."""
        path = RESULTS_DIR / "SUMMARY.md"
        lines = [
            "# Ragify RAG Benchmark",
            "",
            f"- Generated: {datetime.now(timezone.utc).isoformat()}",
            f"- Queries: {baseline.get('total_queries', 0)}",
            f"- Top-K: {rerank.get('top_k', 5)}",
            f"- Reranker: {rerank.get('reranker', '-')} "
            f"({rerank.get('reranker_backend', '-')}, "
            f"initial_k={rerank.get('initial_k', 25)})",
            "",
            "| Metric | Baseline | With Reranker | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]

        rows = [
            ("Doc Recall@K", "doc_recall_at_k"),
            ("MRR@K", "mrr_at_k"),
            ("NDCG@K", "ndcg_at_k"),
            ("Chunk Relevance@K", "chunk_relevance_at_k"),
            ("Answer Fidelity", "generation_score"),
            ("Faithfulness (LLM judge)", "faithfulness_rate"),
            ("Avg Retrieval (ms)", "avg_retrieval_latency_ms"),
            ("Avg Rerank (ms)", "avg_rerank_latency_ms"),
            ("Avg Latency (ms)", "avg_latency_ms"),
            ("p95 Latency (ms)", "p95_latency_ms"),
        ]
        for name, key in rows:
            base = baseline.get(key)
            rer = rerank.get(key)
            if base is None and rer is None:
                continue
            base_s = f"{base:.4f}" if base is not None else "-"
            rer_s = f"{rer:.4f}" if rer is not None else "-"
            if base is not None and rer is not None:
                delta_s = f"{rer - base:+.4f}"
            else:
                delta_s = "-"
            lines.append(f"| {name} | {base_s} | {rer_s} | {delta_s} |")

        path.write_text("\n".join(lines) + "\n")
        print(f"Saved summary -> {path}")


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
