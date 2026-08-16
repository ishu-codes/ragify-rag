import statistics
import time
from typing import Any, Dict, List

from src.utils.logger import QueryLogger

# ---------------------------
# Data Structures
# ---------------------------

class Chunk:
    def __init__(self, id: str, text: str, doc_id: str):
        self.id = id
        self.text = text
        self.doc_id = doc_id


# ---------------------------
# Retrieval Engine Wrapper
# ---------------------------

class RAGRetriever:
    def __init__(self, embed_fn, vector_db, reranker=None):
        self.embed_fn = embed_fn
        self.vector_db = vector_db
        self.reranker = reranker

    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        q_vec = self.embed_fn(query)
        chunks = self.vector_db.search(q_vec, k=20)  # initial fetch

        if self.reranker:
            chunks = self.reranker(query, chunks)

        return chunks[:k]


# ---------------------------
# Evaluation Logic
# ---------------------------

class Evaluator:
    def __init__(self, retriever, llm_generate=None, llm_judge=None):
        self.retriever = retriever
        self.llm_generate = llm_generate
        self.llm_judge = llm_judge

    # ---- Document-level recall ----
    def doc_recall(self, retrieved: List[Chunk], relevant_docs: List[str]) -> int:
        retrieved_docs = {c.doc_id for c in retrieved}
        return int(any(doc in retrieved_docs for doc in relevant_docs))

    # ---- Chunk-level relevance (keyword-based) ----
    def chunk_relevance_keyword(self, chunks: List[Chunk], expected_points: List[str]) -> int:
        relevant = 0
        for c in chunks:
            text = c.text.lower()
            if any(point.lower() in text for point in expected_points):
                relevant += 1
        return relevant

    # ---- Chunk-level relevance (LLM-as-judge) ----
    def chunk_relevance_llm(self, query: str, chunks: List[Chunk]) -> int:
        if not self.llm_judge:
            return 0

        relevant = 0
        for c in chunks:
            prompt = f"""
            Query: {query}
            Chunk: {c.text}

            Is this chunk useful to answer the query? Answer YES or NO.
            """
            result = self.llm_judge(prompt).strip().upper()
            if "YES" in result:
                relevant += 1
        return relevant

    # ---- Generation evaluation ----
    def evaluate_generation(self, query: str, chunks: List[Chunk], expected_points: List[str]) -> Dict[str, Any]:
        if not self.llm_generate:
            return {}

        context = "\n\n".join([c.text for c in chunks])
        answer = self.llm_generate(query, context)

        # simple scoring: keyword coverage
        score = sum(1 for p in expected_points if p.lower() in answer.lower())
        total = len(expected_points)

        return {
            "answer": answer,
            "score": score,
            "total": total
        }

    # ---- Full evaluation ----
    from src.utils.logger import QueryLogger


    def run(self, queries, k=5, run_id="default"):
        logger = QueryLogger(run_id)

        doc_hits = []
        chunk_scores = []
        gen_scores = []
        latencies = []

        for q in queries:
            start = time.time()

            retrieved = self.retriever.retrieve(q["query"], k=k)

            latency = time.time() - start
            latencies.append(latency)

            # doc-level
            doc_hit = self.doc_recall(retrieved, q["relevant_docs"])
            doc_hits.append(doc_hit)

            # chunk-level
            chunk_rel = self.chunk_relevance_keyword(retrieved, q["expected_points"])
            chunk_score = chunk_rel / k
            chunk_scores.append(chunk_score)

            # generation
            gen = self.evaluate_generation(q["query"], retrieved, q["expected_points"])
            gen_score = None
            answer = None

            if gen:
                gen_score = gen["score"] / gen["total"] if gen["total"] else 0
                gen_scores.append(gen_score)
                answer = gen["answer"]

            # LOG PER QUERY
            logger.log({
                "query_id": q["id"],
                "query": q["query"],
                "doc_hit": doc_hit,
                "chunk_score": chunk_score,
                "latency": latency,
                "retrieved_doc_ids": [c.doc_id for c in retrieved],
                "retrieved_chunks": [c.text[:200] for c in retrieved],  # truncate
                "generation_score": gen_score,
                "answer": answer
            })

        return {
            "doc_recall@k": sum(doc_hits) / len(doc_hits),
            "chunk_relevance@k": sum(chunk_scores) / len(chunk_scores),
            "generation_score": sum(gen_scores) / len(gen_scores) if gen_scores else None,
            "avg_latency": statistics.mean(latencies),
            "p95_latency": statistics.quantiles(latencies, n=20)[-1] if len(latencies) > 1 else latencies[0],
        }
