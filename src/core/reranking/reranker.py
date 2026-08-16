import io
import logging
import os
from contextlib import redirect_stderr
from typing import Sequence

from langchain_core.documents import Document

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

for lib in ["transformers", "transformers.modeling_utils"]:
    logging.getLogger(lib).setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

_dummy_stderr = io.StringIO()
reranker_instance = None


class Reranker:
    def __init__(
        self,
        model: str | None = None,
        backend: str = "transformers",
        silent: bool = False,
    ):
        from src.ragify.utils.config import RERANKER_MODEL

        self.model_name = model if model is not None else RERANKER_MODEL
        self.backend = backend
        self._tokenizer = None
        self._model = None
        self.silent = silent
        self._load_model()

    def _load_model(self):
        if self.model_name is None:
            return

        try:
            with redirect_stderr(_dummy_stderr):
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
                self._model.eval()
                if not self.silent:
                    logger.info(f"Loaded transformers reranker: {self.model_name}")
        except Exception as e:
            if not self.silent:
                logger.warning(f"Could not load reranker model {self.model_name}: {e}")
            self._tokenizer = None
            self._model = None

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    def rerank(
        self, query: str, docs: Sequence[Document], top_k: int = 4
    ) -> list[Document]:
        if not docs:
            return list(docs)[:top_k]

        if self._model is None:
            print("Warning: Reranker model not loaded, skipping rerank")
            return list(docs)[:top_k]

        return self._rerank_transformers(query, docs, top_k)

    def _rerank_transformers(
        self, query: str, docs: Sequence[Document], top_k: int = 4
    ) -> list[Document]:
        try:
            import torch

            pairs = [[query, doc.page_content] for doc in docs]
            inputs = self._tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt"
            )

            with torch.no_grad():
                scores = self._model(**inputs).logits.view(-1)

            ranked = sorted(
                zip(docs, scores.tolist()), key=lambda x: x[1], reverse=True
            )
            return [doc for doc, _ in ranked[:top_k]]
        except Exception as e:
            print(f"Reranking failed: {e}")
            return list(docs)[:top_k]


def get_reranker(backend: str | None = None, silent: bool = False):
    from src.ragify.utils.config import RERANKER_BACKEND

    global reranker_instance
    if backend is None:
        backend = RERANKER_BACKEND

    if reranker_instance is None or reranker_instance.backend != backend:
        reranker_instance = Reranker(backend=backend, silent=silent)

    return reranker_instance


reranker = get_reranker(silent=True)
