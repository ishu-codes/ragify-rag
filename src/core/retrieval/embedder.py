"""Embedding backends, selected with ``EMBED_BACKEND``.

``transformers`` (default)
    Loads the upstream safetensors weights in-process with ``torch``. No model
    server is needed at runtime.

``ollama``
    Delegates to a local Ollama server (``OLLAMA_HOST``), which serves a
    GGUF-quantised build of the model as a separate process. Kept as a fallback.

Both expose the LangChain ``Embeddings`` interface, so the vector store works
with either one without changes. Models are loaded lazily on first use.
"""

import threading
from typing import Sequence

from langchain_core.embeddings import Embeddings

from src.core.utils.config import (
    EMBED_BACKEND,
    EMBED_BATCH_SIZE,
    EMBED_MODEL,
    EMBED_MODEL_HF,
)

# BGE retrieval models are trained with an instruction prefix on the *query*
# side only: passages are embedded bare, queries carry the instruction.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_MAX_SEQ_LENGTH = 512


def configure_embedder_threads(workers: int) -> None:
    """Pin torch's intra-op thread count when fanning out across workers.

    torch defaults to roughly one thread per core. With a pool of N workers each
    starting its own forward pass, letting every pass grab all cores
    oversubscribes the CPU and total throughput drops, so the parallelism is
    left to the pool.
    """
    if workers <= 1:
        return
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001 - purely an optimisation
        pass_load_lock
        _load_lock


class EmbeddingError(RuntimeError):
    """Raised when the active backend cannot produce embeddings.

    Backend-neutral, so callers such as ingestion do not need to know which
    backend is configured.
    """


class Embedder(Embeddings):
    """Embedding facade over the configured backend.

    Subclasses ``langchain_core.embeddings.Embeddings`` rather than merely
    implementing the same methods: ``QdrantVectorStore`` validates its
    ``embedding`` argument with ``isinstance(..., Embeddings)``, so a class that
    only duck-types the interface is rejected with "Invalid `embeddings` type."
    """

    def __init__(
        self,
        model: str | None = None,
        backend: str | None = None,
        query_instruction: str | None = None,
    ):
        self.backend = (backend or EMBED_BACKEND).strip().lower()
        if self.backend not in ("ollama", "transformers"):
            raise ValueError(
                f"Unknown EMBED_BACKEND {self.backend!r}; "
                "expected 'ollama' or 'transformers'"
            )

        # The default model differs per backend: the Ollama name is a community
        # repack, not a valid Hugging Face repo id.
        default_model = (
            EMBED_MODEL_HF if self.backend == "transformers" else EMBED_MODEL
        )
        self.model = model or default_model
        self.query_instruction = (
            BGE_QUERY_INSTRUCTION if query_instruction is None else query_instruction
        )

        # Populated lazily so importing this module never loads a model.
        self._ollama_client = None
        self._torch = None
        self._tokenizer = None
        self._model = None
        # Ingest fans out across threads, so loading has to be race-free.
        self._load_lock = threading.Lock()
        self._batch_size = EMBED_BATCH_SIZE

    # ------------------------------------------------------------------ ollama

    def _get_ollama_client(self):
        if self._ollama_client is None:
            from langchain_ollama import OllamaEmbeddings

            self._ollama_client = OllamaEmbeddings(model=self.model)
        return self._ollama_client

    def _encode_ollama(self, input: str | Sequence[str]) -> list[list[float]]:
        from ollama import embed as ollama_embed

        result = ollama_embed(model=self.model, input=input)
        return result.get("embeddings")

    # ------------------------------------------------------------ transformers

    def _load_transformers(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:  # another thread won the race
                return

            import torch
            from transformers import AutoModel, AutoTokenizer

            self._torch = torch
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            model = AutoModel.from_pretrained(self.model)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model

    def _encode_transformers_batched(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode in fixed-size batches.

        A single forward pass over every input would size its activations by the
        whole corpus, so callers can hand over an arbitrarily long list.
        """
        if len(texts) <= self._batch_size:
            return self._encode_transformers(texts)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(
                self._encode_transformers(texts[start : start + self._batch_size])
            )
        return vectors

    def _encode_transformers(self, texts: Sequence[str]) -> list[list[float]]:
        self._load_transformers()
        torch = self._torch

        inputs = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=_MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = self._model(**inputs).last_hidden_state

        # BGE pools by mean over token embeddings, ignoring padding.
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        return torch.nn.functional.normalize(pooled, p=2, dim=1).tolist()

    # ----------------------------------------------------------------- public

    @property
    def client(self):
        """LangChain-compatible embedding object for the active backend."""
        if self.backend == "transformers":
            # Embedder subclasses Embeddings, so it can be handed to the vector
            # store directly.
            return self
        return self._get_ollama_client()

    def encode(self, input: str | Sequence[str]) -> list[list[float]]:
        try:
            if self.backend == "transformers":
                texts = [input] if isinstance(input, str) else list(input)
                return self._encode_transformers_batched(texts)
            return self._encode_ollama(input)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(
                f"{self.backend} embedding failed for model {self.model}: {exc}"
            ) from exc

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        if self.backend == "transformers":
            return self.encode([self.query_instruction + text])[0]
        return self.encode(text)[0]


embeddings = Embedder()
