from typing import Sequence

from langchain_ollama import OllamaEmbeddings
from ollama import embed as ollama_embed

from src.ragify.utils.config import EMBED_MODEL


class Embedder:
    def __init__(self, model: str = EMBED_MODEL):
        self.model = model
        self._client = OllamaEmbeddings(model=model)

    @property
    def client(self) -> OllamaEmbeddings:
        return self._client

    def encode(self, input: str | Sequence[str]) -> list[list[float]]:
        result = ollama_embed(model=self.model, input=input)
        return result.get("embeddings")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.encode(text)[0]


embeddings = Embedder()
