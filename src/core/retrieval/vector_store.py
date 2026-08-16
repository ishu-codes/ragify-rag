from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.ragify.retrieval.embedder import embeddings
from src.ragify.utils.config import VECTOR_SIZE, VECTORDB_URL


class VectorStoreManager:
    def __init__(
        self,
        url: str = VECTORDB_URL,
        api_key: str | None = None,
        vector_size: int = VECTOR_SIZE,
    ):
        self.url = url
        self.api_key = api_key
        self.vector_size = vector_size
        self._client = QdrantClient(url=url, api_key=api_key, timeout=10)
        self._stores: dict[str, QdrantVectorStore] = {}

    def create_collection(self, collection_name: str) -> None:
        if collection_name not in [
            c.name for c in self._client.get_collections().collections
        ]:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def get_or_create(
        self,
        collection_name: str,
        documents: list[Document] | None = None,
    ) -> QdrantVectorStore:
        if collection_name in self._stores:
            return self._stores[collection_name]

        # Check if collection exists on the server
        collection_exists = False
        try:
            self._client.get_collection(collection_name=collection_name)
            collection_exists = True
        except Exception:
            # Collection doesn't exist
            collection_exists = False

        if documents:
            store = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=embeddings.client,
                url=self.url,
                collection_name=collection_name,
            )
        elif collection_exists:
            # Use existing collection from server
            store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings.client,
                url=self.url,
                collection_name=collection_name,
            )
        else:
            # Create new empty collection
            self.create_collection(collection_name)
            store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings.client,
                url=self.url,
                collection_name=collection_name,
            )

        self._stores[collection_name] = store
        return store

    def get(self, collection_name: str) -> QdrantVectorStore | None:
        return self._stores.get(collection_name)

    @property
    def client(self) -> QdrantClient:
        return self._client

    def delete_collection(self, collection_name: str) -> bool:
        return self._client.delete_collection(collection_name)

    def insert_documents(
        self,
        collection_name: str,
        documents: list[Document],
    ) -> None:
        store = self.get_or_create(collection_name)
        store.add_documents(documents)

    def insert_points(
        self,
        collection_name: str,
        points: list[PointStruct],
    ) -> None:
        self._client.upsert(
            collection_name=collection_name,
            points=points,
        )


vector_store_manager = VectorStoreManager()
