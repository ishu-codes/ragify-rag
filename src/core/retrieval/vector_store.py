import uuid

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.core.retrieval.embedder import embeddings
from src.core.utils.config import VECTORDB_API_KEY, VECTOR_SIZE, VECTORDB_URL
from src.core.utils.logger import get_logger

logger = get_logger("ragify.vector_store")

_EMBED_BATCH_SIZE = 64


class VectorStoreManager:
    def __init__(
        self,
        url: str = VECTORDB_URL,
        api_key: str | None = VECTORDB_API_KEY or None,
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
                api_key=self.api_key,
            )
        elif collection_exists:
            # Use existing collection from server
            store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings.client,
                url=self.url,
                collection_name=collection_name,
                api_key=self.api_key,
            )
        else:
            # Create new empty collection
            self.create_collection(collection_name)
            store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings.client,
                url=self.url,
                collection_name=collection_name,
                api_key=self.api_key,
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
    ) -> list[str]:
        """Embed and upsert documents in batches.

        Returns the point ids in document order so callers can persist a
        mapping from chunk text to the vector database for monitoring.
        """
        if not documents:
            return []

        # Ensure the collection exists without embedding anything (the old
        # path passed documents into get_or_create, which embedded each chunk
        # through langchain one call at a time).
        self.get_or_create(collection_name)

        # Batch embeddings and upserts: one embed call + one blocking upsert
        # per batch instead of one round-trip per chunk.
        point_ids: list[str] = []
        for start in range(0, len(documents), _EMBED_BATCH_SIZE):
            batch = documents[start : start + _EMBED_BATCH_SIZE]
            vectors = embeddings.encode([d.page_content for d in batch])
            points = []
            for index, (d, vector) in enumerate(zip(batch, vectors)):
                point_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{collection_name}:{d.page_content}:{start + index}",
                    )
                )
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        # Keep langchain's payload contract so the retriever
                        # (QdrantVectorStore) reconstructs Documents correctly.
                        payload={"page_content": d.page_content, "metadata": d.metadata},
                    )
                )
            self._client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
            point_ids.extend(point.id for point in points)
            logger.info(
                "upsert_batch",
                extra={
                    "collection": collection_name,
                    "points": len(points),
                    "batch_index": start // _EMBED_BATCH_SIZE,
                },
            )
        return point_ids

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
