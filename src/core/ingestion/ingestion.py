from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ragify.retrieval import vector_store_manager
from src.ragify.utils.config import MAX_TOKENS, OVERLAP


class DocumentIngester:
    def __init__(
        self,
        chunk_size: int = MAX_TOKENS,
        chunk_overlap: int = OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @property
    def splitter(self) -> RecursiveCharacterTextSplitter:
        return self._splitter

    def split_text(self, text: str) -> list[str]:
        return self._splitter.split_text(text)

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return self._splitter.split_documents(documents)

    def create_chunks(
        self, text: str, source_name: str, metadata: dict | None = None
    ) -> list[Document]:
        chunks = self.split_text(text)
        meta = metadata or {}
        return [
            Document(
                page_content=chunk, metadata={**meta, "source": source_name, "chunk": i}
            )
            for i, chunk in enumerate(chunks)
        ]

    def index_documents(self, documents: list[Document], collection_name: str) -> None:
        vector_store_manager.get_or_create(collection_name, documents=documents)

    def process_and_index(
        self,
        text: str,
        collection_name: str,
        source_name: str,
        metadata: dict | None = None,
    ) -> list[Document]:
        chunks = self.create_chunks(text, source_name, metadata)
        self.index_documents(chunks, collection_name)
        return chunks


ingester = DocumentIngester()
