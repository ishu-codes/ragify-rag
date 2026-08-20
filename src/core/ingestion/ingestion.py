from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.retrieval import vector_store_manager
from src.core.utils.config import MAX_TOKENS, OVERLAP
from src.core.utils.logger import get_logger
from src.core.utils.text_quality import is_degenerate

logger = get_logger("ragify.ingestion")


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
        raw_chunks = self.split_text(text)
        chunks = [chunk for chunk in raw_chunks if not is_degenerate(chunk)]
        if len(chunks) != len(raw_chunks):
            logger.warning(
                "skipped_degenerate_chunks",
                extra={
                    "source": source_name,
                    "skipped": len(raw_chunks) - len(chunks),
                    "kept": len(chunks),
                },
            )
        meta = metadata or {}
        return [
            Document(
                page_content=chunk, metadata={**meta, "source": source_name, "chunk": i}
            )
            for i, chunk in enumerate(chunks)
        ]

    def index_documents(self, documents: list[Document], collection_name: str) -> None:
        vector_store_manager.insert_documents(collection_name, documents)

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
