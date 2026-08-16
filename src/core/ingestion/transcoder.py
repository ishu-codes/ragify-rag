from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from src.ragify.ingestion.ingestion import ingester
from src.ragify.retrieval import vector_store_manager
from src.ragify.utils.config import MAX_TOKENS, OVERLAP


class PDFTranscoder:
    def __init__(self):
        self._md_converter = None

    @property
    def md_converter(self):
        if self._md_converter is None:
            from markitdown import MarkItDown

            self._md_converter = MarkItDown(enable_plugins=False)
        return self._md_converter

    def convert_to_markdown(self, path: str) -> str:
        result = self.md_converter.convert(path)
        return result.text_content

    def extract_metadata(self, text: str) -> dict[str, str]:
        meta: dict[str, str] = {}
        text_lower = text.lower()

        abstract_start = text_lower.find("abstract")
        if abstract_start != -1:
            abstract_end = text_lower.find("introduction")
            if abstract_end == -1:
                abstract_end = min(abstract_start + 800, len(text))
            abstract = text[abstract_start:abstract_end].strip()[:500]
            meta["abstract"] = abstract

        return meta

    def process_pdf(
        self,
        pdf_path: str,
        collection_name: str,
        source_name: str | None = None,
    ) -> list[Document]:
        source_name = source_name or Path(pdf_path).stem

        content = self.convert_to_markdown(pdf_path)
        metadata = self.extract_metadata(content)

        chunks: list[Document] = []

        if "abstract" in metadata:
            chunks.append(
                Document(
                    page_content=metadata["abstract"],
                    metadata={"source": source_name, "chunk_type": "abstract"},
                )
            )

        content_text = content
        content_lower = content.lower()
        if "abstract" in content_lower:
            intro_idx = content_lower.find("introduction")
            if intro_idx != -1:
                content_text = content[intro_idx:]

        max_tokens = 512
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens,
            chunk_overlap=OVERLAP,
        )
        content_chunks = text_splitter.split_text(content_text)

        for i, chunk in enumerate(content_chunks):
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source_name,
                        "chunk_type": "content",
                        "chunk_index": i,
                    },
                )
            )

        vector_store_manager.get_or_create(collection_name, documents=chunks)
        return chunks


transcoder = PDFTranscoder()
