"""gRPC server exposing ragify-rag services to other processes (e.g. the API).

Run with:

    python -m src.grpc

or the ``ragify-server`` console script. The port/host can be configured with
``RAGIFY_GRPC_HOST`` and ``RAGIFY_GRPC_PORT``.
"""

import tempfile
from concurrent import futures
from os import getenv
from pathlib import Path

import grpc
from langchain_core.documents import Document

from src.core.ingestion.chunk_processor import process_section
from src.core.ingestion.grobid_ingestion import GrobidIngestor
from src.core.ingestion.transcoder import transcoder
from src.core.retrieval import vector_store_manager

from . import ragify_pb2, ragify_pb2_grpc

RAGIFY_GRPC_HOST = getenv("RAGIFY_GRPC_HOST", "0.0.0.0")
RAGIFY_GRPC_PORT = int(getenv("RAGIFY_GRPC_PORT", "50051"))
RAGIFY_GRPC_MAX_WORKERS = int(getenv("RAGIFY_GRPC_MAX_WORKERS", "10"))

_MARKDOWN_KINDS = {"md", "markdown", "txt", "text"}


def _convert_bytes_to_markdown(filename: str, content: bytes) -> str:
    with tempfile.TemporaryDirectory(prefix="ragify-md-") as tmp_dir:
        path = Path(tmp_dir) / (Path(filename).name or "document")
        path.write_bytes(content)
        markdown = transcoder.convert_to_markdown(str(path))
    if not markdown:
        raise ValueError("No valid content extracted")
    return markdown


class VectorStoreService(ragify_pb2_grpc.VectorStoreServiceServicer):
    """Wraps src.core.retrieval.vector_store_manager."""

    def CreateCollection(self, request, context):
        vector_store_manager.create_collection(str(request.workspace_id))
        return ragify_pb2.Empty()

    def DeleteCollection(self, request, context):
        vector_store_manager.delete_collection(str(request.workspace_id))
        return ragify_pb2.Empty()

    def InsertDocuments(self, request, context):
        documents = [
            Document(page_content=doc.text, metadata=dict(doc.metadata))
            for doc in request.documents
        ]
        vector_store_manager.insert_documents(str(request.workspace_id), documents)
        return ragify_pb2.InsertDocumentsResponse(inserted=len(documents))


class IngestionService(ragify_pb2_grpc.IngestionServiceServicer):
    """Wraps src.core.ingestion chunking / conversion / indexing."""

    def ProcessSection(self, request, context):
        chunks = process_section(list(request.paragraphs))
        return ragify_pb2.ProcessSectionResponse(chunks=chunks)

    def ConvertToMarkdown(self, request, context):
        markdown = _convert_bytes_to_markdown(request.filename, request.content)
        return ragify_pb2.ConvertToMarkdownResponse(markdown=markdown)

    def ProcessDocument(self, request, context):
        if not request.content:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "File content is empty")

        kind = (request.kind or "").lower()
        if kind in _MARKDOWN_KINDS:
            text = request.content.decode("utf-8", errors="replace")
        else:
            text = _convert_bytes_to_markdown(request.filename, request.content)

        chunks = process_section([text])
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": request.filename,
                    "chunk": str(index),
                    "chunk_type": request.chunk_type or kind,
                },
            )
            for index, chunk in enumerate(chunks)
        ]
        vector_store_manager.insert_documents(str(request.workspace_id), documents)
        return ragify_pb2.ProcessDocumentResponse(chunks=len(documents))

    def IngestPdf(self, request, context):
        results = []
        processed = 0
        failed = 0

        for pdf in request.files:
            try:
                with tempfile.TemporaryDirectory(prefix="ragify-pdf-") as tmp_dir:
                    pdf_path = Path(tmp_dir) / (Path(pdf.name).name or "document.pdf")
                    pdf_path.write_bytes(pdf.content)
                    GrobidIngestor(str(request.workspace_id), str(pdf_path.parent)).ingest()
                results.append(ragify_pb2.PdfResult(name=pdf.name, ok=True))
                processed += 1
            except Exception as exc:  # per-file failures don't kill the batch
                results.append(
                    ragify_pb2.PdfResult(name=pdf.name, ok=False, error=str(exc))
                )
                failed += 1

        return ragify_pb2.IngestPdfResponse(
            processed=processed,
            failed=failed,
            results=results,
        )


class RagService(ragify_pb2_grpc.RagServiceServicer):
    """Wraps the src.core.generation graph (retrieval + evaluation + generation)."""

    _ROLE_MESSAGE_TYPES = {
        "user": "human",
        "human": "human",
        "assistant": "ai",
        "ai": "ai",
        "system": "system",
    }

    def Query(self, request, context):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from src.core.generation import builder

        message_types = {
            "human": HumanMessage,
            "ai": AIMessage,
            "system": SystemMessage,
        }
        messages = []
        for msg in request.messages:
            kind = self._ROLE_MESSAGE_TYPES.get(msg.role, "human")
            messages.append(message_types[kind](content=msg.content))

        result = builder.invoke(
            {
                "messages": messages,
                "workspace_id": request.workspace_id,
            }
        )
        answer = result["messages"][-1].content
        return ragify_pb2.QueryResponse(answer=answer)


def build_server() -> grpc.Server:
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=RAGIFY_GRPC_MAX_WORKERS)
    )
    ragify_pb2_grpc.add_VectorStoreServiceServicer_to_server(
        VectorStoreService(), server
    )
    ragify_pb2_grpc.add_IngestionServiceServicer_to_server(IngestionService(), server)
    ragify_pb2_grpc.add_RagServiceServicer_to_server(RagService(), server)
    return server


def serve(host: str | None = None, port: int | None = None) -> None:
    """Start the ragify-rag gRPC server and block until terminated."""
    host = host or RAGIFY_GRPC_HOST
    port = port or RAGIFY_GRPC_PORT

    server = build_server()
    bound = server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"[ragify-grpc] listening on {host}:{bound}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("[ragify-grpc] shutting down")
        server.stop(0)
