import json
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from grobid2json import convert_xml_to_json
from grobid_client.grobid_client import GrobidClient
from langchain_core.documents import Document
from ollama import ResponseError

from src.core.ingestion.chunk_processor import process_section
from src.core.retrieval import vector_store_manager
from src.core.utils.config import ARTIFACTS_ROOT
from src.core.utils.logger import get_logger
from src.core.utils.text_quality import is_degenerate
from src.utils.files import get_file_content
from src.utils.json import load_json_file, save_to_json
from src.utils.threads import run_in_threads

logger = get_logger("ragify.grobid")


class GrobidIngestor:
    def __init__(self, collection_name: str, input_dir: str, output_dir: str|None = None) -> None:
        self._collection_name = collection_name
        # _input_dir for pdf
        self._input_dir = input_dir

        # _output_dir for xml and json
        self._output_dir = input_dir if output_dir is None else output_dir
        self._grobid_client = GrobidClient()
        # Persisted artifacts (pdf / xml / json / chunks) for debugging, under
        # source/workspace/{workspace_id} in the rag directory.
        self._artifacts_root = Path(ARTIFACTS_ROOT) / str(collection_name)


    def ingest(self):
        self._convert_pdf_to_xml()  # using grobid
        self._convert_xml_to_json()  # using grobid2json
        self._persist_artifacts()  # keep pdf/xml/json for inspection
        self._ingest_json_files()  # into qdrant vector db


    # File conversions
    def _convert_pdf_to_xml(self) -> None:
        output_dir_path = Path(self._output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        self._grobid_client.process(
            service="processFulltextDocument",
            input_path=self._input_dir,
            output=str(output_dir_path),
            consolidate_header=True,
        )

    def _convert_xml_to_json(self) -> None:
        xmls = Path(self._output_dir).glob("*.grobid.tei.xml")
        run_in_threads(
            self._load_xml_and_save_to_json,
            xmls
        )

    def _load_xml_and_save_to_json(self, file_path: str) -> None:
        from bs4 import BeautifulSoup

        paper_id = ".".join(Path(file_path).stem.split(".")[:2])
        xml_data = get_file_content(file_path) or ''
        soup = BeautifulSoup(xml_data, "xml")

        self._sanitize_figures_for_grobid2json(soup)
        json_content = convert_xml_to_json(soup, paper_id, "")
        save_to_json(f'{self._output_dir}/{paper_id}.json', json_content.as_json())

    def _persist_artifacts(self) -> None:
        """Copy PDF, XML and JSON artifacts into source/workspace/{workspace_id}."""
        target_dirs = {
            "pdf": self._artifacts_root / "pdf",
            "xml": self._artifacts_root / "xml",
            "json": self._artifacts_root / "json",
        }
        for directory in target_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        sources = [
            (Path(self._input_dir).glob("*.pdf"), target_dirs["pdf"]),
            (Path(self._output_dir).glob("*.grobid.tei.xml"), target_dirs["xml"]),
            (Path(self._output_dir).glob("*.json"), target_dirs["json"]),
        ]
        for files, target_dir in sources:
            for file_path in files:
                try:
                    shutil.copy2(file_path, target_dir / file_path.name)
                except OSError as exc:
                    logger.warning(
                        "artifact_copy_failed",
                        extra={"file": str(file_path), "error": str(exc)},
                    )

        logger.info(
            "artifacts_persisted",
            extra={
                "workspace_id": self._collection_name,
                "root": str(self._artifacts_root),
            },
        )

    @staticmethod
    def _filter_chunks(chunks: list[str], paper_id: str, section: str) -> list[str]:
        """Drop degenerate chunks (broken extraction) before indexing."""
        kept = [chunk for chunk in chunks if not is_degenerate(chunk)]
        if len(kept) != len(chunks):
            logger.warning(
                "skipped_degenerate_chunks",
                extra={
                    "paper_id": paper_id,
                    "section": section,
                    "skipped": len(chunks) - len(kept),
                    "kept": len(kept),
                },
            )
        return kept

    def _save_chunks(self, paper_id: str, docs: list[Document], point_ids: list[str]) -> None:
        """Write the indexed chunks (with vector point ids) to a JSONL file."""
        if len(point_ids) != len(docs):
            logger.warning(
                "chunk_point_id_mismatch",
                extra={"paper_id": paper_id, "docs": len(docs), "points": len(point_ids)},
            )

        chunks_dir = self._artifacts_root / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        target = chunks_dir / f"{paper_id}.jsonl"
        with open(target, "w", encoding="utf-8") as handle:
            for doc, point_id in zip(docs, point_ids):
                record = {
                    "point_id": point_id,
                    "char_count": len(doc.page_content),
                    "text": doc.page_content,
                    **doc.metadata,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "chunks_saved",
            extra={
                "paper_id": paper_id,
                "chunks": len(docs),
                "path": str(target),
            },
        )

    @staticmethod
    def _sanitize_figures_for_grobid2json(soup) -> None:
        """Work around grobid2json assuming every non-table figure has a
        ``<head>`` followed by a ``<label>``.

        Script-style PDFs (e.g. ``endgame_script.pdf``) can produce figures
        whose following ``<label>`` is missing, which crashes
        ``extract_figures_and_tables_from_tei_xml`` with
        ``TypeError: 'NoneType' object is not iterable``. Give those figures a
        placeholder label so conversion keeps working.
        """
        for fig in soup.find_all("figure"):
            if fig.get("type") == "table" or not fig.get("xml:id"):
                continue
            head = fig.findNext("head")
            if head is None or head.findNext("label") is not None:
                continue
            label = soup.new_tag("label")
            label.string = fig.get("xml:id", "")
            head.insert_after(label)

    # File ingestion
    def _ingest_json_files(self) -> None:
        json_files_path = Path(self._output_dir).glob("*.json")
        run_in_threads(
            self._load_paper_and_ingest,
            json_files_path
        )

    def _load_paper_and_ingest(self, file_path: str) -> None:
        json_content = load_json_file(file_path)
        self._ingest_paper_from_json(json_content)

    def _ingest_paper_from_json(self, paper: dict[str, Any]) -> dict[str, str]:
        docs = []
        parent_store = {}

        if not isinstance(paper, dict):
            return parent_store

        paper_id = paper.get("paper_id", "unknown")
        metadata = paper.get("metadata") or {}
        title = metadata.get("title", "Untitled")

        abstract = paper.get("abstract") or []
        body_text = paper.get("body_text") or []
        back_matter = paper.get("back_matter") or []

        abstract_texts = [a.get("text", "") for a in abstract if isinstance(a, dict)]
        abstract_text = " ".join([t for t in abstract_texts if t])

        if abstract_text:
            parent_id = f"{paper_id}_abstract"
            parent_store[parent_id] = abstract_text

            chunks = self._filter_chunks(
                process_section([abstract_text]), paper_id, "abstract"
            )
            docs.extend([
                Document(
                    page_content=chunk,
                    metadata={
                        "id": str(uuid.uuid4()),
                        "paper_id": paper_id,
                        "title": title,
                        "section": "abstract",
                        "parent_id": parent_id,
                        "abstract_index": i,
                        "level": "child",
                    },
                )
                for (i, chunk) in enumerate(chunks)
            ])


        sections = defaultdict(list)

        for para in body_text:
            if not isinstance(para, dict):
                continue
            sections[para.get("section", "unknown")].append(para.get("text", ""))

        for para in back_matter:
            if not isinstance(para, dict):
                continue
            sections[para.get("section", "back_matter")].append(para.get("text", ""))

        for sec_name, paras in sections.items():
            parent_id = f"{paper_id}_{sec_name}"
            full_text = " ".join(paras)
            parent_store[parent_id] = full_text

            chunks = self._filter_chunks(process_section(paras), paper_id, sec_name)
            docs.extend([
                Document(
                    page_content=chunk,
                    metadata={
                        "id": str(uuid.uuid4()),
                        "paper_id": paper_id,
                        "title": title,
                        "section": sec_name,
                        "parent_id": parent_id,
                        "chunk_index": i,
                        "level": "child",
                    },
                )
                for (i, chunk) in enumerate(chunks)
            ])

        try:
            if docs:
                point_ids = vector_store_manager.insert_documents(
                    self._collection_name, docs
                )
                self._save_chunks(paper_id, docs, point_ids)

        except ResponseError as err:
            logger.exception("ingest_failed", extra={"paper_id": paper_id, "error": str(err)})

        return parent_store
