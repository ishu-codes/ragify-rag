import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from grobid2json import convert_xml_to_json
from grobid_client.grobid_client import GrobidClient
from langchain_core.documents import Document
from ollama import ResponseError

from src.ragify.ingestion.chunk_processor import process_section
from src.ragify.retrieval import vector_store_manager
from src.utils.files import get_file_content
from src.utils.json import load_json_file, save_to_json
from src.utils.threads import run_in_threads


class GrobidIngestor:
    def __init__(self, collection_name: str, input_dir: str, output_dir: str|None = None) -> None:
        self._collection_name = collection_name
        # _input_dir for pdf
        self._input_dir = input_dir

        # _output_dir for xml and json
        self._output_dir = input_dir if output_dir is None else output_dir
        self._grobid_client = GrobidClient()


    def ingest(self):
        self._convert_pdf_to_xml()  # using grobid
        self._convert_xml_to_json()  # using grobid2json
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

        json_content = convert_xml_to_json(soup, paper_id, "")
        save_to_json(f'{self._output_dir}/{paper_id}.json', json_content.as_json())


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

        paper_id = paper.get("paper_id", "unknown")
        title = paper.get("metadata", {}).get("title", "Untitled")

        abstract_texts = [a.get("text", "") for a in paper.get("abstract", [])]
        abstract_text = " ".join([t for t in abstract_texts if t])

        if abstract_text:
            parent_id = f"{paper_id}_abstract"
            parent_store[parent_id] = abstract_text

            chunks = process_section([abstract_text])
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

        for para in paper.get("body_text", []):
            sections[para.get("section", "unknown")].append(para.get("text", ""))

        for para in paper.get("back_matter", []):
            sections[para.get("section", "back_matter")].append(para.get("text", ""))

        for sec_name, paras in sections.items():
            parent_id = f"{paper_id}_{sec_name}"
            full_text = " ".join(paras)
            parent_store[parent_id] = full_text

            chunks = process_section(paras)
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
                vector_store_manager.insert_documents(self._collection_name, docs)

        except ResponseError as err:
            print(f"Error in {paper_id}: {err}")

        finally:
            return parent_store
