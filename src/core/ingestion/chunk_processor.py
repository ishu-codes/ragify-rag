import numpy as np
import tiktoken
from qdrant_client import QdrantClient

from src.ragify.retrieval.embedder import embeddings
from src.ragify.utils.config import VECTORDB_URL

SIM_THRESHOLD = 0.75
MAX_CHUNK_TOKENS = 4096


client = QdrantClient(url=VECTORDB_URL)
tokenizer = tiktoken.get_encoding("cl100k_base")

def token_len(text: str) -> int:
    return len(tokenizer.encode(text))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in text.split(". ") if s.strip()]


def semantic_chunk(text: str) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    safe_sentences = [s for s in sentences if token_len(s) <= MAX_CHUNK_TOKENS]
    if not safe_sentences:
        return []

    try:
        emb = embeddings.encode(safe_sentences)
    except Exception:
        return safe_sentences

    chunks = []
    curr = [safe_sentences[0]]

    for i in range(1, len(safe_sentences)):
        sim = cosine(emb[i - 1], emb[i])
        if sim < SIM_THRESHOLD:
            chunks.append(" ".join(curr))
            curr = []
        curr.append(safe_sentences[i])

    if curr:
        chunks.append(" ".join(curr))

    return chunks


def dynamic_overlap(size: int) -> int:
    if size < 200:
        return 20
    elif size < 500:
        return 50
    return 100


def split_by_tokens(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[str]:
    words = text.split()
    chunks = []
    current = []
    current_len = 0

    for word in words:
        word_len = token_len(word)
        if current_len + word_len > max_tokens and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = word_len
        else:
            current.append(word)
            current_len += word_len

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if token_len(c) <= max_tokens]


def merge_chunks(chunks: list[str]) -> list[str]:
    merged = []
    for i, c in enumerate(chunks):
        if i > 0:
            overlap = dynamic_overlap(token_len(c))
            merged[-1] += " " + c[:overlap]
        merged.append(c)
    return merged


def process_section(paragraphs: list[str]) -> list[str]:
    text = " ".join(paragraphs)
    t = token_len(text)

    if t < 300:
        return [text]

    elif t < 1500:
        chunks = []
        for p in paragraphs:
            if token_len(p) < 200:
                chunks.append(p)
            else:
                chunks.extend(split_sentences(p))
        chunks = merge_chunks(chunks)

    else:
        chunks = merge_chunks(semantic_chunk(text))

    final_chunks = []
    for chunk in chunks:
        if token_len(chunk) > MAX_CHUNK_TOKENS:
            final_chunks.extend(split_by_tokens(chunk))
        else:
            final_chunks.append(chunk)

    return final_chunks
