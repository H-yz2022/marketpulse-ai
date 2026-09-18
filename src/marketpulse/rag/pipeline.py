"""Retrieval-augmented generation over SEC filing text using ChromaDB + Claude.

Chunking and retrieval work fully offline (Chroma's default embedding
function downloads a small local model the first time it runs, then it's
cached). Only the final answer-generation step calls the Anthropic API, and
it's isolated behind a swappable `generate_fn` so tests/demos can run
without a real API key.
"""
from __future__ import annotations

from typing import Callable, Optional

from marketpulse.config import settings

_CLIENT = None
_COLLECTION_NAME = "filings"


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks for embedding/retrieval."""
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks


def _get_collection():
    global _CLIENT
    import chromadb

    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _CLIENT.get_or_create_collection(_COLLECTION_NAME)


def index_document(doc_id: str, text: str, metadata: dict) -> int:
    """Chunk a filing and add it to the vector store. Returns chunk count."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    collection = _get_collection()
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metadatas = [dict(metadata, chunk_index=i) for i in range(len(chunks))]
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return len(chunks)


def retrieve(query: str, n_results: int = 4, where: Optional[dict] = None) -> list[dict]:
    """Retrieve the most relevant indexed chunks for a natural-language query."""
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=n_results, where=where)
    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(docs)
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits


def _default_generate_answer(question: str, context_chunks: list[str]) -> str:
    """Call the Anthropic API to answer a question grounded in retrieved chunks."""
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env, or pass a custom "
            "`generate_fn` to `answer_question` (useful for tests/offline demos)."
        )
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are a financial research assistant. Answer the question using ONLY "
        "the filing excerpts below. Cite which excerpt(s) you used. If the "
        "excerpts don't contain the answer, say so explicitly.\n\n"
        f"Filing excerpts:\n{context}\n\nQuestion: {question}"
    )
    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def answer_question(
    question: str,
    n_results: int = 4,
    where: Optional[dict] = None,
    generate_fn: Optional[Callable[[str, list[str]], str]] = None,
) -> dict:
    """Retrieve relevant filing chunks and generate a grounded answer.

    Pass `generate_fn` to swap in a mock/local model for tests or demos
    that shouldn't require an Anthropic API key.
    """
    hits = retrieve(question, n_results=n_results, where=where)
    context_chunks = [h["text"] for h in hits]
    if not context_chunks:
        return {
            "answer": (
                "No indexed filings matched this question yet. Run the ingestion "
                "and indexing pipeline first (see scripts/run_pipeline.py)."
            ),
            "sources": [],
        }
    generate = generate_fn or _default_generate_answer
    answer = generate(question, context_chunks)
    return {"answer": answer, "sources": hits}
