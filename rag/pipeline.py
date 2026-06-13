from pathlib import Path

from config.settings import settings
from rag.loader import load_pdf
from rag.chunker import chunk_pages
from rag.vector_store import add_chunks, collection_count
from rag.retriever import retrieve_as_context
from rag.generator import generate_answer, generate_standalone_question
from rag.query_transform import (multi_query_retrieve, rag_fusion_retrieve, hyde_retrieve)


RETRIEVAL_MODES = {
    "simple": "Pure cosine similarity search (fastest baseline)",
    "multi_query": "Multi-Query — generate multiple query variations and deduplicate results",
    "rag_fusion": "RAG-Fusion — Multi-Query + Reciprocal Rank Fusion (RRF) re-ranking",
    "hyde": "HyDE — Hypothetical Document Embedding (generate fake answer, then embed it for retrieval)",
}


def index_pdf(
    file_path: str | Path,
    original_filename: str | None = None,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
    show_progress: bool = True,
) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if show_progress:
        print(f"\n Indexing: {path.name}")

    pages = load_pdf(path, source_name=original_filename)
    if show_progress:
        print(f"Read {len(pages)} pages")

    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if show_progress:
        print(f"Split into {len(chunks)} chunks"
              f"(avg {sum(c['token_count'] for c in chunks) // max(len(chunks), 1)} tokens/chunk)")

    added = add_chunks(chunks, show_progress=show_progress)
    total = collection_count()

    if show_progress:
        print(f"Database currently has a total of {total} chunks.\n")

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "added": added,
        "total_in_db": total,
    }


def _build_context(docs: list[dict]) -> str:
    if not docs:
        return ""
    parts = [
        f"[{i}] Source: {doc['source']} (page {doc['page'] + 1})\n{doc['text']}"
        for i, doc in enumerate(docs, 1)
    ]
    return "\n\n---\n\n".join(parts)


def query(
    question: str,
    retrieval_mode: str = "simple",
    top_k: int = settings.TOP_K,
    min_score: float = 0.0,
    chat_history: list[tuple[str, str]] | None = None,
    stream: bool = False,
) -> dict:
    standalone_q = question
    if chat_history:
        standalone_q = generate_standalone_question(chat_history, question)

    if retrieval_mode == "multi_query":
        docs = multi_query_retrieve(standalone_q, top_k=top_k)
        context = _build_context(docs)
    elif retrieval_mode == "rag_fusion":
        docs = rag_fusion_retrieve(standalone_q, top_k=top_k)
        context = _build_context(docs)
    elif retrieval_mode == "hyde":
        docs = hyde_retrieve(standalone_q, top_k=top_k)
        context = _build_context(docs)
    else:
        context, docs = retrieve_as_context(
            standalone_q,
            top_k=top_k,
            min_score=min_score,
        )

    answer = generate_answer(
        question=standalone_q,
        context=context,
        stream=stream,
    )

    return {
        "answer": answer,
        "sources": docs,
        "standalone_question": standalone_q,
        "retrieval_mode": retrieval_mode,
        "context_found": len(docs) > 0,
    }


def query_stream(
    question: str,
    retrieval_mode: str = "simple",
    top_k: int = settings.TOP_K,
    chat_history: list[tuple[str, str]] | None = None,
):
    result = query(
        question=question,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        chat_history=chat_history,
        stream=True,
    )

    return result["answer"]