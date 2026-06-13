from config.settings import settings
from rag.vector_store import similarity_search

def retrieve(
    query: str,
    top_k: int = settings.TOP_K,
    min_score: float = 0.0,
    source_filter: str | None = None,
) -> list[dict]:
    where = {"source": source_filter} if source_filter else None

    docs = similarity_search(query, top_k=top_k, where=where)

    if min_score > 0:
        docs = [d for d in docs if d["score"] >= min_score]

    docs.sort(key=lambda x: x["score"], reverse=True)

    return docs


def retrieve_as_context(
    query: str,
    top_k: int = settings.TOP_K,
    min_score: float = 0.0,
    source_filter: str | None = None,
    separator: str = "\n\n---\n\n",
) -> tuple[str, list[dict]]:

    docs = retrieve(
        query,
        top_k=top_k,
        min_score=min_score,
        source_filter=source_filter,
    )

    if not docs:
        return "", []

    parts = []
    for i, doc in enumerate(docs, 1):
        header = f"[{i}] Source: {doc['source']} (page {doc['page'] + 1}, score: {doc['score']})"
        parts.append(f"{header}\n{doc['text']}")

    context = separator.join(parts)
    return context, docs
