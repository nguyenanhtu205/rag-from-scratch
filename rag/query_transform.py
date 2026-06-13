from config.settings import settings
from rag.retriever import retrieve
from rag.embedder import embed_query
from rag.vector_store import get_collection
from rag.generator import (generate_multi_queries,generate_hypothetical_document)

def multi_query_retrieve(
    question: str,
    n_queries: int = 3,
    top_k: int = settings.TOP_K,
) -> list[dict]:
    queries = generate_multi_queries(question, n=n_queries)

    seen: set[str] = set()
    all_docs: list[dict] = []
    for q in queries:
        for doc in retrieve(q, top_k=top_k):
            if doc["text"] not in seen:
                seen.add(doc["text"])
                all_docs.append(doc)

    all_docs.sort(key=lambda x: x["score"], reverse=True)
    return all_docs[:top_k]


def reciprocal_rank_fusion(
    results_list: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    store: dict[str, dict] = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            key = doc["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k)
            store.setdefault(key, doc)
    return [
        {**store[t], "rrf_score": round(s, 6)}
        for t, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


def rag_fusion_retrieve(
    question: str,
    n_queries: int = 3,
    top_k: int = settings.TOP_K,
) -> list[dict]:
    queries = generate_multi_queries(question, n=n_queries)
    all_results = [retrieve(q, top_k=top_k) for q in queries]
    return reciprocal_rank_fusion(all_results)[:top_k]


def hyde_retrieve(
    question: str,
    top_k: int = settings.TOP_K,
) -> list[dict]:
    hypo_doc = generate_hypothetical_document(question)
    hypo_vec = embed_query(hypo_doc)

    collection = get_collection()
    results = collection.query(
        query_embeddings=[hypo_vec],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs  = results["documents"]
    metas = results["metadatas"]
    dists = results["distances"]

    if docs is None or metas is None or dists is None:
        return []

    return [
        {
            "text":   doc,
            "source": meta.get("source", ""),
            "page":   meta.get("page", -1),
            "score":  round(1 - dist, 4),
        }
        for doc, meta, dist in zip(docs[0], metas[0], dists[0])
    ]
