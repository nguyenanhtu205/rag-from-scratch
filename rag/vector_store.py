import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from rag.embedder import embed_texts, embed_query


_client = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DB_PATH),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection(name: str = settings.CHROMA_COLLECTION) -> chromadb.Collection:
    client = _get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    chunks: list[dict],
    collection_name: str = settings.CHROMA_COLLECTION,
    show_progress: bool = True,
) -> int:
    if not chunks:
        return 0

    collection = get_collection(collection_name)

    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("All chunks already exist, skipping.")
        return 0

    texts      = [c["text"]     for c in new_chunks]
    ids        = [c["chunk_id"] for c in new_chunks]
    metadatas  = [
        {"source": c["source"], "page": c["page"], "token_count": c["token_count"]}
        for c in new_chunks
    ]

    if show_progress:
        print(f"Embedding {len(texts)} chunks…")

    embeddings = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT", show_progress=show_progress)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    if show_progress:
        print(f"Saved {len(new_chunks)} chunks to ChromaDB.")

    return len(new_chunks)


def similarity_search(
    query: str,
    top_k: int = settings.TOP_K,
    collection_name: str = settings.CHROMA_COLLECTION,
    where: dict | None = None,
) -> list[dict]:
    collection = get_collection(collection_name)
    query_embedding = embed_query(query)

    kwargs: dict = dict(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []
    distances = results.get("distances") or []

    docs = []
    for doc, meta, dist in zip(documents[0], metadatas[0], distances[0]):
        docs.append({
            "text": doc,
            "source": meta.get("source", "") if meta else "",
            "page": meta.get("page", -1) if meta else -1,
            "score": round(1 - dist, 4),
        })

    return docs


def collection_count(collection_name: str = settings.CHROMA_COLLECTION) -> int:
    return get_collection(collection_name).count()


def delete_collection(collection_name: str = settings.CHROMA_COLLECTION) -> None:
    client = _get_client()
    client.delete_collection(collection_name)
    print(f"Deleted collection '{collection_name}'.")
