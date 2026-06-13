import requests
from google import genai
from google.genai import types

from config.settings import settings

def _ollama_embed(texts: list[str]) -> list[list[float]]:
    url = f"{settings.OLLAMA_BASE_URL}/api/embed"
    resp = requests.post(
        url,
        json={"model": settings.OLLAMA_EMBED_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]   # list[list[float]]


def _ollama_embed_one(text: str) -> list[float]:
    return _ollama_embed([text])[0]


_GEMINI_BATCH = 100
_RETRY_DELAY  = 2


client = genai.Client(api_key=settings.GEMINI_API_KEY)


def _gemini_embed(texts: list[str], task_type: str) -> list[list[float]]:
    all_vecs = []

    for i in range(0, len(texts), _GEMINI_BATCH):
        batch = texts[i : i + _GEMINI_BATCH]

        res = client.models.embed_content(
            model=settings.GEMINI_EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )

        if not res.embeddings:
            raise ValueError("Gemini returned empty embeddings")

        all_vecs.extend(
            [e.values or [] for e in res.embeddings]
        )

    return all_vecs


def _gemini_embed_one(text: str, task_type: str) -> list[float]:
    res = client.models.embed_content(
        model=settings.GEMINI_EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )

    if not res.embeddings or not res.embeddings[0].values:
        raise ValueError("Empty embedding result from Gemini")

    return res.embeddings[0].values


def embed_texts(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    show_progress: bool = False,
) -> list[list[float]]:
    if not texts:
        return []

    if settings.is_ollama:
        if show_progress:
            print(f"  [Ollama/{settings.OLLAMA_EMBED_MODEL}] Embedding {len(texts)} texts…")
        BATCH = 50
        result = []
        for i in range(0, len(texts), BATCH):
            if show_progress and len(texts) > BATCH:
                print(f"    batch {i // BATCH + 1}/{-(-len(texts)//BATCH)}")
            result.extend(_ollama_embed(texts[i : i + BATCH]))
        return result
    else:
        if show_progress:
            print(f"  [Gemini/{settings.GEMINI_EMBED_MODEL}] Embedding {len(texts)} texts…")
        return _gemini_embed(texts, task_type)


def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    if settings.is_ollama:
        return _ollama_embed_one(text)
    return _gemini_embed_one(text, task_type)


def embed_query(question: str) -> list[float]:
    if settings.is_ollama:
        return _ollama_embed_one(question)
    return _gemini_embed_one(question, "RETRIEVAL_QUERY")
