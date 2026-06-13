import json
from typing import Generator

import requests
from google import genai
from google.genai import types

from config.settings import settings


_gemini_client = None


def _get_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add it to your .env file or set PROVIDER=ollama."
            )
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


RAG_PROMPT_TEMPLATE = """You are an AI assistant specialized in answering questions based on PDF documents.

Answer the question STRICTLY based on the provided context. If the context does not contain enough information, say "I could not find this information in the document."

DO NOT fabricate information outside the context.

=== CONTEXT ===
{context}

=== QUESTION ===
{question}

=== ANSWER ==="""

STANDALONE_Q_TEMPLATE = """Given the chat history and a new user question, rewrite the new question into a standalone question that can be understood without the history.

Return only the rewritten question, with no explanation.

=== HISTORY ===
{history}

=== NEW QUESTION ===
{question}

=== STANDALONE QUESTION ==="""

MULTI_QUERY_TEMPLATE = """Generate {n} different questions that have the same meaning as the original question below.
Purpose: retrieve documents from multiple perspectives to avoid missing relevant information.

Return a JSON array with exactly {n} questions. Example: ["question 1", "question 2", "question 3"]
Return ONLY JSON, no extra text.

Original question: {question}"""

HYDE_TEMPLATE = """Write a short passage (3–5 sentences) as if it were taken from a professional document that answers the question below. Only write the content, no introduction or conclusion.

Question: {question}

Passage:"""


def _ollama_generate(prompt: str, stream: bool = False):
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": settings.OLLAMA_LLM_MODEL,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": settings.TEMPERATURE,
            "num_predict": settings.MAX_OUTPUT_TOKENS,
        },
    }

    if not stream:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["response"]

    def _stream() -> Generator[str, None, None]:
        with requests.post(url, json=payload, stream=True, timeout=180) as respt:
            respt.raise_for_status()
            for line in respt.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if token := chunk.get("response", ""):
                        yield token
                    if chunk.get("done"):
                        break

    return _stream()


_GEMINI_CONFIG = types.GenerateContentConfig(
    temperature=settings.TEMPERATURE,
    max_output_tokens=settings.MAX_OUTPUT_TOKENS,
)


def _gemini_generate(prompt: str, stream: bool = False):
    client = _get_client()

    if not stream:
        resp = client.models.generate_content(
            model=settings.GEMINI_LLM_MODEL,
            contents=prompt,
            config=_GEMINI_CONFIG,
        )
        return resp.text

    def _stream() -> Generator[str, None, None]:
        for chunk in client.models.generate_content_stream(
            model=settings.GEMINI_LLM_MODEL,
            contents=prompt,
            config=_GEMINI_CONFIG,
        ):
            text = chunk.text
            if text:
                yield text

    return _stream()


def _generate(prompt: str, stream: bool = False):
    if settings.is_ollama:
        return _ollama_generate(prompt, stream=stream)
    return _gemini_generate(prompt, stream=stream)


def generate_answer(question: str, context: str, stream: bool = False):
    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context if context else "(No context available.)",
        question=question,
    )
    return _generate(prompt, stream=stream)


def generate_standalone_question(
    chat_history: list[tuple[str, str]],
    current_question: str,
) -> str:
    if not chat_history:
        return current_question
    history_text = "\n".join(
        f"User: {u}\nAssistant: {a}" for u, a in chat_history[-3:]
    )
    prompt = STANDALONE_Q_TEMPLATE.format(
        history=history_text,
        question=current_question,
    )
    result = _generate(prompt, stream=False)
    return result.strip() if isinstance(result, str) else current_question


def generate_multi_queries(question: str, n: int = 3) -> list[str]:
    prompt = MULTI_QUERY_TEMPLATE.format(n=n, question=question)
    try:
        raw = _generate(prompt, stream=False)

        if not isinstance(raw, str):
            return [question]

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        queries = json.loads(raw.strip())
        return [question] + [q for q in queries if q != question]

    except json.JSONDecodeError:
        return [question]
    except (ValueError, TypeError):
        return [question]


def generate_hypothetical_document(question: str) -> str:
    prompt = HYDE_TEMPLATE.format(question=question)
    result = _generate(prompt, stream=False)
    return result.strip() if isinstance(result, str) else question