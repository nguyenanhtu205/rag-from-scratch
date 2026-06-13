import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_by_separators(text: str, separators: list[str]) -> list[str]:
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            return [p.strip() for p in parts if p.strip()]

    return list(text)


def chunk_text(
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    paragraphs = _split_by_separators(text, separators)

    chunks: list[str] = []
    current_tokens: list[int] = []
    current_text_parts: list[str] = []

    def flush():
        nonlocal current_tokens, current_text_parts
        if not current_text_parts:
            return

        chunk_str = " ".join(current_text_parts)
        chunks.append(chunk_str.strip())

        all_tokens = _ENCODING.encode(chunk_str)
        overlap_tokens = all_tokens[-chunk_overlap:] if chunk_overlap else []
        overlap_text = _ENCODING.decode(overlap_tokens)

        current_text_parts = [overlap_text] if overlap_text.strip() else []
        current_tokens = overlap_tokens

    for para in paragraphs:
        para_tokens = _ENCODING.encode(para)

        if len(para_tokens) > chunk_size:
            flush()

            sub_chunks = chunk_text(
                para,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators[1:] if len(separators) > 1 else [""],
            )
            chunks.extend(sub_chunks)
            continue

        if len(current_tokens) + len(para_tokens) > chunk_size:
            flush()

        current_text_parts.append(para)
        current_tokens.extend(para_tokens)

    if current_text_parts:
        flush()

    return [c for c in chunks if c.strip()]


def chunk_pages(
        pages: list[dict],
        chunk_size: int = 512,
        chunk_overlap: int = 64,
) -> list[dict]:
    result: list[dict] = []
    for page in pages:
        text_chunks = chunk_text(
            page["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for i, chunk in enumerate(text_chunks):
            result.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"],
                "chunk_id": f"{page['source']}::page{page['page']}::chunk{i}",
                "token_count": count_tokens(chunk),
            })
    return result
