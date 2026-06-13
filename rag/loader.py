import re
from pathlib import Path
from pypdf import PdfReader

def _clean(text: str) -> str:
    text = re.sub(r'\n{2,}', '<<PARA>>', text)
    text = re.sub(r'\n', ' ', text)
    text = text.replace('<<PARA>>', '\n\n')

    lines = [re.sub(r' {2,}', ' ', line).strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def load_pdf(file_path: str | Path, source_name: str | None = None) -> list[dict]:
    path = Path(file_path)
    source = source_name if source_name else path.name
    reader = PdfReader(str(path))

    pages = []
    for i, page in enumerate(reader.pages):
        raw  = page.extract_text() or ""
        text = _clean(raw)
        pages.append({
            "page": i,
            "text": text,
            "source": source,
        })

    return [p for p in pages if p["text"]]


def load_pdf_as_full_text(file_path: str | Path) -> str:
    pages = load_pdf(file_path)
    return "\n\n".join(p["text"] for p in pages)
