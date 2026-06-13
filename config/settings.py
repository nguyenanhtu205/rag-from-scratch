from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROVIDER: Literal["ollama", "gemini"] = "ollama"

    GEMINI_API_KEY: str
    GEMINI_EMBED_MODEL: str = "models/gemini-embedding-001"
    GEMINI_LLM_MODEL: str = "models/gemini-2.5-flash"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBED_MODEL: str = "bge-m3"
    OLLAMA_LLM_MODEL: str = "qwen3:4b"

    TEMPERATURE: float = 0.0
    MAX_OUTPUT_TOKENS: int = 2048

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    TOP_K: int = 5

    CHROMA_DB_PATH: Path = Path("./chroma_db")
    CHROMA_COLLECTION: str = "pdf_rag"

    @property
    def embed_model(self) -> str:
        return self.OLLAMA_EMBED_MODEL if self.PROVIDER == "ollama" else self.GEMINI_EMBED_MODEL

    @property
    def llm_model(self) -> str:
        return self.OLLAMA_LLM_MODEL if self.PROVIDER == "ollama" else self.GEMINI_LLM_MODEL

    @property
    def is_ollama(self) -> bool:
        return self.PROVIDER == "ollama"

    class Config:
        env_file = str(Path(__file__).resolve().parent.parent / ".env")
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
