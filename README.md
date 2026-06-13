# RAG PDF from Scratch

## Stack

- **Embeddings:** bge-m3 or gemini-embedding-001
- **LLM:** qwen3:4b or gemini-2.5-flash
- **Vector Database:** ChromaDB
- **UI:** Streamlit

---

## Installation

```bash
git clone https://github.com/nguyenanhtu205/rag-from-scratch.git
cd rag-from-scratch
python -m venv .venv
.venv\Scripts\activate     
pip install -r requirements.txt
```

Create a `.env` file from the template:

```bash
cp .env.example .env
```

---

## Configure API Key

Add your Gemini API key to `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

You can get a free API key from Google AI Studio. The application always loads the Gemini API key from `.env`, regardless of which provider you use.

---

## Running with Ollama

**Step 1** — Install Ollama, then pull the required models:

```bash
ollama pull bge-m3
ollama pull qwen3:4b
```

**Step 2** — Open `config/settings.py` and set:

```python
PROVIDER: Literal["ollama", "gemini"] = "ollama"
```

**Step 3** — Start the application:

```bash
streamlit run app.py
```

---

## Running with Gemini

**Step 1** — Open `config/settings.py` and set:

```python
PROVIDER: Literal["ollama", "gemini"] = "gemini"
```

**Step 2** — Start the application:

```bash
streamlit run app.py
```

---

## Notes When Switching Providers

ChromaDB stores vectors based on the dimensionality of the embedding model. Since different embedding models may produce vectors with different dimensions, existing indexes are not compatible when switching providers.

When changing the provider, delete the `chroma_db/` directory and re-index your PDF files.

```bash
rm -r -fo chroma_db
```