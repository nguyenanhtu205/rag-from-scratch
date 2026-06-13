import streamlit as st
import tempfile
import os

from rag.pipeline import index_pdf, query, RETRIEVAL_MODES

st.set_page_config(
    page_title="PDF RAG Assistant",
    layout="wide",
)

st.title("RAG on PDFs with Ollama / Gemini")

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Configuration")
    retrieval_mode = st.selectbox(
        "Retrieval mode",
        options=list(RETRIEVAL_MODES.keys()),
        format_func=lambda x: f"{RETRIEVAL_MODES[x]}",
        index=0,
    )
    top_k = st.slider("Top-K retrieval chunks", min_value=1, max_value=10, value=5)

    st.divider()
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose PDF file", type=["pdf"])

    if uploaded_file is not None and not isinstance(uploaded_file, list):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        file_name = uploaded_file.name
        if st.button("Index this PDF", use_container_width=True):
            with st.spinner(f"Processing {file_name}..."):
                try:
                    result = index_pdf(tmp_path, original_filename=uploaded_file.name, show_progress=False)
                    st.session_state.indexed_files.add(file_name)
                    st.success(f"Index successful! {result['chunks']} chunks have been added.")
                except Exception as e:
                    st.error(f"Error: {e}")
            os.unlink(tmp_path)

    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

if st.session_state.indexed_files:
    st.info(f"Indexed: {', '.join(st.session_state.indexed_files)}")
else:
    st.warning("No files indexed yet. Please upload and index a PDF in the sidebar.")

for user_msg, assistant_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.write(assistant_msg)

if prompt := st.chat_input("Ask a question about the PDF content..."):
    if not st.session_state.indexed_files:
        st.error("Please index at least one PDF before asking questions.")
        st.stop()

    st.session_state.chat_history.append((prompt, ""))
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""
        sources_display = []

        try:
            history = st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 1 else []
            result = query(
                question=str(prompt),
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                chat_history=history,
                stream=True,
            )

            answer_gen = result["answer"]
            for chunk in answer_gen:
                full_answer += chunk
                placeholder.markdown(full_answer + "▌")
            placeholder.markdown(full_answer)

            with st.spinner("Fetching sources..."):
                result_non_stream = query(
                    question=str(prompt),
                    retrieval_mode=retrieval_mode,
                    top_k=top_k,
                    chat_history=history,
                    stream=False,
                )
                sources = result_non_stream.get("sources", [])
                if sources:
                    with st.expander("Sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f"**{i}.** *{src['source']}* – page {src['page']+1} (similarity: {src['score']:.2f})"
                            )
                            st.text(src["text"][:500] + ("..." if len(src["text"]) > 500 else ""))
                            st.divider()

            st.session_state.chat_history[-1] = (prompt, full_answer)

        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.session_state.chat_history.pop()