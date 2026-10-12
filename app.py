import os
import tempfile
from pathlib import Path
from turtle import done

import streamlit as st

from graph import build_graph
from ingestion import build_vectorstore

st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")
st.title("🤖 Agentic RAG")

STEP_LABELS = {
    "route": "🧭 Routing question",
    "retrieve": "🔎 Retrieving passages",
    "grade_documents": "⚖️ Grading relevance",
    "rewrite_query": "✏️ Rewriting query",
    "generate": "💬 Generating answer",
    "direct_answer": "💬 Answering directly",
    "not_found": "🚫 Nothing relevant found",
}

# ---------- sidebar ----------
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Google API key", type="password", value=os.environ.get("GOOGLE_API_KEY", "")
    )
    uploaded = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])

    if st.button("Process document", type="primary", disabled=not (uploaded and api_key)):
        with st.spinner("Chunking and embedding..."):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = tmp.name
                bar = st.progress(0.0, text="Starting...")
                def on_progress(done, total):
                    bar.progress(done / total, text=f"Embedded {done}/{total} chunks")
                vs, n_chunks = build_vectorstore(tmp_path, api_key, on_progress)
                st.session_state.graph = build_graph(vs, api_key)
                st.session_state.doc_name = uploaded.name
                st.session_state.messages = []
                st.success(f"Indexed {n_chunks} chunks from {uploaded.name}")
            except Exception as e:
                st.error(f"Failed to process document: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    if "doc_name" in st.session_state:
        st.caption(f"Active document: **{st.session_state.doc_name}**")

# ---------- chat ----------
st.session_state.setdefault("messages", [])

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.caption(s["label"])
                    st.write(s["text"])

if "graph" not in st.session_state:
    st.info("Add your API key, upload a document, and click **Process document** to begin.")
    st.stop()

if question := st.chat_input("Ask something about your document"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer, docs = "", []
        try:
            with st.status("Thinking...", expanded=True) as status:
                for update in st.session_state.graph.stream(
                    {"question": question}, stream_mode="updates"
                ):
                    for node, out in update.items():
                        label = STEP_LABELS.get(node, node)
                        if node == "rewrite_query":
                            label += f": `{out['query']}`"
                        elif node == "grade_documents":
                            label += f": {len(out['documents'])} relevant"
                        status.write(label)
                        if "generation" in out:
                            answer = out["generation"]
                        if node == "grade_documents":
                            docs = out["documents"]
                status.update(label="Done", state="complete", expanded=False)
        except Exception as e:
            answer = f"Something went wrong: {e}"

        st.markdown(answer)
        sources = []
        if docs and answer and not answer.startswith("I couldn't find"):
            for d in docs:
                page = d.metadata.get("page")
                label = f"{Path(d.metadata.get('source', '')).name or 'document'}" + (
                    f" — page {page + 1}" if isinstance(page, int) else ""
                )
                sources.append({"label": label, "text": d.page_content})
            with st.expander("Sources"):
                for s in sources:
                    st.caption(s["label"])
                    st.write(s["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
