import os
import tempfile
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

import config
from graph import NOT_FOUND_MESSAGE, build_graph
from ingestion import (
    build_vectorstore,
    index_exists,
    load_index,
    save_index,
    summarize_chunks,
)

st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="wide")
st.title("🤖 Agentic RAG")

STEP_LABELS = {
    "route": "🧭 Routing question",
    "contextualize": "🧵 Standing question on its own",
    "retrieve": "🔎 Retrieving passages",
    "grade_documents": "⚖️ Grading relevance",
    "rewrite_query": "✏️ Rewriting query",
    "generate": "💬 Generating answer",
    "greeting": "👋 Greeting",
    "off_topic": "🚫 Off-topic",
    "not_found": "🚫 Nothing relevant found",
}


def history_for_graph(messages: list) -> list:
    """Prior turns only (exclude the user message just appended)."""
    prior = messages[:-1] if messages else []
    prior = prior[-config.MAX_HISTORY_MESSAGES :]
    out = []
    for m in prior:
        if m["role"] == "user":
            out.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            out.append(AIMessage(content=m["content"]))
    return out


def activate_index(vs, api_key: str, summary: str, doc_name: str, reset_chat: bool) -> None:
    st.session_state.graph = build_graph(vs, api_key, summary)
    st.session_state.doc_name = doc_name
    st.session_state.doc_summary = summary
    if reset_chat:
        st.session_state.messages = []


# ---------- sidebar ----------
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Google API key", type="password", value=os.environ.get("GOOGLE_API_KEY", "")
    )
    uploaded = st.file_uploader("Upload a document", type=["pdf", "txt", "md"])

    if st.button("Process document", type="primary", disabled=not (uploaded and api_key)):
        with st.spinner("Chunking, embedding, and summarizing..."):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uploaded.name).suffix
                ) as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = tmp.name
                bar = st.progress(0.0, text="Starting...")

                def on_progress(done, total):
                    bar.progress(done / total, text=f"Embedded {done}/{total} chunks")

                vs, chunks = build_vectorstore(tmp_path, api_key, on_progress)
                summary = summarize_chunks(chunks, api_key)
                save_index(
                    vs,
                    {
                        "doc_name": uploaded.name,
                        "n_chunks": len(chunks),
                        "summary": summary,
                    },
                )
                activate_index(vs, api_key, summary, uploaded.name, reset_chat=True)
                st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}")
            except Exception as e:
                st.error(f"Failed to process document: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    if api_key and "graph" not in st.session_state and index_exists():
        try:
            vs, meta = load_index(api_key)
            activate_index(
                vs,
                api_key,
                meta.get("summary", ""),
                meta.get("doc_name", "saved document"),
                reset_chat=False,
            )
            st.caption("Loaded saved index from disk.")
        except Exception as e:
            st.warning(f"Could not load saved index: {e}")

    if "doc_name" in st.session_state:
        st.caption(f"Active document: **{st.session_state.doc_name}**")
        if st.session_state.get("doc_summary"):
            with st.expander("Document summary"):
                st.write(st.session_state.doc_summary)

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
                    {
                        "question": question,
                        "history": history_for_graph(st.session_state.messages),
                    },
                    stream_mode="updates",
                ):
                    for node, out in update.items():
                        label = STEP_LABELS.get(node, node)
                        if node in {"rewrite_query", "contextualize"} and out.get("query"):
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
        if docs and answer and answer != NOT_FOUND_MESSAGE:
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
