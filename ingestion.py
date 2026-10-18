"""Load -> split -> embed (in rate-limit-safe batches) -> store (FAISS)."""
import json
import time
from pathlib import Path
from typing import Callable, Optional

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import prompts

BATCH_SIZE = 80  # stay under the free-tier limit of 100 requests/minute
WAIT_SECONDS = 62  # pause between batches


def load_document(path: str):
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader  # needs pypdf

        return PyPDFLoader(path).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(path, encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {suffix}. Use PDF, TXT or MD.")


def _with_retry(fn, *args, retries: int = 3):
    """Retry when Google returns a 429 / RESOURCE_EXHAUSTED error."""
    for attempt in range(retries):
        try:
            return fn(*args)
        except Exception as e:
            rate_limited = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
            if rate_limited and attempt < retries - 1:
                time.sleep(WAIT_SECONDS)
                continue
            raise


def build_vectorstore(
    path: str,
    api_key: str,
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[FAISS, list[Document]]:
    """Returns (vectorstore, chunks)."""
    docs = load_document(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No text could be extracted from the document.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL, google_api_key=api_key
    )

    store = None
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        if store is None:
            store = _with_retry(FAISS.from_documents, batch, embeddings)
        else:
            _with_retry(store.add_documents, batch)
        done = min(i + BATCH_SIZE, total)
        if progress:
            progress(done, total)
        if done < total:
            time.sleep(WAIT_SECONDS)
    return store, chunks


def summarize_chunks(chunks: list[Document], api_key: str) -> str:
    """One-pass summary so the router knows the whole document, not just the first pages."""
    parts: list[str] = []
    used = 0
    for chunk in chunks:
        remaining = config.SUMMARY_INPUT_CHARS - used
        if remaining <= 0:
            break
        piece = chunk.page_content[:remaining]
        parts.append(piece)
        used += len(piece)
    text = "\n\n".join(parts)

    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL, google_api_key=api_key, temperature=0
    )
    chain = (
        ChatPromptTemplate.from_messages(
            [
                ("system", prompts.SUMMARIZE_SYSTEM),
                ("human", prompts.SUMMARIZE_HUMAN),
            ]
        )
        | llm
        | StrOutputParser()
    )
    return _with_retry(chain.invoke, {"text": text}).strip()


def index_exists() -> bool:
    return Path(config.INDEX_DIR).exists() and Path(config.METADATA_PATH).exists()


def save_index(store: FAISS, metadata: dict) -> None:
    index_dir = Path(config.INDEX_DIR)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    Path(config.METADATA_PATH).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def load_index(api_key: str) -> tuple[FAISS, dict]:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL, google_api_key=api_key
    )
    store = FAISS.load_local(
        config.INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    metadata = json.loads(Path(config.METADATA_PATH).read_text(encoding="utf-8"))
    return store, metadata
