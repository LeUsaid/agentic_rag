"""Load -> split -> embed (in rate-limit-safe batches) -> store (FAISS)."""
import time
from pathlib import Path
from typing import Callable, Optional

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

BATCH_SIZE = 80      # stay under the free-tier limit of 100 requests/minute
WAIT_SECONDS = 62    # pause between batches


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
) -> tuple[FAISS, int]:
    """Returns (vectorstore, number_of_chunks)."""
    docs = load_document(path)
#2
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No text could be extracted from the document.")
#3
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
    return store, total