"""Central settings. Adjust model names here if Google renames/retires them."""

LLM_MODEL = "gemini-3.5-flash-lite"
# LLM_MODEL = "gemini-3.8-flash"
EMBEDDING_MODEL = "models/gemini-embedding-2"
# EMBEDDING_MODEL = "models/gemini-embedding-001"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
TOP_K = 4
MAX_REWRITES = 1  # how many times the agent may rewrite the query and retry
MAX_HISTORY_MESSAGES = 8  # prior chat turns sent into the graph
SUMMARY_INPUT_CHARS = 24000  # how much document text the summarizer may see

INDEX_DIR = "data/faiss_index"
METADATA_PATH = "data/index_meta.json"
