"""Central settings. Adjust model names here if Google renames/retires them."""

LLM_MODEL = "gemini-3.5-flash"
#LLM_MODEL = "gemini-3.8-flash"
EMBEDDING_MODEL = "models/gemini-embedding-2"
# EMBEDDING_MODEL = "models/gemini-embedding-001"

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
TOP_K = 4
MAX_REWRITES = 2  # how many times the agent may rewrite the query and retry
