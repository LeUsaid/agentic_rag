
Stack: Gemini (LLM) · Google embeddings · FAISS · LangGraph · Streamlit

## Run
```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key"   # or paste it in the sidebar
streamlit run app.py
```
Upload a PDF/TXT/MD, click **Process document**, then chat.

## Agent flow
route -> retrieve -> grade_documents -> generate
                          └─ no relevant docs -> rewrite_query -> retrieve (max 2 retries) -> not_found

**Tune models, chunking, top-k and retry count in `config.py`.
