RAG Chatbot — Backend

Quick start (development)

1) Ensure Python 3.10+ is installed.
2) Create and populate `backend/.env` with your OpenAI key:

   OPENAI_API_KEY=sk-...

3) Install Python dependencies (adjust as needed):

   pip install -r requirements.txt

   # If you don't have a requirements file, at minimum install:
   pip install fastapi uvicorn python-dotenv langchain-openai langgraph langchain-core langchain-community faiss-cpu

4) Build the FAISS index (one-time — may take several minutes):

   py -c "from backend.utils import get_kb_retriever; get_kb_retriever(); print('FAISS ready')"

5) Start the API (development):

   py -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

6) Endpoints

- POST /chat — body: { message, thread_id, user_id }
- GET /memory/{user_id} — returns stored facts for the user
- GET /threads/{thread_id} — debug reader for checkpoint DB

Notes

- The backend reads `backend/.env` automatically. If starting uvicorn from another folder, ensure `backend/.env` exists or set `OPENAI_API_KEY` in the environment.
- If you want to run without embeddings, the service falls back to a no-op retriever (will return no docs).

Troubleshooting

- If memory or checkpoint calls report sqlite context manager errors, make sure you have the latest changes where `store` is a persistent `SqliteStore` instance and the checkpointer uses an explicit sqlite3 connection.

Contact

- For next steps I scaffolded a minimal Next.js frontend in `frontend/` to call the `/chat` endpoint.