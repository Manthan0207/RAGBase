# RAGBase

> A RAG chatbot scaffold. Swap in your documents, update the prompt, and you have a real RAG system.

## Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI + LangGraph |
| Retrieval | FAISS (persisted to disk) |
| Memory | SQLite (per-user, cross-session) |
| Frontend | Next.js |
| Streaming | Server-Sent Events (SSE) |


## What is implemented now

- Chat UI with streaming responses
- Per-thread chat history stored in SQLite
- Thread history reload on refresh using `thread_id`
- Long-term memory storage in SQLite
- Knowledge base retrieval from FAISS
- Backend endpoints for chat, streaming chat, memory, and thread history

## What makes it a "dummy" chatbot

Right now the assistant is focused on ML / DL / maths style answers and uses the local knowledge base. It is intentionally kept simple so the app is easy to understand and extend.

That means:
- the data flow is real
- the persistence is real
- the retrieval layer is real
- but the bot logic is still small and easy to modify

## How to turn it into a fuller RAG chatbot

Minimal changes:
1. Add or replace documents in `backend/kb/`.
2. Rebuild the FAISS index.
3. Update the prompt in `backend/prompt_helpers.py`.
4. If needed, replace the routing rules in `backend/decisions.py`.
5. Expand the frontend if you want a richer UX.

## Project structure

- `backend/` — FastAPI app, LangGraph workflow, persistence, retrieval
- `frontend/` — Next.js chatbot UI


## Run locally

### Backend

```powershell
py -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Notes

- The frontend loads previous chats from the backend using `thread_id`.
- If you refresh the page with the same `thread_id`, you will see the previous conversation again.
- The project is intentionally small and readable so you can evolve it into a real RAG chatbot without starting over.
