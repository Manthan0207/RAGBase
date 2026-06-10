from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Any, List, Dict
import asyncio
import sqlite3
import json
from langchain_core.messages import HumanMessage

from backend.main import build_graph
from backend.config import CHECKPOINT_DB, MEMORY_DB, store
import backend.main as backend_main
from backend.config import _memory_conn
from backend.retrieve_node import retrieve_docs as retrieve_docs_node
from backend.chat_nodes import chat_node
from backend.prompt_helpers import format_user_details, format_docs, build_chat_prompt, call_llm
from backend.config import llm

app = FastAPI(title="RAG Chatbot API")

# CORS - allow your frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"
    user_id: Optional[str] = "default"


class MemoryEntry(BaseModel):
    category: str
    key: str
    value: Any


class StreamChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = "default"
    user_id: Optional[str] = "default"


# Build the graph once on startup to reuse expensive resources
_graph = None


def _ensure_thread_history_table() -> None:
    conn = sqlite3.connect(CHECKPOINT_DB)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_thread_messages_thread_id ON thread_messages(thread_id, id)")
        conn.commit()
    finally:
        conn.close()


def _append_thread_messages(thread_id: str, user_id: str, user_message: str, assistant_message: str) -> None:
    conn = sqlite3.connect(CHECKPOINT_DB)
    try:
        conn.executemany(
            "INSERT INTO thread_messages (thread_id, user_id, role, content) VALUES (?, ?, ?, ?)",
            [
                (thread_id, user_id, "user", user_message),
                (thread_id, user_id, "assistant", assistant_message),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def _get_thread_messages(thread_id: str):
    conn = sqlite3.connect(CHECKPOINT_DB)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(
            "SELECT thread_id, user_id, role, content, created_at FROM thread_messages WHERE thread_id = ? ORDER BY id ASC",
            (thread_id,),
        )
        rows = [dict(row) for row in cur.fetchall()]
        return rows
    finally:
        conn.close()


@app.on_event("startup")
async def startup_event():
    global _graph
    await asyncio.to_thread(_ensure_thread_history_table)
    # build_graph is blocking; run in a thread
    _graph = await asyncio.to_thread(build_graph)


@app.on_event("shutdown")
def shutdown_event():
    # Close the persistent memory DB connection
    try:
        if _memory_conn:
            try:
                _memory_conn.close()
                print("Closed memory sqlite connection")
            except Exception:
                pass
    except NameError:
        pass

    # Close the checkpoint connection if available
    try:
        if getattr(backend_main, 'checkpoint_conn', None):
            try:
                backend_main.checkpoint_conn.close()
                print("Closed checkpoint sqlite connection")
            except Exception:
                pass
    except Exception:
        pass


@app.post("/chat")
async def chat(req: ChatRequest):
    if _graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")
    try:
        human = HumanMessage(content=req.message)
        # graph.invoke is typically blocking, so run in a thread
        res = await asyncio.to_thread(
            _graph.invoke,
            {"messages": [human]},
            {"configurable": {"thread_id": req.thread_id, "user_id": req.user_id}},
        )

        # messages -> list of message objects (HumanMessage/AIMessage)
        messages_out = []
        for m in res.get("messages", []):
            # try to extract content and role if present
            content = getattr(m, "content", str(m))
            role = getattr(m, "__class__", None)
            messages_out.append({"content": content})

        # docs (if returned by graph) -> list of Document objects
        docs = []
        for d in res.get("docs", []) or []:
            try:
                docs.append({"page_content": d.page_content, "metadata": getattr(d, "metadata", {})})
            except Exception:
                docs.append({"raw": str(d)})

        summary = res.get("summary")

        assistant_text = ""
        if messages_out:
            assistant_text = messages_out[-1]["content"]
        await asyncio.to_thread(_append_thread_messages, req.thread_id, req.user_id, req.message, assistant_text)

        return {"messages": messages_out, "docs": docs, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: StreamChatRequest):
    if _graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")

    async def event_generator():
        try:
            state = {"messages": [HumanMessage(content=req.message)]}

            # Run retrieval and user details fetch in parallel
            docs_state, user_details_text = await asyncio.gather(
                asyncio.to_thread(retrieve_docs_node, state),
                asyncio.to_thread(format_user_details, store, req.user_id),
            )

            docs = docs_state.get("docs", []) or []
            docs_text = format_docs(docs)          # ← only once, no duplicate

            prompt = build_chat_prompt(
                question=req.message,
                docs_text=docs_text,
                user_details_text=user_details_text,
                summary_text="",
            )

            if llm is None:
                raise RuntimeError("LLM not configured; set OPENAI_API_KEY or init llm.")

            final_text = ""
            async for chunk in llm.astream(prompt):
                token = getattr(chunk, "content", None)
                if token:
                    final_text += token
                    yield f"data: {json.dumps({'token': token})}\n\n"

            await asyncio.to_thread(_append_thread_messages, req.thread_id, req.user_id, req.message, final_text)
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/memory/{user_id}")
def get_memory(user_id: str):
    """Return stored facts for a user (namespace = ('user', user_id, category))."""
    try:
        results = []
        for itm in store.search(("user", user_id)):
            results.append(
                {
                    "category": itm.namespace[2],
                    "key": itm.key,
                    "value": itm.value.get("data") if isinstance(itm.value, dict) else itm.value,
                }
            )
        return {"user_id": user_id, "facts": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/{user_id}")
def post_memory(user_id: str, entry: MemoryEntry):
    """Upsert a memory fact for a user."""
    try:
        store.put(namespace=("user", user_id, entry.category), key=entry.key, value={"data": entry.value})
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}")
def get_thread_rows(thread_id: str):
    """
    Best-effort reader: open the checkpoint SQLite DB and return any rows
    that contain the thread_id in any column. Table/schema may vary by
    langgraph/sqlite implementation, so this is a generic helper for debugging.
    """
    try:
        conn = sqlite3.connect(CHECKPOINT_DB)
        curs = conn.cursor()

        # list tables
        curs.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in curs.fetchall()]

        matches = {}
        for table in tables:
            try:
                curs.execute(f"SELECT * FROM {table} LIMIT 1000")
                cols = [c[0] for c in curs.description] if curs.description else []
                rows = [dict(zip(cols, row)) for row in curs.fetchall()]
                # filter rows that contain thread_id anywhere (string match)
                filtered = []
                for r in rows:
                    if any(thread_id in (str(v) if v is not None else "") for v in r.values()):
                        filtered.append(r)
                if filtered:
                    matches[table] = filtered
            except Exception:
                # ignore tables we can't read
                continue

        conn.close()
        return {"thread_id": thread_id, "tables_with_matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history/{thread_id}")
def get_chat_history(thread_id: str):
    try:
        return {"thread_id": thread_id, "messages": _get_thread_messages(thread_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))