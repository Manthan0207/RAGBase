"""Configuration and initialization for the RAG chatbot.

This file exposes `llm`, a persistent `store` backed by SQLite, and
configuration variables for the checkpoint DB. Environment variables
control DB filenames:

- `SQLITE_MEMORY_DB` — path to the long-term memory DB (default: memory.db)
- `SQLITE_CHECKPOINT_DB` — path to the checkpoint DB (default: checkpoints.db)
"""
import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.store.sqlite import SqliteStore

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
load_dotenv(override=True)  # fallback: also load from current working directory

# LLM configuration (allow overriding model via env)
# Initialize the LLM lazily and fail gracefully when API keys are not set,
# so importing this module doesn't crash during local tests.
try:
	llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"))
except Exception as exc:  # pragma: no cover - runtime environment dependent
	llm = None
	print("Warning: ChatOpenAI could not be initialized (missing/invalid credentials). llm set to None.")

# Database filenames (can be full connection strings accepted by the adapters)
MEMORY_DB = os.getenv("SQLITE_MEMORY_DB", "memory.db")
CHECKPOINT_DB = os.getenv("SQLITE_CHECKPOINT_DB", "checkpoints.db")

# Initialize persistent store for long-term memory
_memory_conn = sqlite3.connect(MEMORY_DB, check_same_thread=False, isolation_level=None)
store = SqliteStore(_memory_conn)  # for long term mem
