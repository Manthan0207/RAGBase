"""Graph state and configuration."""
from typing import Annotated, List, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph import add_messages
"""
# `store` should be provided by `backend.config` so that persistence
# (SQLite) vs in-memory behavior is controlled centrally. This file only
# defines the `Chat` state TypedDict used by the graph.
"""
class Chat(TypedDict):
    """State for the chat graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    docs: list[Document]
    summary: Optional[str]


