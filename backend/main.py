"""Main graph construction and execution."""
import sys
from pathlib import Path

# Allow both module and direct script execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backend.graph import Chat
    from backend.chat_nodes import retrieve_docs_node, chat_node, summarize_node
    from backend.decisions import should_retrieve, should_summarize
    from backend.config import store, CHECKPOINT_DB
else:
    from .graph import Chat
    from .chat_nodes import retrieve_docs_node, chat_node, summarize_node
    from .decisions import should_retrieve, should_summarize
    from .config import store, CHECKPOINT_DB

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Expose the checkpoint connection so callers can close it on shutdown
checkpoint_conn = None
checkpoint_saver = None


def build_graph():
    """Build and compile the LangGraph workflow."""
    builder = StateGraph(Chat)

    # Add nodes
    builder.add_node("retrieve_docs", retrieve_docs_node)
    builder.add_node("chat_node", chat_node)
    builder.add_node("summarize_node", summarize_node)

    # Add edges
    builder.add_conditional_edges(
        START,
        should_retrieve,
        {
            True: "retrieve_docs",
            False: "chat_node"
        }
    )
    builder.add_edge("retrieve_docs", "chat_node")
    builder.add_conditional_edges(
        "chat_node",
        should_summarize,
        {
            True: "summarize_node",
            False: END
        }
    )
    builder.add_edge("summarize_node", END)

    # Compile with checkpointing. Create a sqlite3 connection and pass a
    # persistent SqliteSaver instance so the checkpointer remains valid
    # for the lifetime of the process.
    global checkpoint_conn, checkpoint_saver
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    checkpoint_conn = conn
    checkpoint_saver = SqliteSaver(conn)
    graph = builder.compile(checkpointer=checkpoint_saver, store=store)
    
    return graph


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    
    graph = build_graph()
    
    # Example usage
    res = graph.invoke(
        {"messages": [HumanMessage(content="hi?")]},
        config={
            "configurable": {
                "thread_id": "test_thread_4",
                "user_id": "user_67"
            }
        }
    )
    
    print("Messages:")
    print(res["messages"])
