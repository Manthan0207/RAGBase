"""Retrieval node for fetching documents from vector store."""
from .graph import Chat
from .utils import retriever_setup , get_kb_retriever


def retrieve_docs(state: Chat) -> dict:
    """Retrieve documents based on the latest user message."""
    retriever = get_kb_retriever()
    query = state["messages"][-1].content
    print(f"Query: {query}")

    return {
        "docs": retriever.invoke(query)
    }