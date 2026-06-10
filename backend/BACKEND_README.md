# Backend Module Structure

## Files Overview

- **config.py** — Global configuration (LLM, store initialization)
- **graph.py** — Chat state definition
- **models.py** — Pydantic models (ShouldRetrieve, LTMData, etc.)
- **utils.py** — Utility functions (document loading, vector store setup)
- **retrieve_node.py** — Document retrieval node
- **chat_nodes.py** — Chat, summarization, and retrieval nodes
- **decisions.py** — Decision nodes (should_retrieve, should_summarize)
- **main.py** — Graph construction and execution

## How to Run

```bash
# From the backend directory
python -m main
```

Or in a notebook:

```python
from backend.main import build_graph
from langchain_core.messages import HumanMessage

graph = build_graph()

res = graph.invoke(
    {"messages": [HumanMessage(content="what is L1 regularization?")]},
    config={"configurable": {"thread_id": "test_1", "user_id": "user_123"}}
)

print(res["messages"][-1].content)
```

## Module Dependencies

- `config` → used by `chat_nodes.py`, `decisions.py`
- `graph` → imported by all nodes
- `models` → used by `decisions.py`, `chat_nodes.py`
- `utils` → used by `retrieve_node.py`
- `decisions` → used by `main.py`
- `chat_nodes` → used by `main.py`
- `retrieve_node` → used by `main.py`
