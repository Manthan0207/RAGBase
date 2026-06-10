"""Decision nodes for routing in the LangGraph workflow."""
import re
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from .graph import Chat
from .models import ShouldRetrieve
from .config import llm


ML_ROUTING_KEYWORDS = (
    "machine learning",
    "deep learning",
    "regularization",
    "lasso",
    "ridge",
    "neural network",
    "neural networks",
    "regression",
    "classification",
    "optimization",
    "gradient",
    "loss",
    "backprop",
    "overfitting",
    "underfitting",
    "feature",
    "model",
)


def _looks_like_ml_query(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ML_ROUTING_KEYWORDS)


def should_retrieve(state: Chat) -> bool:
    """Decide whether to retrieve documents for the user query."""
    if len(state.get("messages", [])) == 0:
        return False
    
    last_message = state["messages"][-1]

    if isinstance(last_message, HumanMessage):
        if _looks_like_ml_query(last_message.content):
            return True

        # If the query is ambiguous, fall back to the LLM router.
        res = llm.with_structured_output(ShouldRetrieve).invoke(
            "You are a helpful assistant which helps user to learn Machine Learning, Deep Learning and Machine learning maths. "
            "If the query is related to these topics then return true else return false. "
            f"Query: {last_message.content}"
        )
        return res.retrieve
    
    return False


def should_summarize(state: Chat) -> bool:
    """Decide whether to summarize the conversation."""
    return len(state["messages"]) >= 10
