"""Chat and summarization nodes for the LangGraph workflow."""
from typing import Optional
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from .graph import Chat
from .models import SummarizeAndSaveToLTM
from .config import llm, store
from .retrieve_node import retrieve_docs as retrieve_docs_node
from .prompt_helpers import format_user_details, format_docs, build_chat_prompt, call_llm
from .prompt_helpers import build_summary_prompt, call_llm




# def chat_node(state : Chat , config : RunnableConfig ,store : BaseStore) -> dict :
#     user_id = config.get("configurable", {}).get("user_id", "default")
    
#     # Load stored user details
#     d = {}
#     for itm in store.search(("user", user_id)):
#         category = itm.namespace[2]
#         if category not in d:
#             d[category] = {}
#         d[category][itm.key] = itm.value['data']

#     docs = state.get("docs") or []

#     prompt = """You are a specialized Machine Learning assistant.

# You have access to a knowledge base containing ONLY:
# - Machine Learning concepts and algorithms
# - Deep Learning architectures and techniques
# - ML Mathematics (linear algebra, calculus, probability, statistics)
# - Model training, evaluation, and optimization
# - Neural networks, backpropagation, activation functions

# BEHAVIOR:
# - If the user asks about ML/DL/math → answer strictly using the retrieved documents only
# - If the answer is not found in the documents → say "I don't have enough information in my knowledge base to answer this."
# - If the user asks something outside ML/DL scope (weather, sports, cooking etc) → politely say "I can only answer questions related to Machine Learning, Deep Learning, and ML Mathematics."
# - If the user is greeting or sharing personal info → respond naturally and friendly
# - Never make up ML facts or concepts not present in the documents
# """

#     if docs:
#         prompt += "\n\nRetrieved Documents:\n\n" + "\n\n".join(
#             [f"Document {i}: {doc.page_content}" for i, doc in enumerate(docs, 1)]
#         )
#     else:
#         prompt += "\n\nRetrieved Documents:\n\nNone"

#     prompt += f"\n\nUser Details: {d if d else 'None'}"
#     prompt += f"\n\nQuestion: {state['messages'][-1].content}"

#     summary = state.get('summary', '')
#     messages_text = "\n".join(msg.content for msg in state["messages"])
#     chat_context = f"{summary}\n{messages_text}".strip()
#     prompt += f"\n\nChat Context:\n\n{chat_context}" if chat_context else ""
    
#     print(prompt)
#     res = llm.invoke(prompt)
#     return {"messages": [AIMessage(content=res.content)]}


def chat_node(state: Chat, config: RunnableConfig, store: BaseStore) -> dict:
    user_id = config.get("configurable", {}).get("user_id", "default")
    user_details_text = format_user_details(store, user_id)
    docs = state.get("docs") or []
    docs_text = format_docs(docs)
    summary = state.get("summary", "")
    question = state["messages"][-1].content

    prompt = build_chat_prompt(
        question=question,
        docs_text=docs_text,
        user_details_text=user_details_text,
        summary_text=summary
    )

    res = call_llm(llm, prompt)  # returns a normal LLM response object
    return {"messages": [AIMessage(content=res.content)]}


# def summarize_node(state: Chat, config: RunnableConfig, store: BaseStore) -> dict:
#     """Summarize conversation and extract long-term memory."""
#     existing_summary = state.get("summary", "")
#     user_id = config.get("configurable", {}).get("user_id", "default")

#     prompt = """You are maintaining a concise, evolving summary of a conversation AND extracting important information for long-term memory.

# You must do TWO things:

# ---
# TASK 1 — UPDATE SUMMARY:
# - Combine the existing summary with the new conversation.
# - Preserve important context from the existing summary.
# - If new information updates or contradicts old information, MODIFY accordingly.
# - Remove redundant, outdated, or less relevant details.
# - Keep only the most important and up-to-date information.
# - Ensure the final summary is clear, coherent, and under 100 words.
# - Always capture the USER'S INTENT — what the user is trying to ask, solve, or achieve.
# - Track key decisions, problems, and outcomes.

# Focus on:
# - User intent (main problem/question)
# - Key facts and context
# - Decisions or solutions discussed

# ---
# TASK 2 — EXTRACT LONG-TERM MEMORY:
# Check if the conversation contains any personal information worth remembering across sessions.

# Categories:
# - details: personal facts (name, age, location, occupation)
# - preferences: likes, dislikes, preferred tools, style choices
# - goals: what the user is building, learning, or trying to achieve

# Rules:
# - Only store clear, explicit facts — not assumptions
# - If nothing worth storing, set should_store to false and ltm_data to null
# - Extract ALL clear facts found in the conversation, not just one
# - Each fact gets its own entry in ltm_data
# - If a fact was already mentioned in the existing summary, do NOT store it again
# - For details/preferences: use consistent keys like 'name', 'location' so updates overwrite cleanly
# - For goals: use descriptive keys like 'goal_learn_ml', 'goal_build_chatbot' since users can have multiple
# """

#     if existing_summary:
#         prompt += "\n\nExisting Summary:\n" + existing_summary

#     prompt += "\n\nNew Conversation:\n" + "\n".join(
#         "User: " + msg.content if isinstance(msg, HumanMessage) else "AI: " + msg.content
#         for msg in state["messages"]
#     )

#     llm_with_structured_output = llm.with_structured_output(SummarizeAndSaveToLTM)
#     res: SummarizeAndSaveToLTM = llm_with_structured_output.invoke(prompt)
#     messages_to_delete = state["messages"][:-2]  # Keep the last 2 messages for context

#     updatesSummary = res.summary
#     ltm_output = res.ltm_output
    
#     if ltm_output and ltm_output.should_store:
#         for item in ltm_output.ltm_data or []:
#             store.put(
#                 namespace=("user", user_id, item.category),
#                 key=item.key,
#                 value={"data": item.text}
#             )

#     return {
#         "summary": updatesSummary,
#         "messages": [RemoveMessage(id=msg.id) for msg in messages_to_delete]
#     }


def summarize_node(state: Chat, config: RunnableConfig, store: BaseStore) -> dict:
    existing_summary = state.get("summary", "")
    user_id = config.get("configurable", {}).get("user_id", "default")

    prompt = build_summary_prompt(existing_summary, state["messages"])
    messages_to_delete = state["messages"][:-2]

    res = call_llm(llm, prompt, structured_model=SummarizeAndSaveToLTM)
    updatesSummary = res.summary
    ltm_output = res.ltm_output
    
    if ltm_output and ltm_output.should_store:
        for item in ltm_output.ltm_data or []:
            store.put(
                namespace=("user", user_id, item.category),
                key=item.key,
                value={"data": item.text}
            )

    return {
        "summary": updatesSummary,
        "messages": [RemoveMessage(id=msg.id) for msg in messages_to_delete]
    }