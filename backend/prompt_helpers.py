from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig


MAX_DOCS_IN_PROMPT = 3
MAX_DOC_CHARS = 1200

def format_user_details(store, user_id: str) -> str:
    d = {}
    for itm in store.search(("user", user_id)):
        category = itm.namespace[2]
        d.setdefault(category, {})[itm.key] = itm.value["data"]
    return str(d) if d else "None"

def format_docs(docs: Optional[List[Document]]) -> str:
    if not docs:
        return "None"
    trimmed = []
    for i, doc in enumerate(docs[:MAX_DOCS_IN_PROMPT], 1):
        content = doc.page_content[:MAX_DOC_CHARS]
        if len(doc.page_content) > MAX_DOC_CHARS:
            content += "\n...[truncated]"
        trimmed.append(f"Document {i}: {content}")
    return "\n\n".join(trimmed)

def build_chat_prompt(question: str,
                      docs_text: str,
                      user_details_text: str,
                      summary_text: str,
                      behavior_config: Optional[str] = None) -> str:
    base = behavior_config or (
        "You are a specialized Machine Learning assistant.\n\n"
        "You have access to a knowledge base containing ONLY:\n"
        "- Machine Learning concepts and algorithms\n"
        "- Deep Learning architectures and techniques\n"
        "- ML Mathematics (linear algebra, calculus, probability, statistics)\n"
        "- Model training, evaluation, and optimization\n"
        "- Neural networks, backpropagation, activation functions\n\n"
        "BEHAVIOR:\n"
        "- If the user asks about ML/DL/math → answer strictly using the retrieved documents only\n"
        "- If the answer is not found in the documents → say \"I don't have enough information in my knowledge base to answer this.\"\n"
        "- If the user asks something outside ML/DL scope → politely say you can only answer ML/DL/math questions\n"
        "- If the user is greeting or sharing personal info → respond naturally and friendly\n"
        "- Never invent facts not present in the documents\n"
    )
    prompt = base
    prompt += "\n\nRetrieved Documents:\n\n" + docs_text
    prompt += f"\n\nUser Details: {user_details_text}"
    prompt += f"\n\nQuestion: {question}"
    if summary_text:
        prompt += f"\n\nChat Context:\n\n{summary_text}"
    return prompt

def build_summary_prompt(existing_summary: str, messages: List[BaseMessage]) -> str:
    prompt = (
        "You are maintaining a concise, evolving summary of a conversation AND extracting important information for long-term memory.\n\n"
        "You must do TWO things:\n\n"
        "TASK 1 — UPDATE SUMMARY:\n- Combine the existing summary with the new conversation.\n- Preserve important context from the existing summary.\n- If new information updates or contradicts old information, MODIFY accordingly.\n- Remove redundant, outdated, or less relevant details.\n- Keep only the most important and up-to-date information.\n- Ensure the final summary is clear, coherent, and under 500 words.\n- Always capture the USER'S INTENT.\n\n"
        "TASK 2 — EXTRACT LONG-TERM MEMORY:\n- Check conversation for personal info worth remembering.\n- Categories: details, preferences, goals.\n- Only store clear explicit facts; do not invent.\n"
    )
    if existing_summary:
        prompt += "\n\nExisting Summary:\n" + existing_summary
    prompt += "\n\nNew Conversation:\n" + "\n".join(
        "User: " + m.content if m.__class__.__name__ == "HumanMessage" else "AI: " + m.content
        for m in messages
    )
    return prompt

def call_llm(llm, prompt: str, structured_model=None):
    if llm is None:
        raise RuntimeError("LLM not configured; set OPENAI_API_KEY or init llm.")
    if structured_model is None:
        return llm.invoke(prompt)
    llm_struct = llm.with_structured_output(structured_model)
    return llm_struct.invoke(prompt)