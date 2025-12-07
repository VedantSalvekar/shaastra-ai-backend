from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import get_settings
from app.schemas.rag import (
    AnswerRequest,
    AnswerResponse,
    AnswerSource,
    SearchRequest,
)
from app.services.vector_store import search as vector_search


def generate_rag_answer(request: AnswerRequest) -> AnswerResponse:
    """
    High-level RAG pipeline:
      1) Retrieve relevant chunks from Qdrant.
      2) Build a safe, Ireland-specific prompt.
      3) Call OpenAI to generate an answer.
      4) Return answer + the sources used.
    """
    settings = get_settings()

    # Retrieve context from Qdrant via our existing search service
    search_req = SearchRequest(
        collection=request.collection,
        query=request.question,
        top_k=request.top_k,
        filter=request.filter,
    )
    search_res = vector_search(search_req)

    if not search_res.results:
       
        empty_answer = (
            "I could not find any relevant information in the current knowledge base "
            "to answer your question confidently. Please check the official Irish "
            "government or Citizens Information websites for the latest details."
        )
        return AnswerResponse(
            answer=empty_answer,
            sources=[],
        )

    # Build context text for the LLM
    # We include short markers so we can later map citations if needed.
    context_blocks: List[str] = []
    for idx, item in enumerate(search_res.results, start=1):
        meta = item.metadata or {}
        label_parts = []
        if "source" in meta:
            label_parts.append(str(meta["source"]))
        if "topic" in meta:
            label_parts.append(str(meta["topic"]))
        if "subtopic" in meta:
            label_parts.append(str(meta["subtopic"]))
        label = " / ".join(label_parts) if label_parts else f"source_{idx}"

        block = (
            f"[Source {idx}: {label}]\n"
            f"{item.text}"
        )
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

   
    llm = ChatOpenAI(
        model=settings.chat_model_name,
        temperature=0.2,
        api_key=settings.openai_api_key,
    )

    # Design a safe, task-specific prompt
    system_prompt = (
        "You are an assistant helping people understand everyday life-admin and "
        "legal procedures in Ireland (immigration stamps, PPS numbers, work rules, tax, etc.). "
        "You must answer ONLY using the context provided. Do not assume or invent rules. "
        "If the context does not fully answer the question or seems outdated, say that clearly.\n\n"
        "Important:\n"
        "- You are NOT a lawyer and this is NOT legal advice.\n"
        "- Always encourage the user to verify critical decisions on official Irish government "
        "websites or with a qualified advisor.\n"
        "- Be concise, practical, and use simple English suitable for international students."
    )

    human_prompt = (
        f"Context from trusted Irish information sources:\n\n"
        f"{context_text}\n\n"
        f"User question: {request.question}\n\n"
        f"Using ONLY the information in the context, answer the question. "
        f"If you are not fully sure, say so and suggest where they can double-check."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    response = llm.invoke(messages)

    # Build AnswerResponse with mapped sources
    sources: List[AnswerSource] = [
        AnswerSource(
            text=item.text,
            metadata=item.metadata or {},
            score=item.score,
        )
        for item in search_res.results
    ]

    return AnswerResponse(
        answer=response.content,
        sources=sources,
    )
