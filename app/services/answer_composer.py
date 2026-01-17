# app/services/answer_composer.py
"""
This service composes the final answer using the retrieved context.

It takes chunks from legal knowledge and/or user documents,
and uses an LLM to generate a helpful answer with citations.

IMPORTANT: The LLM is instructed to ONLY use information from the provided context.
This prevents hallucinations (making up facts).
"""

from typing import List
from openai import OpenAI
from app.schemas.langgraph_state import RetrievalChunk, Citation, CitationType
from app.core.config import settings


def compose_answer(
    question: str,
    legal_chunks: List[RetrievalChunk],
    user_chunks: List[RetrievalChunk],
    document_summary: str = ""
) -> tuple[str, List[Citation]]:
    """
    Generate an answer to the user's question using the retrieved context.
    
    How it works:
    1. Combine all the retrieved chunks into a context string
    2. Create a strict prompt telling the LLM to ONLY use the provided context
    3. Ask the LLM to generate an answer with citations
    4. Extract citations from the chunks that were actually used
    
    Args:
        question: The user's original question
        legal_chunks: Retrieved chunks from legal knowledge base
        user_chunks: Retrieved chunks from user's documents
        document_summary: Summary of what documents the user has uploaded
    
    Returns:
        tuple: (answer_text, citations_list)
        - answer_text: The generated answer
        - citations_list: List of Citation objects showing sources
    
    Example:
        answer, citations = compose_answer(
            question="Can I work with Stamp 4?",
            legal_chunks=[...],  # Chunks about Stamp 4 work rules
            user_chunks=[...],   # User's IRP showing their stamp type
            document_summary="User has uploaded: Resume.pdf (contains Stamp 1G)"
        )
    """
    
    # ========== STEP 1: Check if we have any context ==========
    if not legal_chunks and not user_chunks:
        # No context available - we can't answer
        return (
            "I don't have enough information to answer your question. "
            "Please try rephrasing your question or providing more context.",
            []
        )
    
    # ========== STEP 2: Build the context string ==========
    context_parts = []
    
    # Add document summary if available (tells AI what documents exist)
    if document_summary:
        context_parts.append("=== USER'S UPLOADED DOCUMENTS ===")
        context_parts.append(document_summary)
        context_parts.append("")  # Empty line for spacing
    
    # Add legal knowledge context
    if legal_chunks:
        context_parts.append("=== LEGAL KNOWLEDGE BASE ===")
        for i, chunk in enumerate(legal_chunks, 1):
            # Include the source URL if available
            source = chunk.metadata.get("source_url", "Unknown source")
            title = chunk.metadata.get("title", "Legal document")
            context_parts.append(f"\n[Legal Source {i}]")
            context_parts.append(f"Title: {title}")
            context_parts.append(f"URL: {source}")
            context_parts.append(f"Content: {chunk.text}\n")
    
    # Add user document context
    if user_chunks:
        context_parts.append("\n=== CONTENT FROM USER'S DOCUMENTS ===")
        for i, chunk in enumerate(user_chunks, 1):
            # Include document metadata
            doc_id = chunk.metadata.get("doc_id", "Unknown")
            title = chunk.metadata.get("title", "User document")
            filename = chunk.metadata.get("filename", "")
            context_parts.append(f"\n[User Document {i}]")
            context_parts.append(f"Document: {title}")
            if filename:
                context_parts.append(f"Filename: {filename}")
            context_parts.append(f"Content: {chunk.text}\n")
    
    full_context = "\n".join(context_parts)
    
    # ========== STEP 3: Build the prompt for the LLM ==========
    # This is a STRICT prompt that prevents hallucinations
    system_prompt = """You are an Irish immigration assistant. Your job is to help users understand Irish immigration law and their personal documents.

CRITICAL RULES:
1. ONLY use information from the provided context
2. If the context doesn't contain the answer, say "I don't have enough information"
3. NEVER make up facts, dates, or legal requirements
4. Always cite your sources using [Legal Source X] or [User Document X] format
5. Be clear and helpful, but accurate above all
6. If you reference information, cite which source it came from

When answering:
- Start with a direct answer to the question
- Explain relevant details from the context
- Cite sources as you go (e.g., "According to [Legal Source 1], ...")
- If user's documents are relevant, reference them specifically
- End with any important caveats or next steps"""

    user_prompt = f"""Context:
{full_context}

User's Question: {question}

Instructions:
1. Answer the question using ONLY the context provided above
2. Cite your sources using [Legal Source X] or [User Document X] format
3. Be helpful but accurate - if you're not sure, say so
4. If the user's documents are mentioned in the context, reference them specifically

Answer:"""

    # ========== STEP 4: Call the LLM ==========
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4 for better reasoning
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for more factual responses
            max_tokens=800  # Enough for a detailed answer
        )
        
        answer_text = response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"[ERROR] Answer composition failed: {e}")
        return (
            "I encountered an error while generating your answer. Please try again.",
            []
        )
    
    # ========== STEP 5: Build citations list ==========
    citations = []
    
    # Create citations from legal chunks
    for i, chunk in enumerate(legal_chunks, 1):
        # Check if this source was actually mentioned in the answer
        # (We look for [Legal Source X] in the answer text)
        if f"[Legal Source {i}]" in answer_text or i <= 2:  # Always include top 2 sources
            citations.append(
                Citation(
                    type=CitationType.LEGAL,
                    title=chunk.metadata.get("title", "Legal document"),
                    url=chunk.metadata.get("source_url"),
                    snippet=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                )
            )
    
    # Create citations from user document chunks
    for i, chunk in enumerate(user_chunks, 1):
        # Check if this source was actually mentioned in the answer
        if f"[User Document {i}]" in answer_text or i <= 2:  # Always include top 2 sources
            citations.append(
                Citation(
                    type=CitationType.USER_DOC,
                    title=chunk.metadata.get("title", "User document"),
                    doc_id=chunk.metadata.get("doc_id"),
                    snippet=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                )
            )
    
    print(f"[INFO] Generated answer with {len(citations)} citations")
    
    return answer_text, citations

