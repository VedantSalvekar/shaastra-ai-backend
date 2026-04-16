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
    document_summary: str = "",
    conversation_history: List[dict] = None
) -> tuple[str, List[Citation], List[str]]:
    """
    Generate an answer to the user's question using the retrieved context.
    
    How it works:
    1. Combine all the retrieved chunks into a context string
    2. Add conversation history for context
    3. Create a strict prompt telling the LLM to ONLY use the provided context
    4. Ask the LLM to think step-by-step (chain of thought)
    5. Generate answer with citations
    6. Extract reasoning steps and citations
    
    Args:
        question: The user's original question
        legal_chunks: Retrieved chunks from legal knowledge base
        user_chunks: Retrieved chunks from user's documents
        document_summary: Summary of what documents the user has uploaded
        conversation_history: Previous messages for context
    
    Returns:
        tuple: (answer_text, citations_list, reasoning_steps)
        - answer_text: The generated answer
        - citations_list: List of Citation objects showing sources
        - reasoning_steps: List of thinking steps the AI took
    
    Example:
        answer, citations, steps = compose_answer(
            question="Am I garda vetted?",
            legal_chunks=[...],
            user_chunks=[...],
            document_summary="User has uploaded: Resume.pdf",
            conversation_history=[{"role": "user", "content": "check my documents"}]
        )
    """
    
    # ========== STEP 1: Check if we have any context ==========
    if not legal_chunks and not user_chunks:
        # No context available - we can't answer
        return (
            "I don't have enough information to answer your question. "
            "Please try rephrasing your question or providing more context.",
            [],
            ["No relevant context found"]
        )
    
    # ========== STEP 2: Build conversation history context ==========
    history_context = ""
    if conversation_history and len(conversation_history) > 0:
        history_context = "\n=== CONVERSATION HISTORY ===\n"
        for msg in conversation_history[-6:]:  # Last 6 messages (3 exchanges)
            role = msg["role"].upper()
            content = msg["content"]
            history_context += f"{role}: {content}\n"
        history_context += "\n"
    
    # ========== STEP 3: Build the context string ==========
    context_parts = []
    
    # Add conversation history first
    if history_context:
        context_parts.append(history_context)
    
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
    
    # ========== STEP 4: Build the prompt for the LLM ==========
    # This is a STRICT prompt that prevents hallucinations AND includes reasoning
    system_prompt = """You are an Irish immigration assistant. Your job is to help users understand Irish immigration law and their personal documents.

CRITICAL RULES:
1. ONLY use information from the provided context
2. If the context truly has no relevant text for the question, say clearly that the excerpts do not mention it—do not claim you "cannot find" something that literally appears in a snippet (job titles, names, dates, etc.)
3. NEVER invent facts, dates, or legal requirements not present in the context
4. Always cite your sources using [Legal Source X] or [User Document X] in the ANSWER section
5. Be direct: for questions about the user's own uploads (cover letter, CV, IRP, etc.), quote or paraphrase the exact detail from the excerpt first, then cite [User Document X]
6. PDF text may contain OCR/font artifacts (e.g. "/" instead of "t" in words like "position" → "posi/on"). Infer the intended wording when it is obvious from context; still cite the document
7. For legal questions with several cases (e.g. student vs worker), summarize what the legal sources say for each case instead of saying only "I'm not sure." You may ask one short clarifying question at the end if status is unknown
8. **THINK STEP BY STEP before answering** - show your reasoning in THINKING only; keep ANSWER decisive and readable
9. Consider the conversation history to understand what the user is referring to

When answering:
1. **First, think through the question step by step** (use "THINKING:" section)
2. Understand what the user is asking (consider conversation history if present)
3. Identify which documents/sources contain the answer
4. Extract the relevant information (including job titles, employer names, hours, stamp types when present)
5. **Then provide your answer** (use "ANSWER:" section)—state the fact or rule plainly, then cite
6. Cite sources in the ANSWER (e.g., "According to [Legal Source 1], ..." or "Your cover letter states ... [User Document 1]")
7. If user's documents are relevant, reference them specifically
8. End with brief caveats only when the sources say the rule depends on status you do not have

Format your response EXACTLY like this:
```
THINKING:
- Step 1: [Your first reasoning step]
- Step 2: [Your second reasoning step]
- Step 3: [Your third reasoning step]

ANSWER:
[Your actual answer here with citations]
```"""

    user_prompt = f"""Context:
{full_context}

User's Current Question: {question}

Instructions:
1. **THINK STEP BY STEP** first - show your reasoning process
2. Answer the question using ONLY the context provided above
3. Consider the conversation history (if present) to understand what the user is asking
4. In ANSWER: always include at least one [Legal Source N] or [User Document N] citation when you use that text
5. If the answer is in the user's document excerpts, state it directly (e.g. role, company) rather than describing what you would "look for"
6. For general law questions, give the rule from legal sources; note different categories (e.g. Stamp 2 vs work permit) when sources distinguish them

Remember: Format your response with "THINKING:" section first, then "ANSWER:" section.

Response:"""

    # ========== STEP 5: Call the LLM ==========
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4 for better reasoning
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for more factual responses
            max_tokens=1000  # Enough for reasoning + answer
        )
        
        full_response = response.choices[0].message.content.strip()
        
        # ========== STEP 6: Extract thinking steps and answer ==========
        reasoning_steps = []
        answer_text = full_response
        
        # Try to separate THINKING and ANSWER sections
        if "THINKING:" in full_response and "ANSWER:" in full_response:
            parts = full_response.split("ANSWER:")
            thinking_part = parts[0].replace("THINKING:", "").strip()
            answer_text = parts[1].strip()
            
            # Extract individual thinking steps
            thinking_lines = thinking_part.split("\n")
            for line in thinking_lines:
                line = line.strip()
                if line and (line.startswith("-") or line.startswith("•") or line.startswith("Step")):
                    # Remove bullet points and clean up
                    clean_step = line.lstrip("-•").strip()
                    if clean_step:
                        reasoning_steps.append(clean_step)
                        print(f"[THINKING] {clean_step}")  # Log thinking steps
        else:
            # If AI didn't follow format, treat entire response as answer
            answer_text = full_response
            reasoning_steps = ["AI generated direct answer without explicit reasoning steps"]
        
    except Exception as e:
        print(f"[ERROR] Answer composition failed: {e}")
        return (
            "I encountered an error while generating your answer. Please try again.",
            [],
            [f"Error during answer generation: {str(e)}"]
        )
    
    # ========== STEP 7: Build citations list ==========
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
    
    print(f"[INFO] Generated answer with {len(citations)} citations and {len(reasoning_steps)} reasoning steps")
    
    return answer_text, citations, reasoning_steps

