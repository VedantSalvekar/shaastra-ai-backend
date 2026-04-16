# app/services/intent_classifier.py
"""
This service classifies what the user is asking about.
It determines whether they want:
- General legal information (LEGAL_ONLY)
- Information from their uploaded documents (USER_ONLY)
- Both (MIXED)

This is the first step in our LangGraph pipeline.
"""

from openai import OpenAI
from app.schemas.langgraph_state import IntentType
from app.core.config import settings


def classify_intent(question: str, user_has_docs: bool, conversation_history: list[dict] = None) -> IntentType:
    """
    Determines what type of question the user is asking.
    
    How it works:
    1. First, we check for obvious keywords like "my document", "my letter", etc.
    2. Then we use an LLM (AI model) to make a smart decision
    3. We also consider whether the user has uploaded any documents
    4. We consider conversation history for context (e.g., "check now" refers to previous context)
    
    Args:
        question: The user's question (e.g., "What does my Revenue letter say?")
        user_has_docs: Does this user have any uploaded documents? (True/False)
        conversation_history: Previous messages in the conversation (for context)
    
    Returns:
        IntentType: One of LEGAL_ONLY, USER_ONLY, MIXED, or UNKNOWN
    
    Examples:
        - "How do I apply for Stamp 4?" → LEGAL_ONLY
        - "What does my Revenue letter say?" → USER_ONLY
        - "Can I work given my current stamp?" → MIXED (needs both docs and legal info)
        - "check now" (after "am I garda vetted?") → USER_ONLY (needs conversation context)
    """
    
    # ========== STEP 1: Quick keyword check ==========
    # These are words that suggest the user is asking about THEIR documents
    user_doc_keywords = [
        "my document", "my letter", "my file", "my upload",
        "my revenue", "my bank statement", "my visa", "my stamp",
        "my application", "my irp", "my permit", "check now",
        "am i", "do i have", "my resume", "my cv"
    ]
    
    question_lower = question.lower()
    mentions_user_docs = any(keyword in question_lower for keyword in user_doc_keywords)
    
    # If user mentions their docs but hasn't uploaded any, they need to upload first
    if mentions_user_docs and not user_has_docs:
        return IntentType.USER_ONLY  # Will trigger a clarification: "Please upload your documents first"
    
    # If they mention their docs AND have uploaded docs, it's at least USER_ONLY or MIXED
    # If they don't mention their docs, it's probably LEGAL_ONLY
    
    # ========== STEP 2: Build context from conversation history ==========
    conversation_context = ""
    if conversation_history and len(conversation_history) > 0:
        conversation_context = "\n\nRecent conversation history:\n"
        for msg in conversation_history[-4:]:  # Last 4 messages (2 exchanges)
            role = msg["role"].upper()
            content = msg["content"][:200]  # Truncate long messages
            conversation_context += f"{role}: {content}\n"
    
    # ========== STEP 3: Use AI to classify more accurately ==========
    # Create prompt for the AI
    classification_prompt = f"""You are an intent classifier for an Irish immigration assistant.

User's current question: "{question}"
User has uploaded documents: {user_has_docs}
{conversation_context}

**IMPORTANT: Consider the conversation history to understand context.**
- If the user says "check now" or "am I" after asking about documents, they want to check THEIR documents
- If they ask follow-up questions like "do I have it?", they're referring to their previous question

Classify the intent as ONE of:
- LEGAL_ONLY: Question about general Irish immigration/legal information
- USER_ONLY: Question specifically about the user's uploaded documents
- MIXED: Question requires both legal knowledge AND the user's documents
- UNKNOWN: Cannot determine the intent

Examples:
- "How do I apply for Stamp 4?" → LEGAL_ONLY
- "What does my Revenue letter say?" → USER_ONLY
- "Given my current stamp, can I work 40 hours?" → MIXED
- "Am I eligible for citizenship based on my IRP?" → MIXED
- "check now" (after asking "am I garda vetted?") → USER_ONLY
- "do I have it?" (after asking about something) → USER_ONLY

Think step by step:
1. Does the question mention the user's specific documents? (keywords: "my", "my document", "my letter", "am I", "do I")
2. Does the question ask about general legal rules or processes?
3. Does it need both?
4. Does conversation history provide context?

Respond with ONLY one word: LEGAL_ONLY, USER_ONLY, MIXED, or UNKNOWN"""

    try:
        # Call OpenAI API
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap model for classification
            messages=[
                {"role": "system", "content": "You are an intent classification system. Respond with only one word."},
                {"role": "user", "content": classification_prompt}
            ],
            temperature=0,  # No randomness, we want consistent classification
            max_tokens=10  # We only need one word back
        )
        
        # Extract the classification from the response
        classification = response.choices[0].message.content.strip().upper()
        
        # Convert to our IntentType enum
        if "LEGAL_ONLY" in classification:
            return IntentType.LEGAL_ONLY
        elif "USER_ONLY" in classification:
            return IntentType.USER_ONLY
        elif "MIXED" in classification:
            return IntentType.MIXED
        else:
            return IntentType.UNKNOWN
            
    except Exception as e:
        # If something goes wrong with the AI, fall back to keyword-based classification
        print(f"[WARN] Intent classification failed: {e}. Using fallback logic.")
        
        if mentions_user_docs:
            # If they mention their docs, assume they want to use them
            return IntentType.USER_ONLY
        else:
            # Otherwise, default to searching legal knowledge
            return IntentType.LEGAL_ONLY


def should_retrieve_legal(intent: IntentType) -> bool:
    """
    Helper function: Should we search the legal knowledge base for this intent?
    
    Returns True if intent is LEGAL_ONLY or MIXED
    """
    return intent in [IntentType.LEGAL_ONLY, IntentType.MIXED]


def should_retrieve_user_docs(intent: IntentType) -> bool:
    """
    Helper function: Should we search the user's documents for this intent?
    
    Returns True if intent is USER_ONLY or MIXED
    """
    return intent in [IntentType.USER_ONLY, IntentType.MIXED]

