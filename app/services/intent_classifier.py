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


# Phrases that clearly reference the CONTENT of the user's own uploaded file.
# Ambiguous phrases like "my visa", "am I", "do I have" are deliberately excluded:
# they appear constantly in general legal questions ("how many hours can I work",
# "do I need insurance") and must not force a user-document intent.
EXPLICIT_USER_DOC_PHRASES = [
    "my document", "my documents", "my file", "my files",
    "my upload", "my uploaded", "uploaded document",
    "in my document", "on my document", "from my document",
    "my resume", "my cv", "my cover letter", "my pdf",
    "what does my", "check my document", "read my",
]


def references_uploaded_document(question: str) -> bool:
    """True only if the question explicitly refers to the user's uploaded file."""
    question_lower = question.lower()
    return any(phrase in question_lower for phrase in EXPLICIT_USER_DOC_PHRASES)


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
        - "How do I register my immigration permission?" → LEGAL_ONLY (general process, not doc content)
        - "What does my Revenue letter say?" → USER_ONLY
        - "Based on my IRP stamp, can I work full-time?" → MIXED (needs a fact from their doc + legal rule)
        - "check now" (after "am I garda vetted?") → USER_ONLY (needs conversation context)
    """
    
    # ========== STEP 1: Quick keyword check ==========
    mentions_user_docs = references_uploaded_document(question)
    
    # If user explicitly references their own document but hasn't uploaded any,
    # they need to upload first.
    if mentions_user_docs and not user_has_docs:
        return IntentType.USER_ONLY  # Will trigger a clarification: "Please upload your documents first"
    
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

Decide the intent using ONE guiding test:
**"To answer this, do I need to READ a fact from the user's own uploaded document?"**

- NO  → LEGAL_ONLY. The question is about general Irish rules, processes,
        eligibility, entitlements or how-to. This is the DEFAULT. Questions
        phrased with "I", "me", or "my" are STILL legal_only when they ask
        about the general rule (e.g. "how do I register", "how many hours can
        I work", "do I need insurance", "what documents do I need").
- YES, and ONLY the document → USER_ONLY. The question asks what the user's
        uploaded document says/contains (dates, names, stamp printed on it).
- YES, document fact AND a legal rule → MIXED. Answering needs a specific fact
        from their document PLUS a general rule.

IMPORTANT:
- "User has uploaded documents" is only context. Do NOT choose USER_ONLY or
  MIXED merely because documents exist. Choose them only if the question itself
  depends on reading the user's document.
- Use conversation history for short follow-ups: "check now" / "do I have it?"
  after a question about a document → USER_ONLY.

Examples:
- "How do I apply for Stamp 4?" → LEGAL_ONLY
- "How do I register my immigration permission after arriving?" → LEGAL_ONLY
- "How long is the landing stamp valid on a student visa?" → LEGAL_ONLY
- "How many hours can I work on Stamp 2?" → LEGAL_ONLY
- "Do I need private health insurance for my student visa?" → LEGAL_ONLY
- "What documents do I need to open a bank account?" → LEGAL_ONLY
- "What does my Revenue letter say?" → USER_ONLY
- "What is the expiry date on my visa letter?" → USER_ONLY
- "Based on my IRP stamp, can I work full-time in summer?" → MIXED
- "Given the employer on my cover letter, am I allowed to work there?" → MIXED
- "check now" (after asking "am I garda vetted?") → USER_ONLY

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

