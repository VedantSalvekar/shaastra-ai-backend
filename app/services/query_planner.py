# app/services/query_planner.py
"""
This service generates optimized search queries.

Why do we need this?
- User asks: "Can I work given my current stamp?"
- To get good search results, we need TWO different queries:
  1. Legal query: "Stamp 4 work hour limitations Ireland employment rules"
  2. User doc query: "immigration stamp type IRP card validity work permit"

This service uses AI to generate these optimized queries.
"""

from openai import OpenAI
from app.schemas.langgraph_state import IntentType
from app.core.config import settings


def plan_retrieval(question: str, intent: IntentType) -> tuple[str, str]:
    """
    Generates two optimized search queries: one for legal knowledge, one for user documents.
    
    How it works:
    1. We give the AI the original question and the intent
    2. The AI generates two specialized queries optimized for different searches
    3. We return both queries to be used in retrieval
    
    Args:
        question: The user's original question
        intent: What type of question is this (LEGAL_ONLY, USER_ONLY, MIXED, UNKNOWN)
    
    Returns:
        tuple: (legal_query, user_doc_query)
        - legal_query: Optimized query for searching legal knowledge base
        - user_doc_query: Optimized query for searching user's documents
    
    Example:
        Input: "Can I work 40 hours with my current visa?"
        Output:
            legal_query: "Stamp 4 work hour limitations Ireland employment restrictions visa"
            user_doc_query: "visa type immigration stamp IRP card work permit employment authorization"
    """
    
    # Build a prompt for the AI to generate queries
    query_planning_prompt = f"""You are a query optimization system for an Irish immigration assistant.

User's question: "{question}"
Intent: {intent.value}

Your task: Generate TWO optimized search queries:

1. LEGAL_QUERY: For searching a legal knowledge base (Irish immigration law, work permits, visas, etc.)
   - Should contain keywords related to Irish immigration law
   - Should be optimized for finding relevant legal information
   - Should include synonyms and related terms

2. USER_DOC_QUERY: For searching the user's uploaded documents (IRP cards, visa letters, bank statements, etc.)
   - Should contain keywords that would appear in official documents
   - Should look for document types, stamps, dates, permits, etc.
   - Should be specific to finding personal information in documents

Guidelines:
- Keep queries concise but information-rich (5-10 keywords each)
- Include Irish immigration terminology
- Think about what text would actually appear in documents vs legal guides

Respond in EXACTLY this format:
LEGAL_QUERY: [your legal query here]
USER_DOC_QUERY: [your user doc query here]

Example:
User asks: "Can I work 40 hours with my current visa?"
LEGAL_QUERY: Stamp 4 work hour limitations Ireland employment restrictions visa conditions
USER_DOC_QUERY: immigration stamp type IRP card work permit conditions employment authorization validity"""

    try:
        # Call OpenAI API
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Fast model for query generation
            messages=[
                {"role": "system", "content": "You are a query optimization system. Generate concise, keyword-rich search queries."},
                {"role": "user", "content": query_planning_prompt}
            ],
            temperature=0.3,  # Slight creativity for query variations
            max_tokens=150  # Enough for two queries
        )
        
        # Parse the response
        content = response.choices[0].message.content.strip()
        
        # Extract the two queries from the response
        legal_query = question  # Default fallback
        user_doc_query = question  # Default fallback
        
        for line in content.split('\n'):
            if line.startswith('LEGAL_QUERY:'):
                legal_query = line.replace('LEGAL_QUERY:', '').strip()
            elif line.startswith('USER_DOC_QUERY:'):
                user_doc_query = line.replace('USER_DOC_QUERY:', '').strip()
        
        # If we didn't get proper queries, use the original question
        if not legal_query or legal_query == "":
            legal_query = question
        if not user_doc_query or user_doc_query == "":
            user_doc_query = question
        
        print(f"[INFO] Query planning:")
        print(f"  Original: {question}")
        print(f"  Legal query: {legal_query}")
        print(f"  User doc query: {user_doc_query}")
        
        return legal_query, user_doc_query
        
    except Exception as e:
        # If something goes wrong, just use the original question for both
        print(f"[WARN] Query planning failed: {e}. Using original question.")
        return question, question


def optimize_legal_query(question: str) -> str:
    """
    Helper function: Generate only the legal query (for LEGAL_ONLY intent).
    This is a shortcut when we know we only need legal knowledge.
    """
    legal_query, _ = plan_retrieval(question, IntentType.LEGAL_ONLY)
    return legal_query


def optimize_user_doc_query(question: str) -> str:
    """
    Helper function: Generate only the user doc query (for USER_ONLY intent).
    This is a shortcut when we know we only need user documents.
    """
    _, user_doc_query = plan_retrieval(question, IntentType.USER_ONLY)
    return user_doc_query

