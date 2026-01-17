# app/schemas/langgraph_state.py
"""
This file defines the data structures (schemas) used in our LangGraph orchestration.
Think of these as the "blueprint" for what data flows through our AI pipeline.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel


# ============================================================================
# ENUMS (Predefined choices/categories)
# ============================================================================

class IntentType(str, Enum):
    """
    Defines what type of question the user is asking.
    This helps us decide which knowledge bases to search.
    
    LEGAL_ONLY: Question about general legal/immigration info (e.g., "How do I apply for Stamp 4?")
    USER_ONLY: Question about user's uploaded documents (e.g., "What does my Revenue letter say?")
    MIXED: Question needs both legal knowledge AND user documents (e.g., "Given my stamp, can I work 40 hours?")
    UNKNOWN: We're not sure what the user wants yet
    """
    LEGAL_ONLY = "legal_only"
    USER_ONLY = "user_only"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class CitationType(str, Enum):
    """
    Where did we get the information from?
    
    LEGAL: From our legal knowledge database (CitizensInformation, IrishImmigration, etc.)
    USER_DOC: From the user's uploaded documents
    """
    LEGAL = "legal"
    USER_DOC = "user_doc"


# ============================================================================
# CITATION MODEL (Source of information)
# ============================================================================

class Citation(BaseModel):
    """
    Represents a source/citation for information in the answer.
    Think of this like a footnote or bibliography entry.
    """
    type: CitationType  # Is this from legal knowledge or user's doc?
    title: str  # Name of the source (e.g., "Revenue Notice 2025" or "CitizensInformation: Work Permits")
    snippet: str  # The actual text excerpt we used from this source
    
    # Optional fields depending on citation type:
    url: Optional[str] = None  # Web URL (for legal sources)
    doc_id: Optional[str] = None  # Document ID (for user documents)
    
    class Config:
        # This allows the model to be used with Pydantic v2
        from_attributes = True


# ============================================================================
# RETRIEVAL CONTEXT (Information we found)
# ============================================================================

class RetrievalChunk(BaseModel):
    """
    Represents a "chunk" of text we retrieved from our vector database.
    When we search, we get back multiple chunks ranked by relevance.
    """
    text: str  # The actual content of this chunk
    score: float  # How relevant is this chunk? (0.0 to 1.0, higher is better)
    metadata: Dict[str, Any]  # Extra info like source URL, doc_id, title, etc.
    
    class Config:
        from_attributes = True


# ============================================================================
# LANGGRAPH STATE (The main data structure)
# ============================================================================

class LangGraphState(BaseModel):
    """
    This is the "state" that flows through our entire LangGraph pipeline.
    Each node in the graph reads from this state and writes back to it.
    
    Think of it like a bucket of information that gets passed from step to step,
    with each step adding more information to the bucket.
    """
    
    # ========== INPUT (What we start with) ==========
    user_id: str  # Who is asking the question? (for security: only show their docs)
    question: str  # The actual question the user asked
    session_id: Optional[str] = None  # Chat session ID (for conversation history)
    
    # ========== CLASSIFICATION (What type of question?) ==========
    intent: Optional[IntentType] = None  # What type of question is this?
    
    # ========== QUERY PLANNING (What should we search for?) ==========
    legal_query: Optional[str] = None  # Optimized search query for legal knowledge
    user_doc_query: Optional[str] = None  # Optimized search query for user documents
    
    # ========== RETRIEVAL (What information did we find?) ==========
    legal_context_chunks: List[RetrievalChunk] = []  # Chunks from legal knowledge base
    user_context_chunks: List[RetrievalChunk] = []  # Chunks from user's documents
    
    # ========== ANSWER GENERATION (Creating the response) ==========
    draft_answer: Optional[str] = None  # First version of the answer
    final_answer: Optional[str] = None  # Final, validated answer
    citations: List[Citation] = []  # Sources we used for the answer
    
    # ========== QUALITY CONTROL (Is the answer good enough?) ==========
    needs_clarification: bool = False  # Do we need to ask the user for more info?
    clarifying_question: Optional[str] = None  # What should we ask the user?
    
    # ========== METADATA (Extra info for debugging/logging) ==========
    error: Optional[str] = None  # If something went wrong, store error here
    metadata: Dict[str, Any] = {}  # Any other info we want to track
    
    class Config:
        from_attributes = True


# ============================================================================
# API RESPONSE MODEL (What we send back to the frontend)
# ============================================================================

class OrchestrationResponse(BaseModel):
    """
    This is what our API endpoint returns to the frontend.
    It's a clean, structured response that the UI can easily display.
    """
    answer: str  # The final answer to show the user
    citations: List[Citation]  # Sources to display as "References" or "Sources Used"
    needs_clarification: bool  # Should we show a follow-up question?
    clarifying_question: Optional[str] = None  # The follow-up question text
    metadata: Dict[str, Any] = {}  # Any extra info (for debugging or analytics)
    
    class Config:
        from_attributes = True

