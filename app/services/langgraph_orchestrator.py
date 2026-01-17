# app/services/langgraph_orchestrator.py
"""
This is the MAIN orchestrator that ties everything together using LangGraph.

LangGraph is a framework for building workflows with AI. Think of it like a flowchart:
- Each box (node) does one specific task
- Arrows (edges) determine which box comes next
- The "state" flows through the boxes, getting updated at each step

Our graph looks like this:

START → classify_intent → plan_queries → [retrieve legal / retrieve user docs] → compose_answer → quality_gate → END
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session
import operator

from app.schemas.langgraph_state import (
    LangGraphState, 
    IntentType, 
    RetrievalChunk, 
    Citation,
    OrchestrationResponse
)
from app.services.intent_classifier import classify_intent, should_retrieve_legal, should_retrieve_user_docs
from app.services.query_planner import plan_retrieval
from app.services.vector_store import search_with_filters
from app.services.answer_composer import compose_answer
from app.services.quality_gate import validate_answer
from app.models.document import Document


# ============================================================================
# STATE DEFINITION (with proper handling for concurrent updates)
# ============================================================================

class GraphState(TypedDict):
    """
    State for our LangGraph workflow.
    
    Important: Fields that can be updated by multiple nodes running in parallel
    need to use Annotated with a reducer function. This tells LangGraph how to
    combine multiple updates to the same field.
    
    For example, when both retrieve_legal and retrieve_user_docs run in parallel,
    they both try to update the state. We use operator.add to combine their updates.
    """
    # Basic input fields (single value, no concurrent updates)
    user_id: str
    question: str
    session_id: str | None
    
    # Classification (single value)
    intent: IntentType | None
    
    # Query planning (single value)
    legal_query: str | None
    user_doc_query: str | None
    
    # Retrieval results (can be updated by parallel nodes - use Annotated with reducer)
    # The operator.add will concatenate lists from both retrieval nodes
    legal_context_chunks: Annotated[list[RetrievalChunk], operator.add]
    user_context_chunks: Annotated[list[RetrievalChunk], operator.add]
    
    # Answer composition (single value)
    draft_answer: str | None
    final_answer: str | None
    citations: list[Citation]
    
    # Quality control (single value)
    needs_clarification: bool
    clarifying_question: str | None
    
    # Metadata (single value)
    error: str | None
    metadata: dict


# ============================================================================
# HELPER FUNCTION: Check if user has documents
# ============================================================================

def user_has_documents(user_id: str, db: Session) -> tuple[bool, list[dict]]:
    """
    Check if a user has uploaded any documents and return document information.
    
    This is used by the intent classifier to make better decisions.
    If user asks "What does my letter say?" but has no documents, we need to tell them to upload first.
    
    Returns:
        tuple: (has_docs: bool, documents: list[dict])
        - has_docs: True if user has at least one document
        - documents: List of dicts with document metadata (id, title, doc_type, filename)
    """
    try:
        # Query database for this user's documents
        user_docs = db.query(Document).filter(Document.user_id == int(user_id)).all()
        
        if not user_docs:
            return (False, [])
        
        # Build a list of document metadata for the AI to reference
        doc_info = []
        for doc in user_docs:
            doc_info.append({
                "id": str(doc.id),
                "title": doc.title,
                "doc_type": doc.doc_type or "unknown",
                "filename": doc.storage_key or "",
                "status": doc.status.value
            })
        
        return (True, doc_info)
    except Exception as e:
        print(f"[ERROR] Failed to get user documents: {e}")
        return (False, [])


# ============================================================================
# NODE FUNCTIONS (Each node in our graph)
# ============================================================================

def node_classify_intent(state: dict, db: Session) -> dict:
    """
    NODE 1: Classify the user's intent.
    
    This node determines what type of question the user is asking:
    - LEGAL_ONLY: General legal/immigration question
    - USER_ONLY: Question about their uploaded documents
    - MIXED: Needs both legal knowledge and their documents
    - UNKNOWN: Not sure, will default to legal search
    
    Input state: user_id, question
    Output state: adds 'intent' and metadata about user's documents
    """
    print("\n[NODE] Classifying intent...")
    
    # Check if user has uploaded documents and get their details
    has_docs, user_docs = user_has_documents(state["user_id"], db)
    
    # Classify the intent using our AI classifier
    intent = classify_intent(state["question"], has_docs)
    
    print(f"[NODE] Intent classified as: {intent.value}")
    print(f"[NODE] User has documents: {has_docs}")
    if has_docs:
        print(f"[NODE] User's documents: {[doc['title'] for doc in user_docs]}")
    
    # Build a summary of user's documents for the AI to reference
    doc_summary = ""
    if user_docs:
        doc_summary = "User has uploaded the following documents:\n"
        for doc in user_docs:
            doc_summary += f"- {doc['title']}"
            if doc['doc_type']:
                doc_summary += f" (Type: {doc['doc_type']})"
            if doc['filename']:
                doc_summary += f" [File: {doc['filename']}]"
            doc_summary += "\n"
    
    # Update state with the classification and document info
    return {
        **state,
        "intent": intent,
        "metadata": {
            **state.get("metadata", {}),
            "user_has_docs": has_docs,
            "user_documents": user_docs,
            "document_summary": doc_summary
        }
    }


def node_plan_queries(state: dict) -> dict:
    """
    NODE 2: Plan retrieval queries.
    
    This node takes the original question and generates two optimized search queries:
    - legal_query: For searching legal knowledge base
    - user_doc_query: For searching user's documents
    
    Input state: question, intent
    Output state: adds 'legal_query' and 'user_doc_query'
    """
    print("\n[NODE] Planning retrieval queries...")
    
    # Generate optimized queries
    legal_query, user_doc_query = plan_retrieval(
        state["question"],
        state["intent"]
    )
    
    print(f"[NODE] Legal query: {legal_query}")
    print(f"[NODE] User doc query: {user_doc_query}")
    
    # Update state with the queries
    return {
        **state,
        "legal_query": legal_query,
        "user_doc_query": user_doc_query
    }


def node_retrieve_legal(state: dict) -> dict:
    """
    NODE 3: Retrieve from legal knowledge base.
    
    This node searches our legal knowledge database (CitizensInformation, IrishImmigration, etc.)
    for relevant information.
    
    Input state: legal_query
    Output state: adds 'legal_context_chunks'
    
    IMPORTANT: Returns chunks as a list to be ADDED to existing state (for parallel execution)
    """
    print("\n[NODE] Retrieving from legal knowledge base...")
    
    try:
        # Search legal knowledge collection
        chunks = search_with_filters(
            collection="legal-knowledge",
            query=state["legal_query"],
            top_k=5,  # Get top 5 most relevant chunks
            # No user_id filter needed for legal knowledge (it's public)
        )
        
        print(f"[NODE] Retrieved {len(chunks)} legal chunks")
        if chunks:
            print(f"[NODE] Top chunk score: {chunks[0].score:.3f}")
        
        # Return ONLY the new data (LangGraph will merge with existing state)
        return {
            "legal_context_chunks": chunks
        }
    
    except Exception as e:
        print(f"[ERROR] Legal retrieval failed: {e}")
        # Continue with empty results rather than failing completely
        return {
            "legal_context_chunks": []
        }


def node_retrieve_user_docs(state: dict) -> dict:
    """
    NODE 4: Retrieve from user's documents.
    
    This node searches the user's uploaded documents for relevant information.
    SECURITY: Always filters by user_id to ensure users only see their own documents.
    
    Input state: user_doc_query, user_id
    Output state: adds 'user_context_chunks'
    
    IMPORTANT: Returns chunks as a list to be ADDED to existing state (for parallel execution)
    """
    print("\n[NODE] Retrieving from user documents...")
    
    try:
        # Search user documents collection with MANDATORY user_id filter
        chunks = search_with_filters(
            collection="user-documents",
            query=state["user_doc_query"],
            top_k=5,  # Get top 5 most relevant chunks
            user_id=state["user_id"],  # SECURITY: Filter by user_id
            # Could also add doc_id or doc_type filters here if needed
        )
        
        print(f"[NODE] Retrieved {len(chunks)} user document chunks")
        if chunks:
            print(f"[NODE] Top chunk score: {chunks[0].score:.3f}")
        
        # Return ONLY the new data (LangGraph will merge with existing state)
        return {
            "user_context_chunks": chunks
        }
    
    except Exception as e:
        print(f"[ERROR] User doc retrieval failed: {e}")
        # Continue with empty results rather than failing completely
        return {
            "user_context_chunks": []
        }


def node_compose_answer(state: dict) -> dict:
    """
    NODE 5: Compose the answer.
    
    This node uses an LLM to generate an answer based on the retrieved context.
    It also extracts citations from the sources used.
    
    Input state: question, legal_context_chunks, user_context_chunks, metadata (document_summary)
    Output state: adds 'draft_answer' and 'citations'
    """
    print("\n[NODE] Composing answer...")
    
    # Get document summary from metadata (if available)
    document_summary = state.get("metadata", {}).get("document_summary", "")
    
    # Generate answer with citations
    answer, citations = compose_answer(
        question=state["question"],
        legal_chunks=state.get("legal_context_chunks", []),
        user_chunks=state.get("user_context_chunks", []),
        document_summary=document_summary
    )
    
    print(f"[NODE] Generated answer ({len(answer)} chars)")
    print(f"[NODE] Extracted {len(citations)} citations")
    
    return {
        **state,
        "draft_answer": answer,
        "citations": citations
    }


def node_quality_gate(state: dict) -> dict:
    """
    NODE 6: Quality gate validation.
    
    This node validates the answer and determines if we need clarification.
    It checks for common issues like missing context, low confidence, etc.
    
    Input state: draft_answer, legal_context_chunks, user_context_chunks, intent
    Output state: adds 'final_answer', 'needs_clarification', 'clarifying_question'
    """
    print("\n[NODE] Running quality gate validation...")
    
    # Validate the answer
    is_valid, clarifying_question = validate_answer(
        draft_answer=state["draft_answer"],
        legal_chunks=state.get("legal_context_chunks", []),
        user_chunks=state.get("user_context_chunks", []),
        intent=state["intent"],
        user_has_docs=state.get("metadata", {}).get("user_has_docs", False)
    )
    
    if is_valid:
        # Answer is good - use it as final answer
        print("[NODE] Answer passed quality gate ✓")
        return {
            **state,
            "final_answer": state["draft_answer"],
            "needs_clarification": False,
            "clarifying_question": None
        }
    else:
        # Answer needs clarification
        print(f"[NODE] Answer needs clarification: {clarifying_question}")
        return {
            **state,
            "final_answer": clarifying_question,  # Send clarifying question as the response
            "needs_clarification": True,
            "clarifying_question": clarifying_question
        }


# ============================================================================
# ROUTING FUNCTIONS (Determine which path to take)
# ============================================================================

def route_after_planning(state: dict) -> list[str]:
    """
    Routing function: Decide which retrieval nodes to run.
    
    Based on the intent, we might need:
    - Legal retrieval only
    - User doc retrieval only
    - Both (in parallel)
    
    Returns a list of node names to execute next.
    """
    intent = state["intent"]
    next_nodes = []
    
    if should_retrieve_legal(intent):
        next_nodes.append("retrieve_legal")
    
    if should_retrieve_user_docs(intent):
        next_nodes.append("retrieve_user_docs")
    
    # If we don't know what to do, default to legal retrieval
    if not next_nodes:
        next_nodes.append("retrieve_legal")
    
    print(f"[ROUTING] Will execute: {next_nodes}")
    return next_nodes


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def create_orchestration_graph(db: Session):
    """
    Creates and returns the LangGraph workflow.
    
    This function defines the structure of our AI workflow:
    1. What nodes exist
    2. How they connect to each other
    3. What data flows between them
    
    Returns:
        Compiled LangGraph that can be executed
    """
    
    # Create a new state graph with our GraphState TypedDict
    # This properly handles concurrent updates from parallel nodes
    graph = StateGraph(GraphState)
    
    # ========== ADD NODES ==========
    # Each node is a function that processes the state
    graph.add_node("classify_intent", lambda state: node_classify_intent(state, db))
    graph.add_node("plan_queries", node_plan_queries)
    graph.add_node("retrieve_legal", node_retrieve_legal)
    graph.add_node("retrieve_user_docs", node_retrieve_user_docs)
    graph.add_node("compose_answer", node_compose_answer)
    graph.add_node("quality_gate", node_quality_gate)
    
    # ========== ADD EDGES (Connections between nodes) ==========
    
    # Start → classify_intent
    graph.set_entry_point("classify_intent")
    
    # classify_intent → plan_queries
    graph.add_edge("classify_intent", "plan_queries")
    
    # plan_queries → [retrieve_legal and/or retrieve_user_docs]
    # This is a CONDITIONAL edge - it decides at runtime which nodes to execute
    graph.add_conditional_edges(
        "plan_queries",
        route_after_planning,
        {
            "retrieve_legal": "retrieve_legal",
            "retrieve_user_docs": "retrieve_user_docs"
        }
    )
    
    # Both retrieval nodes → compose_answer
    graph.add_edge("retrieve_legal", "compose_answer")
    graph.add_edge("retrieve_user_docs", "compose_answer")
    
    # compose_answer → quality_gate
    graph.add_edge("compose_answer", "quality_gate")
    
    # quality_gate → END
    graph.add_edge("quality_gate", END)
    
    # ========== COMPILE THE GRAPH ==========
    # This validates the graph structure and prepares it for execution
    compiled_graph = graph.compile()
    
    print("[INFO] LangGraph orchestration workflow created successfully")
    return compiled_graph


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def process_question(
    question: str,
    user_id: str,
    db: Session,
    session_id: str | None = None
) -> OrchestrationResponse:
    """
    Main function to process a user's question through the entire LangGraph pipeline.
    
    This is what external code (like your API endpoint) should call.
    
    How it works:
    1. Create the initial state with the question and user_id
    2. Build the graph
    3. Execute the graph (it flows through all the nodes automatically)
    4. Extract the final result
    5. Return a clean response
    
    Args:
        question: The user's question
        user_id: The user's ID (for document filtering)
        db: Database session (for checking if user has docs)
        session_id: Optional chat session ID
    
    Returns:
        OrchestrationResponse with answer, citations, and clarification status
    
    Example:
        response = process_question(
            question="Can I work with Stamp 4?",
            user_id="123",
            db=db_session
        )
        print(response.answer)
        print(response.citations)
    """
    
    print("\n" + "="*80)
    print(f"[ORCHESTRATOR] Processing question: {question}")
    print(f"[ORCHESTRATOR] User ID: {user_id}")
    print("="*80)
    
    # ========== STEP 1: Create initial state ==========
    initial_state = {
        "user_id": str(user_id),
        "question": question,
        "session_id": session_id,
        "intent": None,
        "legal_query": None,
        "user_doc_query": None,
        "legal_context_chunks": [],
        "user_context_chunks": [],
        "draft_answer": None,
        "final_answer": None,
        "citations": [],
        "needs_clarification": False,
        "clarifying_question": None,
        "error": None,
        "metadata": {}
    }
    
    try:
        # ========== STEP 2: Create and execute the graph ==========
        graph = create_orchestration_graph(db)
        
        # Execute the graph - it will flow through all nodes automatically
        print("\n[ORCHESTRATOR] Executing graph...")
        final_state = graph.invoke(initial_state)
        
        # ========== STEP 3: Extract results from final state ==========
        print("\n[ORCHESTRATOR] Graph execution completed")
        print(f"[ORCHESTRATOR] Final answer: {final_state['final_answer'][:100]}...")
        print(f"[ORCHESTRATOR] Citations: {len(final_state['citations'])}")
        print(f"[ORCHESTRATOR] Needs clarification: {final_state['needs_clarification']}")
        
        # ========== STEP 4: Return formatted response ==========
        return OrchestrationResponse(
            answer=final_state["final_answer"] or "I encountered an issue processing your question.",
            citations=final_state["citations"],
            needs_clarification=final_state["needs_clarification"],
            clarifying_question=final_state.get("clarifying_question"),
            metadata={
                "intent": final_state["intent"].value if final_state.get("intent") else "unknown",
                "legal_chunks_found": len(final_state.get("legal_context_chunks", [])),
                "user_chunks_found": len(final_state.get("user_context_chunks", []))
            }
        )
    
    except Exception as e:
        # If anything goes wrong, return an error response
        print(f"\n[ERROR] Orchestration failed: {e}")
        import traceback
        traceback.print_exc()
        
        return OrchestrationResponse(
            answer="I encountered an error processing your question. Please try again or rephrase your question.",
            citations=[],
            needs_clarification=False,
            clarifying_question=None,
            metadata={"error": str(e)}
        )

