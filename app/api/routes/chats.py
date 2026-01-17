from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from uuid import UUID
from pydantic import BaseModel
from typing import Optional, List

from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage, ChatRole
from app.schemas.chat import ChatSessionRead, ChatMessageRead
from app.services.langgraph_orchestrator import process_question

router = APIRouter()

class CreateChatSessionRequest(BaseModel):
    title: Optional[str] = None

@router.post("/chats", response_model=ChatSessionRead, status_code=status.HTTP_201_CREATED)
def create_chat_session(
    request: CreateChatSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatSession:
    session = ChatSession(
        user_id=current_user.id,
        title=request.title or "New Chat"
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

@router.get("/chats", response_model=list[ChatSessionRead])
def list_user_chats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ChatSession]:
    sessions = db.query(ChatSession).filter(ChatSession.user_id == current_user.id).order_by(ChatSession.updated_at.desc()).all()
    return sessions

@router.get("/chats/{session_id}/messages", response_model=list[ChatMessageRead])
def list_chat_messages(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ChatMessage]:
    """
    Get all messages in a chat session.
    Returns messages in chronological order (oldest first).
    """
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found",
        )
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).all()
    return messages


class CreateMessageRequest(BaseModel):
    """Request model for creating a new message in a chat session."""
    content: str  # The user's question/message


@router.post("/chats/{session_id}/messages", response_model=ChatMessageRead, status_code=status.HTTP_201_CREATED)
def create_chat_message(
    session_id: UUID,
    request: CreateMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ChatMessage:
    """
    Create a new message in a chat session and get an AI-generated response.
    
    This endpoint:
    1. Validates that the chat session exists and belongs to the user
    2. Saves the user's message to the database
    3. Processes the question through the LangGraph orchestrator
    4. Generates an AI response with citations
    5. Saves the AI response to the database
    6. Returns the AI response to the user
    
    The LangGraph orchestrator handles:
    - Intent classification (legal vs user docs vs mixed)
    - Query optimization
    - Retrieval from legal knowledge and/or user documents
    - Answer composition with citations
    - Quality validation
    """
    
    # ========== STEP 1: Verify chat session exists and belongs to user ==========
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found",
        )
    
    # ========== STEP 2: Save user's message to database ==========
    user_message = ChatMessage(
        session_id=session_id,
        role=ChatRole.user,
        content=request.content,
        citations=None  # User messages don't have citations
    )
    db.add(user_message)
    db.commit()
    db.refresh(user_message)
    
    try:
        # ========== STEP 3: Process question through LangGraph orchestrator ==========
        print(f"\n[API] Processing question: {request.content}")
        
        orchestration_result = process_question(
            question=request.content,
            user_id=str(current_user.id),
            db=db,
            session_id=str(session_id)
        )
        
        # ========== STEP 4: Prepare citations for database storage ==========
        # Convert Citation objects to dict format for JSONB storage
        citations_data = None
        if orchestration_result.citations:
            citations_data = {
                "sources": [
                    {
                        "type": citation.type.value,
                        "title": citation.title,
                        "snippet": citation.snippet,
                        "url": citation.url,
                        "doc_id": citation.doc_id
                    }
                    for citation in orchestration_result.citations
                ],
                "needs_clarification": orchestration_result.needs_clarification,
                "metadata": orchestration_result.metadata
            }
        
        # ========== STEP 5: Save AI response to database ==========
        assistant_message = ChatMessage(
            session_id=session_id,
            role=ChatRole.assistant,
            content=orchestration_result.answer,
            citations=citations_data
        )
        db.add(assistant_message)
        
        # Update session's updated_at timestamp
        session.updated_at = func.now()
        
        db.commit()
        db.refresh(assistant_message)
        
        print(f"[API] Successfully created message pair (user + assistant)")
        
        # ========== STEP 6: Return assistant's response ==========
        return assistant_message
    
    except Exception as e:
        # If something goes wrong with AI processing, still save an error message
        print(f"[ERROR] Failed to process question: {e}")
        
        # Create an error response message
        error_message = ChatMessage(
            session_id=session_id,
            role=ChatRole.assistant,
            content="I encountered an error processing your question. Please try again or rephrase your question.",
            citations={"error": str(e)}
        )
        db.add(error_message)
        db.commit()
        db.refresh(error_message)
        
        return error_message

