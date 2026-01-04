from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from uuid import UUID
from pydantic import BaseModel
from typing import Optional

from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage
from app.schemas.chat import ChatSessionRead, ChatMessageRead

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
    session = db.query(ChatSession).filter(ChatSession.id == session_id, ChatSession.user_id == current_user.id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found",
        )
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).all()
    return messages

