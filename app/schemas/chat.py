from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
from app.models.chat_message import ChatRole

class ChatSessionRead(BaseModel):
    id: UUID
    user_id: int
    title: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatMessageRead(BaseModel):
    id: UUID
    session_id: UUID
    role: ChatRole
    content: str
    citations: dict | None
    created_at: datetime

    class Config:
        from_attributes = True

