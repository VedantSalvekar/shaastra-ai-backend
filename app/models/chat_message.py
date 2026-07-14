from sqlalchemy import ForeignKey, DateTime, Text, Enum, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
import uuid
import enum

from app.db.session import Base


class ChatRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True, nullable=False)

    role: Mapped[ChatRole] = mapped_column(Enum(ChatRole, name="chat_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    citations: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    session = relationship("ChatSession", back_populates="messages")


Index("ix_chat_messages_session_created_at", ChatMessage.session_id, ChatMessage.created_at)

