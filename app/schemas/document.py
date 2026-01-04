from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
from app.models.document import DocumentStatus

class DocumentRead(BaseModel):
    id: UUID
    user_id: int
    title: str
    doc_type: str | None
    storage_key: str | None
    mime_type: str | None
    size_bytes: int | None
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

