# app/schemas/user_docs.py
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class UserDocTextIn(BaseModel):
    """
    Ingest a single logical user document as plain text.
    File upload -> text extraction will call this later.
    """

    user_id: str = Field(..., description="Logical user identifier")
    doc_id: Optional[str] = Field(
        None,
        description="Stable document id; if not provided, backend will generate one.",
    )
    title: str = Field(..., description="Human-friendly title for the document")
    text: str = Field(..., description="Full extracted text of the document")
    extra_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata (e.g. doc_type, source_system).",
    )


class UserDocIngestResponse(BaseModel):
    doc_id: str
    chunks_indexed: int

