# app/api/user_docs.py
from fastapi import APIRouter, HTTPException, status, Query

from app.schemas.user_docs import UserDocTextIn, UserDocIngestResponse
from app.services.user_docs import ingest_user_document_text

router = APIRouter(prefix="/user-docs", tags=["user-docs"])


@router.post("/ingest-text", response_model=UserDocIngestResponse)
def ingest_user_doc_text_endpoint(
    payload: UserDocTextIn,
    force: bool = Query(False, description="Force re-ingestion even if document exists"),
) -> UserDocIngestResponse:
    """
    Ingest a user document as raw text into the 'user-documents' collection.
    
    For now, this endpoint expects text. Later, the file-upload endpoint will
    extract text and call this same service.
    
    Args:
        payload: User document data including user_id, title, and text
        force: If True, re-ingest even if document with same doc_id already exists
        
    Returns:
        UserDocIngestResponse with doc_id and chunks_indexed count
    """
    try:
        return ingest_user_document_text(payload, force=force)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest user document: {e}",
        )

