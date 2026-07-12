# app/api/user_docs.py
from fastapi import APIRouter, HTTPException, status, Query, UploadFile, File, Form, Depends
from typing import Optional
from sqlalchemy.orm import Session

from app.schemas.user_docs import UserDocTextIn, UserDocIngestResponse
from app.services.user_docs import (
    ingest_user_document_text,
    reindex_all_user_documents,
    reindex_user_document,
)
from app.services.file_extraction import extract_text_from_file
from app.api.deps import get_db, get_current_user
from app.models.user import User

router = APIRouter(prefix="/user-docs", tags=["user-docs"])


@router.post("/ingest-text", response_model=UserDocIngestResponse)
def ingest_user_doc_text_endpoint(
    payload: UserDocTextIn,
    force: bool = Query(False, description="Force re-ingestion even if document exists"),
) -> UserDocIngestResponse:
    
    try:
        return ingest_user_document_text(payload, force=force)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest user document: {e}",
        )


@router.post("/upload-file", response_model=UserDocIngestResponse)
async def upload_user_document_file(
    file: UploadFile = File(..., description="File to upload (PDF, DOCX, or TXT)"),
    title: Optional[str] = Form(None, description="Document title (uses filename if not provided)"),
    doc_id: Optional[str] = Form(None, description="Optional document ID for updates"),
    doc_type: Optional[str] = Form(None, description="Document type (e.g., bank_statement, visa_application)"),
    force: bool = Query(False, description="Force re-ingestion even if document exists"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserDocIngestResponse:
    """
    Upload and ingest a user document file (PDF, DOCX, or TXT).
    
    This endpoint:
    1. Accepts a file upload
    2. Extracts text from the file
    3. Saves metadata to PostgreSQL
    4. Indexes the text in Qdrant for RAG queries
    
    Supported file types:
    - PDF (.pdf)
    - Microsoft Word (.docx)
    - Plain text (.txt)
    
    Args:
        file: The file to upload
        title: Document title (defaults to filename)
        doc_id: Optional stable document ID (generated if not provided)
        doc_type: Optional document type for metadata
        force: Force re-ingestion even if document exists
        
    Returns:
        UserDocIngestResponse with doc_id and chunks_indexed count
        
    Example:
        ```
        curl -X POST "http://localhost:8000/user-docs/upload-file" \
          -H "Authorization: Bearer YOUR_TOKEN" \
          -F "file=@/path/to/document.pdf" \
          -F "title=My Bank Statement" \
          -F "doc_type=bank_statement"
        ```
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size is {max_size / (1024*1024)}MB",
            )
        
        # Extract text from file
        try:
            extracted_text = extract_text_from_file(
                file_content=file_content,
                filename=file.filename or "unknown",
                content_type=file.content_type,
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )
        except ImportError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Server configuration error: {e}",
            )
        
        # Validate extracted text
        if not extracted_text or len(extracted_text.strip()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to extract meaningful text from the file. File may be empty or corrupted.",
            )
        
        # Prepare metadata
        extra_metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size": len(file_content),
        }
        if doc_type:
            extra_metadata["doc_type"] = doc_type
        
        # Create ingestion payload
        payload = UserDocTextIn(
            user_id=str(current_user.id),
            doc_id=doc_id,
            title=title or file.filename or "Untitled Document",
            text=extracted_text,
            extra_metadata=extra_metadata,
        )
        
        # Ingest the document (saves to both Qdrant and PostgreSQL)
        return ingest_user_document_text(payload, force=force, db=db, user_id=current_user.id)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process file upload: {str(e)}",
        )


@router.post("/{document_id}/reindex", response_model=UserDocIngestResponse)
def reindex_user_document_endpoint(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> UserDocIngestResponse:
    """
    Rebuild the Qdrant vectors for one of the current user's documents from the
    text stored in the database. Useful if the vector store was cleared/lost.
    """
    try:
        return reindex_user_document(document_id, db=db, user_id=current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reindex document: {e}",
        )


@router.post("/reindex-all", response_model=list[UserDocIngestResponse])
def reindex_all_user_documents_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[UserDocIngestResponse]:
    """
    Rebuild Qdrant vectors for all of the current user's documents that have
    stored text. Documents uploaded before text persistence must be re-uploaded.
    """
    try:
        return reindex_all_user_documents(db=db, user_id=current_user.id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reindex documents: {e}",
        )

