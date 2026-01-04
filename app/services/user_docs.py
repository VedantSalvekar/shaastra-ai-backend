# app/services/user_docs.py
import uuid
from typing import Dict, Any
from sqlalchemy.orm import Session

from app.schemas.user_docs import UserDocTextIn, UserDocIngestResponse
from app.schemas.rag import IngestTextRequest
from app.services.ingestion import ingest_text
from app.services.vector_store import delete_by_doc_id, doc_id_exists
from app.models.document import Document, DocumentStatus


def ingest_user_document_text(
    payload: UserDocTextIn, 
    force: bool = False,
    db: Session | None = None,
    user_id: int | None = None
) -> UserDocIngestResponse:
    """
    Ingest or re-ingest a single user document into the 'user-documents' collection.
    
    Args:
        payload: User document data to ingest
        force: If True, re-ingest even if document already exists
        
    - If doc_id is provided, we treat it as stable and replace existing chunks.
    - If doc_id is missing, we generate one.
    - If force is False and doc_id exists, we skip re-ingestion.
    """
    # 1) Decide doc_id
    doc_id = payload.doc_id or f"userdoc_{uuid.uuid4().hex}"

    # 2) Check if document already exists (unless force re-ingestion is requested)
    if not force and doc_id_exists("user-documents", doc_id):
        print(f"[INFO] Document {doc_id} already exists. Skipping re-ingestion.")
        # Return 0 chunks indexed since we skipped
        return UserDocIngestResponse(
            doc_id=doc_id,
            chunks_indexed=0,
        )

    # 3) Delete existing chunks for this doc in Qdrant (if force=True or first time)
    if force or payload.doc_id:
        try:
            delete_by_doc_id("user-documents", doc_id)
            print(f"[INFO] Deleted existing chunks for user doc {doc_id}")
        except Exception as e:
            # Non-fatal; just log
            print(f"[WARN] Failed to delete existing chunks for user doc {doc_id}: {e}")

    # 4) Build metadata
    metadata: Dict[str, Any] = {
        "doc_id": doc_id,
        "user_id": payload.user_id,
        "title": payload.title,
        "source": "user_upload",
    }
    metadata.update(payload.extra_metadata or {})

    # 5) Call existing ingestion pipeline
    req = IngestTextRequest(
        collection="user-documents",
        text=payload.text,
        metadata=metadata,
        chunk_size=800,
        chunk_overlap=80,
    )

    chunks_indexed = ingest_text(req)
    print(f"[OK] Indexed {chunks_indexed} chunks for user doc {doc_id}")

    # 6) Save document metadata to PostgreSQL database
    if db is not None and user_id is not None:
        try:
            existing_doc = db.query(Document).filter(Document.id == uuid.UUID(doc_id)).first()
            
            if existing_doc:
                existing_doc.title = payload.title
                existing_doc.doc_type = payload.extra_metadata.get("doc_type")
                existing_doc.mime_type = payload.extra_metadata.get("content_type")
                existing_doc.size_bytes = payload.extra_metadata.get("file_size")
                existing_doc.status = DocumentStatus.indexed if chunks_indexed > 0 else DocumentStatus.failed
            else:
                doc_record = Document(
                    id=uuid.UUID(doc_id),
                    user_id=user_id,
                    title=payload.title,
                    doc_type=payload.extra_metadata.get("doc_type"),
                    storage_key=payload.extra_metadata.get("filename"),
                    mime_type=payload.extra_metadata.get("content_type"),
                    size_bytes=payload.extra_metadata.get("file_size"),
                    status=DocumentStatus.indexed if chunks_indexed > 0 else DocumentStatus.failed,
                )
                db.add(doc_record)
            
            db.commit()
            print(f"[OK] Saved document metadata to database: {doc_id}")
        except Exception as e:
            print(f"[WARN] Failed to save document to database: {e}")
            db.rollback()

    return UserDocIngestResponse(
        doc_id=doc_id,
        chunks_indexed=chunks_indexed,
    )

