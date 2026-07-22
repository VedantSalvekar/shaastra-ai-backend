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
    # Decide doc_id
    doc_id = payload.doc_id or f"userdoc_{uuid.uuid4().hex}"

    # Check if document already exists (unless force re-ingestion is requested)
    if not force and doc_id_exists("user-documents", doc_id):
        print(f"[INFO] Document {doc_id} already exists. Skipping re-ingestion.")
        # Return 0 chunks indexed since we skipped
        return UserDocIngestResponse(
            doc_id=doc_id,
            chunks_indexed=0,
        )

    # Delete existing chunks for this doc in Qdrant 
    if force or payload.doc_id:
        try:
            delete_by_doc_id("user-documents", doc_id)
            print(f"[INFO] Deleted existing chunks for user doc {doc_id}")
        except Exception as e:
            # Non-fatal; just log
            print(f"[WARN] Failed to delete existing chunks for user doc {doc_id}: {e}")

    # Build metadata
    metadata: Dict[str, Any] = {
        "doc_id": doc_id,
        "user_id": payload.user_id,
        "title": payload.title,
        "source": "user_upload",
    }
    metadata.update(payload.extra_metadata or {})

    # Call existing ingestion pipeline
    req = IngestTextRequest(
        collection="user-documents",
        text=payload.text,
        metadata=metadata,
        chunk_size=800,
        chunk_overlap=80,
    )

    chunks_indexed = ingest_text(req)
    print(f"[OK] Indexed {chunks_indexed} chunks for user doc {doc_id}")

    # Save document metadata to PostgreSQL database
    if db is not None and user_id is not None:
        try:
            # Extract UUID from doc_id (handles both "userdoc_<uuid>" and plain UUID formats)
            uuid_str = doc_id.replace("userdoc_", "") if doc_id.startswith("userdoc_") else doc_id
            doc_uuid = uuid.UUID(uuid_str)
            
            existing_doc = db.query(Document).filter(Document.id == doc_uuid).first()
            
            if existing_doc:
                existing_doc.title = payload.title
                existing_doc.doc_type = payload.extra_metadata.get("doc_type")
                existing_doc.mime_type = payload.extra_metadata.get("content_type")
                existing_doc.size_bytes = payload.extra_metadata.get("file_size")
                existing_doc.extracted_text = payload.text
                existing_doc.status = DocumentStatus.indexed if chunks_indexed > 0 else DocumentStatus.failed
            else:
                doc_record = Document(
                    id=doc_uuid,
                    user_id=user_id,
                    title=payload.title,
                    doc_type=payload.extra_metadata.get("doc_type"),
                    storage_key=payload.extra_metadata.get("filename"),
                    mime_type=payload.extra_metadata.get("content_type"),
                    size_bytes=payload.extra_metadata.get("file_size"),
                    extracted_text=payload.text,
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


def _normalize_doc_uuid(document_id: str) -> uuid.UUID:
    """Accept either the DB UUID or the 'userdoc_<uuid>' form."""
    raw = document_id.replace("userdoc_", "") if document_id.startswith("userdoc_") else document_id
    return uuid.UUID(raw)


def reindex_user_document(document_id: str, db: Session, user_id: int) -> UserDocIngestResponse:
    """
    Rebuild the Qdrant vectors for a single user document from the text stored
    in PostgreSQL. Scoped to the owning user so no one can reindex another
    user's document.

    Raises ValueError if the document is not found for this user, or if no
    stored text exists (in which case the user must re-upload the file).
    """
    doc_uuid = _normalize_doc_uuid(document_id)

    doc = (
        db.query(Document)
        .filter(Document.id == doc_uuid, Document.user_id == user_id)
        .first()
    )
    if doc is None:
        raise ValueError("Document not found for this user.")
    if not doc.extracted_text:
        raise ValueError(
            "No stored text for this document. It was uploaded before text "
            "persistence was enabled — please re-upload the file."
        )

    payload = UserDocTextIn(
        user_id=str(user_id),
        doc_id=f"userdoc_{doc.id}",
        title=doc.title,
        text=doc.extracted_text,
        extra_metadata={
            "doc_type": doc.doc_type,
            "filename": doc.storage_key,
            "content_type": doc.mime_type,
            "file_size": doc.size_bytes,
        },
    )
    return ingest_user_document_text(payload, force=True, db=db, user_id=user_id)


def reindex_all_user_documents(db: Session, user_id: int) -> list[UserDocIngestResponse]:
    """Rebuild Qdrant vectors for every document this user has stored text for."""
    docs = db.query(Document).filter(Document.user_id == user_id).all()
    results: list[UserDocIngestResponse] = []
    for doc in docs:
        if not doc.extracted_text:
            print(f"[WARN] Skipping reindex for {doc.id}: no stored text (re-upload needed)")
            continue
        results.append(reindex_user_document(str(doc.id), db=db, user_id=user_id))
    return results

def _qdrant_doc_id_candidates(document_id: str, doc: Document) -> list[str]:
    doc_uuid = _normalize_doc_uuid(document_id)
    return list({
        f"userdoc_{doc_uuid.hex}",
        f"userdoc_{doc_uuid}",
        str(doc_uuid),
        document_id,
    })

def delete_user_document(document_id: str, db: Session, user_id: int) -> None:
    doc_uuid = _normalize_doc_uuid(document_id)
    doc = (
        db.query(Document)
        .filter(Document.id == doc_uuid, Document.user_id == user_id)
        .first()
    )
    if doc is None: 
        raise ValueError("Document not found for this user.")
    for qdrant_doc_id in _qdrant_doc_id_candidates(document_id, doc):
        try:
            delete_by_doc_id("user-documents", qdrant_doc_id)
        except Exception as e:
            print(f"[WARN] Failed to delete Qdrant chunks for user doc {qdrant_doc_id}: {e}")
    
    db.delete(doc)
    db.commit()