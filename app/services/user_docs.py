# app/services/user_docs.py
import uuid
from typing import Dict, Any

from app.schemas.user_docs import UserDocTextIn, UserDocIngestResponse
from app.schemas.rag import IngestTextRequest
from app.services.ingestion import ingest_text
from app.services.vector_store import delete_by_doc_id, doc_id_exists


def ingest_user_document_text(payload: UserDocTextIn, force: bool = False) -> UserDocIngestResponse:
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

    return UserDocIngestResponse(
        doc_id=doc_id,
        chunks_indexed=chunks_indexed,
    )

