# app/ingestion/legal_ingest.py
from typing import Optional

from app.ingestion.legal_sources import LEGAL_SOURCES, LegalSource
from app.ingestion.html_loader import load_and_clean, HtmlLoadError
from app.schemas.rag import IngestTextRequest
from app.services.ingestion import ingest_text
from app.services.vector_store import delete_by_doc_id, doc_id_exists



def ingest_single_source(source: LegalSource, dry_run: bool = False, force: bool = False) -> Optional[int]:
    """
    Ingest a single legal source URL into the 'legal' collection.

    Args:
        source: The legal source to ingest
        dry_run: If True, only preview without indexing
        force: If True, re-ingest even if document already exists

    Returns number of chunks indexed, or None if failed.
    """
    print(f"\n=== Ingesting: {source.url} ({source.provider} / {source.topic} / {source.subtopic}) ===")

    doc_id = source.url
    
    # Check if document already exists (unless force re-ingestion is requested)
    if not force and not dry_run:
        if doc_id_exists("legal-knowledge", doc_id):
            print(f"[SKIP] Document already exists in collection. Use force=True to re-ingest.")
            return 0

    try:
        text = load_and_clean(source.url)
    except HtmlLoadError as e:
        print(f"[ERROR] {e}")
        return None

    if dry_run:
        # Just show a preview for debugging
        preview = text[:600].replace("\n", " ")
        print(f"[DRY RUN] Loaded {len(text)} characters. Preview:")
        print(preview + ("..." if len(text) > 600 else ""))
        return 0

    # If force=True and document exists, delete existing chunks first
    if force:
        try:
            delete_by_doc_id("legal-knowledge", doc_id)
            print(f"[INFO] Deleted existing chunks for doc_id={doc_id}")
        except Exception as e:
            print(f"[WARN] Failed to delete existing chunks for {doc_id}: {e}")

    request = IngestTextRequest(
        collection="legal-knowledge",
        text=text,
        metadata={
            "doc_id": doc_id,
            "source": source.provider,
            "topic": source.topic,
            "subtopic": source.subtopic,
            "url": source.url,
            "description": source.description,
        },
        # these chunk settings are decent defaults for legal text
        chunk_size=800,
        chunk_overlap=80,
    )

    count = ingest_text(request)
    print(f"[OK] Indexed {count} chunks for {source.url}")
    return count


def ingest_all_sources(dry_run: bool = False, force: bool = False) -> None:
    """
    Ingest all configured legal sources.
    
    Args:
        dry_run: If True, only preview without indexing
        force: If True, re-ingest even if documents already exist
    """
    total_chunks = 0
    skipped = 0
    for src in LEGAL_SOURCES:
        result = ingest_single_source(src, dry_run=dry_run, force=force)
        if isinstance(result, int):
            if result == 0 and not dry_run:
                skipped += 1
            else:
                total_chunks += result

    print(f"\n=== Done. Total chunks indexed: {total_chunks} | Skipped: {skipped} ===")


if __name__ == "__main__":
    # Change dry_run to True if you just want to see text preview.
    # Change force to True to re-ingest all documents even if they exist.
    ingest_all_sources(dry_run=False, force=False)
