# app/ingestion/legal_ingest.py
from typing import Optional

from app.ingestion.html_loader import HtmlLoadError, load_page
from app.ingestion.legal_sources import LEGAL_SOURCES, LegalSource
from app.schemas.rag import IngestTextRequest
from app.services.ingestion import ingest_text
from app.services.vector_store import delete_by_doc_id, doc_id_exists


def _build_metadata(source: LegalSource, page) -> dict:
    return {
        "doc_id": page.canonical_url,
        "source": source.provider,
        "topic": source.topic,
        "subtopic": source.subtopic,
        "title": page.title or source.title,
        "source_url": page.canonical_url,
        "url": page.canonical_url,
        "authority_tier": source.authority_tier,
        "description": source.description,
        "content_hash": page.content_hash,
        "fetched_at": page.fetched_at,
    }


def ingest_single_source(
    source: LegalSource,
    dry_run: bool = False,
    force: bool = False,
) -> Optional[int]:
    """
    Ingest a single legal source URL into the legal-knowledge collection.

    Returns number of chunks indexed, 0 if skipped, or None if failed.
    """
    print(
        f"\n=== Ingesting: {source.url} "
        f"({source.provider} / {source.topic} / {source.subtopic}) ==="
    )

    try:
        page = load_page(source.url)
    except HtmlLoadError as e:
        print(f"[ERROR] {e}")
        return None

    doc_id = page.canonical_url
    print(f"[INFO] title={page.title!r} chars={len(page.text)} hash={page.content_hash[:12]}...")

    if not force and not dry_run and doc_id_exists("legal-knowledge", doc_id):
        print("[SKIP] Document already exists in collection. Use force=True to re-ingest.")
        return 0

    if dry_run:
        preview = page.text[:600].replace("\n", " ")
        print(f"[DRY RUN] Loaded {len(page.text)} characters. Preview:")
        print(preview + ("..." if len(page.text) > 600 else ""))
        return 0

    if force:
        try:
            delete_by_doc_id("legal-knowledge", doc_id)
            print(f"[INFO] Deleted existing chunks for doc_id={doc_id}")
        except Exception as e:
            print(f"[WARN] Failed to delete existing chunks for {doc_id}: {e}")

    request = IngestTextRequest(
        collection="legal-knowledge",
        text=page.text,
        metadata=_build_metadata(source, page),
        chunk_size=800,
        chunk_overlap=80,
    )

    count = ingest_text(request)
    print(f"[OK] Indexed {count} chunks for {doc_id}")
    return count


def ingest_all_sources(dry_run: bool = False, force: bool = False) -> None:
    """Ingest all configured legal sources."""
    total_chunks = 0
    skipped = 0
    failed = 0

    for src in LEGAL_SOURCES:
        result = ingest_single_source(src, dry_run=dry_run, force=force)
        if result is None:
            failed += 1
        elif result == 0 and not dry_run:
            skipped += 1
        else:
            total_chunks += result

    print(
        f"\n=== Done. Total chunks indexed: {total_chunks} | "
        f"Skipped: {skipped} | Failed: {failed} ==="
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    ingest_all_sources(dry_run=False, force=False)
