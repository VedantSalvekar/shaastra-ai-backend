# app/ingestion/legal_ingest.py
from typing import Optional

from app.ingestion.legal_sources import LEGAL_SOURCES, LegalSource
from app.ingestion.html_loader import load_and_clean, HtmlLoadError
from app.schemas.rag import IngestTextRequest
from app.services.ingestion import ingest_text


def ingest_single_source(source: LegalSource, dry_run: bool = False) -> Optional[int]:
    """
    Ingest a single legal source URL into the 'legal' collection.

    Returns number of chunks indexed, or None if failed.
    """
    print(f"\n=== Ingesting: {source.url} ({source.provider} / {source.topic} / {source.subtopic}) ===")

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

    request = IngestTextRequest(
        collection="legal-knowledge",
        text=text,
        metadata={
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


def ingest_all_sources(dry_run: bool = False) -> None:
    """
    Ingest all configured legal sources.
    """
    total_chunks = 0
    for src in LEGAL_SOURCES:
        result = ingest_single_source(src, dry_run=dry_run)
        if isinstance(result, int):
            total_chunks += result

    print(f"\n=== Done. Total chunks indexed: {total_chunks} ===")


if __name__ == "__main__":
    # Change dry_run to True if you just want to see text preview.
    ingest_all_sources(dry_run=False)
