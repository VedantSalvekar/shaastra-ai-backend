# app/services/ingestion.py
from typing import List

try:  # langchain>=0.2
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - fallback for older langchain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.schemas.rag import (
    IngestTextRequest,
    TextChunkIn,
)
from app.services.vector_store import index_chunks


def ingest_text(request: IngestTextRequest) -> int:
    """
    Ingest a large text blob into Qdrant by:
      - splitting into overlapping chunks
      - attaching shared metadata to each chunk
      - delegating to the vector_store.index_chunks service

    Returns the number of chunks actually indexed.
    """
    # 1) Set up a text splitter tuned for legal-style content
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ],
    )

    # 2) Split text into chunks
    raw_chunks: list[str] = splitter.split_text(request.text)

    # 3) Wrap into TextChunkIn with shared metadata
    chunks: List[TextChunkIn] = [
        TextChunkIn(text=chunk, metadata=dict(request.metadata))
        for chunk in raw_chunks
        if chunk.strip()
    ]

    if not chunks:
        return 0

    # 4) Delegate to vector store
    count = index_chunks(request.collection, chunks)
    return count
