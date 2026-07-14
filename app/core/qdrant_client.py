from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse 
from app.core.config import get_settings

def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )

def ensure_collection(collection_name: str) -> None:
    """Create the collection with the correct vector config if it doesn't exist."""
    settings = get_settings()
    client = get_qdrant_client()

    collections = client.get_collections().collections
    existing = {c.name for c in collections}

    if collection_name in existing:
        return

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        print(f"[INFO] Created Qdrant collection: {collection_name}")
    except UnexpectedResponse as e:
        # Another request may have created it concurrently.
        if "already exists" in str(e):
            return
        raise