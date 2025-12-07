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
    settings = get_settings()
    client = get_qdrant_client()

    collections = client.get_collections().collections
    existing = {c.name for c in collections}

    if collection_name not in existing:
        return 
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
    except UnexpectedResponse as e:
        
        if "already exists" in str(e):
            return
        raise 