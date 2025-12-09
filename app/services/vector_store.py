from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.core.config import get_settings
from app.core.qdrant_client import get_qdrant_client, ensure_collection
from app.schemas.rag import (CollectionName, TextChunkIn, IndexRequest, SearchRequest, SearchResultItem, SearchResponse)

def _map_collection_name(collection: CollectionName) -> str:
    settings = get_settings()
    if collection == "legal-knowledge":
        return settings.qdrant_collection_legal
    elif collection == "user-documents":
        return settings.qdrant_collection_user_docs
    else:
        raise ValueError(f"Unknown collection name: {collection}")
    
def _get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model_name,
        openai_api_key=settings.openai_api_key,
    )

def index_chunks(collection: CollectionName, chunks: List[TextChunkIn]) -> int:

    if not chunks: 
        return 0
    
    settings = get_settings()
    qdrant_collection = _map_collection_name(collection)

    ensure_collection(qdrant_collection)

    texts = [c.text for c in chunks]
    metadatas: List[Dict[str, Any]] = [c.metadata for c in chunks]
    embeddings = _get_embeddings()

    _=QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=qdrant_collection,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )

    return len(chunks)

def search(request: SearchRequest) -> SearchResponse:
    """
    Perform semantic search over the given collection using LangChain's Qdrant wrapper.
    """
    settings = get_settings()
    qdrant_collection = _map_collection_name(request.collection)

    embeddings = _get_embeddings()
    client = get_qdrant_client()

    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=qdrant_collection,
        embeddings=embeddings,
    )

    
    docs_and_scores = vectorstore.similarity_search_with_score(
        request.query,
        k=request.top_k,
    )

    results: List[SearchResultItem] = []
    for doc, score in docs_and_scores:
        results.append(
            SearchResultItem(
                text=doc.page_content,
                score=float(score),
                metadata=doc.metadata or {},
            )
        )

    return SearchResponse(results=results)


def doc_id_exists(collection: CollectionName, doc_id: str) -> bool:
    """
    Check if a document with the given doc_id already exists in the collection.
    
    Returns True if at least one point with this doc_id exists, False otherwise.
    """
    try:
        settings = get_settings()
        client = get_qdrant_client()

     
        qdrant_collection = _map_collection_name(collection)

   
        flt = Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        )

        # Use scroll to check if any points exist with this doc_id
        # We only need to check if at least one exists, so limit=1
        result = client.scroll(
            collection_name=qdrant_collection,
            scroll_filter=flt,
            limit=1,
        )
        
        # result is a tuple of (points, next_page_offset)
        points = result[0] if result else []
        return len(points) > 0
    except Exception as e:
      
        print(f"[DEBUG] doc_id_exists check failed: {e}")
        return False


def delete_by_doc_id(collection: CollectionName, doc_id: str) -> None:
    """
    Delete all points in a collection that belong to a given logical document.

    We identify a document by `doc_id` stored in the payload (metadata).
    For our legal ingestion, doc_id = source URL.
    """
    settings = get_settings()
    client = get_qdrant_client()

    qdrant_collection = _map_collection_name(collection)


    flt = Filter(
        must=[
            FieldCondition(
                key="metadata.doc_id",
                match=MatchValue(value=doc_id),
            )
        ]
    )

    client.delete(
        collection_name=qdrant_collection,
        points_selector=flt,
    )
