from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.core.config import get_settings
from app.core.qdrant_client import get_qdrant_client, ensure_collection
from app.schemas.rag import (CollectionName, TextChunkIn, IndexRequest, SearchRequest, SearchResultItem, SearchResponse)
from app.schemas.langgraph_state import RetrievalChunk

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


def search_with_filters(
    collection: CollectionName,
    query: str,
    top_k: int = 5,
    user_id: Optional[str] = None,
    doc_id: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> List[RetrievalChunk]:
    """
    Search with metadata filtering - CRITICAL for user isolation and security.
    
    This function is similar to the regular search() function, but it adds metadata filters
    to ensure users can ONLY see their own documents.
    
    How it works:
    1. Build a filter based on the parameters (user_id, doc_id, doc_type)
    2. Execute the search with these filters applied
    3. Return results as RetrievalChunk objects
    
    SECURITY NOTE: When searching user-documents collection, user_id filter is MANDATORY.
    This ensures users can never see other users' documents, even if the AI tries to access them.
    
    Args:
        collection: Which collection to search ("legal-knowledge" or "user-documents")
        query: The search query text
        top_k: How many results to return (default: 5)
        user_id: Filter by user_id (REQUIRED for user-documents collection)
        doc_id: Optional filter by specific document ID
        doc_type: Optional filter by document type (e.g., "revenue_letter", "bank_statement")
    
    Returns:
        List of RetrievalChunk objects containing text, score, and metadata
        
    Example:
        # Search user's documents (user_id is MANDATORY)
        results = search_with_filters(
            collection="user-documents",
            query="immigration stamp",
            top_k=5,
            user_id="user_123"  # This filter is enforced in code, not by the LLM
        )
        
        # Search legal knowledge (no user_id needed, it's public)
        results = search_with_filters(
            collection="legal-knowledge",
            query="Stamp 4 work hours",
            top_k=5
        )
    """
    settings = get_settings()
    qdrant_collection = _map_collection_name(collection)
    
    # ========== SECURITY CHECK ==========
    # If searching user-documents, user_id MUST be provided
    if collection == "user-documents" and not user_id:
        raise ValueError(
            "user_id is REQUIRED when searching user-documents collection. "
            "This is a security measure to prevent users from seeing other users' documents."
        )
    
    # ========== BUILD FILTER ==========
    # Build a list of filter conditions based on what was provided
    filter_conditions = []
    
    if user_id:
        # Filter by user_id (stored in metadata.user_id)
        filter_conditions.append(
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value=user_id)
            )
        )
        print(f"[INFO] Applying user_id filter: {user_id}")
    
    if doc_id:
        # Filter by specific document ID
        filter_conditions.append(
            FieldCondition(
                key="metadata.doc_id",
                match=MatchValue(value=doc_id)
            )
        )
        print(f"[INFO] Applying doc_id filter: {doc_id}")
    
    if doc_type:
        # Filter by document type (e.g., "revenue_letter")
        filter_conditions.append(
            FieldCondition(
                key="metadata.doc_type",
                match=MatchValue(value=doc_type)
            )
        )
        print(f"[INFO] Applying doc_type filter: {doc_type}")
    
    # Create the filter object (only if we have conditions)
    qdrant_filter = None
    if filter_conditions:
        qdrant_filter = Filter(must=filter_conditions)
    
    # ========== EXECUTE SEARCH ==========
    embeddings = _get_embeddings()
    client = get_qdrant_client()
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=qdrant_collection,
        embeddings=embeddings,
    )
    
    # Perform the search with filters
    # Note: LangChain's Qdrant wrapper doesn't directly support filters in similarity_search_with_score,
    # so we need to use the underlying client
    if qdrant_filter:
        # Use the raw Qdrant client for filtered search
        query_vector = embeddings.embed_query(query)
        search_result = client.search(
            collection_name=qdrant_collection,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
        )
        
        # Convert to our format
        results = []
        for scored_point in search_result:
            # Extract metadata (Qdrant stores it under 'payload')
            metadata = scored_point.payload.get("metadata", {}) if scored_point.payload else {}
            text = scored_point.payload.get("page_content", "") if scored_point.payload else ""
            
            results.append(
                RetrievalChunk(
                    text=text,
                    score=float(scored_point.score),
                    metadata=metadata
                )
            )
    else:
        # No filters, use the regular LangChain method
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_k)
        results = []
        for doc, score in docs_and_scores:
            results.append(
                RetrievalChunk(
                    text=doc.page_content,
                    score=float(score),
                    metadata=doc.metadata or {}
                )
            )
    
    print(f"[INFO] Search returned {len(results)} results from {qdrant_collection}")
    return results
