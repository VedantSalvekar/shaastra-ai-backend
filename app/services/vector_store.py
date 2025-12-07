from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant as QdrantVectorStore

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