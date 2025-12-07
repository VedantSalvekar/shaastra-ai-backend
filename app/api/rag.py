from fastapi import APIRouter, HTTPException, status
from app.schemas.rag import IndexRequest, SearchRequest, SearchResponse, IngestTextRequest, AnswerRequest, AnswerResponse
from app.services import vector_store
from app.services.ingestion import ingest_text
from app.services.rag_answer import generate_rag_answer

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/index", status_code=status.HTTP_201_CREATED)
def index_text(request: IndexRequest) -> dict: 
    """
    Index text chunks into the Qdrant collection.
    """
    try:
        count = vector_store.index_chunks(request.collection, request.chunks)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index chunks: {str(e)}"
        )
    return {"indexed": count}

@router.post("/ingest-text", status_code=status.HTTP_201_CREATED)
def ingest_raw_text(request: IngestTextRequest) -> dict:
    try:
        count = ingest_text(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest text: {str(e)}"
        )
    return {"indexed": count}

@router.post("/search", response_model=SearchResponse)
def semantic_search(request: SearchRequest) -> SearchResponse:
    """
    Perform semantic search over the specified Qdrant collection.
    """
    try:
        return vector_store.search(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )
    return response

@router.post("/answer", response_model=AnswerResponse)
def rag_answer(request: AnswerRequest) -> AnswerResponse:

    try:
        return generate_rag_answer(request)
    except HTTPException:
        raise  
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {str(e)}"
        )