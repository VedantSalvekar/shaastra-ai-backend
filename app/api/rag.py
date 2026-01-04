from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session
from uuid import UUID
from typing import Optional

from app.schemas.rag import IndexRequest, SearchRequest, SearchResponse, IngestTextRequest, AnswerRequest, AnswerResponse
from app.services import vector_store
from app.services.ingestion import ingest_text
from app.services.rag_answer import generate_rag_answer
from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.models.chat_session import ChatSession
from app.models.chat_message import ChatMessage, ChatRole

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
def rag_answer(
    request: AnswerRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnswerResponse:
    try:
        response = generate_rag_answer(request)
        
        # Save to database if user is authenticated
        try:
            if request.session_id:
                session = db.query(ChatSession).filter(
                    ChatSession.id == request.session_id,
                    ChatSession.user_id == current_user.id
                ).first()
                if not session:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Chat session not found"
                    )
            else:
                session = ChatSession(
                    user_id=current_user.id,
                    title=request.question[:50] + "..." if len(request.question) > 50 else request.question
                )
                db.add(session)
                db.flush()
            
            user_msg = ChatMessage(
                session_id=session.id,
                role=ChatRole.user,
                content=request.question,
            )
            db.add(user_msg)
            
            assistant_msg = ChatMessage(
                session_id=session.id,
                role=ChatRole.assistant,
                content=response.answer,
                citations={"sources": [s.model_dump() for s in response.sources]} if response.sources else None,
            )
            db.add(assistant_msg)
            
            db.commit()
            print(f"[OK] Saved chat messages to session {session.id}")
        except HTTPException:
            raise
        except Exception as e:
            print(f"[WARN] Failed to save chat to database: {e}")
            db.rollback()
        
        return response
    except HTTPException:
        raise  
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {str(e)}"
        )