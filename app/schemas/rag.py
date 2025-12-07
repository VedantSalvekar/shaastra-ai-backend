from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

CollectionName = Literal["legal-knowledge", "user-documents"]

class TextChunkIn(BaseModel):
    text: str = Field(..., description="The raw text content to index")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (e.g. source_url, topic, user_id, doc_type)",
    )

class IndexRequest(BaseModel):
    collection: CollectionName
    chunks: List[TextChunkIn]

class SearchRequest(BaseModel):
    
    collection: CollectionName
    query: str
    top_k: int = Field(5, ge=1, le=50)
    filter: Optional[Dict[str, Any]] = None


class SearchResultItem(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResultItem]

class IngestTextRequest(BaseModel):
    collection: CollectionName
    text: str = Field(..., description="The raw text content to ingest")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (e.g. source_url, topic, user_id, doc_type)",
    )
    chunk_size: int = Field(500, ge=100, le=5000, description="Size of each text chunk")
    chunk_overlap: int = Field(50, ge=0, le=1000, description="Overlap between text chunks")

class AnswerSource(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float

class AnswerRequest(BaseModel):
    collection: CollectionName
    question: str = Field(..., description="The user's question to answer")
    top_k: int = Field(5, ge=1, le=20, description="Number of relevant chunks to retrieve")
    filter: Optional[Dict[str, Any]] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[AnswerSource]
