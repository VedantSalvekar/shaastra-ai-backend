from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    embedding_model_name: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    chat_model_name: str = "gpt-4o-mini"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None

    qdrant_collection_legal: str = "legal-knowledge"
    qdrant_collection_user_docs: str = "user-documents"

    class Config:
        env_file = ".env" 
        extra = "ignore"

@lru_cache()
def get_settings() -> Settings:
    return Settings()