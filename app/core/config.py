from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI Settings
    openai_api_key: str
    embedding_model_name: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    chat_model_name: str = "gpt-4o-mini"

    # Qdrant Settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_legal: str = "legal-knowledge"
    qdrant_collection_user_docs: str = "user-documents"

    # Database Settings (Optional - for future user auth)
    DATABASE_URL: str = "postgresql+psycopg2://shaastra:shaastra_password@localhost:5432/shaastra_db"

    # JWT Settings (Optional - for future user auth)
    JWT_SECRET: str = "super-secret-change-me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore"
        # Make .env file optional - won't crash if missing or unreadable
        env_ignore_empty = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()