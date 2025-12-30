from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.core.config import get_settings
settings = get_settings()

engine =create_engine(
    settings.DATABASE_URL,
    future = True,
    echo = False,
)

SessionLocal = sessionmaker(
    autocommit = False,
    autoflush = False,
    bind = engine,
    future = True,
)

Base = declarative_base()

