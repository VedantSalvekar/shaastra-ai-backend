from langfuse import Langfuse, get_client

from app.core.config import settings

_initialized = False


def init_langfuse() -> None:
    global _initialized
    if _initialized:
        return
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return
    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    _initialized = True


def is_langfuse_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def flush_langfuse() -> None:
    if not is_langfuse_enabled():
        return
    get_client().flush()
