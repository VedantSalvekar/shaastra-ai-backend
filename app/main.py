from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health, rag, user_docs

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"

]


app=FastAPI(
    title="Shaastra AI",
    description="RAG-based AI assistant with legal knowledge and user document management.",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup() -> None:
    try:
        from app.db.session import engine
        from app.db.base import Base
        Base.metadata.create_all(bind=engine)
        print("[INFO] Database initialized successfully")
    except ImportError as e:
        print(f"[INFO] Database not configured (missing psycopg2). Skipping DB initialization.")
    except Exception as e:
        print(f"[WARN] Database initialization failed: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(rag.router)
app.include_router(user_docs.router)

try:
    from app.api.routes import auth, documents, chats
    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(documents.router, tags=["documents"])
    app.include_router(chats.router, tags=["chats"])
    print("[INFO] Auth routes enabled")
except ImportError as e:
    print(f"[ERROR] Auth routes disabled - Import failed: {e}")
except Exception as e:
    print(f"[ERROR] Auth routes failed to load: {e}")