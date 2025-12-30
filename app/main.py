from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health, rag, user_docs
from app.db.session import engine
from app.db.base import Base
from app.api.routes import auth

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"

]


app=FastAPI(
    title="My API",
    description="This is a sample API built with FastAPI.",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)

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
app.include_router(auth.router, prefix="/auth", tags=["auth"])