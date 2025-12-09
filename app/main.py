from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import health, rag, user_docs

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"

]


app=FastAPI(
    title="My API",
    description="This is a sample API built with FastAPI.",
    version="1.0.0"
)
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