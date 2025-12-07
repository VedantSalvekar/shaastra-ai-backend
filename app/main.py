from fastapi import FastAPI
from app.api import health, rag

app=FastAPI(
    title="My API",
    description="This is a sample API built with FastAPI.",
    version="1.0.0"
)

app.include_router(health.router)
app.include_router(rag.router)