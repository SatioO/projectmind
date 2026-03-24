import logging
from dotenv import load_dotenv
from fastapi import FastAPI

from routes import ingestion
from repository.connection import check_db, close_db

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="GraphMind",
    description="Advanced RAG ingestion pipeline",
    version="0.1.0",
)


@app.on_event("startup")
async def startup():
    await check_db()


@app.on_event("shutdown")
async def shutdown():
    await close_db()


app.include_router(ingestion.router)
