import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from routes import ingestion
from repository.connection import check_db, close_db, init_db

load_dotenv()
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await check_db()
    yield
    await close_db()


app = FastAPI(
    title="GraphMind",
    description="Advanced RAG ingestion pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingestion.router)
