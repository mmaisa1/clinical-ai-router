# app/main.py
"""
Application entry point. 
Wires together the pipeline, routers, and startup/shutdown lifecycle.

Run locally:
    uvicorn app.main:app --reload

Docs available at /docs once running.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import APP_NAME, APP_VERSION
from app.pipeline.pipeline import ClassificationPipeline
from app.routers import public, admin


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load all models once, attach to app.state
    print("Loading classification pipeline...")
    app.state.pipeline = ClassificationPipeline()
    print("Pipeline loaded. API ready.")

    yield

    # Shutdown: nothing to clean up yet (no DB connections open currently)
    print("Shutting down.")


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan,
)

app.include_router(public.router)
app.include_router(admin.router)