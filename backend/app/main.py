from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, tickets
from app.config import get_model_path
from app.services.ml_predictor import IntentPredictor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if getattr(app.state, "intent_predictor", None) is None:
        model_path = get_model_path()
        try:
            app.state.intent_predictor = IntentPredictor.from_joblib(
                model_path
            )
            app.state.model_error = ""
        except (FileNotFoundError, TypeError, ValueError) as error:
            app.state.intent_predictor = None
            app.state.model_error = str(error)
    yield


app = FastAPI(
    title="SupportLens API",
    version="0.2.0",
    description="Customer-support intent classification and triage API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(tickets.router, prefix="/api/v1")
