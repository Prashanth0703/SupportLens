from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, tickets

app = FastAPI(
    title="SupportLens API",
    version="0.1.0",
    description="Backend API for the SupportLens portfolio project.",
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
