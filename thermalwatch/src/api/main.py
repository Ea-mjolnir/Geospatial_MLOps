"""
ThermalWatch FastAPI — Inference Server
========================================
Endpoints:
  GET  /health              → health check
  POST /predict/wildfire    → wildfire risk prediction
  POST /predict/solar       → solar farm health prediction
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routers import wildfire, solar, health

app = FastAPI(
    title="ThermalWatch API",
    description="Wildfire risk + Solar farm health monitoring via thermal imagery",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(wildfire.router, prefix="/predict", tags=["Wildfire"])
app.include_router(solar.router, prefix="/predict", tags=["Solar"])


@app.get("/")
def root():
    return {
        "name": "ThermalWatch API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/predict/wildfire",
            "/predict/solar",
        ],
    }
