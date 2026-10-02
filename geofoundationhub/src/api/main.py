"""
GeoFoundationHub — FastAPI Gateway
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import health as health_router
from src.api.routers import registry as registry_router
from src.api.routers import inference as inference_router
from src.api.routers import routing as routing_router
from src.api.routers import tenants as tenants_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("GeoFoundationHub starting...")
    from src.registry.registry import get_registry
    from src.auth.auth import get_tenant_manager
    from src.serving.model_loader import load_all_models
    import threading

    registry = get_registry()
    tm       = get_tenant_manager()
    log.info(f"Registry: {registry.summary()['total']} models")
    log.info(f"Tenants:  {tm.summary()['total_tenants']}")

    # Load models in background — server starts immediately
    def load_models_bg():
        adapters = load_all_models(registry=registry)
        log.info(f"Loaded:   {len(adapters)} model adapters")

    t = threading.Thread(target=load_models_bg, daemon=True)
    t.start()
    log.info("Server ready — models loading in background...")
    yield
    log.info("GeoFoundationHub shutting down...")


app = FastAPI(
    title="GeoFoundationHub",
    description="Universal Geospatial Foundation Model Registry and Serving Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router.router,    tags=["Health"])
app.include_router(registry_router.router,  prefix="/registry", tags=["Registry"])
app.include_router(inference_router.router, prefix="/infer",    tags=["Inference"])
app.include_router(routing_router.router,   prefix="/routing",  tags=["Routing"])
app.include_router(tenants_router.router,   prefix="/tenants",  tags=["Tenants"])


@app.get("/")
def root():
    return {
        "name":     "GeoFoundationHub",
        "version":  "1.0.0",
        "models":   ["Prithvi-EO-2.0-300M", "SatMAE", "RemoteCLIP"],
        "patterns": ["A/B Testing", "Shadow Deployment", "Canary Rollout", "Multi-tenant"],
        "docs":     "/docs",
    }
