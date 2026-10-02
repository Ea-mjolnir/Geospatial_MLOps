"""Health check router."""
from datetime import datetime
from fastapi import APIRouter
from src.registry.registry import get_registry
from src.routing.router import get_router
from src.auth.auth import get_tenant_manager

router = APIRouter()

@router.get("/health")
def health():
    registry = get_registry()
    router_  = get_router()
    tm       = get_tenant_manager()
    return {
        "status":    "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "registry":  registry.summary(),
        "routing":   router_.stats(),
        "tenants":   tm.summary()["total_tenants"],
    }
