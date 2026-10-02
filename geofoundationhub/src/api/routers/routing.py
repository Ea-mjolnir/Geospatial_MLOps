"""Routing configuration router."""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from src.routing.router import get_router

router = APIRouter()

class ABConfig(BaseModel):
    champion_id:   str
    challenger_id: str
    split:         float = 0.5

class ShadowConfig(BaseModel):
    champion_id:   str
    challenger_id: str

class CanaryConfig(BaseModel):
    baseline_id: str
    canary_id:   str
    weight:      float = 0.05

class TenantRoute(BaseModel):
    tenant_id: str
    model_id:  str

@router.get("/stats")
def routing_stats():
    return get_router().stats()

@router.post("/ab")
def configure_ab(config: ABConfig):
    get_router().configure_ab(
        config.champion_id, config.challenger_id, config.split,
    )
    return {"status": "A/B configured", "config": config}

@router.post("/shadow")
def configure_shadow(config: ShadowConfig):
    get_router().configure_shadow(
        config.champion_id, config.challenger_id,
    )
    return {"status": "Shadow configured", "config": config}

@router.post("/canary")
def configure_canary(config: CanaryConfig):
    get_router().configure_canary(
        config.baseline_id, config.canary_id, config.weight,
    )
    return {"status": "Canary configured", "config": config}

@router.post("/tenant")
def add_tenant_route(route: TenantRoute):
    get_router().add_tenant_route(route.tenant_id, route.model_id)
    return {"status": "Tenant route added", "route": route}
