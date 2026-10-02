"""Tenant management router."""
from fastapi import APIRouter, HTTPException
from src.auth.auth import get_tenant_manager

router = APIRouter()

@router.get("/")
def list_tenants():
    return get_tenant_manager().summary()

@router.post("/", status_code=201)
def create_tenant(tenant_id: str, name: str):
    return get_tenant_manager().create_api_key(tenant_id, name)

@router.post("/{tenant_id}/model")
def assign_model(tenant_id: str, model_id: str):
    success = get_tenant_manager().set_model(tenant_id, model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")
    return {"tenant_id": tenant_id, "model_id": model_id}
