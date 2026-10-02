"""Model registry router."""
from typing import Optional
from fastapi import APIRouter, HTTPException
from src.registry.registry import get_registry
from src.registry.models import (
    ModelRegistration, ModelUpdateRequest,
    ModelStatus, SensorType, TaskType,
)

router = APIRouter()

@router.get("/")
def list_models(
    status:      Optional[str] = None,
    sensor_type: Optional[str] = None,
    task_type:   Optional[str] = None,
):
    reg = get_registry()
    return reg.list(
        status=ModelStatus(status) if status else None,
        sensor_type=SensorType(sensor_type) if sensor_type else None,
        task_type=TaskType(task_type) if task_type else None,
    )

@router.get("/summary")
def registry_summary():
    return get_registry().summary()

@router.get("/{model_id}")
def get_model(model_id: str):
    model = get_registry().get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return model

@router.post("/", status_code=201)
def register_model(req: ModelRegistration):
    return get_registry().register(req)

@router.patch("/{model_id}")
def update_model(model_id: str, req: ModelUpdateRequest):
    model = get_registry().update(model_id, req)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return model
