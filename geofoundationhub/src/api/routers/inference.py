"""Inference router."""
import uuid
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from src.routing.router import get_router
from src.registry.registry import get_registry
from src.auth.auth import get_tenant_manager
from models.base import InferenceInput

router = APIRouter()

class InferRequest(BaseModel):
    image:       List[float]
    channels:    int = 6
    height:      int = 224
    width:       int = 224
    sensor_type: str = "optical"
    patch_id:    Optional[str] = None
    model_id:    Optional[str] = None

class CompareRequest(BaseModel):
    image:     List[float]
    channels:  int = 6
    height:    int = 224
    width:     int = 224
    model_ids: List[str]
    patch_id:  Optional[str] = None

@router.post("/")
def infer(
    req:       InferRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    tenant_id = None
    if x_api_key:
        tenant = get_tenant_manager().get_by_key(x_api_key)
        if tenant:
            tenant_id = tenant.tenant_id

    inputs = InferenceInput(
        image=req.image,
        channels=req.channels,
        height=req.height,
        width=req.width,
        sensor_type=req.sensor_type,
        patch_id=req.patch_id or str(uuid.uuid4())[:8],
    )

    try:
        output, decision = get_router().route(
            inputs=inputs,
            tenant_id=tenant_id,
            request_id=str(uuid.uuid4())[:8],
        )
        return {
            "output":  output,
            "routing": {
                "pattern":   decision.pattern,
                "model_id":  decision.champion_id,
                "tenant_id": decision.tenant_id,
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

@router.post("/compare")
def compare_models(req: CompareRequest):
    router_ = get_router()
    results = {}
    inputs  = InferenceInput(
        image=req.image,
        channels=req.channels,
        height=req.height,
        width=req.width,
        patch_id=req.patch_id or str(uuid.uuid4())[:8],
    )
    for model_id in req.model_ids:
        if model_id not in router_._adapters:
            results[model_id] = {"error": f"Model {model_id} not loaded"}
            continue
        try:
            output = router_._run_inference(model_id, inputs)
            results[model_id] = {
                "latency_ms":    output.latency_ms,
                "embedding_dim": len(output.embedding or []),
                "embedding":     output.embedding,
                "model_name":    output.model_name,
                "version":       output.model_version,
            }
        except Exception as e:
            results[model_id] = {"error": str(e)}
    return {"patch_id": inputs.patch_id, "results": results}
