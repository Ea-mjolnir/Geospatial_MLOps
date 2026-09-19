"""Health check router."""
import torch
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status":    "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "device":    "cuda" if torch.cuda.is_available() else "cpu",
        "gpu":       torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
