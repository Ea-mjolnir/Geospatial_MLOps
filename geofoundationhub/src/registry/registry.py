"""
GeoFoundationHub — In-Memory Model Registry
=============================================
Stores and manages registered foundation models.
In production this would be backed by PostgreSQL.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional
from src.registry.models import (
    ModelMetadata, ModelRegistration,
    ModelUpdateRequest, ModelStatus,
    SensorType, TaskType,
)

log = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for geospatial foundation models.
    Supports CRUD + filtering by sensor type, task, status.
    """

    def __init__(self):
        self._models: Dict[str, ModelMetadata] = {}
        self._seed_default_models()

    def _seed_default_models(self):
        """Pre-register known foundation models."""

        # Prithvi-EO-2.0-300M
        self.register(ModelRegistration(
            name="Prithvi-EO-2.0-300M",
            version="2.0.0",
            description="IBM/NASA geospatial foundation model trained on HLS data",
            architecture="ViT-L",
            parameters_m=300.0,
            sensor_types=[SensorType.OPTICAL, SensorType.MULTISPECTRAL],
            task_types=[
                TaskType.SEGMENTATION,
                TaskType.CHANGE_DETECTION,
                TaskType.EMBEDDING,
            ],
            input_channels=6,
            input_size=224,
            embedding_dim=1024,
            hf_model_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
            paper_url="https://arxiv.org/abs/2310.18660",
            tags={"source": "HLS", "org": "IBM/NASA"},
            metrics={"linear_probe_acc": 0.847},
        ))

        # SatMAE
        self.register(ModelRegistration(
            name="SatMAE",
            version="1.0.0",
            description="Masked Autoencoder for satellite imagery (Stanford)",
            architecture="ViT-L",
            parameters_m=307.0,
            sensor_types=[SensorType.MULTISPECTRAL],
            task_types=[
                TaskType.CLASSIFICATION,
                TaskType.SEGMENTATION,
                TaskType.EMBEDDING,
            ],
            input_channels=3,
            input_size=224,
            embedding_dim=1024,
            hf_model_id="stanfordMMLab/SatMAE",
            paper_url="https://arxiv.org/abs/2207.08051",
            tags={"source": "fMoW", "org": "Stanford"},
            metrics={"fmow_acc": 0.782},
        ))

        # RemoteCLIP
        self.register(ModelRegistration(
            name="RemoteCLIP",
            version="1.0.0",
            description="CLIP model fine-tuned on remote sensing data",
            architecture="ViT-B/32",
            parameters_m=151.0,
            sensor_types=[SensorType.OPTICAL],
            task_types=[
                TaskType.CLASSIFICATION,
                TaskType.EMBEDDING,
            ],
            input_channels=3,
            input_size=224,
            embedding_dim=512,
            hf_model_id="CLIP-RS/RemoteCLIP-ViT-B-32",
            paper_url="https://arxiv.org/abs/2306.11029",
            tags={"source": "RS5M", "org": "CLIP-RS"},
            metrics={"zero_shot_acc": 0.731},
        ))

        log.info(f"✅ Registry seeded with {len(self._models)} models")

    def register(self, req: ModelRegistration) -> ModelMetadata:
        """Register a new model."""
        model_id = str(uuid.uuid4())[:8]
        model    = ModelMetadata(
            model_id=model_id,
            **req.dict(),
        )
        self._models[model_id] = model
        log.info(f"Registered: {model.name} v{model.version} [{model_id}]")
        return model

    def get(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model by ID."""
        return self._models.get(model_id)

    def get_by_name(self, name: str, version: Optional[str] = None) -> Optional[ModelMetadata]:
        """Get model by name and optional version."""
        for m in self._models.values():
            if m.name == name:
                if version is None or m.version == version:
                    return m
        return None

    def list(
        self,
        status:      Optional[ModelStatus] = None,
        sensor_type: Optional[SensorType] = None,
        task_type:   Optional[TaskType] = None,
    ) -> List[ModelMetadata]:
        """List models with optional filters."""
        models = list(self._models.values())
        if status:
            models = [m for m in models if m.status == status]
        if sensor_type:
            models = [m for m in models if sensor_type in m.sensor_types]
        if task_type:
            models = [m for m in models if task_type in m.task_types]
        return models

    def update(self, model_id: str, req: ModelUpdateRequest) -> Optional[ModelMetadata]:
        """Update model metadata."""
        model = self._models.get(model_id)
        if not model:
            return None
        data = model.dict()
        if req.status is not None:
            data['status'] = req.status
        if req.canary_weight is not None:
            data['canary_weight'] = req.canary_weight
        if req.metrics is not None:
            data['metrics'].update(req.metrics)
        if req.tags is not None:
            data['tags'].update(req.tags)
        data['updated_at'] = datetime.utcnow()
        self._models[model_id] = ModelMetadata(**data)
        return self._models[model_id]

    def delete(self, model_id: str) -> bool:
        """Remove model from registry."""
        if model_id in self._models:
            del self._models[model_id]
            return True
        return False

    def get_active_models(self) -> List[ModelMetadata]:
        """Get all active + canary models for routing."""
        return [
            m for m in self._models.values()
            if m.status in [ModelStatus.ACTIVE, ModelStatus.CANARY]
        ]

    def get_shadow_models(self) -> List[ModelMetadata]:
        """Get all shadow models."""
        return [
            m for m in self._models.values()
            if m.status == ModelStatus.SHADOW
        ]

    def summary(self) -> Dict:
        """Registry summary statistics."""
        models = list(self._models.values())
        return {
            "total":      len(models),
            "active":     sum(1 for m in models if m.status == ModelStatus.ACTIVE),
            "shadow":     sum(1 for m in models if m.status == ModelStatus.SHADOW),
            "canary":     sum(1 for m in models if m.status == ModelStatus.CANARY),
            "inactive":   sum(1 for m in models if m.status == ModelStatus.INACTIVE),
            "models":     [f"{m.name} v{m.version} [{m.status}]" for m in models],
        }


# Global singleton
_registry = None

def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
