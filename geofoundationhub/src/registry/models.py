"""
GeoFoundationHub — Model Registry
===================================
Defines the data models for registered
geospatial foundation models.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ModelStatus(str, Enum):
    ACTIVE    = "active"
    SHADOW    = "shadow"
    CANARY    = "canary"
    INACTIVE  = "inactive"
    DEPRECATED = "deprecated"


class SensorType(str, Enum):
    OPTICAL      = "optical"       # Sentinel-2, Landsat
    SAR          = "sar"           # Sentinel-1
    THERMAL      = "thermal"       # Landsat thermal
    MULTISPECTRAL = "multispectral"
    HYPERSPECTRAL = "hyperspectral"
    MULTIMODAL   = "multimodal"    # multiple sensors


class TaskType(str, Enum):
    CLASSIFICATION     = "classification"
    SEGMENTATION       = "segmentation"
    CHANGE_DETECTION   = "change_detection"
    OBJECT_DETECTION   = "object_detection"
    EMBEDDING          = "embedding"
    REGRESSION         = "regression"


class ModelMetadata(BaseModel):
    """Metadata for a registered foundation model."""
    model_id:       str  = Field(..., description="Unique model identifier")
    name:           str  = Field(..., description="Human readable name")
    version:        str  = Field(..., description="Model version e.g. v1.0.0")
    description:    str  = Field(..., description="Model description")
    architecture:   str  = Field(..., description="e.g. ViT-L, ResNet-50")
    parameters_m:   float = Field(..., description="Parameters in millions")
    sensor_types:   List[SensorType] = Field(..., description="Supported sensor types")
    task_types:     List[TaskType]   = Field(..., description="Supported tasks")
    input_channels: int  = Field(..., description="Number of input channels")
    input_size:     int  = Field(224, description="Input image size")
    embedding_dim:  int  = Field(..., description="Output embedding dimension")
    status:         ModelStatus = Field(default=ModelStatus.ACTIVE)
    canary_weight:  float = Field(default=0.0, description="Traffic % for canary [0-1]")
    checkpoint_path: Optional[str] = Field(default=None)
    hf_model_id:    Optional[str] = Field(default=None, description="HuggingFace model ID")
    paper_url:      Optional[str] = Field(default=None)
    tags:           Dict[str, str] = Field(default_factory=dict)
    metrics:        Dict[str, float] = Field(default_factory=dict)
    registered_at:  datetime = Field(default_factory=datetime.utcnow)
    updated_at:     datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True


class ModelRegistration(BaseModel):
    """Request to register a new model."""
    name:           str
    version:        str
    description:    str
    architecture:   str
    parameters_m:   float
    sensor_types:   List[SensorType]
    task_types:     List[TaskType]
    input_channels: int
    input_size:     int = 224
    embedding_dim:  int
    checkpoint_path: Optional[str] = None
    hf_model_id:    Optional[str] = None
    paper_url:      Optional[str] = None
    tags:           Dict[str, str] = {}
    metrics:        Dict[str, float] = {}


class ModelUpdateRequest(BaseModel):
    """Request to update model status or metadata."""
    status:         Optional[ModelStatus] = None
    canary_weight:  Optional[float] = None
    metrics:        Optional[Dict[str, float]] = None
    tags:           Optional[Dict[str, str]] = None
