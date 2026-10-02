"""
GeoFoundationHub — Base Model Adapter
=======================================
All geospatial foundation models must
implement this interface.

This ensures a unified API regardless
of the underlying model architecture.
"""

import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

log = logging.getLogger(__name__)


class InferenceInput(BaseModel):
    """Standard input for all GeoAI models."""
    image:        List[float]     # flattened image array
    channels:     int             # number of channels
    height:       int = 224
    width:        int = 224
    sensor_type:  str = "optical"
    metadata:     Dict[str, Any] = {}
    patch_id:     Optional[str] = None


class InferenceOutput(BaseModel):
    """Standard output from all GeoAI models."""
    model_id:     str
    model_name:   str
    model_version: str
    embedding:    Optional[List[float]] = None
    predictions:  Dict[str, Any] = {}
    latency_ms:   float = 0.0
    patch_id:     Optional[str] = None
    metadata:     Dict[str, Any] = {}


class BaseModelAdapter(ABC):
    """
    Abstract base class for all foundation model adapters.

    Every model (Prithvi, SatMAE, RemoteCLIP) must:
    1. Implement load() to load weights
    2. Implement predict() to run inference
    3. Implement health_check() to verify model is ready

    This pattern ensures:
    → Unified API across all models ✅
    → Easy to add new models ✅
    → Router can swap models transparently ✅
    """

    def __init__(self, model_id: str, model_name: str, version: str):
        self.model_id    = model_id
        self.model_name  = model_name
        self.version     = version
        self._loaded     = False
        self._load_time  = None

    @abstractmethod
    def load(self, checkpoint_path: Optional[str] = None) -> None:
        """Load model weights. Must set self._loaded = True."""
        pass

    @abstractmethod
    def predict(self, inputs: InferenceInput) -> InferenceOutput:
        """Run inference on input. Must return InferenceOutput."""
        pass

    @abstractmethod
    def get_embedding(self, inputs: InferenceInput) -> List[float]:
        """Extract embedding vector from input."""
        pass

    def health_check(self) -> Dict[str, Any]:
        """Check if model is ready for inference."""
        return {
            "model_id":   self.model_id,
            "model_name": self.model_name,
            "version":    self.version,
            "loaded":     self._loaded,
            "load_time":  self._load_time,
            "status":     "ready" if self._loaded else "not_loaded",
        }

    def timed_predict(self, inputs: InferenceInput) -> InferenceOutput:
        """Run predict() and record latency."""
        if not self._loaded:
            raise RuntimeError(
                f"Model {self.model_name} not loaded. Call load() first."
            )
        t0     = time.time()
        output = self.predict(inputs)
        output.latency_ms = round((time.time() - t0) * 1000, 2)
        log.info(
            f"{self.model_name} inference: "
            f"{output.latency_ms}ms | "
            f"patch={inputs.patch_id}"
        )
        return output

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"name={self.model_name}, "
            f"version={self.version}, "
            f"loaded={self._loaded})"
        )
