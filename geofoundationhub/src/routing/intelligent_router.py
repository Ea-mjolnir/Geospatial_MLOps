"""
GeoFoundationHub — Intelligent Router
=======================================
Routes requests to the best foundation model
based on input data characteristics.

No model_id needed from user — system decides
automatically based on:
  - sensor_type (optical, multispectral, thermal, sar)
  - channels (1, 3, 6, etc.)
  - image statistics (future enhancement)

Scoring logic:
  Each model scored 0-10 per request
  Highest score wins
"""

import logging
from typing import Dict, List, Optional
from models.base import InferenceInput
from src.registry.models import ModelMetadata, SensorType

log = logging.getLogger(__name__)


# Model capability profiles
MODEL_PROFILES = {
    "Prithvi-EO-2.0-300M": {
        "optimal_channels":  [6],
        "optimal_sensors":   ["thermal", "multispectral"],
        "good_sensors":      ["optical"],
        "description":       "Best for 6-band HLS multispectral + thermal",
    },
    "SatMAE": {
        "optimal_channels":  [3],
        "optimal_sensors":   ["multispectral", "sar"],
        "good_sensors":      ["optical"],
        "description":       "Best for 3-band multispectral + SAR imagery",
    },
    "RemoteCLIP": {
        "optimal_channels":  [3],
        "optimal_sensors":   ["optical"],
        "good_sensors":      ["multispectral"],
        "description":       "Best for RGB optical imagery + zero-shot",
    },
}


def score_model(
    model_name: str,
    channels:   int,
    sensor_type: str,
) -> float:
    """
    Score a model for a given request.
    Returns score 0-10 (higher = better fit).
    """
    profile = MODEL_PROFILES.get(model_name)
    if not profile:
        return 1.0  # unknown model gets low score

    score = 0.0

    # Channel match
    if channels in profile["optimal_channels"]:
        score += 4.0
    elif abs(channels - profile["optimal_channels"][0]) <= 3:
        score += 1.0

    # Sensor type match
    if sensor_type in profile["optimal_sensors"]:
        score += 4.0
    elif sensor_type in profile["good_sensors"]:
        score += 2.0

    # Special cases
    if model_name == "Prithvi-EO-2.0-300M":
        if sensor_type == "thermal":
            score += 2.0  # Prithvi is best for thermal
        if channels == 6:
            score += 1.0  # HLS native format

    if model_name == "RemoteCLIP":
        if sensor_type == "optical" and channels == 3:
            score += 2.0  # RemoteCLIP native RGB

    if model_name == "SatMAE":
        if sensor_type == "sar":
            score += 2.0  # SatMAE handles SAR best

    return round(score, 2)


def select_best_model(
    inputs:    InferenceInput,
    adapters:  Dict[str, object],
    registry_models: List[ModelMetadata],
) -> Optional[str]:
    """
    Select the best model for a given input.

    Args:
        inputs:          InferenceInput with channels + sensor_type
        adapters:        Dict of loaded model_id → adapter
        registry_models: List of registered ModelMetadata

    Returns:
        model_id of best model, or None if no models loaded
    """
    if not adapters:
        return None

    scores = {}
    reasoning = {}

    for model in registry_models:
        if model.model_id not in adapters:
            continue  # skip unloaded models

        s = score_model(
            model_name=model.name,
            channels=inputs.channels,
            sensor_type=inputs.sensor_type,
        )
        scores[model.model_id]    = s
        reasoning[model.model_id] = {
            "name":        model.name,
            "score":       s,
            "channels":    inputs.channels,
            "sensor_type": inputs.sensor_type,
        }

    if not scores:
        return None

    best_id = max(scores, key=scores.get)

    log.info(
        f"Intelligent routing: "
        f"channels={inputs.channels} "
        f"sensor={inputs.sensor_type} → "
        f"{reasoning[best_id]['name']} "
        f"(score={scores[best_id]})"
    )

    # Log all scores for transparency
    for mid, info in reasoning.items():
        log.debug(f"  {info['name']}: {info['score']}")

    return best_id
