"""Wildfire risk prediction router."""

import torch
import numpy as np
import logging
from fastapi import APIRouter, HTTPException
from src.api.schemas.wildfire import WildfireRequest, WildfireResponse
from src.api.dependencies import load_wildfire_model

log    = logging.getLogger(__name__)
router = APIRouter()

ALERT_LABELS = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "HIGH"}

# Normalization constants
THERMAL_MEAN = 35.0
THERMAL_STD  = 15.0
WEATHER_MEAN = np.array([290.0, 0.0, 0.0, 275.0, 2.0], dtype=np.float32)
WEATHER_STD  = np.array([15.0, 3.0, 3.0, 15.0, 3.0], dtype=np.float32)
OSM_MAX      = np.array([4000, 7000, 200, 200, 100, 200, 200], dtype=np.float32)


@router.post("/wildfire", response_model=WildfireResponse)
def predict_wildfire(request: WildfireRequest):
    """
    Predict wildfire risk from thermal patch + weather + OSM features.

    - **thermal_patch**: flattened 224x224 thermal image (50,176 float values)
    - **weather**: ERA5 features [t2m, u10, v10, d2m, wind_speed]
    - **osm**: OSM features [roads, buildings, forest, water, powerlines, residential, farmland]
    """
    # Validate input
    if len(request.thermal_patch) != 224 * 224:
        raise HTTPException(
            status_code=422,
            detail=f"thermal_patch must have {224*224} values, got {len(request.thermal_patch)}"
        )
    if len(request.weather) != 5:
        raise HTTPException(status_code=422, detail="weather must have 5 values")
    if len(request.osm) != 7:
        raise HTTPException(status_code=422, detail="osm must have 7 values")

    try:
        backbone, head, device = load_wildfire_model()

        # Preprocess thermal
        thermal = np.array(request.thermal_patch, dtype=np.float32).reshape(1, 224, 224)
        thermal = np.nan_to_num(thermal, nan=0.0)
        thermal = (thermal - THERMAL_MEAN) / (THERMAL_STD + 1e-6)
        thermal = torch.from_numpy(thermal).unsqueeze(0).to(device)  # [1,1,224,224]

        # Preprocess weather
        weather = np.array(request.weather, dtype=np.float32)
        weather = (weather - WEATHER_MEAN) / (WEATHER_STD + 1e-6)
        weather = torch.from_numpy(weather).unsqueeze(0).to(device)  # [1,5]

        # Preprocess OSM
        osm = np.array(request.osm, dtype=np.float32)
        osm = np.clip(osm / (OSM_MAX + 1e-6), 0, 1)
        osm = torch.from_numpy(osm).unsqueeze(0).to(device)  # [1,7]

        # Inference
        with torch.no_grad():
            emb   = backbone(thermal=thermal, weather=weather, osm=osm)
            preds = head(emb)

        risk_score     = float(preds['risk_score'].cpu().item())
        alert_logits   = preds['alert_level'].cpu()
        alert_probs    = torch.softmax(alert_logits, dim=-1)
        alert_level    = int(alert_logits.argmax(-1).item())
        confidence     = float(alert_probs.max().item())
        spread_prob    = float(preds['spread_prob'].cpu().item())
        structure_risk = float(preds['structure_risk'].cpu().item())

        return WildfireResponse(
            patch_id=request.patch_id,
            risk_score=round(risk_score, 4),
            alert_level=alert_level,
            alert_label=ALERT_LABELS[alert_level],
            spread_prob=round(spread_prob, 4),
            structure_risk=round(structure_risk, 4),
            confidence=round(confidence, 4),
        )

    except Exception as e:
        log.error(f'Wildfire prediction error: {e}')
        raise HTTPException(status_code=500, detail=str(e))
