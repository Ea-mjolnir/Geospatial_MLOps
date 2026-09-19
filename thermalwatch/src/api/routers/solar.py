"""Solar farm health prediction router."""

import torch
import numpy as np
import logging
from fastapi import APIRouter, HTTPException
from src.api.schemas.solar import SolarRequest, SolarResponse
from src.api.dependencies import load_solar_model

log    = logging.getLogger(__name__)
router = APIRouter()

THERMAL_MEAN = 45.0
THERMAL_STD  = 20.0
WEATHER_MEAN = np.array([290.0, 0.0, 0.0, 275.0, 2.0], dtype=np.float32)
WEATHER_STD  = np.array([15.0, 3.0, 3.0, 15.0, 3.0], dtype=np.float32)


def get_health_status(efficiency: float, hotspot: float, maintenance: bool) -> str:
    if maintenance or hotspot > 0.7:
        return "CRITICAL"
    elif efficiency < 0.5 or hotspot > 0.4:
        return "DEGRADED"
    elif efficiency < 0.7:
        return "FAIR"
    else:
        return "HEALTHY"


@router.post("/solar", response_model=SolarResponse)
def predict_solar(request: SolarRequest):
    """
    Predict solar farm health from thermal patch + weather features.

    - **thermal_patch**: flattened 224x224 thermal image (50,176 float values)
    - **weather**: NSRDB features [GHI, DNI, DHI, wind_speed, temp]
    """
    if len(request.thermal_patch) != 224 * 224:
        raise HTTPException(
            status_code=422,
            detail=f"thermal_patch must have {224*224} values, got {len(request.thermal_patch)}"
        )
    if len(request.weather) != 5:
        raise HTTPException(status_code=422, detail="weather must have 5 values")

    try:
        backbone, head, device = load_solar_model()

        # Preprocess thermal
        thermal = np.array(request.thermal_patch, dtype=np.float32).reshape(1, 224, 224)
        thermal = np.nan_to_num(thermal, nan=0.0)
        thermal = (thermal - THERMAL_MEAN) / (THERMAL_STD + 1e-6)
        thermal = torch.from_numpy(thermal).unsqueeze(0).to(device)

        # Preprocess weather
        weather = np.array(request.weather, dtype=np.float32)
        weather = (weather - WEATHER_MEAN) / (WEATHER_STD + 1e-6)
        weather = torch.from_numpy(weather).unsqueeze(0).to(device)

        # No OSM for solar
        osm = torch.zeros(1, 7).to(device)

        with torch.no_grad():
            emb   = backbone(thermal=thermal, weather=weather, osm=osm)
            preds = head(emb)

        efficiency  = float(preds['efficiency_score'].cpu().item())
        hotspot     = float(preds['hotspot_score'].cpu().item())
        degradation = float(preds['degradation_rate'].cpu().item())
        maint_logit = float(preds['maintenance_flag'].cpu().item())
        maint_prob  = float(torch.sigmoid(torch.tensor(maint_logit)).item())
        maint_flag  = maint_prob > 0.5

        return SolarResponse(
            patch_id=request.patch_id,
            efficiency_score=round(efficiency, 4),
            hotspot_score=round(hotspot, 4),
            degradation_rate=round(degradation, 4),
            maintenance_flag=maint_flag,
            maintenance_prob=round(maint_prob, 4),
            health_status=get_health_status(efficiency, hotspot, maint_flag),
        )

    except Exception as e:
        log.error(f'Solar prediction error: {e}')
        raise HTTPException(status_code=500, detail=str(e))
