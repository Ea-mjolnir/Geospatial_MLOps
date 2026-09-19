"""Wildfire prediction request/response schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional


class WildfireRequest(BaseModel):
    """
    Input for wildfire risk prediction.
    thermal_patch: flattened 224x224 thermal image (float list)
    weather:       [t2m, u10, v10, d2m, wind_speed]
    osm:           [roads, buildings, forest, water, powerlines, residential, farmland]
    """
    thermal_patch:  List[float] = Field(..., description="Flattened 224x224 thermal patch (50176 values)")
    weather:        List[float] = Field(default=[0.0]*5, description="ERA5 weather features [5]")
    osm:            List[float] = Field(default=[0.0]*7, description="OSM features [7]")
    patch_id:       Optional[str] = Field(default=None, description="Optional patch identifier")


class WildfireResponse(BaseModel):
    """Wildfire risk prediction output."""
    patch_id:       Optional[str]
    risk_score:     float = Field(..., description="Fire risk score [0-1]")
    alert_level:    int   = Field(..., description="Alert level [0=none, 1=low, 2=medium, 3=high]")
    alert_label:    str   = Field(..., description="Alert level label")
    spread_prob:    float = Field(..., description="Fire spread probability [0-1]")
    structure_risk: float = Field(..., description="Structure risk score [0-1]")
    confidence:     float = Field(..., description="Model confidence [0-1]")
