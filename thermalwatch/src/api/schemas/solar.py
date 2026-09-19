"""Solar farm health prediction request/response schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional


class SolarRequest(BaseModel):
    """
    Input for solar farm health prediction.
    thermal_patch: flattened 224x224 thermal image (float list)
    weather:       [GHI, DNI, DHI, wind_speed, temp]
    """
    thermal_patch:  List[float] = Field(..., description="Flattened 224x224 thermal patch (50176 values)")
    weather:        List[float] = Field(default=[0.0]*5, description="NSRDB weather features [5]")
    patch_id:       Optional[str] = Field(default=None, description="Optional patch identifier")


class SolarResponse(BaseModel):
    """Solar farm health prediction output."""
    patch_id:         Optional[str]
    efficiency_score: float = Field(..., description="Panel efficiency score [0-1]")
    hotspot_score:    float = Field(..., description="Hotspot severity score [0-1]")
    degradation_rate: float = Field(..., description="Degradation rate [0-1]")
    maintenance_flag: bool  = Field(..., description="Maintenance required flag")
    maintenance_prob: float = Field(..., description="Maintenance probability [0-1]")
    health_status:    str   = Field(..., description="Overall health status")
