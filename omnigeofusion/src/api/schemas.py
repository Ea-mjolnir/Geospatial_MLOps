"""
OmniGeoFusion — API Schemas
============================
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────
class Modality(str, Enum):
    sentinel2 = 'sentinel2'
    sentinel1 = 'sentinel1'
    lidar     = 'lidar'
    thermal   = 'thermal'
    osm       = 'osm'
    iot       = 'iot'


class DisasterType(str, Enum):
    flood     = 'flood'
    earthquake = 'earthquake'
    fire      = 'fire'
    storm     = 'storm'


class CropType(str, Enum):
    wheat     = 'wheat'
    maize     = 'maize'
    potato    = 'potato'
    sugar_beet = 'sugar_beet'
    grass     = 'grass'
    unknown   = 'unknown'


class ChangeType(str, Enum):
    no_change       = 'no_change'
    construction    = 'construction'
    demolition      = 'demolition'
    vegetation_loss = 'vegetation_loss'
    flood_damage    = 'flood_damage'
    road_change     = 'road_change'


class DamageLevel(str, Enum):
    no_damage = 'no_damage'
    minor     = 'minor'
    moderate  = 'moderate'
    severe    = 'severe'
    destroyed = 'destroyed'


class StressType(str, Enum):
    healthy          = 'healthy'
    water_stress     = 'water_stress'
    nutrient_stress  = 'nutrient_stress'
    pest_stress      = 'pest_stress'


# ── Common ────────────────────────────────────────────────────
class BBox(BaseModel):
    min_lon: float = Field(..., ge=-180, le=180)
    min_lat: float = Field(..., ge=-90,  le=90)
    max_lon: float = Field(..., ge=-180, le=180)
    max_lat: float = Field(..., ge=-90,  le=90)

    @validator('max_lon')
    def max_lon_gt_min(cls, v, values):
        if 'min_lon' in values and v <= values['min_lon']:
            raise ValueError('max_lon must be > min_lon')
        return v

    @validator('max_lat')
    def max_lat_gt_min(cls, v, values):
        if 'min_lat' in values and v <= values['min_lat']:
            raise ValueError('max_lat must be > min_lat')
        return v

    @property
    def area_km2(self) -> float:
        lon_diff = self.max_lon - self.min_lon
        lat_diff = self.max_lat - self.min_lat
        return lon_diff * lat_diff * 111 * 111

    def to_list(self) -> List[float]:
        return [
            self.min_lon, self.min_lat,
            self.max_lon, self.max_lat
        ]


class GeoJSONGeometry(BaseModel):
    type:        str
    coordinates: Any


class GeoJSONFeature(BaseModel):
    type:       str = 'Feature'
    geometry:   GeoJSONGeometry
    properties: Dict[str, Any] = {}


class GeoJSONFeatureCollection(BaseModel):
    type:     str = 'FeatureCollection'
    features: List[GeoJSONFeature] = []


# ── Task A: Urban Change Detection ────────────────────────────
class UrbanChangeRequest(BaseModel):
    bbox:       BBox
    date_t1:    str = Field(
        ..., description='T1 date (YYYY-MM-DD)'
    )
    date_t2:    str = Field(
        ..., description='T2 date (YYYY-MM-DD)'
    )
    modalities: List[Modality] = Field(
        default=[
            Modality.sentinel2, Modality.sentinel1,
            Modality.lidar, Modality.osm
        ]
    )

    class Config:
        schema_extra = {
            'example': {
                'bbox': {
                    'min_lon': 4.85, 'min_lat': 52.35,
                    'max_lon': 4.95, 'max_lat': 52.40,
                },
                'date_t1':    '2020-07-07',
                'date_t2':    '2023-07-07',
                'modalities': [
                    'sentinel2', 'sentinel1',
                    'lidar', 'osm'
                ],
            }
        }


class UrbanChangeResponse(BaseModel):
    task:              str = 'urban_change'
    bbox:              BBox
    date_t1:           str
    date_t2:           str
    day_gap:           int
    modalities_used:   List[str]
    fusion_confidence: float
    change_detected:   bool
    change_score:      float = Field(
        ..., ge=0, le=1,
        description='Change magnitude [0=stable, 1=high change]'
    )
    change_type:       ChangeType
    area_affected_m2:  float
    volume_change_m3:  Optional[float] = None
    geometry:          Optional[GeoJSONFeatureCollection] = None
    processing_time_ms: int
    model_version:     str


# ── Task B: Flood/Disaster Assessment ─────────────────────────
class FloodDamageRequest(BaseModel):
    bbox:          BBox
    disaster_type: DisasterType = DisasterType.flood
    date_pre:      str = Field(
        ..., description='Pre-event date (YYYY-MM-DD)'
    )
    date_post:     str = Field(
        ..., description='Post-event date (YYYY-MM-DD)'
    )
    modalities:    List[Modality] = Field(
        default=[
            Modality.sentinel2, Modality.sentinel1,
            Modality.lidar, Modality.osm, Modality.iot
        ]
    )

    class Config:
        schema_extra = {
            'example': {
                'bbox': {
                    'min_lon': 4.40, 'min_lat': 51.88,
                    'max_lon': 4.55, 'max_lat': 51.95,
                },
                'disaster_type': 'flood',
                'date_pre':      '2023-06-01',
                'date_post':     '2023-06-15',
                'modalities':    [
                    'sentinel2', 'sentinel1',
                    'lidar', 'osm', 'iot'
                ],
            }
        }


class FloodDamageResponse(BaseModel):
    task:                  str = 'flood_damage'
    bbox:                  BBox
    disaster_type:         str
    date_pre:              str
    date_post:             str
    modalities_used:       List[str]
    fusion_confidence:     float
    flood_detected:        bool
    flood_fraction:        float = Field(
        ..., ge=0, le=1,
        description='Fraction of area flooded'
    )
    mean_flood_depth_m:    float
    max_flood_depth_m:     float
    flooded_area_km2:      float
    dominant_damage_level: DamageLevel
    buildings_affected:    int
    road_accessible_pct:   float
    population_exposed:    Optional[int] = None
    geometry:              Optional[GeoJSONFeatureCollection] = None
    processing_time_ms:    int
    model_version:         str


# ── Task C: Precision Agriculture ─────────────────────────────
class AgricultureRequest(BaseModel):
    bbox:       BBox
    date:       str = Field(
        ..., description='Observation date (YYYY-MM-DD)'
    )
    crop_type:  CropType = CropType.unknown
    modalities: List[Modality] = Field(
        default=[
            Modality.sentinel2, Modality.sentinel1,
            Modality.lidar, Modality.thermal,
            Modality.iot
        ]
    )

    class Config:
        schema_extra = {
            'example': {
                'bbox': {
                    'min_lon': 5.30, 'min_lat': 52.40,
                    'max_lon': 5.60, 'max_lat': 52.60,
                },
                'date':      '2023-07-15',
                'crop_type': 'wheat',
                'modalities': [
                    'sentinel2', 'sentinel1',
                    'lidar', 'thermal', 'iot'
                ],
            }
        }


class AgricultureResponse(BaseModel):
    task:              str = 'agriculture'
    bbox:              BBox
    date:              str
    crop_type:         str
    modalities_used:   List[str]
    fusion_confidence: float
    ndvi:              float = Field(
        ..., ge=-1, le=1,
        description='Vegetation health index'
    )
    lai:               float = Field(
        ..., ge=0, le=10,
        description='Leaf Area Index'
    )
    crop_height_m:     float = Field(
        ..., ge=0, le=5,
        description='Estimated crop height (meters)'
    )
    soil_moisture:     float = Field(
        ..., ge=0, le=1,
        description='Volumetric water content'
    )
    cwsi:              float = Field(
        ..., ge=0, le=1,
        description='Crop Water Stress Index'
    )
    et_mm_day:         float = Field(
        ..., ge=0, le=15,
        description='Evapotranspiration (mm/day)'
    )
    stress_type:       StressType
    stress_severity:   float = Field(
        ..., ge=0, le=1
    )
    irrigation_recommended: bool
    geometry:          Optional[GeoJSONFeatureCollection] = None
    processing_time_ms: int
    model_version:     str


# ── Health + Info ─────────────────────────────────────────────
class HealthResponse(BaseModel):
    status:        str = 'healthy'
    version:       str
    device:        str
    models_loaded: List[str]


class ModelInfoResponse(BaseModel):
    name:          str = 'OmniGeoFusion'
    version:       str
    backbone:      str
    modalities:    List[str]
    tasks:         List[str]
    encoders:      Dict[str, str]
    parameters_m:  float
    training_data: str
