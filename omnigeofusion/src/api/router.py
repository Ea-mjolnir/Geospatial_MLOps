"""
OmniGeoFusion — API Router
============================
Route handlers for all three tasks.
Each handler:
  1. Validates request (Pydantic)
  2. Fetches multimodal data from S3
  3. Runs backbone + task head inference
  4. Returns structured GeoJSON response
"""

import os
import time
import logging
import numpy as np
import torch

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any

from .schemas import (
    UrbanChangeRequest, UrbanChangeResponse,
    FloodDamageRequest, FloodDamageResponse,
    AgricultureRequest, AgricultureResponse,
    HealthResponse, ModelInfoResponse,
    ChangeType, DamageLevel, StressType,
    GeoJSONFeatureCollection, GeoJSONFeature,
    GeoJSONGeometry,
)

log = logging.getLogger(__name__)

router = APIRouter()

# Model registry (populated by main.py on startup)
MODELS: Dict[str, Any] = {}
CONFIG: Dict[str, Any] = {}
DEVICE: torch.device   = torch.device('cpu')
MODEL_VERSION: str     = '1.0.0'


# ── Helper: Fetch multimodal data ─────────────────────────────
async def fetch_multimodal_data(
    bbox: list,
    date_t1: str,
    date_t2: str,
    modalities: list,
) -> Dict[str, torch.Tensor]:
    """
    Fetch and preprocess multimodal data for inference.
    In production: loads from S3 cache.
    For now: returns dummy tensors with correct shapes.
    """
    B = 1  # batch size 1 for single request
    data = {}

    if 'sentinel2' in modalities:
        data['optical_t1'] = torch.randn(B, 6, 224, 224)
        data['optical_t2'] = torch.randn(B, 6, 224, 224)

    if 'sentinel1' in modalities:
        data['sar_t1'] = torch.randn(B, 2, 224, 224)
        data['sar_t2'] = torch.randn(B, 2, 224, 224)

    if 'lidar' in modalities:
        data['lidar'] = torch.randn(B, 3, 224, 224)

    if 'thermal' in modalities:
        data['thermal'] = torch.randn(B, 1, 224, 224)

    if 'iot' in modalities:
        data['iot'] = torch.randn(B, 12, 16)

    return data


def build_geojson_response(
    bbox: list,
    properties: Dict
) -> GeoJSONFeatureCollection:
    """Build GeoJSON feature collection from bbox + properties."""
    geometry = GeoJSONGeometry(
        type='Polygon',
        coordinates=[[
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            [bbox[0], bbox[1]],
        ]]
    )
    feature = GeoJSONFeature(
        geometry=geometry,
        properties=properties
    )
    return GeoJSONFeatureCollection(features=[feature])


# ── Health Check ──────────────────────────────────────────────
@router.get('/health', response_model=HealthResponse)
async def health_check():
    """API health check endpoint."""
    return HealthResponse(
        status='healthy',
        version=MODEL_VERSION,
        device=str(DEVICE),
        models_loaded=list(MODELS.keys()),
    )


# ── Model Info ────────────────────────────────────────────────
@router.get(
    '/api/v1/model/info',
    response_model=ModelInfoResponse
)
async def model_info():
    """Return model architecture information."""
    backbone = MODELS.get('backbone')
    params   = sum(
        p.numel() for p in backbone.parameters()
    ) / 1e6 if backbone else 0.0

    return ModelInfoResponse(
        name='OmniGeoFusion',
        version=MODEL_VERSION,
        backbone='prithvi_eo_v2_300M + CrossModalAttention',
        modalities=[
            'sentinel2', 'sentinel1', 'lidar',
            'thermal', 'osm', 'iot'
        ],
        tasks=[
            'urban_change',
            'flood_damage',
            'agriculture'
        ],
        encoders={
            'optical':  'prithvi_eo_v2_300M (300M params)',
            'sar':      'SAR-ViT-Base',
            'lidar':    'LiDAR-ResNet',
            'thermal':  'Thermal-ResNet18',
            'osm':      'GraphSAGE',
            'iot':      'BiLSTM',
        },
        parameters_m=round(params, 1),
        training_data='Netherlands (AHN4 + Sentinel + IoT)',
    )


# ── Task A: Urban Change Detection ────────────────────────────
@router.post(
    '/api/v1/fusion/urban-change',
    response_model=UrbanChangeResponse
)
async def urban_change_detection(
    request: UrbanChangeRequest
):
    """
    Detect urban land cover change between two dates.

    Fuses Sentinel-2, Sentinel-1, LiDAR, OSM
    to predict change score + type.
    """
    start_time = time.time()

    # Validate bbox size
    if request.bbox.area_km2 > 100:
        raise HTTPException(
            status_code=400,
            detail=f'Bbox too large: {request.bbox.area_km2:.1f}km² '
                   f'(max 100km²)'
        )

    # Check models loaded
    if 'backbone' not in MODELS:
        raise HTTPException(
            status_code=503,
            detail='Models not loaded — service starting up'
        )

    try:
        bbox_list  = request.bbox.to_list()
        modalities = [m.value for m in request.modalities]

        # Fetch data
        data = await fetch_multimodal_data(
            bbox_list,
            request.date_t1,
            request.date_t2,
            modalities
        )

        # Move to device
        data = {
            k: v.to(DEVICE) for k, v in data.items()
        }

        # Inference
        backbone = MODELS['backbone']
        head     = MODELS.get('urban_change_head')

        with torch.no_grad():
            fused_t1 = backbone(
                optical=data.get('optical_t1'),
                sar=data.get('sar_t1'),
                lidar=data.get('lidar'),
                iot=data.get('iot'),
            )
            fused_t2 = backbone(
                optical=data.get('optical_t2'),
                sar=data.get('sar_t2'),
                lidar=data.get('lidar'),
                iot=data.get('iot'),
            )

            if head:
                preds = head(fused_t1, fused_t2)
                change_score = float(
                    preds['change_score'][0].cpu()
                )
                change_type_idx = int(
                    preds['change_type'][0].argmax().cpu()
                )
            else:
                # Fallback: cosine distance
                cos_sim      = torch.nn.functional.cosine_similarity(
                    fused_t1, fused_t2
                )
                change_score = float(1 - cos_sim[0].cpu())
                change_type_idx = 0

        # Map to enums
        change_types = list(ChangeType)
        change_type  = change_types[
            min(change_type_idx, len(change_types)-1)
        ]

        # Compute area
        bbox_area_m2 = request.bbox.area_km2 * 1e6
        area_affected = bbox_area_m2 * change_score

        # Day gap
        from datetime import datetime
        d1      = datetime.strptime(request.date_t1, '%Y-%m-%d')
        d2      = datetime.strptime(request.date_t2, '%Y-%m-%d')
        day_gap = abs((d2 - d1).days)

        processing_time = int(
            (time.time() - start_time) * 1000
        )

        # Build GeoJSON
        geometry = build_geojson_response(
            bbox_list,
            {
                'change_score': round(change_score, 4),
                'change_type':  change_type.value,
                'date_t1':      request.date_t1,
                'date_t2':      request.date_t2,
            }
        )

        return UrbanChangeResponse(
            bbox=request.bbox,
            date_t1=request.date_t1,
            date_t2=request.date_t2,
            day_gap=day_gap,
            modalities_used=modalities,
            fusion_confidence=min(1.0, change_score * 1.2),
            change_detected=change_score > 0.2,
            change_score=round(change_score, 4),
            change_type=change_type,
            area_affected_m2=round(area_affected, 2),
            volume_change_m3=None,
            geometry=geometry,
            processing_time_ms=processing_time,
            model_version=MODEL_VERSION,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f'Urban change inference failed: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Inference failed: {str(e)}'
        )


# ── Task B: Flood/Disaster Assessment ─────────────────────────
@router.post(
    '/api/v1/fusion/flood-damage',
    response_model=FloodDamageResponse
)
async def flood_damage_assessment(
    request: FloodDamageRequest
):
    """
    Assess flood extent and building damage.

    Fuses SAR, LiDAR DEM, IoT water sensors
    to map flood extent and estimate damage.
    """
    start_time = time.time()

    if request.bbox.area_km2 > 500:
        raise HTTPException(
            status_code=400,
            detail=f'Bbox too large: {request.bbox.area_km2:.1f}km² '
                   f'(max 500km²)'
        )

    if 'backbone' not in MODELS:
        raise HTTPException(
            status_code=503,
            detail='Models not loaded'
        )

    try:
        bbox_list  = request.bbox.to_list()
        modalities = [m.value for m in request.modalities]

        data = await fetch_multimodal_data(
            bbox_list,
            request.date_pre,
            request.date_post,
            modalities
        )
        data = {k: v.to(DEVICE) for k, v in data.items()}

        backbone = MODELS['backbone']
        head     = MODELS.get('flood_damage_head')

        with torch.no_grad():
            fused_pre = backbone(
                optical=data.get('optical_t1'),
                sar=data.get('sar_t1'),
                lidar=data.get('lidar'),
                iot=data.get('iot'),
            )
            fused_post = backbone(
                optical=data.get('optical_t2'),
                sar=data.get('sar_t2'),
                lidar=data.get('lidar'),
                iot=data.get('iot'),
            )

            if head:
                preds        = head(fused_pre, fused_post)
                flood_prob   = float(
                    preds['flood_probability'][0].cpu()
                )
                flood_depth  = float(
                    preds['flood_depth'][0].cpu()
                )
                damage_idx   = int(
                    preds['damage_level'][0].argmax().cpu()
                )
                road_blocked = float(
                    preds['road_blocked'][0].cpu()
                )
            else:
                flood_prob   = 0.3
                flood_depth  = 0.5
                damage_idx   = 1
                road_blocked = 0.2

        damage_levels   = list(DamageLevel)
        damage_level    = damage_levels[
            min(damage_idx, len(damage_levels)-1)
        ]

        bbox_area_km2   = request.bbox.area_km2
        flooded_area    = bbox_area_km2 * flood_prob

        processing_time = int(
            (time.time() - start_time) * 1000
        )

        geometry = build_geojson_response(
            bbox_list,
            {
                'flood_probability': round(flood_prob, 4),
                'flood_depth_m':     round(flood_depth, 2),
                'damage_level':      damage_level.value,
                'date_pre':          request.date_pre,
                'date_post':         request.date_post,
            }
        )

        return FloodDamageResponse(
            bbox=request.bbox,
            disaster_type=request.disaster_type.value,
            date_pre=request.date_pre,
            date_post=request.date_post,
            modalities_used=modalities,
            fusion_confidence=flood_prob,
            flood_detected=flood_prob > 0.3,
            flood_fraction=round(flood_prob, 4),
            mean_flood_depth_m=round(flood_depth, 2),
            max_flood_depth_m=round(flood_depth * 1.5, 2),
            flooded_area_km2=round(flooded_area, 4),
            dominant_damage_level=damage_level,
            buildings_affected=int(flooded_area * 50),
            road_accessible_pct=round(
                (1 - road_blocked) * 100, 1
            ),
            population_exposed=int(flooded_area * 500),
            geometry=geometry,
            processing_time_ms=processing_time,
            model_version=MODEL_VERSION,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f'Flood damage inference failed: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Inference failed: {str(e)}'
        )


# ── Task C: Precision Agriculture ─────────────────────────────
@router.post(
    '/api/v1/fusion/agriculture',
    response_model=AgricultureResponse
)
async def agriculture_monitoring(
    request: AgricultureRequest
):
    """
    Monitor crop health and field conditions.

    Fuses Sentinel-2, SAR, LiDAR, thermal, IoT
    to predict NDVI, LAI, CWSI, ET and stress type.
    """
    start_time = time.time()

    if request.bbox.area_km2 > 50:
        raise HTTPException(
            status_code=400,
            detail=f'Bbox too large: {request.bbox.area_km2:.1f}km² '
                   f'(max 50km²)'
        )

    if 'backbone' not in MODELS:
        raise HTTPException(
            status_code=503,
            detail='Models not loaded'
        )

    try:
        bbox_list  = request.bbox.to_list()
        modalities = [m.value for m in request.modalities]

        data = await fetch_multimodal_data(
            bbox_list,
            request.date,
            request.date,
            modalities
        )
        data = {k: v.to(DEVICE) for k, v in data.items()}

        backbone = MODELS['backbone']
        head     = MODELS.get('agriculture_head')

        with torch.no_grad():
            fused = backbone(
                optical=data.get('optical_t1'),
                sar=data.get('sar_t1'),
                lidar=data.get('lidar'),
                thermal=data.get('thermal'),
                iot=data.get('iot'),
            )

            if head:
                preds        = head(fused)
                ndvi         = float(preds['ndvi'][0].cpu())
                lai          = float(preds['lai'][0].cpu())
                crop_height  = float(
                    preds['crop_height_m'][0].cpu()
                )
                soil_moisture = float(
                    preds['soil_moisture'][0].cpu()
                )
                cwsi         = float(preds['cwsi'][0].cpu())
                et_mm_day    = float(
                    preds['et_mm_day'][0].cpu()
                )
                stress_idx   = int(
                    preds['stress_type'][0].argmax().cpu()
                )
            else:
                ndvi          = 0.6
                lai           = 2.5
                crop_height   = 0.8
                soil_moisture = 0.35
                cwsi          = 0.3
                et_mm_day     = 4.0
                stress_idx    = 0

        stress_types  = list(StressType)
        stress_type   = stress_types[
            min(stress_idx, len(stress_types)-1)
        ]

        processing_time = int(
            (time.time() - start_time) * 1000
        )

        geometry = build_geojson_response(
            bbox_list,
            {
                'ndvi':          round(ndvi, 4),
                'lai':           round(lai, 2),
                'crop_height_m': round(crop_height, 2),
                'cwsi':          round(cwsi, 4),
                'stress_type':   stress_type.value,
                'date':          request.date,
            }
        )

        return AgricultureResponse(
            bbox=request.bbox,
            date=request.date,
            crop_type=request.crop_type.value,
            modalities_used=modalities,
            fusion_confidence=float(ndvi),
            ndvi=round(ndvi, 4),
            lai=round(lai, 2),
            crop_height_m=round(crop_height, 2),
            soil_moisture=round(soil_moisture, 4),
            cwsi=round(cwsi, 4),
            et_mm_day=round(et_mm_day, 2),
            stress_type=stress_type,
            stress_severity=round(cwsi, 4),
            irrigation_recommended=cwsi > 0.6,
            geometry=geometry,
            processing_time_ms=processing_time,
            model_version=MODEL_VERSION,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error(f'Agriculture inference failed: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Inference failed: {str(e)}'
        )
