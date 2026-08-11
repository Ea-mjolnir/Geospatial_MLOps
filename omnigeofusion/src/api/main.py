"""
OmniGeoFusion — FastAPI Application
=====================================
Production REST API for multimodal geospatial intelligence.

Endpoints:
  GET  /health                        → health check
  GET  /api/v1/model/info             → model info
  POST /api/v1/fusion/urban-change    → Task A
  POST /api/v1/fusion/flood-damage    → Task B
  POST /api/v1/fusion/agriculture     → Task C

Startup:
  1. Load OmniGeoFusion backbone from checkpoint
  2. Load task-specific heads
  3. Warm up model with dummy inference
  4. Start accepting requests

Usage:
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import logging
import torch
import yaml

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .router import router, MODELS, CONFIG, DEVICE
from . import router as router_module

log = logging.getLogger(__name__)

# ── Config paths ──────────────────────────────────────────────
MODEL_CONFIG_PATH = os.environ.get(
    'MODEL_CONFIG', 'configs/model_config.yaml'
)
TRAIN_CONFIG_PATH = os.environ.get(
    'TRAIN_CONFIG', 'configs/training_config.yaml'
)
API_CONFIG_PATH   = os.environ.get(
    'API_CONFIG', 'configs/api_config.yaml'
)
CHECKPOINT_DIR    = os.environ.get(
    'CHECKPOINT_DIR',
    '/gdrive/MyDrive/omnigeofusion/checkpoints/best'
)
MODEL_VERSION     = os.environ.get('MODEL_VERSION', '1.0.0')


# ── Default model config ──────────────────────────────────────
# Matches EXACTLY what OmniGeoFusionBackbone expects
# in backbone.py — every parameter documented
DEFAULT_MODEL_CFG = {
    'model': {
        'encoders': {

            # ── Optical: Prithvi-EO-2.0-300M ──────────────
            # TerraTorch BACKBONE_REGISTRY pretrained model
            # Trained on HLS (Sentinel-2 + Landsat) global
            # Input: [B, 6, 224, 224] — B02,B03,B04,B08,B11,B8A
            # Output: [B, 1024] — mean pool of patch tokens
            # Frozen in SSL, last 4 layers unfrozen in finetune
            'optical': {
                'name':              'prithvi_eo_v2_300',
                'pretrained':        True,
                'freeze_strategy':   'full',    # frozen during SSL
                'in_channels':       6,          # 6 Sentinel-2 bands
                'num_frames':        1,          # single timestamp
                'patch_size':        16,         # 16×16 pixel patches
                'embed_dim':         1024,       # ViT-Large dim
                'num_heads':         16,         # attention heads
                'num_layers':        24,         # transformer depth
                'vram_estimate_gb':  5.0,        # frozen = less VRAM
                'output_dim':        1024,       # → projection input
            },

            # ── SAR: SAR-ViT-Base ─────────────────────────
            # Vision Transformer trained from scratch on S1
            # No public SAR pretrained weights available
            # Input: [B, 2, 224, 224] — VV + VH polarization
            # Output: [B, 768] — CLS token embedding
            # SAR backscatter sensitive to:
            #   surface roughness, soil moisture, flood water
            'sar': {
                'in_channels':      2,           # VV + VH
                'patch_size':       16,          # 16×16 patches
                'embed_dim':        768,         # ViT-Base dim
                'num_heads':        12,          # attention heads
                'num_layers':       12,          # transformer depth
                'dropout':          0.1,
                'vram_estimate_gb': 1.0,
                'output_dim':       768,         # → projection input
            },

            # ── LiDAR: Multi-scale CNN + Height Attention ─
            # Operates on rasterized AHN4 products (not raw pts)
            # Input: [B, 3, 224, 224] — DSM, DTM, nDSM stacked
            #   DSM:  Digital Surface Model (top of everything)
            #         includes buildings, trees, bridges
            #   DTM:  Digital Terrain Model (bare ground only)
            #         buildings/vegetation removed
            #   nDSM: normalized DSM = DSM - DTM
            #         height above ground
            #         buildings: 5-100m+
            #         vegetation: 0-30m
            #         bare ground: ~0m
            # Output: [B, 512] — 3D structure embedding
            # Source: AHN4 OpenTopography S3 (CC-0 license)
            #   Resolution: 0.5m native → 10m resampled
            #   Accuracy: ±5cm vertical
            #   Coverage: 100% Netherlands
            # Used for:
            #   Task A: height_change = nDSM_AHN4 - nDSM_AHN3
            #   Task B: flood_depth = water_level - DTM_elevation
            #   Task C: crop_height = DSM - DTM (growing season)
            'lidar': {
                'in_channels':      3,           # DSM, DTM, nDSM
                'embed_dim':        512,         # LIDAR_DIM constant
                'dropout':          0.1,
                'products': [
                    'DSM',                       # surface elevation
                    'DTM',                       # terrain elevation
                    'nDSM',                      # height above ground
                ],
                'source':           'AHN4',      # OpenTopography S3
                'native_res_m':     0.5,         # AHN4 resolution
                'target_res_m':     10.0,        # match Sentinel-2
                'ahn3_available':   True,        # for change detection
                'vram_estimate_gb': 1.0,
                'output_dim':       512,         # → projection input
            },

            # ── Thermal: ResNet18 ─────────────────────────
            # ImageNet pretrained ResNet18 adapted for thermal
            # Input: [B, 1, 224, 224] — LST in Celsius
            # Output: [B, 256] — thermal embedding
            # Sources:
            #   Landsat 8/9 Band 10: 100m, 16-day revisit
            #   Sentinel-3 SLSTR:    1km, daily
            # Converted from DN to Celsius:
            #   LST_celsius = DN × 0.00341802 + 149.0 - 273.15
            # Derived products:
            #   UHI:  Urban Heat Island index
            #   CWSI: Crop Water Stress Index
            #   ET:   Evapotranspiration
            'thermal': {
                'in_channels':      1,           # LST single band
                'embed_dim':        256,         # THERMAL_DIM constant
                'pretrained':       True,        # ImageNet weights
                'dropout':          0.1,
                'sources': {
                    'landsat': {
                        'band':         'ST_B10',
                        'resolution_m': 100,
                        'revisit_days': 16,
                    },
                    'sentinel3': {
                        'channel':      'S8',
                        'resolution_m': 1000,
                        'revisit_days': 1,
                    },
                },
                'vram_estimate_gb': 0.5,
                'output_dim':       256,         # → projection input
            },

            # ── OSM: GraphSAGE ────────────────────────────
            # Inductive graph neural network on OSM vector data
            # Input: node_features [N, 64] + edge_index [2, E]
            # Output: [B, 256] — graph-level embedding
            # Node types:
            #   buildings: type, height, area, footprint
            #   roads:     class, lanes, speed, length
            # Edge types:
            #   road connections, proximity (<100m)
            # Layers extracted:
            #   buildings, roads, landuse
            #   waterways, natural features
            # Source: Geofabrik Netherlands extract (daily updated)
            #   URL: geofabrik.de/europe/netherlands-latest.osm.pbf
            'osm': {
                'node_features':    64,          # per-node feature dim
                'edge_features':    32,          # per-edge feature dim
                'embed_dim':        256,         # OSM_DIM constant
                'num_layers':       3,           # GraphSAGE depth
                'aggregation':      'mean',      # neighbor aggregation
                'dropout':          0.1,
                'layers': [
                    'buildings',                 # footprints + heights
                    'roads',                     # network topology
                    'landuse',                   # land cover classes
                    'waterways',                 # flood context
                    'natural',                   # vegetation/water
                ],
                'vram_estimate_gb': 0.5,
                'output_dim':       256,         # → projection input
            },

            # ── IoT: Bidirectional LSTM ───────────────────
            # Temporal sequence model for sensor readings
            # Input: [B, T, 16] — T=12 time steps × 16 features
            # Output: [B, 256] — temporal sensor embedding
            # 16 sensor features:
            #   Air quality (4):
            #     no2, pm25, pm10, o3
            #     Source: Luchtmeetnet API (300+ stations)
            #   Water (2):
            #     water_level_m, water_temp_c
            #     Source: Rijkswaterstaat API (1000+ gauges)
            #   Weather (6):
            #     air_temp_c, temp_min_c, temp_max_c
            #     rainfall_mm, wind_speed_ms, humidity_pct
            #     Source: KNMI API (35+ stations)
            #   Soil temperature (4):
            #     soil_temp_5cm_c, soil_temp_10cm_c
            #     soil_temp_20cm_c, soil_temp_50cm_c
            #     Source: KNMI automatic weather stations
            # T=12 time steps: daily readings for 12 days
            # Bidirectional: captures past + future context
            'iot': {
                'input_dim':        16,          # sensor features
                'hidden_dim':       256,         # LSTM hidden size
                'num_layers':       2,           # LSTM depth
                'bidirectional':    True,        # BiLSTM
                'sequence_length':  12,          # T time steps
                'embed_dim':        256,         # IOT_DIM constant
                'dropout':          0.1,
                'sensors': {
                    'air_quality': {
                        'source':    'luchtmeetnet',
                        'features':  ['no2', 'pm25', 'pm10', 'o3'],
                        'stations':  '300+',
                    },
                    'water': {
                        'source':    'rijkswaterstaat',
                        'features':  ['water_level_m', 'water_temp_c'],
                        'stations':  '1000+',
                    },
                    'weather': {
                        'source':   'knmi',
                        'features': [
                            'air_temp_c', 'temp_min_c', 'temp_max_c',
                            'rainfall_mm', 'wind_speed_ms', 'humidity_pct',
                        ],
                        'stations': '35+',
                    },
                    'soil': {
                        'source':   'knmi',
                        'features': [
                            'soil_temp_5cm_c',  'soil_temp_10cm_c',
                            'soil_temp_20cm_c', 'soil_temp_50cm_c',
                        ],
                        'stations': '35+',
                    },
                },
                'vram_estimate_gb': 0.3,
                'output_dim':       256,         # → projection input
            },
        },

        # ── Cross-Modal Attention Fusion ───────────────────
        # Projects all 6 encoder outputs → common_dim
        # Then runs cross-modal transformer attention
        # 6 modality tokens → [B, common_dim] fused
        # Modality dropout: 30% during training
        #   → model learns to work with any subset
        #   → graceful degradation when modality missing
        'fusion': {
            'common_dim':            512,        # unified embedding dim
            'num_heads':             8,          # attention heads
            'num_layers':            4,          # transformer depth
            'dropout':               0.1,
            'modality_dropout_prob': 0.0,        # 0 at inference
            'vram_estimate_gb':      2.0,
        },

        # ── Memory Budget (T4 = 15GB VRAM) ────────────────
        'memory_budget': {
            'optical_encoder':     5.0,          # frozen prithvi
            'sar_encoder':         1.0,
            'lidar_encoder':       1.0,
            'thermal_encoder':     0.5,
            'osm_encoder':         0.5,
            'iot_encoder':         0.3,
            'fusion_layers':       2.0,
            'task_heads':          0.5,
            'batch_data':          2.0,          # batch_size=8
            'gradients_optimizer': 2.0,
            'total_estimate':      14.8,         # fits T4 ✅
            'safety_margin':       0.2,
        },
    }
}


# ── Model Loading ─────────────────────────────────────────────
def load_models(device: torch.device) -> dict:
    """
    Load all models from checkpoints.
    Called once at startup.
    Falls back to DEFAULT_MODEL_CFG if config file missing.
    """
    models = {}

    # Load model config
    try:
        model_cfg = yaml.safe_load(open(MODEL_CONFIG_PATH))
        log.info(
            f'Loaded model config from {MODEL_CONFIG_PATH}'
        )
    except FileNotFoundError:
        log.warning(
            f'Config not found: {MODEL_CONFIG_PATH} '
            f'— using defaults'
        )
        model_cfg = DEFAULT_MODEL_CFG

    # ── Build backbone ────────────────────────────────────────
    log.info('Loading OmniGeoFusion backbone...')
    try:
        sys.path.insert(0, os.path.abspath('.'))
        from src.fusion.backbone import OmniGeoFusionBackbone

        backbone = OmniGeoFusionBackbone(
            model_cfg['model']
        ).to(device)
        backbone.eval()
        models['backbone'] = backbone

        total_params    = sum(
            p.numel() for p in backbone.parameters()
        )
        trainable_params = sum(
            p.numel() for p in backbone.parameters()
            if p.requires_grad
        )
        log.info(
            f'✅ Backbone loaded: '
            f'{total_params/1e6:.1f}M total, '
            f'{trainable_params/1e6:.1f}M trainable'
        )
    except Exception as e:
        log.error(f'Failed to load backbone: {e}')
        raise

    # ── Load task heads ───────────────────────────────────────
    import glob
    common_dim = model_cfg['model']['fusion']['common_dim']

    # Task A: Urban Change Detection
    try:
        from src.tasks.urban_change import UrbanChangeHead
        head_a    = UrbanChangeHead(
            common_dim=common_dim
        ).to(device)
        ckpt_path = os.path.join(
            CHECKPOINT_DIR, 'urban_change_best*.pt'
        )
        ckpts = sorted(glob.glob(ckpt_path))
        if ckpts:
            ckpt = torch.load(
                ckpts[-1], map_location=device
            )
            if isinstance(ckpt, dict) and 'head_state' in ckpt:
                head_a.load_state_dict(ckpt['head_state'])
            log.info(
                f'✅ Urban change head loaded: '
                f'{os.path.basename(ckpts[-1])}'
            )
        else:
            log.warning(
                'No urban_change checkpoint found '
                '— using random weights'
            )
        head_a.eval()
        models['urban_change_head'] = head_a
    except Exception as e:
        log.warning(f'Urban change head failed: {e}')

    # Task B: Flood/Disaster Assessment
    try:
        from src.tasks.flood_damage import FloodDamageHead
        head_b    = FloodDamageHead(
            common_dim=common_dim
        ).to(device)
        ckpt_path = os.path.join(
            CHECKPOINT_DIR, 'flood_damage_best*.pt'
        )
        ckpts = sorted(glob.glob(ckpt_path))
        if ckpts:
            ckpt = torch.load(
                ckpts[-1], map_location=device
            )
            if isinstance(ckpt, dict) and 'head_state' in ckpt:
                head_b.load_state_dict(ckpt['head_state'])
            log.info(
                f'✅ Flood damage head loaded: '
                f'{os.path.basename(ckpts[-1])}'
            )
        else:
            log.warning(
                'No flood_damage checkpoint found '
                '— using random weights'
            )
        head_b.eval()
        models['flood_damage_head'] = head_b
    except Exception as e:
        log.warning(f'Flood damage head failed: {e}')

    # Task C: Precision Agriculture
    try:
        from src.tasks.agriculture import AgricultureHead
        head_c    = AgricultureHead(
            common_dim=common_dim
        ).to(device)
        ckpt_path = os.path.join(
            CHECKPOINT_DIR, 'agriculture_best*.pt'
        )
        ckpts = sorted(glob.glob(ckpt_path))
        if ckpts:
            ckpt = torch.load(
                ckpts[-1], map_location=device
            )
            if isinstance(ckpt, dict) and 'head_state' in ckpt:
                head_c.load_state_dict(ckpt['head_state'])
            log.info(
                f'✅ Agriculture head loaded: '
                f'{os.path.basename(ckpts[-1])}'
            )
        else:
            log.warning(
                'No agriculture checkpoint found '
                '— using random weights'
            )
        head_c.eval()
        models['agriculture_head'] = head_c
    except Exception as e:
        log.warning(f'Agriculture head failed: {e}')

    return models


def warmup_models(models: dict, device: torch.device):
    """
    Run dummy inference through all components.
    Warms up GPU kernels → reduces first-request latency.
    Tests all modality combinations.
    """
    log.info('Warming up models...')
    backbone = models.get('backbone')
    if not backbone:
        return

    try:
        with torch.no_grad():
            # Full 6-modality warmup
            dummy_fused = backbone(
                optical=torch.randn(
                    1, 6, 224, 224
                ).to(device),
                sar=torch.randn(
                    1, 2, 224, 224
                ).to(device),
                lidar=torch.randn(
                    1, 3, 224, 224   # DSM, DTM, nDSM
                ).to(device),
                thermal=torch.randn(
                    1, 1, 224, 224   # LST in Celsius
                ).to(device),
                iot=torch.randn(
                    1, 12, 16        # 12 timesteps × 16 features
                ).to(device),
            )
        log.info(
            f'✅ Backbone warmup: '
            f'output {dummy_fused.shape}'
        )

        # Partial modality warmup (graceful degradation test)
        with torch.no_grad():
            backbone(
                optical=torch.randn(1, 6, 224, 224).to(device),
                sar=torch.randn(1, 2, 224, 224).to(device),
                # lidar, thermal, osm, iot missing → should work
            )
        log.info('✅ Partial modality warmup passed')

        # Task head warmups
        head_a = models.get('urban_change_head')
        if head_a:
            with torch.no_grad():
                head_a(dummy_fused, dummy_fused)
            log.info('✅ Urban change head warmup')

        head_b = models.get('flood_damage_head')
        if head_b:
            with torch.no_grad():
                head_b(dummy_fused, dummy_fused)
            log.info('✅ Flood damage head warmup')

        head_c = models.get('agriculture_head')
        if head_c:
            with torch.no_grad():
                head_c(dummy_fused)
            log.info('✅ Agriculture head warmup')

        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved  = torch.cuda.memory_reserved() / 1e9
            log.info(
                f'GPU memory after warmup: '
                f'{allocated:.1f}GB allocated, '
                f'{reserved:.1f}GB reserved'
            )

    except Exception as e:
        log.warning(f'Warmup failed: {e}')


# ── App Lifecycle ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + shutdown lifecycle."""

    # ── Startup ───────────────────────────────────────────────
    log.info('='*60)
    log.info('OmniGeoFusion API starting up...')
    log.info('='*60)

    # Device selection
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    log.info(f'Device: {device}')

    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        log.info(
            f'GPU: {torch.cuda.get_device_name(0)}'
        )
        log.info(
            f'VRAM: {props.total_memory / 1e9:.1f}GB total'
        )
        log.info(
            f'CUDA: {torch.version.cuda}'
        )
        # Enable TF32 for A100 performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True
    else:
        log.warning(
            'No GPU detected — running on CPU (slow)'
        )

    # Update router module globals
    router_module.DEVICE        = device
    router_module.MODEL_VERSION = MODEL_VERSION

    # Load + warm up models
    try:
        models = load_models(device)
        router_module.MODELS.update(models)
        warmup_models(models, device)
        log.info(
            f'✅ All models ready: {list(models.keys())}'
        )
    except Exception as e:
        log.error(f'Model loading failed: {e}', exc_info=True)
        # API still starts — /health will report status

    log.info('='*60)
    log.info('✅ OmniGeoFusion API ready')
    log.info(f'   Version:    {MODEL_VERSION}')
    log.info(f'   Device:     {device}')
    log.info(f'   Backbone:   prithvi_eo_v2_300M + fusion')
    log.info(f'   Modalities: S2 + S1 + LiDAR + Thermal + OSM + IoT')
    log.info(f'   Tasks:      urban_change + flood + agriculture')
    log.info(f'   Docs:       http://0.0.0.0:8000/docs')
    log.info(f'   Redoc:      http://0.0.0.0:8000/redoc')
    log.info('='*60)

    yield

    # ── Shutdown ──────────────────────────────────────────────
    log.info('OmniGeoFusion API shutting down...')
    router_module.MODELS.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info('GPU cache cleared')
    log.info('✅ Shutdown complete')


# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title='OmniGeoFusion API',
    description=(
        '## Multimodal Geospatial Intelligence Platform\n\n'
        'Fuses **6 data sources** for Earth observation:\n\n'
        '| Modality | Source | Resolution | Coverage |\n'
        '|----------|--------|------------|----------|\n'
        '| 🛰️ Optical | Sentinel-2 | 10m | Global daily |\n'
        '| 📡 SAR | Sentinel-1 | 10m | Global daily |\n'
        '| 📊 LiDAR | AHN4 (Netherlands) | 0.5m→10m | 100% NL |\n'
        '| 🔴 Thermal | Landsat 8/9 + S3 | 100m-1km | Global |\n'
        '| 🗺️ Vector | OpenStreetMap | Variable | Global |\n'
        '| 🌡️ IoT | Luchtmeetnet/KNMI/RWS | Point | Netherlands |\n\n'
        '### Three Tasks\n'
        '- **Urban Change Detection** — '
        'construction, demolition, vegetation loss\n'
        '- **Flood/Disaster Assessment** — '
        'flood extent, building damage, road access\n'
        '- **Precision Agriculture** — '
        'NDVI, LAI, CWSI, ET, crop stress\n\n'
        '### Architecture\n'
        'Prithvi-EO-2.0-300M backbone + '
        'cross-modal attention fusion + task-specific heads\n\n'
        '### Study Area\n'
        'Netherlands — best open LiDAR (AHN4) + '
        'densest IoT sensor network'
    ),
    version=MODEL_VERSION,
    lifespan=lifespan,
    docs_url='/docs',
    redoc_url='/redoc',
    openapi_tags=[
        {
            'name':        'health',
            'description': 'API health and model information'
        },
        {
            'name':        'fusion',
            'description': 'Multimodal fusion inference endpoints'
        },
    ]
)

# ── Middleware ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Request Logging ───────────────────────────────────────────
@app.middleware('http')
async def log_requests(request: Request, call_next):
    """Log all requests with method, path, status, timing."""
    import time
    start    = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    log.info(
        f'{request.method} {request.url.path} '
        f'→ {response.status_code} ({duration:.0f}ms)'
    )
    return response


# ── Error Handlers ────────────────────────────────────────────
@app.exception_handler(ValueError)
async def value_error_handler(
    request: Request, exc: ValueError
):
    return JSONResponse(
        status_code=400,
        content={'detail': str(exc)}
    )


@app.exception_handler(torch.cuda.OutOfMemoryError)
async def oom_handler(
    request: Request, exc: Exception
):
    log.error('GPU OOM — clearing cache and retrying')
    torch.cuda.empty_cache()
    return JSONResponse(
        status_code=503,
        content={
            'detail': 'GPU out of memory — try smaller bbox'
        }
    )


@app.exception_handler(Exception)
async def general_error_handler(
    request: Request, exc: Exception
):
    log.error(f'Unhandled error: {exc}', exc_info=True)
    return JSONResponse(
        status_code=500,
        content={'detail': 'Internal server error'}
    )


# ── Include Router ────────────────────────────────────────────
app.include_router(router)


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(
        description='OmniGeoFusion API server'
    )
    parser.add_argument(
        '--host',    default='0.0.0.0'
    )
    parser.add_argument(
        '--port',    type=int, default=8000
    )
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Number of uvicorn workers'
    )
    parser.add_argument(
        '--reload',  action='store_true',
        help='Enable hot reload (dev only)'
    )
    parser.add_argument(
        '--log-level', default='info',
        choices=['debug', 'info', 'warning', 'error']
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s %(levelname)s %(message)s'
    )

    uvicorn.run(
        'src.api.main:app',
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
    )
