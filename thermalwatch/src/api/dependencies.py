"""
Model loading dependencies for FastAPI.
Uses singleton pattern — models loaded once at startup.
Checkpoint paths read from environment variables.
"""

import os
import torch
import logging
from pathlib import Path
from functools import lru_cache

log = logging.getLogger(__name__)

WILDFIRE_CKPT = Path(os.getenv('WILDFIRE_CKPT', 'checkpoints/wildfire_best_phase1.pt'))
SOLAR_CKPT    = Path(os.getenv('SOLAR_CKPT',    'checkpoints/solar_best_phase1.pt'))


@lru_cache(maxsize=1)
def load_wildfire_model():
    """Load wildfire backbone + head. Cached after first call."""
    import sys
    sys.path.insert(0, '.')
    from src.models.backbone.thermal_backbone import ThermalWatchBackbone
    from src.models.training.finetune import WildfireHead

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = ThermalWatchBackbone(pretrained_optical=False).to(device)
    head     = WildfireHead(512).to(device)

    if WILDFIRE_CKPT.exists():
        ckpt = torch.load(WILDFIRE_CKPT, map_location=device)
        backbone.load_state_dict(ckpt['backbone_state'])
        head.load_state_dict(ckpt['head_state'])
        log.info(f'✅ Wildfire model loaded from {WILDFIRE_CKPT}')
    else:
        log.warning(f'⚠️  No checkpoint at {WILDFIRE_CKPT} — using random weights')

    backbone.eval()
    head.eval()
    return backbone, head, device


@lru_cache(maxsize=1)
def load_solar_model():
    """Load solar backbone + head. Cached after first call."""
    import sys
    sys.path.insert(0, '.')
    from src.models.backbone.thermal_backbone import ThermalWatchBackbone
    from src.models.training.finetune import SolarHead

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = ThermalWatchBackbone(pretrained_optical=False).to(device)
    head     = SolarHead(512).to(device)

    if SOLAR_CKPT.exists():
        ckpt = torch.load(SOLAR_CKPT, map_location=device)
        backbone.load_state_dict(ckpt['backbone_state'])
        head.load_state_dict(ckpt['head_state'])
        log.info(f'✅ Solar model loaded from {SOLAR_CKPT}')
    else:
        log.warning(f'⚠️  No checkpoint at {SOLAR_CKPT} — using random weights')

    backbone.eval()
    head.eval()
    return backbone, head, device
