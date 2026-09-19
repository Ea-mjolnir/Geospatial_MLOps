"""
ThermalWatch — Wildfire Risk Head
===================================
Task-specific head for wildfire risk prediction.
Takes 512d fused embedding from ThermalWatchBackbone
and predicts wildfire risk metrics.

Outputs:
  risk_score:    float [0,1] — overall fire risk
  alert_level:   int [0-3]  — low/medium/high/critical
  spread_prob:   float [0,1] — probability of spread
  structure_risk:float [0,1] — structures at risk score

Loss:
  risk_score:    MSE vs NIFC-derived risk label
  alert_level:   CrossEntropy vs threshold-derived label
  spread_prob:   BCEWithLogitsLoss
  structure_risk:MSE vs building density + risk score

Metrics:
  risk_r2:       R² for risk_score prediction
  alert_acc:     accuracy for alert_level
  risk_mae:      mean absolute error for risk_score

Usage:
  from src.models.wildfire.wildfire_head import (
      WildfireHead, WildfireLoss, compute_wildfire_metrics
  )
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

log = logging.getLogger(__name__)

FUSED_DIM   = 512
ALERT_LEVELS = 4  # low, medium, high, critical


# ── Wildfire Head ─────────────────────────────────────────────
class WildfireHead(nn.Module):
    """
    Wildfire risk prediction head.
    Input:  [B, FUSED_DIM] fused backbone embedding
    Output: Dict with risk_score, alert_level,
            spread_prob, structure_risk
    """

    def __init__(
        self,
        input_dim:   int = FUSED_DIM,
        hidden_dim:  int = 256,
        dropout:     float = 0.2,
    ):
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        trunk_out = hidden_dim // 2

        # Output heads
        self.risk_head      = nn.Linear(trunk_out, 1)
        self.alert_head     = nn.Linear(trunk_out, ALERT_LEVELS)
        self.spread_head    = nn.Linear(trunk_out, 1)
        self.structure_head = nn.Linear(trunk_out, 1)

        log.info(
            f'WildfireHead: {input_dim}d → '
            f'risk + alert({ALERT_LEVELS}) + '
            f'spread + structure'
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, FUSED_DIM]
        Returns:
            Dict with:
              risk_score:    [B] in [0,1]
              alert_level:   [B, 4] logits
              spread_prob:   [B] logits
              structure_risk:[B] in [0,1]
        """
        if x.shape[-1] != FUSED_DIM:
            raise ValueError(
                f'WildfireHead expects {FUSED_DIM}d input, '
                f'got {x.shape[-1]}'
            )

        feat = self.trunk(x)

        return {
            'risk_score':     torch.sigmoid(
                self.risk_head(feat)
            ).squeeze(-1),
            'alert_level':    self.alert_head(feat),
            'spread_prob':    torch.sigmoid(
                self.spread_head(feat)
            ).squeeze(-1),
            'structure_risk': torch.sigmoid(
                self.structure_head(feat)
            ).squeeze(-1),
        }


# ── Wildfire Loss ─────────────────────────────────────────────
class WildfireLoss(nn.Module):
    """
    Combined loss for wildfire prediction.

    Weights:
      risk_score:    0.4 (primary objective)
      alert_level:   0.3 (categorical)
      spread_prob:   0.2 (binary)
      structure_risk:0.1 (auxiliary)
    """

    def __init__(self):
        super().__init__()
        self.mse     = nn.MSELoss()
        self.ce      = nn.CrossEntropyLoss()
        self.bce     = nn.BCEWithLogitsLoss()

        self.weights = {
            'risk_score':     0.4,
            'alert_level':    0.3,
            'spread_prob':    0.2,
            'structure_risk': 0.1,
        }

    def forward(
        self,
        preds:   Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            preds:   output of WildfireHead.forward()
            targets: dict with same keys as preds
        Returns:
            Dict with individual losses + total loss
        Raises:
            KeyError if required target missing
        """
        required = ['risk_score', 'alert_level']
        for key in required:
            if key not in targets:
                raise KeyError(
                    f'Missing required target: {key}. '
                    f'Available: {list(targets.keys())}'
                )

        losses = {}

        losses['risk_score'] = self.mse(
            preds['risk_score'],
            targets['risk_score'].float(),
        )

        losses['alert_level'] = self.ce(
            preds['alert_level'],
            targets['alert_level'].long(),
        )

        if 'spread_prob' in targets:
            losses['spread_prob'] = self.bce(
                preds['spread_prob'],
                targets['spread_prob'].float(),
            )
        else:
            losses['spread_prob'] = torch.tensor(
                0.0, device=preds['risk_score'].device
            )

        if 'structure_risk' in targets:
            losses['structure_risk'] = self.mse(
                preds['structure_risk'],
                targets['structure_risk'].float(),
            )
        else:
            losses['structure_risk'] = torch.tensor(
                0.0, device=preds['risk_score'].device
            )

        losses['loss'] = sum(
            self.weights[k] * losses[k]
            for k in self.weights
        )

        return losses


# ── Wildfire Metrics ──────────────────────────────────────────
def compute_wildfire_metrics(
    predictions: Dict[str, torch.Tensor],
    targets:     Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute evaluation metrics for wildfire predictions.

    Returns:
      risk_r2:    R² score for risk_score
      risk_mae:   mean absolute error for risk_score
      alert_acc:  accuracy for alert_level prediction
    """
    metrics = {}

    # Risk R²
    pred_risk = predictions['risk_score'].detach().cpu().numpy()
    true_risk = targets['risk_score'].detach().cpu().numpy()

    ss_res = np.sum((true_risk - pred_risk) ** 2)
    ss_tot = np.sum((true_risk - true_risk.mean()) ** 2)

    if ss_tot < 1e-8:
        raise ValueError(
            'risk_score targets have zero variance. '
            'Check target generation — all values identical.'
        )

    metrics['risk_r2']  = float(1 - ss_res / ss_tot)
    metrics['risk_mae'] = float(
        np.abs(pred_risk - true_risk).mean()
    )

    # Alert accuracy
    pred_alert = predictions['alert_level'].argmax(
        dim=-1
    ).detach().cpu().numpy()
    true_alert = targets['alert_level'].detach().cpu().numpy()

    metrics['alert_acc'] = float(
        (pred_alert == true_alert).mean()
    )

    return metrics


# ── Target Generator ──────────────────────────────────────────
def generate_wildfire_targets(
    nifc_overlap: float,
    viirs_count:  int,
    osm_features: Dict,
    weather:      Dict,
) -> Dict[str, float]:
    """
    Generate weakly supervised targets from available data.

    Args:
        nifc_overlap: fraction of patch covered by fire perimeter
        viirs_count:  number of VIIRS hotspots in patch
        osm_features: dict with road/building/forest counts
        weather:      dict with temperature/wind/humidity

    Returns:
        Dict with risk_score, alert_level,
        spread_prob, structure_risk
    """
    # Risk score: weighted combination
    fire_presence = min(1.0, nifc_overlap)
    hotspot_score = min(1.0, viirs_count / 10.0)

    # Weather risk factor (high temp + low humidity + wind)
    temp_c    = weather.get('temperature_c', 20.0)
    humidity  = weather.get('humidity_pct', 50.0)
    wind_ms   = weather.get('wind_ms', 2.0)

    temp_norm     = min(1.0, max(0.0, (temp_c - 10) / 40))
    humidity_norm = min(1.0, max(0.0, 1 - humidity / 100))
    wind_norm     = min(1.0, wind_ms / 20)

    weather_risk = (
        0.4 * temp_norm +
        0.4 * humidity_norm +
        0.2 * wind_norm
    )

    risk_score = (
        0.5 * fire_presence +
        0.3 * hotspot_score +
        0.2 * weather_risk
    )

    # Alert level (0=low, 1=medium, 2=high, 3=critical)
    if risk_score < 0.25:
        alert_level = 0
    elif risk_score < 0.5:
        alert_level = 1
    elif risk_score < 0.75:
        alert_level = 2
    else:
        alert_level = 3

    # Spread probability (fire + wind)
    spread_prob = min(1.0, (
        fire_presence * 0.6 + wind_norm * 0.4
    ))

    # Structure risk (fire presence + building density)
    building_count  = osm_features.get('building_count', 0)
    building_norm   = min(1.0, building_count / 100)
    structure_risk  = min(1.0, (
        fire_presence * 0.7 + building_norm * 0.3
    ))

    return {
        'risk_score':     float(risk_score),
        'alert_level':    int(alert_level),
        'spread_prob':    float(spread_prob),
        'structure_risk': float(structure_risk),
    }
