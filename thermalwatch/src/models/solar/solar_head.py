"""
ThermalWatch — Solar Health Head
===================================
Task-specific head for solar farm health monitoring.
Takes 512d fused embedding from ThermalWatchBackbone
and predicts solar panel health metrics.

Outputs:
  efficiency_score: float [0,1] — overall panel efficiency
  hotspot_score:    float [0,1] — thermal anomaly severity
  degradation_rate: float [0,1] — panel aging rate
  maintenance_flag: float       — maintenance needed (logit)

Loss:
  efficiency_score: MSE vs EIA-derived efficiency label
  hotspot_score:    MSE vs thermal anomaly score
  degradation_rate: MSE vs year-over-year efficiency drop
  maintenance_flag: BCEWithLogitsLoss

Metrics:
  efficiency_r2:  R² for efficiency_score
  efficiency_mae: mean absolute error
  maintenance_acc:accuracy for maintenance prediction

Usage:
  from src.models.solar.solar_head import (
      SolarHead, SolarLoss, compute_solar_metrics
  )
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict

log = logging.getLogger(__name__)

FUSED_DIM = 512


# ── Solar Head ────────────────────────────────────────────────
class SolarHead(nn.Module):
    """
    Solar farm health prediction head.
    Input:  [B, FUSED_DIM] fused backbone embedding
    Output: Dict with efficiency, hotspot,
            degradation, maintenance outputs
    """

    def __init__(
        self,
        input_dim:  int   = FUSED_DIM,
        hidden_dim: int   = 256,
        dropout:    float = 0.2,
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
        self.efficiency_head   = nn.Linear(trunk_out, 1)
        self.hotspot_head      = nn.Linear(trunk_out, 1)
        self.degradation_head  = nn.Linear(trunk_out, 1)
        self.maintenance_head  = nn.Linear(trunk_out, 1)

        log.info(
            f'SolarHead: {input_dim}d → '
            f'efficiency + hotspot + '
            f'degradation + maintenance'
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
              efficiency_score: [B] in [0,1]
              hotspot_score:    [B] in [0,1]
              degradation_rate: [B] in [0,1]
              maintenance_flag: [B] logit
        """
        if x.shape[-1] != FUSED_DIM:
            raise ValueError(
                f'SolarHead expects {FUSED_DIM}d input, '
                f'got {x.shape[-1]}'
            )

        feat = self.trunk(x)

        return {
            'efficiency_score': torch.sigmoid(
                self.efficiency_head(feat)
            ).squeeze(-1),
            'hotspot_score':    torch.sigmoid(
                self.hotspot_head(feat)
            ).squeeze(-1),
            'degradation_rate': torch.sigmoid(
                self.degradation_head(feat)
            ).squeeze(-1),
            'maintenance_flag': torch.sigmoid(
                self.maintenance_head(feat)
            ).squeeze(-1),
        }


# ── Solar Loss ────────────────────────────────────────────────
class SolarLoss(nn.Module):
    """
    Combined loss for solar health prediction.

    Weights:
      efficiency_score: 0.4 (primary objective)
      hotspot_score:    0.3 (thermal anomaly)
      degradation_rate: 0.2 (aging)
      maintenance_flag: 0.1 (binary alert)
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

        self.weights = {
            'efficiency_score': 0.4,
            'hotspot_score':    0.3,
            'degradation_rate': 0.2,
            'maintenance_flag': 0.1,
        }

    def forward(
        self,
        preds:   Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            preds:   output of SolarHead.forward()
            targets: dict with same keys
        Returns:
            Dict with individual losses + total loss
        Raises:
            KeyError if required target missing
        """
        required = ['efficiency_score']
        for key in required:
            if key not in targets:
                raise KeyError(
                    f'Missing required target: {key}. '
                    f'Available: {list(targets.keys())}'
                )

        losses = {}

        losses['efficiency_score'] = self.mse(
            preds['efficiency_score'],
            targets['efficiency_score'].float(),
        )

        if 'hotspot_score' in targets:
            losses['hotspot_score'] = self.mse(
                preds['hotspot_score'],
                targets['hotspot_score'].float(),
            )
        else:
            losses['hotspot_score'] = torch.tensor(
                0.0, device=preds['efficiency_score'].device
            )

        if 'degradation_rate' in targets:
            losses['degradation_rate'] = self.mse(
                preds['degradation_rate'],
                targets['degradation_rate'].float(),
            )
        else:
            losses['degradation_rate'] = torch.tensor(
                0.0, device=preds['efficiency_score'].device
            )

        if 'maintenance_flag' in targets:
            losses['maintenance_flag'] = self.bce(
                preds['maintenance_flag'],
                targets['maintenance_flag'].float(),
            )
        else:
            losses['maintenance_flag'] = torch.tensor(
                0.0, device=preds['efficiency_score'].device
            )

        losses['loss'] = sum(
            self.weights[k] * losses[k]
            for k in self.weights
        )

        return losses


# ── Solar Metrics ─────────────────────────────────────────────
def compute_solar_metrics(
    predictions: Dict[str, torch.Tensor],
    targets:     Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Compute evaluation metrics for solar predictions.

    Returns:
      efficiency_r2:   R² for efficiency_score
      efficiency_mae:  mean absolute error
      hotspot_r2:      R² for hotspot_score
      maintenance_acc: accuracy for maintenance flag
    """
    metrics = {}

    # Efficiency R²
    pred_eff = (
        predictions['efficiency_score']
        .detach().cpu().numpy()
    )
    true_eff = (
        targets['efficiency_score']
        .detach().cpu().numpy()
    )

    ss_res = np.sum((true_eff - pred_eff) ** 2)
    ss_tot = np.sum((true_eff - true_eff.mean()) ** 2)

    if ss_tot < 1e-8:
        raise ValueError(
            'efficiency_score targets have zero variance. '
            'Check target generation — all values identical.'
        )

    metrics['efficiency_r2']  = float(
        1 - ss_res / ss_tot
    )
    metrics['efficiency_mae'] = float(
        np.abs(pred_eff - true_eff).mean()
    )

    # Hotspot R²
    if 'hotspot_score' in targets:
        pred_h = (
            predictions['hotspot_score']
            .detach().cpu().numpy()
        )
        true_h = (
            targets['hotspot_score']
            .detach().cpu().numpy()
        )
        ss_res_h = np.sum((true_h - pred_h) ** 2)
        ss_tot_h = np.sum(
            (true_h - true_h.mean()) ** 2
        )
        if ss_tot_h > 1e-8:
            metrics['hotspot_r2'] = float(
                1 - ss_res_h / ss_tot_h
            )
        else:
            metrics['hotspot_r2'] = 0.0

    # Maintenance accuracy
    if 'maintenance_flag' in targets:
        pred_m = (
            predictions['maintenance_flag'] > 0
        ).detach().cpu().numpy()
        true_m = (
            targets['maintenance_flag'] > 0.5
        ).detach().cpu().numpy()
        metrics['maintenance_acc'] = float(
            (pred_m == true_m).mean()
        )

    return metrics


# ── Target Generator ──────────────────────────────────────────
def generate_solar_targets(
    eia_efficiency:   float,
    thermal_mean_c:   float,
    thermal_max_c:    float,
    nsrdb_ghi:        float,
    install_year:     int,
    current_year:     int,
) -> Dict[str, float]:
    """
    Generate weakly supervised targets from available data.

    Args:
        eia_efficiency:  EIA generation / expected generation
                         (0-1, derived from NSRDB irradiance)
        thermal_mean_c:  mean panel temperature (°C)
        thermal_max_c:   max panel temperature (°C)
        nsrdb_ghi:       global horizontal irradiance (W/m²)
        install_year:    year panels were installed
        current_year:    current year

    Returns:
        Dict with efficiency_score, hotspot_score,
        degradation_rate, maintenance_flag
    """
    # Efficiency: EIA-derived
    efficiency_score = min(1.0, max(0.0, eia_efficiency))

    # Hotspot score: thermal anomaly above expected
    # Panel temperature = ambient + (GHI × NOCT factor)
    # Typical NOCT factor ~0.03 °C per W/m²
    expected_temp = 25.0 + (nsrdb_ghi * 0.03)
    temp_excess   = max(0.0, thermal_max_c - expected_temp)
    hotspot_score = min(1.0, temp_excess / 30.0)

    # Degradation rate: ~0.5% per year typical
    # Higher degradation = lower efficiency over time
    panel_age        = max(0, current_year - install_year)
    expected_degrade = panel_age * 0.005
    actual_degrade   = max(0.0, 1.0 - efficiency_score)
    degradation_rate = min(
        1.0, actual_degrade / max(expected_degrade, 0.01)
    )

    # Maintenance flag: hotspot OR low efficiency
    maintenance_needed = (
        hotspot_score > 0.5 or efficiency_score < 0.7
    )

    return {
        'efficiency_score': float(efficiency_score),
        'hotspot_score':    float(hotspot_score),
        'degradation_rate': float(degradation_rate),
        'maintenance_flag': float(maintenance_needed),
    }
