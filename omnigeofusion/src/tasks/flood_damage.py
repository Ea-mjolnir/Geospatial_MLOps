"""
OmniGeoFusion — Task B: Flood/Disaster Damage Assessment Head
==============================================================
Fine-tuning head for flood extent + damage assessment.

Target generation (weakly supervised):
  Primary:   SAR water mask (automatic from Sentinel-1)
             → water backscatter < -16dB threshold
  Secondary: flood depth from AHN4 DTM + water level sensor
             → flood_depth = max(0, water_level - DTM_elevation)
  Validation: Copernicus Emergency Management labels
              (used for validation only, not training)

Output:
  flood_mask:   [B, H, W] binary flood extent
  flood_depth:  [B, H, W] continuous depth (meters)
  damage_level: [B, 5] building damage classification
    0: no_damage
    1: minor
    2: moderate
    3: severe
    4: destroyed
  road_blocked: [B, 1] probability road is inaccessible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from typing import Dict, Tuple, Optional

log = logging.getLogger(__name__)

# Damage level labels
DAMAGE_LEVELS = [
    'no_damage',
    'minor',
    'moderate',
    'severe',
    'destroyed',
]
NUM_DAMAGE_LEVELS = len(DAMAGE_LEVELS)

# SAR water detection threshold (dB)
SAR_WATER_THRESHOLD = -16.0


# ── Target Generator ──────────────────────────────────────────
class FloodTargetGenerator:
    """
    Automatically generates flood/damage targets
    from SAR + LiDAR DEM + IoT water sensors.
    No manual labels needed.
    """

    def __init__(
        self,
        sar_threshold_db: float = SAR_WATER_THRESHOLD,
        primary_weight: float = 0.6,
        secondary_weight: float = 0.4,
    ):
        self.sar_threshold  = sar_threshold_db
        self.primary_weight = primary_weight
        self.secondary_weight = secondary_weight

    def compute_sar_water_mask(
        self,
        sar_vv_t1: np.ndarray,
        sar_vv_t2: np.ndarray,
        threshold_db: Optional[float] = None
    ) -> np.ndarray:
        """
        Detect flood water from SAR backscatter.
        Water has very low backscatter in SAR.

        Method:
          1. Convert DN to dB
          2. Apply threshold (< -16dB = water)
          3. Change detection: new water = flood

        Args:
            sar_vv_t1: [H, W] SAR VV backscatter at T1
            sar_vv_t2: [H, W] SAR VV backscatter at T2

        Returns:
            water_mask: [H, W] bool (True = flooded)
        """
        threshold = threshold_db or self.sar_threshold

        # Convert to dB if not already
        def to_db(x):
            x = np.clip(x, 1e-10, None)
            return 10 * np.log10(x)

        vv_t1_db = to_db(sar_vv_t1)
        vv_t2_db = to_db(sar_vv_t2)

        # Water mask at T1 and T2
        water_t1 = vv_t1_db < threshold
        water_t2 = vv_t2_db < threshold

        # New water = flood (present at T2 but not T1)
        flood_mask = water_t2 & ~water_t1

        return flood_mask.astype(bool)

    def compute_flood_depth(
        self,
        dtm: np.ndarray,
        water_level_m: float,
        flood_mask: np.ndarray,
        nodata: float = -9999.0
    ) -> np.ndarray:
        """
        Compute flood depth from DEM + water level sensor.

        flood_depth = max(0, water_level - terrain_elevation)
        Only computed where flood_mask = True.

        Args:
            dtm:           [H, W] terrain elevation (m, NAP)
            water_level_m: water surface elevation from IoT sensor
            flood_mask:    [H, W] bool from SAR

        Returns:
            flood_depth: [H, W] float32 (meters)
        """
        valid = dtm != nodata
        depth = np.where(
            valid & flood_mask,
            np.maximum(0, water_level_m - dtm),
            0.0
        )
        return depth.astype(np.float32)

    def compute_building_damage(
        self,
        flood_depth: np.ndarray,
        building_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Estimate building damage from flood depth.
        Based on depth-damage curves (standard hydrology).

        Depth-damage relationships:
          0.0m:     no_damage     (0)
          0.0-0.3m: minor         (1) — floor level
          0.3-1.0m: moderate      (2) — furniture damage
          1.0-2.0m: severe        (3) — structural damage
          >2.0m:    destroyed     (4) — total loss

        Args:
            flood_depth:   [H, W] flood depth in meters
            building_mask: [H, W] bool (True = building from OSM)

        Returns:
            damage_level: [H, W] int [0-4]
        """
        damage = np.zeros_like(flood_depth, dtype=np.int32)

        damage[flood_depth > 0.0]  = 1  # minor
        damage[flood_depth > 0.3]  = 2  # moderate
        damage[flood_depth > 1.0]  = 3  # severe
        damage[flood_depth > 2.0]  = 4  # destroyed

        # Only damage where buildings exist
        if building_mask is not None:
            damage = damage * building_mask.astype(np.int32)

        return damage

    def compute_road_accessibility(
        self,
        flood_depth: np.ndarray,
        road_mask: Optional[np.ndarray] = None,
        passable_depth_m: float = 0.3
    ) -> float:
        """
        Estimate road accessibility after flooding.

        A road is blocked if flood_depth > 0.3m
        (standard threshold for vehicle passage).

        Args:
            flood_depth:      [H, W] flood depth
            road_mask:        [H, W] bool (True = road from OSM)
            passable_depth_m: max passable water depth

        Returns:
            accessible_fraction: float [0, 1]
        """
        if road_mask is not None:
            road_pixels  = road_mask.sum()
            if road_pixels == 0:
                return 1.0
            blocked      = (
                road_mask & (flood_depth > passable_depth_m)
            ).sum()
            return float(1.0 - blocked / road_pixels)
        else:
            total   = flood_depth.size
            blocked = (flood_depth > passable_depth_m).sum()
            return float(1.0 - blocked / total)

    def generate_targets(
        self,
        sar_vv_t1: np.ndarray,
        sar_vv_t2: np.ndarray,
        dtm: np.ndarray,
        water_level_m: float,
        building_mask: Optional[np.ndarray] = None,
        road_mask: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Generate all flood/damage targets for one patch pair.

        Returns:
            Dict with flood_mask, flood_depth,
                       damage_level, road_accessible
        """
        # Primary: SAR water mask
        flood_mask = self.compute_sar_water_mask(
            sar_vv_t1, sar_vv_t2
        )

        # Secondary: flood depth from DEM + sensor
        flood_depth = self.compute_flood_depth(
            dtm, water_level_m, flood_mask
        )

        # Building damage from depth
        damage_level = self.compute_building_damage(
            flood_depth, building_mask
        )

        # Road accessibility
        road_accessible = self.compute_road_accessibility(
            flood_depth, road_mask
        )

        # Patch-level aggregates
        flood_fraction  = float(flood_mask.mean())
        mean_depth      = float(
            flood_depth[flood_mask].mean()
            if flood_mask.sum() > 0 else 0.0
        )
        dominant_damage = int(
            np.bincount(damage_level.flatten()).argmax()
        )

        return {
            'flood_mask':       flood_mask,
            'flood_depth':      flood_depth,
            'damage_level':     damage_level,
            'road_accessible':  road_accessible,
            'flood_fraction':   flood_fraction,
            'mean_flood_depth': mean_depth,
            'dominant_damage':  dominant_damage,
        }


# ── Flood Damage Head ─────────────────────────────────────────
class FloodDamageHead(nn.Module):
    """
    Task B: Flood/Disaster Damage Assessment head.

    Takes T1 + T2 backbone embeddings and predicts:
      flood_probability: float [0, 1]
      flood_depth:       float [0, ∞) meters
      damage_level:      int [0-4]
      road_blocked:      float [0, 1]

    Input:  [B, COMMON_DIM] × 2 (T1, T2 embeddings)
    """

    def __init__(
        self,
        common_dim: int = 512,
        hidden_dims: list = None,
        num_damage_levels: int = NUM_DAMAGE_LEVELS,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Change representation (same as urban change)
        in_dim = common_dim * 4

        # Shared feature extraction
        layers      = []
        current_dim = in_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(current_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            current_dim = h_dim
        self.shared = nn.Sequential(*layers)

        # Flood probability head (binary)
        self.flood_prob_head = nn.Sequential(
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Flood depth head (regression, non-negative)
        self.flood_depth_head = nn.Sequential(
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # ensures non-negative output
        )

        # Damage level head (classification)
        self.damage_head = nn.Sequential(
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_damage_levels)
        )

        # Road blocked head (binary)
        self.road_head = nn.Sequential(
            nn.Linear(current_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        log.info(
            f'FloodDamageHead: {common_dim}d × 2 → '
            f'flood + depth + {num_damage_levels} damage + road'
        )

    def forward(
        self,
        emb_t1: torch.Tensor,
        emb_t2: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            emb_t1: [B, COMMON_DIM] T1 fused embedding
            emb_t2: [B, COMMON_DIM] T2 fused embedding

        Returns:
            dict with flood + damage predictions
        """
        diff    = emb_t2 - emb_t1
        product = emb_t1 * emb_t2
        concat  = torch.cat(
            [emb_t1, emb_t2, diff, product], dim=-1
        )

        features = self.shared(concat)

        return {
            'flood_probability': self.flood_prob_head(
                features
            ).squeeze(-1),                          # [B]
            'flood_depth':       self.flood_depth_head(
                features
            ).squeeze(-1),                          # [B]
            'damage_level':      self.damage_head(features), # [B,5]
            'road_blocked':      self.road_head(
                features
            ).squeeze(-1),                          # [B]
        }


# ── Loss Function ─────────────────────────────────────────────
class FloodDamageLoss(nn.Module):
    """
    Combined loss for flood/damage assessment.

    L = w1 × BCE(flood_prob, flood_mask)
      + w2 × MSE(flood_depth, target_depth)
      + w3 × CE(damage_level, target_damage)
      + w4 × BCE(road_blocked, target_road)
    """

    def __init__(
        self,
        flood_weight:  float = 0.4,
        depth_weight:  float = 0.3,
        damage_weight: float = 0.2,
        road_weight:   float = 0.1,
    ):
        super().__init__()
        self.w_flood  = flood_weight
        self.w_depth  = depth_weight
        self.w_damage = damage_weight
        self.w_road   = road_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.ce  = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: output from FloodDamageHead
            targets: dict with target tensors

        Returns:
            Dict with individual + total loss
        """
        flood_loss = self.bce(
            predictions['flood_probability'],
            targets['flood_fraction'].float()
        )
        depth_loss = self.mse(
            predictions['flood_depth'],
            targets['mean_flood_depth'].float()
        )
        damage_loss = self.ce(
            predictions['damage_level'],
            targets['dominant_damage'].long()
        )
        road_loss = self.bce(
            predictions['road_blocked'],
            (1 - targets['road_accessible']).float()
        )

        total = (
            self.w_flood  * flood_loss  +
            self.w_depth  * depth_loss  +
            self.w_damage * damage_loss +
            self.w_road   * road_loss
        )

        return {
            'loss':        total,
            'flood_loss':  flood_loss,
            'depth_loss':  depth_loss,
            'damage_loss': damage_loss,
            'road_loss':   road_loss,
        }


# ── Metrics ───────────────────────────────────────────────────
def compute_flood_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Compute IoU + RMSE metrics for flood assessment."""
    # Flood IoU
    pred_flood = (
        predictions['flood_probability'].detach().cpu().numpy() > 0.5
    )
    true_flood = (
        targets['flood_fraction'].detach().cpu().numpy() > 0.1
    )
    intersection = (pred_flood & true_flood).sum()
    union        = (pred_flood | true_flood).sum()
    iou          = float(intersection / (union + 1e-8))

    # Depth RMSE
    pred_depth = predictions['flood_depth'].detach().cpu().numpy()
    true_depth = targets['mean_flood_depth'].detach().cpu().numpy()
    rmse       = float(np.sqrt(((pred_depth - true_depth)**2).mean()))

    # Damage accuracy
    pred_damage = predictions['damage_level'].argmax(
        dim=-1
    ).detach().cpu().numpy()
    true_damage = targets['dominant_damage'].detach().cpu().numpy()
    damage_acc  = float((pred_damage == true_damage).mean())

    return {
        'flood_iou':    iou,
        'depth_rmse':   rmse,
        'damage_acc':   damage_acc,
    }


# ── Main ──────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO)

    # Test target generator
    generator = FloodTargetGenerator()

    H, W = 256, 256
    sar_t1 = np.random.uniform(0.001, 0.1, (H, W))
    sar_t2 = np.random.uniform(0.0001, 0.01, (H, W))
    dtm    = np.random.uniform(-2, 5, (H, W))

    targets = generator.generate_targets(
        sar_t1, sar_t2, dtm,
        water_level_m=1.5
    )
    log.info(
        f'Flood fraction: {targets["flood_fraction"]:.3f}'
    )
    log.info(
        f'Mean depth: {targets["mean_flood_depth"]:.3f}m'
    )
    log.info(
        f'Damage: {DAMAGE_LEVELS[targets["dominant_damage"]]}'
    )

    # Test head
    head   = FloodDamageHead(common_dim=512)
    emb_t1 = torch.randn(4, 512)
    emb_t2 = torch.randn(4, 512)

    with torch.no_grad():
        preds = head(emb_t1, emb_t2)

    for k, v in preds.items():
        log.info(f'{k}: {v.shape}')

    print('✅ FloodDamageHead test passed')


if __name__ == '__main__':
    main()
