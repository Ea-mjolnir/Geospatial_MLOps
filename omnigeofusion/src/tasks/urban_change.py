"""
OmniGeoFusion — Task A: Urban Change Detection Head
=====================================================
Fine-tuning head for urban change detection.
Takes fused backbone embedding from T1 + T2
and predicts change score + change type.

Target generation (weakly supervised):
  spectral_change  = sqrt(d_ndvi² + d_ndwi² + d_ndbi²)
  lidar_change     = AHN4_nDSM - AHN3_nDSM (height diff)
  sar_coherence    = 1 - SAR_coherence (coherence loss)

  combined_target  = w1*spectral + w2*lidar + w3*sar_coherence
  weights: [0.4, 0.4, 0.2]

Output:
  change_score:  float [0,1] — magnitude of change
  change_type:   int [0-5]   — type of change
    0: no_change
    1: construction    (nDSM increase + spectral change)
    2: demolition      (nDSM decrease + spectral change)
    3: vegetation_loss (NDVI decrease, no height change)
    4: flood_damage    (SAR coherence loss + water index)
    5: road_change     (linear feature change)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from typing import Dict, Tuple, Optional

log = logging.getLogger(__name__)

# Change type labels
CHANGE_TYPES = [
    'no_change',
    'construction',
    'demolition',
    'vegetation_loss',
    'flood_damage',
    'road_change',
]

NUM_CHANGE_TYPES = len(CHANGE_TYPES)


# ── Target Generator ──────────────────────────────────────────
class UrbanChangeTargetGenerator:
    """
    Automatically generates change detection targets
    from multimodal data — no manual labels needed.

    Inputs per patch pair (T1, T2):
      Sentinel-2 bands → spectral change indices
      AHN3/AHN4 nDSM   → 3D height change
      Sentinel-1 SAR   → coherence loss

    Output:
      change_score: float [0, 1]
      change_type:  int [0-5]
    """

    def __init__(
        self,
        spectral_weight: float = 0.4,
        lidar_weight: float = 0.4,
        sar_weight: float = 0.2,
    ):
        self.w_spectral = spectral_weight
        self.w_lidar    = lidar_weight
        self.w_sar      = sar_weight

    def compute_spectral_change(
        self,
        bands_t1: np.ndarray,
        bands_t2: np.ndarray,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        Compute spectral change score from Sentinel-2 bands.
        Same approach as Project 1 (SiamesePrithvi).

        Args:
            bands_t1: [6, H, W] S2 bands at T1
            bands_t2: [6, H, W] S2 bands at T2

        Returns:
            spectral_change: [H, W] float32 [0, 1]
        """
        # Bands: B02=0, B03=1, B04=2, B08=3, B11=4, B8A=5
        blue_t1  = bands_t1[0]; blue_t2  = bands_t2[0]
        green_t1 = bands_t1[1]; green_t2 = bands_t2[1]
        red_t1   = bands_t1[2]; red_t2   = bands_t2[2]
        nir_t1   = bands_t1[3]; nir_t2   = bands_t2[3]
        swir_t1  = bands_t1[4]; swir_t2  = bands_t2[4]

        # NDVI: vegetation health
        ndvi_t1 = (nir_t1 - red_t1) / (nir_t1 + red_t1 + eps)
        ndvi_t2 = (nir_t2 - red_t2) / (nir_t2 + red_t2 + eps)
        d_ndvi  = ndvi_t2 - ndvi_t1

        # NDWI: water content
        ndwi_t1 = (green_t1 - nir_t1) / (green_t1 + nir_t1 + eps)
        ndwi_t2 = (green_t2 - nir_t2) / (green_t2 + nir_t2 + eps)
        d_ndwi  = ndwi_t2 - ndwi_t1

        # NDBI: built-up index
        ndbi_t1 = (swir_t1 - nir_t1) / (swir_t1 + nir_t1 + eps)
        ndbi_t2 = (swir_t2 - nir_t2) / (swir_t2 + nir_t2 + eps)
        d_ndbi  = ndbi_t2 - ndbi_t1

        # Combined change magnitude
        change = np.sqrt(d_ndvi**2 + d_ndwi**2 + d_ndbi**2)

        # Normalize to [0, 1]
        change = np.clip(change / (np.sqrt(3) + eps), 0, 1)

        return change.astype(np.float32)

    def compute_lidar_change(
        self,
        ndsm_ahn3: np.ndarray,
        ndsm_ahn4: np.ndarray,
        nodata: float = -9999.0
    ) -> np.ndarray:
        """
        Compute normalized height change from AHN3/AHN4.

        Returns:
            height_change_norm: [H, W] float32 [0, 1]
            Positive = construction, Negative = demolition
        """
        valid  = (ndsm_ahn3 != nodata) & (ndsm_ahn4 != nodata)
        change = np.where(
            valid, ndsm_ahn4 - ndsm_ahn3, 0.0
        )

        # Normalize: ±20m range → [0, 1]
        # 0.5 = no change, >0.5 = construction, <0.5 = demolition
        change_norm = np.clip(change / 20.0 + 0.5, 0, 1)
        return change_norm.astype(np.float32)

    def compute_combined_target(
        self,
        bands_t1: np.ndarray,
        bands_t2: np.ndarray,
        ndsm_ahn3: Optional[np.ndarray] = None,
        ndsm_ahn4: Optional[np.ndarray] = None,
        sar_coherence: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute combined change target and change type.

        Returns:
            change_score: [H, W] float32 [0, 1]
            change_type:  [H, W] int32 [0-5]
        """
        # Spectral change (always available)
        spectral = self.compute_spectral_change(bands_t1, bands_t2)
        total_weight = self.w_spectral

        # Initialize combined target
        combined = self.w_spectral * spectral

        # LiDAR change (if available)
        lidar_change = None
        if ndsm_ahn3 is not None and ndsm_ahn4 is not None:
            lidar_change  = self.compute_lidar_change(
                ndsm_ahn3, ndsm_ahn4
            )
            combined     += self.w_lidar * np.abs(lidar_change - 0.5) * 2
            total_weight += self.w_lidar

        # SAR coherence loss (if available)
        if sar_coherence is not None:
            sar_change    = 1.0 - sar_coherence
            combined     += self.w_sar * sar_change
            total_weight += self.w_sar

        # Normalize by total weight
        combined = combined / total_weight
        combined = np.clip(combined, 0, 1)

        # Determine change type per pixel
        change_type = self._classify_change_type(
            spectral, lidar_change, sar_coherence, combined
        )

        # Per-patch aggregation (mean score)
        patch_score = float(combined.mean())
        patch_type  = int(np.bincount(
            change_type.flatten()
        ).argmax())

        return patch_score, patch_type

    def _classify_change_type(
        self,
        spectral: np.ndarray,
        lidar_change: Optional[np.ndarray],
        sar_coherence: Optional[np.ndarray],
        combined: np.ndarray,
    ) -> np.ndarray:
        """
        Rule-based change type classification.

        Rules:
          no_change:       combined < 0.2
          construction:    lidar_change > 0.6 (height increase)
          demolition:      lidar_change < 0.4 (height decrease)
          vegetation_loss: ndvi decrease + no height change
          flood_damage:    sar_coherence loss + high ndwi
          road_change:     linear spectral change, no height
        """
        H, W       = combined.shape
        change_map = np.zeros((H, W), dtype=np.int32)

        # No change threshold
        no_change_mask = combined < 0.2
        change_map[no_change_mask] = 0  # no_change

        # Significant change
        sig_mask = ~no_change_mask

        if lidar_change is not None:
            # Construction: height increased
            construction = sig_mask & (lidar_change > 0.6)
            change_map[construction] = 1

            # Demolition: height decreased
            demolition = sig_mask & (lidar_change < 0.4)
            change_map[demolition] = 2

        if sar_coherence is not None:
            # Flood: low SAR coherence
            flood = sig_mask & (sar_coherence < 0.3)
            change_map[flood] = 4

        # Remaining significant change = vegetation/road
        unclassified = sig_mask & (change_map == 0)
        change_map[unclassified] = 3  # vegetation_loss default

        return change_map


# ── Urban Change Head ─────────────────────────────────────────
class UrbanChangeHead(nn.Module):
    """
    Task A: Urban Change Detection head.

    Takes T1 + T2 backbone embeddings,
    computes difference representation,
    predicts change score + type.

    Input:  [B, COMMON_DIM] × 2 (T1, T2 embeddings)
    Output:
      change_score: [B, 1]  continuous [0, 1]
      change_type:  [B, 6]  logits for 6 change types
    """

    def __init__(
        self,
        common_dim: int = 512,
        hidden_dims: list = None,
        num_change_types: int = NUM_CHANGE_TYPES,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Change representation:
        # concatenate T1, T2, difference, product
        in_dim = common_dim * 4

        # Shared feature extraction
        layers     = []
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

        # Change score head (regression)
        self.score_head = nn.Sequential(
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Change type head (classification)
        self.type_head = nn.Sequential(
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_change_types)
        )

        log.info(
            f'UrbanChangeHead: {common_dim}d × 2 → '
            f'score + {num_change_types} types'
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
            dict with 'change_score' and 'change_type'
        """
        # Change representation
        diff    = emb_t2 - emb_t1
        product = emb_t1 * emb_t2
        concat  = torch.cat([emb_t1, emb_t2, diff, product], dim=-1)

        # Shared features
        features = self.shared(concat)

        # Predictions
        change_score = self.score_head(features)   # [B, 1]
        change_type  = self.type_head(features)    # [B, 6]

        return {
            'change_score': change_score.squeeze(-1),  # [B]
            'change_type':  change_type,               # [B, 6]
        }


# ── Loss Function ─────────────────────────────────────────────
class UrbanChangeLoss(nn.Module):
    """
    Combined loss for urban change detection.

    L = w_score × MSE(score, target_score)
      + w_type  × CE(type_logits, target_type)
    """

    def __init__(
        self,
        score_weight: float = 0.7,
        type_weight: float = 0.3,
    ):
        super().__init__()
        self.score_weight = score_weight
        self.type_weight  = type_weight
        self.mse          = nn.MSELoss()
        self.ce           = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: {'change_score': [B], 'change_type': [B,6]}
            targets:     {'change_score': [B], 'change_type': [B]}

        Returns:
            Dict with 'loss', 'score_loss', 'type_loss'
        """
        score_loss = self.mse(
            predictions['change_score'],
            targets['change_score']
        )
        type_loss = self.ce(
            predictions['change_type'],
            targets['change_type'].long()
        )

        total = (
            self.score_weight * score_loss +
            self.type_weight  * type_loss
        )

        return {
            'loss':       total,
            'score_loss': score_loss,
            'type_loss':  type_loss,
        }


# ── Metrics ───────────────────────────────────────────────────
def compute_urban_change_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Compute R2 + accuracy metrics."""
    from scipy.stats import pearsonr, spearmanr

    pred_scores = predictions['change_score'].detach().cpu().numpy()
    true_scores = targets['change_score'].detach().cpu().numpy()

    # R2 score
    ss_res = np.sum((true_scores - pred_scores) ** 2)
    ss_tot = np.sum((true_scores - true_scores.mean()) ** 2)
    r2     = 1 - ss_res / (ss_tot + 1e-8)

    # Spearman correlation
    spearman = spearmanr(pred_scores, true_scores)[0]

    # Type accuracy
    pred_types = predictions['change_type'].argmax(dim=-1)
    pred_types = pred_types.detach().cpu().numpy()
    true_types = targets['change_type'].detach().cpu().numpy()
    accuracy   = float((pred_types == true_types).mean())

    return {
        'r2':            float(r2),
        'spearman':      float(spearman),
        'type_accuracy': accuracy,
        'mae':           float(
            np.abs(pred_scores - true_scores).mean()
        ),
    }


# ── Main ──────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO)

    # Test target generator
    generator = UrbanChangeTargetGenerator()

    # Dummy data
    B02 = np.random.randint(100, 3000, (6, 256, 256)).astype(np.float32)
    B12 = np.random.randint(100, 3000, (6, 256, 256)).astype(np.float32)

    score, ctype = generator.compute_combined_target(B02, B12)
    log.info(
        f'Target: score={score:.4f}, '
        f'type={CHANGE_TYPES[ctype]}'
    )

    # Test head
    head = UrbanChangeHead(common_dim=512)
    emb_t1 = torch.randn(4, 512)
    emb_t2 = torch.randn(4, 512)

    with torch.no_grad():
        preds = head(emb_t1, emb_t2)

    log.info(
        f'Predictions: score={preds["change_score"].shape}, '
        f'type={preds["change_type"].shape}'
    )
    print('✅ UrbanChangeHead test passed')


if __name__ == '__main__':
    main()
