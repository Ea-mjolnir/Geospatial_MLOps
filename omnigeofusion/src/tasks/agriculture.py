"""
OmniGeoFusion — Task C: Precision Agriculture Head
===================================================
Fine-tuning head for precision agriculture monitoring.

Target generation (weakly supervised):
  NDVI:         from Sentinel-2 bands (vegetation health)
  LAI:          Leaf Area Index from S2 biophysical processor
  crop_height:  AHN4 DSM - AHN4 DTM (height above ground)
  soil_moisture: from KNMI + WUR IoT sensors
  CWSI:         Crop Water Stress Index from thermal + IoT
  ET:           Evapotranspiration from thermal + NDVI

All targets derived automatically from data physics.
No manual field survey labels needed.

Output:
  ndvi:          float [-1, 1]  vegetation health
  lai:           float [0, 10]  leaf area index
  crop_height_m: float [0, 5]   crop height in meters
  soil_moisture: float [0, 1]   volumetric water content
  cwsi:          float [0, 1]   water stress index
  et_mm_day:     float [0, 15]  evapotranspiration
  stress_flag:   int [0-3]      stress type
    0: healthy
    1: water_stress   (CWSI > 0.6)
    2: nutrient_stress (low NDVI + normal CWSI)
    3: pest_stress    (patchy NDVI decline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from typing import Dict, Tuple, Optional

log = logging.getLogger(__name__)

# Stress type labels
STRESS_TYPES = [
    'healthy',
    'water_stress',
    'nutrient_stress',
    'pest_stress',
]
NUM_STRESS_TYPES = len(STRESS_TYPES)


# ── Target Generator ──────────────────────────────────────────
class AgricultureTargetGenerator:
    """
    Automatically generates precision agriculture targets
    from multimodal data — no manual labels needed.

    Data sources used:
      Sentinel-2  → NDVI, LAI, spectral indices
      AHN4 LiDAR  → crop height (DSM - DTM)
      IoT sensors → soil moisture, temperature
      Thermal     → CWSI, ET
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def compute_ndvi(
        self,
        bands: np.ndarray
    ) -> np.ndarray:
        """
        Compute NDVI from Sentinel-2 bands.
        NDVI = (NIR - RED) / (NIR + RED)

        Args:
            bands: [6, H, W] S2 bands
                   [B02, B03, B04, B08, B11, B8A]

        Returns:
            ndvi: [H, W] float32 [-1, 1]
        """
        red = bands[2]  # B04
        nir = bands[3]  # B08
        ndvi = (nir - red) / (nir + red + self.eps)
        return np.clip(ndvi, -1, 1).astype(np.float32)

    def compute_lai(
        self,
        bands: np.ndarray
    ) -> np.ndarray:
        """
        Estimate Leaf Area Index (LAI) from Sentinel-2.
        Simplified biophysical processor approach.

        LAI correlates strongly with:
          - NIR reflectance (B08)
          - Red edge (B8A)
          - NDVI

        Args:
            bands: [6, H, W] S2 bands

        Returns:
            lai: [H, W] float32 [0, 10]
        """
        red  = bands[2]   # B04
        nir  = bands[3]   # B08
        re   = bands[5]   # B8A (red edge)

        ndvi = (nir - red) / (nir + red + self.eps)

        # Simplified LAI formula (Baret et al. 2007)
        # LAI = -ln((0.69 - NDVI) / 0.59) / 0.91
        # Clipped to valid range
        lai = np.where(
            ndvi < 0.69,
            -np.log(
                np.clip((0.69 - ndvi) / 0.59, 1e-6, 1)
            ) / 0.91,
            8.0  # max LAI for dense canopy
        )
        return np.clip(lai, 0, 10).astype(np.float32)

    def compute_evi(
        self,
        bands: np.ndarray
    ) -> np.ndarray:
        """
        Enhanced Vegetation Index — less saturated than NDVI
        for dense canopies.

        EVI = 2.5 × (NIR - RED) / (NIR + 6×RED - 7.5×BLUE + 1)
        """
        blue = bands[0]  # B02
        red  = bands[2]  # B04
        nir  = bands[3]  # B08

        evi = 2.5 * (nir - red) / (
            nir + 6*red - 7.5*blue + 1 + self.eps
        )
        return np.clip(evi, -1, 1).astype(np.float32)

    def compute_crop_height(
        self,
        dsm: np.ndarray,
        dtm: np.ndarray,
        nodata: float = -9999.0
    ) -> np.ndarray:
        """
        Compute crop height from AHN4 LiDAR.
        crop_height = DSM - DTM (height above ground)

        During growing season:
          Wheat:   0.6-1.2m
          Maize:   1.5-3.0m
          Potato:  0.4-0.8m
          Sugar beet: 0.4-0.6m

        Args:
            dsm: [H, W] Digital Surface Model
            dtm: [H, W] Digital Terrain Model

        Returns:
            crop_height: [H, W] float32 (meters) [0, 5]
        """
        valid  = (dsm != nodata) & (dtm != nodata)
        height = np.where(
            valid,
            np.maximum(0, dsm - dtm),
            0.0
        )
        # Cap at 5m (max crop height)
        return np.clip(height, 0, 5).astype(np.float32)

    def compute_soil_moisture_from_sar(
        self,
        sar_vv: np.ndarray,
        sar_vh: np.ndarray,
        crop_height: np.ndarray
    ) -> np.ndarray:
        """
        Estimate soil moisture from SAR backscatter.
        SAR backscatter increases with soil moisture.

        Uses simplified Water Cloud Model (WCM):
          sigma = A × VWC × cos(theta) × (1 - exp(-2B×VWC/cos(theta)))
                + exp(-2B×VWC/cos(theta)) × sigma_soil

        Simplified version uses empirical relationship:
          soil_moisture ∝ VV - VH ratio (vegetation corrected)

        Returns:
            soil_moisture: [H, W] float32 [0, 1]
        """
        eps = self.eps

        # VV/VH ratio sensitive to soil moisture
        # after vegetation correction
        vv_db = 10 * np.log10(np.clip(sar_vv, eps, None))
        vh_db = 10 * np.log10(np.clip(sar_vh, eps, None))

        # Cross-polarization ratio
        ratio = vv_db - vh_db

        # Vegetation correction using crop height
        veg_correction = np.clip(crop_height / 3.0, 0, 1)
        corrected      = ratio * (1 - 0.5 * veg_correction)

        # Normalize to [0, 1]
        # Typical range: -5 to -15 dB
        sm = np.clip((corrected + 15) / 10, 0, 1)
        return sm.astype(np.float32)

    def compute_cwsi_from_thermal(
        self,
        lst: np.ndarray,
        air_temp_c: float,
        wind_speed_ms: float,
        humidity_pct: float
    ) -> np.ndarray:
        """
        Crop Water Stress Index from thermal LST.
        CWSI = (LST - LST_wet) / (LST_dry - LST_wet)

        Returns:
            cwsi: [H, W] float32 [0, 1]
        """
        vpd     = (1 - humidity_pct/100) * 0.611 * np.exp(
            17.27 * air_temp_c / (air_temp_c + 237.3)
        )
        lst_wet = air_temp_c - 5.0 - wind_speed_ms * 0.5
        lst_dry = air_temp_c + 5.0 + vpd * 2.0

        valid = ~np.isnan(lst)
        cwsi  = np.where(
            valid,
            (lst - lst_wet) / (lst_dry - lst_wet + self.eps),
            0.5  # neutral default
        )
        return np.clip(cwsi, 0, 1).astype(np.float32)

    def classify_stress(
        self,
        ndvi: np.ndarray,
        cwsi: np.ndarray,
        ndvi_threshold: float = 0.4,
        cwsi_threshold: float = 0.6,
    ) -> int:
        """
        Classify crop stress type from NDVI + CWSI.

        Rules:
          healthy:          ndvi > 0.4, cwsi < 0.6
          water_stress:     ndvi > 0.4, cwsi > 0.6
          nutrient_stress:  ndvi < 0.4, cwsi < 0.6
          pest_stress:      ndvi < 0.4, cwsi < 0.6 + patchy

        Returns:
            stress_type: int [0-3]
        """
        mean_ndvi = float(ndvi.mean())
        mean_cwsi = float(cwsi.mean())

        # Spatial variance of NDVI (patchy = pest)
        ndvi_std = float(ndvi.std())

        if mean_ndvi > ndvi_threshold and mean_cwsi < cwsi_threshold:
            return 0  # healthy
        elif mean_cwsi >= cwsi_threshold:
            return 1  # water_stress
        elif ndvi_std > 0.15:
            return 3  # pest_stress (patchy)
        else:
            return 2  # nutrient_stress

    def generate_targets(
        self,
        bands: np.ndarray,
        dsm: Optional[np.ndarray] = None,
        dtm: Optional[np.ndarray] = None,
        sar_vv: Optional[np.ndarray] = None,
        sar_vh: Optional[np.ndarray] = None,
        lst: Optional[np.ndarray] = None,
        iot_features: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate all agriculture targets for one patch.

        Returns:
            Dict with all agriculture targets
        """
        # Core spectral indices (always available)
        ndvi = self.compute_ndvi(bands)
        lai  = self.compute_lai(bands)
        evi  = self.compute_evi(bands)

        # Crop height from LiDAR
        crop_height = np.zeros_like(ndvi)
        if dsm is not None and dtm is not None:
            crop_height = self.compute_crop_height(dsm, dtm)

        # Soil moisture from SAR
        soil_moisture = np.full_like(ndvi, 0.3)
        if sar_vv is not None and sar_vh is not None:
            soil_moisture = self.compute_soil_moisture_from_sar(
                sar_vv, sar_vh, crop_height
            )

        # Override with IoT if available
        if iot_features:
            iot_sm = iot_features.get('soil_moisture', None)
            if iot_sm is not None:
                soil_moisture = np.full_like(ndvi, iot_sm)

        # CWSI from thermal + IoT
        cwsi = np.full_like(ndvi, 0.3)
        if lst is not None and iot_features:
            cwsi = self.compute_cwsi_from_thermal(
                lst,
                air_temp_c=iot_features.get('air_temp_c', 15.0),
                wind_speed_ms=iot_features.get('wind_speed_ms', 3.0),
                humidity_pct=iot_features.get('humidity_pct', 80.0)
            )

        # ET from thermal + NDVI
        et_mm_day = 3.0  # default Netherlands mean
        if lst is not None and iot_features:
            # Simplified ET: healthy crop × available energy
            et_fraction = (1 - cwsi.mean()) * ndvi.mean()
            et_mm_day   = float(et_fraction * 8.0)

        # Stress classification
        stress_type = self.classify_stress(ndvi, cwsi)

        return {
            'ndvi':            float(ndvi.mean()),
            'lai':             float(lai.mean()),
            'evi':             float(evi.mean()),
            'crop_height_m':   float(crop_height.mean()),
            'soil_moisture':   float(soil_moisture.mean()),
            'cwsi':            float(cwsi.mean()),
            'et_mm_day':       et_mm_day,
            'stress_type':     stress_type,
            # Spatial maps for visualization
            'ndvi_map':        ndvi,
            'cwsi_map':        cwsi,
            'crop_height_map': crop_height,
        }


# ── Agriculture Head ──────────────────────────────────────────
class AgricultureHead(nn.Module):
    """
    Task C: Precision Agriculture monitoring head.

    Takes backbone embedding and predicts
    all agriculture indicators simultaneously.

    Input:  [B, COMMON_DIM] fused embedding
    Output: dict of agriculture predictions
    """

    def __init__(
        self,
        common_dim: int = 512,
        hidden_dims: list = None,
        num_stress_types: int = NUM_STRESS_TYPES,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Shared feature extraction
        layers      = []
        current_dim = common_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(current_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            current_dim = h_dim
        self.shared = nn.Sequential(*layers)

        # NDVI head [-1, 1]
        self.ndvi_head = nn.Sequential(
            nn.Linear(current_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        # LAI head [0, 10]
        self.lai_head = nn.Sequential(
            nn.Linear(current_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

        # Crop height head [0, 5]
        self.height_head = nn.Sequential(
            nn.Linear(current_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

        # Soil moisture head [0, 1]
        self.soil_head = nn.Sequential(
            nn.Linear(current_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # CWSI head [0, 1]
        self.cwsi_head = nn.Sequential(
            nn.Linear(current_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # ET head [0, 15]
        self.et_head = nn.Sequential(
            nn.Linear(current_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

        # Stress type head (classification)
        self.stress_head = nn.Sequential(
            nn.Linear(current_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_stress_types)
        )

        log.info(
            f'AgricultureHead: {common_dim}d → '
            f'ndvi + lai + height + soil + cwsi + et + stress'
        )

    def forward(
        self,
        embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embedding: [B, COMMON_DIM] fused backbone embedding

        Returns:
            dict with all agriculture predictions
        """
        features = self.shared(embedding)

        ndvi   = self.ndvi_head(features).squeeze(-1)
        lai    = torch.clamp(
            self.lai_head(features).squeeze(-1), 0, 10
        )
        height = torch.clamp(
            self.height_head(features).squeeze(-1), 0, 5
        )
        soil   = self.soil_head(features).squeeze(-1)
        cwsi   = self.cwsi_head(features).squeeze(-1)
        et     = torch.clamp(
            self.et_head(features).squeeze(-1), 0, 15
        )
        stress = self.stress_head(features)

        return {
            'ndvi':          ndvi,    # [B]
            'lai':           lai,     # [B]
            'crop_height_m': height,  # [B]
            'soil_moisture': soil,    # [B]
            'cwsi':          cwsi,    # [B]
            'et_mm_day':     et,      # [B]
            'stress_type':   stress,  # [B, 4]
        }


# ── Loss Function ─────────────────────────────────────────────
class AgricultureLoss(nn.Module):
    """
    Combined multi-output loss for agriculture.

    L = w1×MSE(ndvi) + w2×MSE(lai) + w3×MSE(height)
      + w4×MSE(soil) + w5×MSE(cwsi) + w6×MSE(et)
      + w7×CE(stress)
    """

    def __init__(
        self,
        ndvi_weight:   float = 0.25,
        lai_weight:    float = 0.15,
        height_weight: float = 0.15,
        soil_weight:   float = 0.15,
        cwsi_weight:   float = 0.15,
        et_weight:     float = 0.10,
        stress_weight: float = 0.05,
    ):
        super().__init__()
        self.w_ndvi   = ndvi_weight
        self.w_lai    = lai_weight
        self.w_height = height_weight
        self.w_soil   = soil_weight
        self.w_cwsi   = cwsi_weight
        self.w_et     = et_weight
        self.w_stress = stress_weight

        self.mse = nn.MSELoss()
        self.ce  = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute combined agriculture loss."""
        losses = {}

        losses['ndvi_loss']   = self.mse(
            predictions['ndvi'],
            targets['ndvi'].float()
        )
        losses['lai_loss']    = self.mse(
            predictions['lai'],
            targets['lai'].float()
        )
        losses['height_loss'] = self.mse(
            predictions['crop_height_m'],
            targets['crop_height_m'].float()
        )
        losses['soil_loss']   = self.mse(
            predictions['soil_moisture'],
            targets['soil_moisture'].float()
        )
        losses['cwsi_loss']   = self.mse(
            predictions['cwsi'],
            targets['cwsi'].float()
        )
        losses['et_loss']     = self.mse(
            predictions['et_mm_day'],
            targets['et_mm_day'].float()
        )
        losses['stress_loss'] = self.ce(
            predictions['stress_type'],
            targets['stress_type'].long()
        )

        total = (
            self.w_ndvi   * losses['ndvi_loss']   +
            self.w_lai    * losses['lai_loss']     +
            self.w_height * losses['height_loss']  +
            self.w_soil   * losses['soil_loss']    +
            self.w_cwsi   * losses['cwsi_loss']    +
            self.w_et     * losses['et_loss']      +
            self.w_stress * losses['stress_loss']
        )
        losses['loss'] = total
        return losses


# ── Metrics ───────────────────────────────────────────────────
def compute_agriculture_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Compute R2 for each agriculture output."""
    metrics = {}

    for key in ['ndvi', 'lai', 'crop_height_m',
                'soil_moisture', 'cwsi', 'et_mm_day']:
        if key not in targets:
            continue
        pred = predictions[key].detach().cpu().numpy()
        true = targets[key].detach().cpu().numpy()

        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2     = 1 - ss_res / (ss_tot + 1e-8)
        mae    = float(np.abs(pred - true).mean())

        metrics[f'{key}_r2']  = float(r2)
        metrics[f'{key}_mae'] = mae

    # Stress accuracy
    if 'stress_type' in targets:
        pred_stress = predictions['stress_type'].argmax(
            dim=-1
        ).detach().cpu().numpy()
        true_stress = targets['stress_type'].detach().cpu().numpy()
        metrics['stress_accuracy'] = float(
            (pred_stress == true_stress).mean()
        )

    # Overall R2 (mean across outputs)
    r2_vals = [v for k, v in metrics.items() if k.endswith('_r2')]
    if r2_vals:
        metrics['mean_r2'] = float(np.mean(r2_vals))

    return metrics


# ── Main ──────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO)

    # Test target generator
    generator = AgricultureTargetGenerator()

    H, W  = 256, 256
    bands = np.random.randint(
        100, 3000, (6, H, W)
    ).astype(np.float32)
    dsm   = np.random.uniform(0.5, 2.0, (H, W))
    dtm   = np.random.uniform(-0.5, 0.5, (H, W))

    iot = {
        'air_temp_c':    18.5,
        'wind_speed_ms': 3.2,
        'humidity_pct':  75.0,
        'soil_moisture': 0.35,
    }

    targets = generator.generate_targets(
        bands, dsm, dtm,
        iot_features=iot
    )

    log.info(f'NDVI:         {targets["ndvi"]:.4f}')
    log.info(f'LAI:          {targets["lai"]:.4f}')
    log.info(f'Crop height:  {targets["crop_height_m"]:.2f}m')
    log.info(f'Soil moisture:{targets["soil_moisture"]:.3f}')
    log.info(f'CWSI:         {targets["cwsi"]:.3f}')
    log.info(
        f'Stress:       {STRESS_TYPES[targets["stress_type"]]}'
    )

    # Test head
    head      = AgricultureHead(common_dim=512)
    embedding = torch.randn(4, 512)

    with torch.no_grad():
        preds = head(embedding)

    for k, v in preds.items():
        log.info(f'{k}: {v.shape}')

    print('✅ AgricultureHead test passed')


if __name__ == '__main__':
    main()
