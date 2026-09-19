"""
ThermalWatch — Shared Thermal Backbone
========================================
Shared encoder backbone for both wildfire risk
and solar health monitoring tasks.

Architecture:
  ThermalEncoder  → Landsat Band 10 (1 channel)
  OpticalEncoder  → Sentinel-2 (6 channels) via Prithvi
  WeatherEncoder  → ERA5 features (5 values)
  OSMEncoder      → OSM feature vector (7 values)
  CrossModalFusion→ 512d fused embedding

SSL Pre-training objectives:
  1. Cross-modal contrastive (thermal ↔ optical)
  2. Temporal contrastive (T1 ↔ T2)

Usage:
  from src.models.backbone.thermal_backbone import (
      ThermalWatchBackbone
  )
  model = ThermalWatchBackbone(cfg)
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Optional

log = logging.getLogger(__name__)

# ── Embedding dimensions ──────────────────────────────────────
THERMAL_DIM = 512
OPTICAL_DIM = 1024   # Prithvi-EO-2.0-300M output
WEATHER_DIM = 128
OSM_DIM     = 64
FUSED_DIM   = 512


# ── 1. Thermal Encoder ────────────────────────────────────────
class ThermalEncoder(nn.Module):
    """
    Encodes Landsat Band 10 thermal imagery.
    Input:  [B, 1, 224, 224] surface temperature (°C)
    Output: [B, THERMAL_DIM]

    Uses ResNet-18 backbone pretrained on ImageNet.
    First conv layer adapted for 1-channel input.
    """

    def __init__(self, embed_dim: int = THERMAL_DIM):
        super().__init__()
        import torchvision.models as models

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Adapt first conv for 1-channel thermal input
        backbone.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7, stride=2,
            padding=3, bias=False
        )
        # Initialize adapted conv with mean of RGB weights
        with torch.no_grad():
            pretrained_weight = (
                models.resnet18(
                    weights=models.ResNet18_Weights.IMAGENET1K_V1
                ).conv1.weight
            )
            backbone.conv1.weight.copy_(
                pretrained_weight.mean(dim=1, keepdim=True)
            )

        # Remove classification head
        in_features        = backbone.fc.in_features
        backbone.fc        = nn.Identity()
        self.backbone      = backbone
        self.projection    = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.embed_dim = embed_dim
        log.info(
            f'ThermalEncoder: 1 channel → {embed_dim}d'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 224, 224] thermal image
        Returns:
            [B, embed_dim]
        """
        if x.shape[1] != 1:
            raise ValueError(
                f'ThermalEncoder expects 1 channel, '
                f'got {x.shape[1]}'
            )
        feat = self.backbone(x)       # [B, 512]
        return self.projection(feat)  # [B, embed_dim]


# ── 2. Optical Encoder (Prithvi) ──────────────────────────────
class OpticalEncoder(nn.Module):
    """
    Encodes Sentinel-2 optical imagery via Prithvi-EO-2.0-300M.
    Input:  [B, 6, 224, 224] S2 bands
    Output: [B, OPTICAL_DIM]

    Loaded via TerraTorch registry.
    Frozen during SSL pre-training.
    Partially unfrozen (last 4 layers) during fine-tuning.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool     = True,
    ):
        super().__init__()

        try:
            from terratorch.registry import BACKBONE_REGISTRY
            self.encoder = BACKBONE_REGISTRY.build(
                'prithvi_eo_v2_300',
                pretrained=pretrained,
                num_frames=1,
                in_chans=6,
            )
            log.info(
                f'OpticalEncoder: Prithvi-EO-2.0-300M loaded'
            )
        except Exception as e:
            raise RuntimeError(
                f'Failed to load Prithvi via TerraTorch: {e}\n'
                f'Install: pip install terratorch'
            )

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            log.info('OpticalEncoder: all layers frozen')

        self.embed_dim = OPTICAL_DIM

    def set_phase(self, phase: str):
        """
        Control freezing per training phase.
        phase='ssl':      freeze all
        phase='finetune': unfreeze last 4 layers
        """
        if phase == 'ssl':
            for param in self.encoder.parameters():
                param.requires_grad = False
            log.info('OpticalEncoder: all layers frozen')

        elif phase == 'finetune':
            # Freeze all first
            for param in self.encoder.parameters():
                param.requires_grad = False

            # Unfreeze last 4 transformer blocks
            if hasattr(self.encoder, 'blocks'):
                blocks = list(self.encoder.blocks)
                n      = len(blocks)
                for block in blocks[n-4:]:
                    for param in block.parameters():
                        param.requires_grad = True
                log.info(
                    f'OpticalEncoder: unfroze last 4/{n} layers'
                )
        else:
            raise ValueError(
                f'Unknown phase: {phase}. '
                f'Use "ssl" or "finetune".'
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 6, 224, 224] Sentinel-2 bands
        Returns:
            [B, OPTICAL_DIM]

        Prithvi expects [B, C, T, H, W] — adds T=1 dim.
        """
        if x.shape[1] != 6:
            raise ValueError(
                f'OpticalEncoder expects 6 channels, '
                f'got {x.shape[1]}'
            )
        # Add time dimension: [B, 6, 224, 224] → [B, 6, 1, 224, 224]
        x    = x.unsqueeze(2)
        out  = self.encoder(x)
        feat = out[-1] if isinstance(out, (list, tuple)) \
               else out
        # Pool patch tokens → [B, embed_dim]
        if feat.dim() == 3:
            return feat[:, 1:, :].mean(dim=1)
        elif feat.dim() == 4:
            return feat.mean(dim=[2, 3])
        else:
            raise ValueError(
                f'Unexpected Prithvi output shape: {feat.shape}'
            )


# ── 3. Weather Encoder ────────────────────────────────────────
class WeatherEncoder(nn.Module):
    """
    Encodes ERA5 weather features.
    Input:  [B, 5] (t2m, u10, v10, humidity, precip)
    Output: [B, WEATHER_DIM]

    Simple MLP — weather features are tabular not spatial.
    """

    def __init__(
        self,
        input_dim: int  = 5,
        embed_dim: int  = WEATHER_DIM,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.embed_dim = embed_dim
        log.info(
            f'WeatherEncoder: {input_dim}d → {embed_dim}d'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 5] weather features
        Returns:
            [B, embed_dim]
        """
        if x.shape[-1] != 5:
            raise ValueError(
                f'WeatherEncoder expects 5 features, '
                f'got {x.shape[-1]}'
            )
        return self.mlp(x)


# ── 4. OSM Encoder ────────────────────────────────────────────
class OSMEncoder(nn.Module):
    """
    Encodes OSM contextual features.
    Input:  [B, 7] (roads, buildings, forest, water,
                    powerlines, residential, farmland)
    Output: [B, OSM_DIM]

    Simple MLP — OSM features are tabular.
    """

    def __init__(
        self,
        input_dim: int = 7,
        embed_dim: int = OSM_DIM,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.embed_dim = embed_dim
        log.info(
            f'OSMEncoder: {input_dim}d → {embed_dim}d'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 7] OSM features
        Returns:
            [B, embed_dim]
        """
        return self.mlp(x)


# ── 5. Modality Projection ────────────────────────────────────
class ModalityProjection(nn.Module):
    """
    Projects each modality embedding to common FUSED_DIM space.
    Enables cross-modal attention.
    """

    def __init__(self, fused_dim: int = FUSED_DIM):
        super().__init__()
        self.thermal_proj = nn.Linear(THERMAL_DIM, fused_dim)
        self.optical_proj = nn.Linear(OPTICAL_DIM, fused_dim)
        self.weather_proj = nn.Linear(WEATHER_DIM, fused_dim)
        self.osm_proj     = nn.Linear(OSM_DIM,     fused_dim)
        self.fused_dim    = fused_dim
        log.info(
            f'ModalityProjection: all → {fused_dim}d'
        )

    def forward(
        self,
        thermal: torch.Tensor,
        optical: torch.Tensor,
        weather: torch.Tensor,
        osm:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Projects + stacks all modalities.
        Returns: [B, 4, fused_dim] (4 modalities)
        """
        t = self.thermal_proj(thermal).unsqueeze(1)
        o = self.optical_proj(optical).unsqueeze(1)
        w = self.weather_proj(weather).unsqueeze(1)
        s = self.osm_proj(osm).unsqueeze(1)
        return torch.cat([t, o, w, s], dim=1)


# ── 6. Cross-Modal Attention ──────────────────────────────────
class CrossModalAttention(nn.Module):
    """
    Fuses 4 modality embeddings via multi-head attention.
    Input:  [B, 4, fused_dim]
    Output: [B, fused_dim]
    """

    def __init__(
        self,
        fused_dim:  int = FUSED_DIM,
        num_heads:  int = 8,
        num_layers: int = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fused_dim,
            nhead=num_heads,
            dim_feedforward=fused_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.fused_dim = fused_dim
        log.info(
            f'CrossModalAttention: 4 modalities × '
            f'{fused_dim}d → {fused_dim}d'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 4, fused_dim]
        Returns:
            [B, fused_dim] (mean pool over modalities)
        """
        out = self.transformer(x)    # [B, 4, fused_dim]
        return out.mean(dim=1)       # [B, fused_dim]


# ── 7. ThermalWatch Backbone ──────────────────────────────────
class ThermalWatchBackbone(nn.Module):
    """
    Full shared backbone for ThermalWatch.

    Encodes thermal + optical + weather + OSM
    into a single 512d fused embedding.

    Used for both:
      - Wildfire risk prediction
      - Solar farm health monitoring

    Training phases:
      ssl:      optical encoder frozen
      finetune: optical last 4 layers unfrozen

    Parameter counts (approximate):
      ThermalEncoder:   11.4M (ResNet-18 + proj)
      OpticalEncoder:  303.9M (Prithvi, mostly frozen)
      WeatherEncoder:    0.01M
      OSMEncoder:        0.01M
      Projection:        2.1M
      CrossModalAttn:    4.2M
      TOTAL:           ~321M (304M frozen optical)
    """

    def __init__(
        self,
        pretrained_optical: bool = True,
        freeze_optical:     bool = True,
        fused_dim:          int  = FUSED_DIM,
        num_heads:          int  = 8,
        num_layers:         int  = 2,
    ):
        super().__init__()

        self.thermal_encoder = ThermalEncoder(THERMAL_DIM)
        self.optical_encoder = OpticalEncoder(
            pretrained=pretrained_optical,
            freeze=freeze_optical,
        )
        self.weather_encoder = WeatherEncoder(5, WEATHER_DIM)
        self.osm_encoder     = OSMEncoder(7, OSM_DIM)
        self.projection      = ModalityProjection(fused_dim)
        self.fusion          = CrossModalAttention(
            fused_dim, num_heads, num_layers
        )

        self._log_params()

    def _log_params(self):
        def count(m):
            total     = sum(p.numel() for p in m.parameters())
            trainable = sum(
                p.numel() for p in m.parameters()
                if p.requires_grad
            )
            return total, trainable

        log.info('ThermalWatchBackbone parameter counts:')
        for name, module in [
            ('thermal',  self.thermal_encoder),
            ('optical',  self.optical_encoder),
            ('weather',  self.weather_encoder),
            ('osm',      self.osm_encoder),
            ('fusion',   self.fusion),
        ]:
            total, train = count(module)
            log.info(
                f'  {name:<10}: '
                f'{total/1e6:.1f}M total, '
                f'{train/1e6:.1f}M trainable'
            )

        total, train = count(self)
        log.info(
            f'  {"TOTAL":<10}: '
            f'{total/1e6:.1f}M total, '
            f'{train/1e6:.1f}M trainable'
        )

    def set_phase(self, phase: str):
        """
        Set training phase.
        phase='ssl':      freeze optical encoder
        phase='finetune': unfreeze last 4 optical layers
        """
        self.optical_encoder.set_phase(phase)
        log.info(f'Backbone phase: {phase}')

    def forward(
        self,
        thermal: torch.Tensor,
        optical: Optional[torch.Tensor] = None,
        weather: Optional[torch.Tensor] = None,
        osm:     Optional[torch.Tensor] = None,
        use_precomputed_optical: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through all encoders + fusion.

        Args:
            thermal: [B, 1, 224, 224] required
            optical: [B, 6, 224, 224] raw S2 OR
                     [B, 1024] pre-computed Prithvi embedding
            weather: [B, 5]           optional
            osm:     [B, 7]           optional
            use_precomputed_optical: if True, optical is
                     already a [B, 1024] embedding —
                     skip OpticalEncoder forward pass

        Missing modalities replaced with zeros.

        Returns:
            [B, FUSED_DIM] fused embedding
        """
        B      = thermal.shape[0]
        device = thermal.device

        if thermal.shape[1] != 1:
            raise ValueError(
                f'thermal must have 1 channel, '
                f'got {thermal.shape[1]}'
            )

        # Encode thermal (always present)
        t_emb = self.thermal_encoder(thermal)

        # Encode optical (zeros if missing)
        if optical is not None:
            if use_precomputed_optical:
                # Already a [B, 1024] Prithvi embedding
                o_emb = optical
            else:
                # Raw S2 [B, 6, 224, 224] → run Prithvi
                o_emb = self.optical_encoder(optical)
        else:
            o_emb = torch.zeros(
                B, OPTICAL_DIM, device=device
            )

        # Encode weather (zeros if missing)
        if weather is not None:
            w_emb = self.weather_encoder(weather)
        else:
            w_emb = torch.zeros(
                B, WEATHER_DIM, device=device
            )

        # Encode OSM (zeros if missing)
        if osm is not None:
            s_emb = self.osm_encoder(osm)
        else:
            s_emb = torch.zeros(
                B, OSM_DIM, device=device
            )

        # Project to common space + fuse
        projected = self.projection(
            t_emb, o_emb, w_emb, s_emb
        )                              # [B, 4, FUSED_DIM]
        fused = self.fusion(projected) # [B, FUSED_DIM]

        return fused
