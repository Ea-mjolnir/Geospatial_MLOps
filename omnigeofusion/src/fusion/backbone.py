"""
OmniGeoFusion — Cross-Modal Fusion Backbone
=============================================
The core multimodal fusion architecture that combines
all 6 data sources into a unified representation.

Architecture:
  1. Per-modality encoders (independent streams)
  2. Modality projection (→ common dimension 512)
  3. Cross-modal attention transformer (fusion)
  4. Shared latent representation

Design decisions:
  - Prithvi v2_300M as optical encoder (frozen in SSL)
  - SAR-ViT for Sentinel-1 (trained from scratch)
  - PointNet++ for LiDAR point clouds
  - ResNet18 for thermal (ImageNet pretrained)
  - GraphSAGE for OSM vector data
  - Temporal LSTM for IoT sensor sequences
  - Modality dropout (30%) for graceful degradation
  - Common dimension: 512 (fits T4 VRAM budget)
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────
COMMON_DIM        = 512     # unified embedding dimension
OPTICAL_DIM       = 1024    # prithvi_v2_300M output dim
SAR_DIM           = 768     # SAR-ViT output dim
LIDAR_DIM         = 512     # PointNet++ output dim
THERMAL_DIM       = 512     # ResNet18 output dim
OSM_DIM           = 256     # GraphSAGE output dim
IOT_DIM           = 256     # LSTM output dim

NUM_MODALITIES    = 6
MODALITY_NAMES    = [
    'optical', 'sar', 'lidar',
    'thermal', 'osm', 'iot'
]


# ── Optical Encoder (Prithvi v2_300M) ─────────────────────────
class OpticalEncoder(nn.Module):
    """
    Prithvi-EO-2.0-300M encoder for Sentinel-2 optical data.
    Loaded via TerraTorch BACKBONE_REGISTRY.

    Input:  [B, 6, 224, 224] — 6 bands, 224×224 patch
    Output: [B, 1024]        — patch embedding

    Strategy:
      - Fully frozen during SSL pre-training
      - Last 4 layers unfrozen during fine-tuning
      - Mean pool patch tokens → [B, 1024]
    """

    def __init__(
        self,
        model_name: str = 'prithvi_eo_v2_300',
        pretrained: bool = True,
        freeze_all: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_all = freeze_all

        try:
            import os
            os.environ['MPLBACKEND'] = 'agg'
            from terratorch.registry import BACKBONE_REGISTRY
            self.encoder = BACKBONE_REGISTRY.build(
                model_name,
                pretrained=pretrained,
                num_frames=1,
                in_chans=6,
            )
            log.info(
                f'✅ Loaded {model_name} via TerraTorch'
            )
        except Exception as e:
            log.warning(
                f'TerraTorch not available: {e} '
                f'— using stub encoder'
            )
            self.encoder = self._build_stub_encoder()

        if freeze_all:
            self.freeze()

    def _build_stub_encoder(self) -> nn.Module:
        """Stub encoder for testing without TerraTorch."""
        return nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, OPTICAL_DIM)
        )

    def freeze(self):
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        log.info('OpticalEncoder: all layers frozen')

    def unfreeze_last_n_layers(self, n: int = 4):
        """Unfreeze last N transformer layers for fine-tuning."""
        # Unfreeze all first (reset)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Find transformer blocks
        blocks = None
        if hasattr(self.encoder, 'blocks'):
            blocks = self.encoder.blocks
        elif hasattr(self.encoder, 'encoder'):
            if hasattr(self.encoder.encoder, 'blocks'):
                blocks = self.encoder.encoder.blocks

        if blocks is not None:
            total = len(blocks)
            for i, block in enumerate(blocks):
                if i >= total - n:
                    for param in block.parameters():
                        param.requires_grad = True
            log.info(
                f'OpticalEncoder: unfroze last {n}/{total} layers'
            )
        else:
            # Fallback: unfreeze last 25% of parameters
            params = list(self.encoder.parameters())
            n_unfreeze = max(1, len(params) // 4)
            for param in params[-n_unfreeze:]:
                param.requires_grad = True
            log.info(
                f'OpticalEncoder: unfroze last '
                f'{n_unfreeze} parameters'
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 6, H, W] Sentinel-2 patch

        Returns:
            embedding: [B, OPTICAL_DIM]
        """
        out = self.encoder(x)

        # Handle different output formats
        if isinstance(out, (list, tuple)):
            feat = out[-1]  # last layer features
        else:
            feat = out

        # Mean pool if spatial output
        if feat.dim() == 4:
            # [B, C, H, W] → [B, C]
            embedding = feat.mean(dim=[2, 3])
        elif feat.dim() == 3:
            # [B, N, C] → [B, C] (ViT patch tokens)
            embedding = feat[:, 1:, :].mean(dim=1)
        else:
            embedding = feat

        return embedding  # [B, OPTICAL_DIM]


# ── SAR Encoder (SAR-ViT) ─────────────────────────────────────
class SAREncoder(nn.Module):
    """
    Vision Transformer encoder for Sentinel-1 SAR data.
    Trained from scratch (no public SAR pretrained weights).

    Input:  [B, 2, 224, 224] — VV + VH polarization
    Output: [B, SAR_DIM]     — SAR embedding

    Why ViT for SAR:
      SAR speckle noise → ViT attention learns to
      distinguish real structures from noise
      Global receptive field → captures coherence patterns
    """

    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = SAR_DIM,
        patch_size: int = 16,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim  = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # CLS token + position embedding
        self.cls_token   = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )
        self.pos_embed   = nn.Parameter(
            torch.zeros(1, 197, embed_dim)  # 196 patches + CLS
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        log.info(
            f'SAREncoder: {in_channels} channels → '
            f'{embed_dim}d embedding'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 2, H, W] SAR VV+VH patch

        Returns:
            embedding: [B, SAR_DIM]
        """
        B = x.shape[0]

        # Patch embedding: [B, embed_dim, H/P, W/P]
        x = self.patch_embed(x)

        # Flatten patches: [B, N, embed_dim]
        x = x.flatten(2).transpose(1, 2)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        # Add position embedding
        n_patches = x.shape[1]
        pos_embed = self.pos_embed[:, :n_patches, :]
        x         = x + pos_embed

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # CLS token as embedding
        return x[:, 0, :]  # [B, SAR_DIM]


# ── LiDAR Encoder (PointNet++) ────────────────────────────────
class LiDAREncoder(nn.Module):
    """
    Simplified PointNet++ encoder for LiDAR elevation data.
    Operates on rasterized DSM/DTM/nDSM products
    (not raw point clouds — for efficiency).

    Input:  [B, 3, H, W] — DSM, DTM, nDSM rasters
    Output: [B, LIDAR_DIM] — 3D structure embedding

    Note: We use rasterized LiDAR (0.5m → 10m resampled)
    rather than raw point clouds for three reasons:
      1. Memory efficiency on T4 GPU
      2. Direct spatial alignment with Sentinel patches
      3. AHN4 provides high-quality 0.5m rasters
    """

    def __init__(
        self,
        in_channels: int = 3,   # DSM, DTM, nDSM
        embed_dim: int = LIDAR_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Multi-scale feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Height-aware attention
        self.height_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.Sigmoid()
        )

        # Global pooling + projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

        log.info(
            f'LiDAREncoder: {in_channels} channels → '
            f'{embed_dim}d embedding'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] LiDAR raster (DSM, DTM, nDSM)

        Returns:
            embedding: [B, LIDAR_DIM]
        """
        # Multi-scale features
        f1 = self.conv1(x)   # [B, 128, H, W]
        f2 = self.conv2(f1)  # [B, 512, H/2, W/2]

        # Height-aware attention
        attn = self.height_attn(f2).unsqueeze(-1).unsqueeze(-1)
        f2   = f2 * attn

        # Global pooling
        out = self.global_pool(f2)      # [B, 512, 1, 1]
        return self.projection(out)     # [B, LIDAR_DIM]


# ── Thermal Encoder (ResNet18) ────────────────────────────────
class ThermalEncoder(nn.Module):
    """
    ResNet18-based encoder for thermal infrared data.
    Uses ImageNet pretrained weights (transfer learning).

    Input:  [B, 1, H, W] — LST raster (°C)
    Output: [B, THERMAL_DIM] — thermal embedding

    Why ResNet18:
      Thermal imagery shares textures with RGB images
      ImageNet pretraining provides good initialization
      Lightweight (11M params) → minimal VRAM
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = THERMAL_DIM,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        try:
            from torchvision.models import resnet18, ResNet18_Weights
            if pretrained:
                backbone = resnet18(
                    weights=ResNet18_Weights.IMAGENET1K_V1
                )
            else:
                backbone = resnet18(weights=None)
        except ImportError:
            log.warning(
                'torchvision not available — using simple CNN'
            )
            backbone = None

        if backbone is not None:
            # Modify first conv for single channel input
            backbone.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2,
                padding=3, bias=False
            )
            # Remove final FC layer
            self.backbone = nn.Sequential(
                *list(backbone.children())[:-1]
            )
            backbone_out = 512
        else:
            # Fallback simple CNN
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            backbone_out = 256

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

        log.info(
            f'ThermalEncoder: {in_channels} channel → '
            f'{embed_dim}d embedding'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] thermal LST patch

        Returns:
            embedding: [B, THERMAL_DIM]
        """
        feat = self.backbone(x)    # [B, 512, 1, 1]
        return self.projection(feat)  # [B, THERMAL_DIM]


# ── OSM Encoder (GraphSAGE) ───────────────────────────────────
class OSMEncoder(nn.Module):
    """
    GraphSAGE encoder for OSM vector data.
    Encodes spatial graph of buildings + roads.

    Input:  node_features [N, 64], edge_index [2, E]
    Output: [B, OSM_DIM] — graph-level embedding

    Why GraphSAGE:
      OSM = graph structure (roads connect buildings)
      GraphSAGE handles variable-size graphs
      Inductive: generalizes to new areas
    """

    def __init__(
        self,
        node_features: int = 64,
        edge_features: int = 32,
        embed_dim: int = OSM_DIM,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers  = num_layers
        self.node_dim    = node_features
        self.embed_dim   = embed_dim

        # Message passing layers (simplified GraphSAGE)
        self.sage_layers = nn.ModuleList()
        in_dim = node_features
        for i in range(num_layers):
            out_dim = embed_dim if i == num_layers-1 else embed_dim
            self.sage_layers.append(
                nn.Sequential(
                    nn.Linear(in_dim * 2, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_dim = out_dim

        # Graph-level readout
        self.readout = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )

        log.info(
            f'OSMEncoder: {node_features}d nodes → '
            f'{embed_dim}d graph embedding'
        )

    def sage_conv(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer: nn.Module
    ) -> torch.Tensor:
        """
        One GraphSAGE convolution step.
        Aggregates neighbor features via mean pooling.
        """
        N = x.shape[0]

        if edge_index.shape[1] == 0:
            # No edges — use self features only
            self_feat = x
            agg_feat  = torch.zeros_like(x)
        else:
            src = edge_index[0]  # source nodes
            dst = edge_index[1]  # destination nodes

            # Aggregate neighbor features (mean)
            agg = torch.zeros_like(x)
            agg.scatter_add_(
                0,
                dst.unsqueeze(-1).expand(-1, x.shape[-1]),
                x[src]
            )
            count = torch.zeros(N, 1, device=x.device)
            count.scatter_add_(
                0,
                dst.unsqueeze(-1),
                torch.ones(len(src), 1, device=x.device)
            )
            agg_feat  = agg / (count + 1e-8)
            self_feat = x

        # Concatenate self + aggregated neighbor
        combined = torch.cat([self_feat, agg_feat], dim=-1)
        return layer(combined)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_features: [N, node_features]
            edge_index:    [2, E]
            batch:         [N] node-to-graph assignment

        Returns:
            embedding: [B, OSM_DIM]
        """
        x = node_features

        # GraphSAGE layers
        for layer in self.sage_layers:
            x = self.sage_conv(x, edge_index, layer)

        # Graph-level readout (mean pool)
        if batch is not None:
            B          = batch.max().item() + 1
            graph_emb  = torch.zeros(
                B, self.embed_dim, device=x.device
            )
            count      = torch.zeros(B, 1, device=x.device)
            graph_emb.scatter_add_(
                0,
                batch.unsqueeze(-1).expand(-1, self.embed_dim),
                x
            )
            count.scatter_add_(
                0, batch.unsqueeze(-1),
                torch.ones(x.shape[0], 1, device=x.device)
            )
            graph_emb = graph_emb / (count + 1e-8)
        else:
            # Single graph — mean all nodes
            graph_emb = x.mean(dim=0, keepdim=True)

        return self.readout(graph_emb)  # [B, OSM_DIM]


# ── IoT Encoder (Temporal LSTM) ───────────────────────────────
class IoTEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for IoT sensor sequences.

    Input:  [B, T, 16] — T time steps × 16 sensor features
    Output: [B, IOT_DIM] — temporal sensor embedding

    Why BiLSTM:
      Sensor readings are temporal sequences
      Bidirectional captures past + future context
      16 features × 12 time steps = 192 inputs
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 2,
        embed_dim: int = IOT_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # BiLSTM output = 2 × hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        log.info(
            f'IoTEncoder: {input_dim}d × T steps → '
            f'{embed_dim}d embedding'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 16] IoT sensor sequence
               or [B, 16] single time step

        Returns:
            embedding: [B, IOT_DIM]
        """
        # Handle single time step
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, 16]

        # BiLSTM
        output, (h_n, _) = self.lstm(x)

        # Concatenate final forward + backward hidden states
        # h_n shape: [num_layers*2, B, hidden_dim]
        h_forward  = h_n[-2]  # last forward layer
        h_backward = h_n[-1]  # last backward layer
        h_concat   = torch.cat([h_forward, h_backward], dim=-1)

        return self.projection(h_concat)  # [B, IOT_DIM]


# ── Modality Projection ───────────────────────────────────────
class ModalityProjection(nn.Module):
    """
    Projects each encoder output to common dimension.
    Allows cross-modal attention to compare embeddings.
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        common_dim: int = COMMON_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, common_dim),
                nn.LayerNorm(common_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for name, dim in input_dims.items()
        })

        log.info(
            f'ModalityProjection: → {common_dim}d common space'
        )

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Project all embeddings to common dimension."""
        return {
            name: self.projections[name](emb)
            for name, emb in embeddings.items()
            if name in self.projections
        }


# ── Cross-Modal Attention Transformer ─────────────────────────
class CrossModalAttention(nn.Module):
    """
    Cross-modal attention transformer for fusing
    embeddings from all 6 modalities.

    Each modality embedding becomes a token.
    Transformer attention learns which modalities
    to attend to for each task.

    With 6 modalities → 6 tokens per sample
    Transformer processes [B, 6, COMMON_DIM]
    """

    def __init__(
        self,
        common_dim: int = COMMON_DIM,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        modality_dropout_prob: float = 0.3,
    ):
        super().__init__()
        self.common_dim            = common_dim
        self.modality_dropout_prob = modality_dropout_prob
        self.num_modalities        = NUM_MODALITIES

        # Modality type embeddings (like token type embeddings in BERT)
        self.modality_embeddings = nn.Embedding(
            NUM_MODALITIES, common_dim
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=common_dim,
            nhead=num_heads,
            dim_feedforward=common_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(common_dim)

        # CLS token for global representation
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, common_dim)
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        log.info(
            f'CrossModalAttention: {NUM_MODALITIES} modalities '
            f'× {common_dim}d → {common_dim}d fused'
        )

    def apply_modality_dropout(
        self,
        tokens: torch.Tensor,
        modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply modality dropout during training.
        Randomly zeros out entire modality tokens.
        Forces model to be robust to missing modalities.

        Args:
            tokens:       [B, M, D] modality tokens
            modality_mask: [B, M] 1=present, 0=dropped

        Returns:
            masked_tokens: [B, M, D]
        """
        mask = modality_mask.unsqueeze(-1).float()
        return tokens * mask

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        training: bool = False
    ) -> torch.Tensor:
        """
        Fuse all modality embeddings via cross-modal attention.

        Args:
            embeddings: dict of {modality_name: [B, COMMON_DIM]}
            training:   apply modality dropout if True

        Returns:
            fused: [B, COMMON_DIM] unified representation
        """
        B = next(iter(embeddings.values())).shape[0]
        device = next(iter(embeddings.values())).device

        tokens     = []
        modality_ids = []
        present_mask = []

        for i, name in enumerate(MODALITY_NAMES):
            if name in embeddings:
                tokens.append(embeddings[name])
                present_mask.append(
                    torch.ones(B, device=device)
                )
            else:
                # Missing modality — use zero embedding
                tokens.append(
                    torch.zeros(B, self.common_dim, device=device)
                )
                present_mask.append(
                    torch.zeros(B, device=device)
                )
            modality_ids.append(i)

        # Stack: [B, M, COMMON_DIM]
        token_stack  = torch.stack(tokens, dim=1)
        mask_stack   = torch.stack(present_mask, dim=1)

        # Add modality type embeddings
        mod_ids = torch.tensor(
            modality_ids, device=device
        ).unsqueeze(0).expand(B, -1)
        token_stack = token_stack + self.modality_embeddings(mod_ids)

        # Modality dropout during training
        if training and self.modality_dropout_prob > 0:
            dropout_mask = (
                torch.rand(B, len(MODALITY_NAMES), device=device)
                > self.modality_dropout_prob
            ).float()
            # Always keep at least one modality
            dropout_mask = torch.where(
                dropout_mask.sum(dim=1, keepdim=True) == 0,
                torch.ones_like(dropout_mask),
                dropout_mask
            )
            # Combine with present mask
            final_mask  = mask_stack * dropout_mask
            token_stack = self.apply_modality_dropout(
                token_stack, final_mask
            )

        # Prepend CLS token
        cls    = self.cls_token.expand(B, -1, -1)
        tokens_with_cls = torch.cat([cls, token_stack], dim=1)

        # Cross-modal transformer
        fused = self.transformer(tokens_with_cls)
        fused = self.norm(fused)

        # Return CLS token as global representation
        return fused[:, 0, :]  # [B, COMMON_DIM]


# ── OmniGeoFusion Backbone ────────────────────────────────────
class OmniGeoFusionBackbone(nn.Module):
    """
    Complete multimodal fusion backbone.
    Combines all 6 encoders + projection + cross-modal attention.

    Input:  dict of modality tensors (any subset)
    Output: [B, COMMON_DIM] unified representation

    Graceful degradation:
      Works with any subset of modalities.
      Missing modalities → zero tokens.
      Model still produces valid output.

    Memory budget on T4 (15GB):
      OpticalEncoder (frozen):  ~5.0GB
      SAREncoder:               ~1.0GB
      LiDAREncoder:             ~1.0GB
      ThermalEncoder:           ~0.5GB
      OSMEncoder:               ~0.5GB
      IoTEncoder:               ~0.3GB
      Projection + Fusion:      ~2.0GB
      Batch data (size=8):      ~2.0GB
      Gradients:                ~2.0GB
      Total:                   ~14.3GB ✅
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg

        # ── Encoders ─────────────────────────────────────────
        self.optical_encoder = OpticalEncoder(
            model_name=model_cfg['encoders']['optical']['name'],
            pretrained=model_cfg['encoders']['optical']['pretrained'],
            freeze_all=model_cfg['encoders']['optical']['freeze_strategy'] == 'full',
        )
        self.sar_encoder = SAREncoder(
            in_channels=model_cfg['encoders']['sar']['in_channels'],
            embed_dim=model_cfg['encoders']['sar']['embed_dim'],
            num_heads=model_cfg['encoders']['sar']['num_heads'],
            num_layers=model_cfg['encoders']['sar']['num_layers'],
        )
        self.lidar_encoder = LiDAREncoder(
            in_channels=3,  # DSM, DTM, nDSM
            embed_dim=model_cfg['encoders']['lidar']['embed_dim'],
        )
        self.thermal_encoder = ThermalEncoder(
            in_channels=model_cfg['encoders']['thermal']['in_channels'],
            embed_dim=model_cfg['encoders']['thermal']['embed_dim'],
        )
        self.osm_encoder = OSMEncoder(
            node_features=model_cfg['encoders']['osm']['node_features'],
            embed_dim=model_cfg['encoders']['osm']['embed_dim'],
            num_layers=model_cfg['encoders']['osm']['num_layers'],
        )
        self.iot_encoder = IoTEncoder(
            input_dim=model_cfg['encoders']['iot']['input_dim'],
            hidden_dim=model_cfg['encoders']['iot']['hidden_dim'],
            embed_dim=model_cfg['encoders']['iot']['embed_dim'],
        )

        # ── Modality Projection ───────────────────────────────
        self.projection = ModalityProjection(
            input_dims={
                'optical': OPTICAL_DIM,
                'sar':     SAR_DIM,
                'lidar':   LIDAR_DIM,
                'thermal': THERMAL_DIM,
                'osm':     OSM_DIM,
                'iot':     IOT_DIM,
            },
            common_dim=model_cfg['fusion']['common_dim'],
        )

        # ── Cross-Modal Attention ─────────────────────────────
        self.fusion = CrossModalAttention(
            common_dim=model_cfg['fusion']['common_dim'],
            num_heads=model_cfg['fusion']['num_heads'],
            num_layers=model_cfg['fusion']['num_layers'],
            dropout=model_cfg['fusion']['dropout'],
            modality_dropout_prob=model_cfg['fusion']['modality_dropout_prob'],
        )

        # Log parameter counts
        self._log_params()

    def _log_params(self):
        """Log parameter counts per component."""
        components = {
            'optical':  self.optical_encoder,
            'sar':      self.sar_encoder,
            'lidar':    self.lidar_encoder,
            'thermal':  self.thermal_encoder,
            'osm':      self.osm_encoder,
            'iot':      self.iot_encoder,
            'fusion':   self.fusion,
        }
        total = 0
        log.info('OmniGeoFusion parameter counts:')
        for name, module in components.items():
            params     = sum(p.numel() for p in module.parameters())
            trainable  = sum(
                p.numel() for p in module.parameters()
                if p.requires_grad
            )
            total     += params
            log.info(
                f'  {name:<12}: {params/1e6:.1f}M total, '
                f'{trainable/1e6:.1f}M trainable'
            )
        log.info(f'  {"TOTAL":<12}: {total/1e6:.1f}M parameters')

    def set_phase(self, phase: str):
        """
        Set training phase.

        phase='ssl':      optical frozen, others train
        phase='finetune': optical last 4 layers unfreeze
        phase='full':     all parameters train
        """
        if phase == 'ssl':
            self.optical_encoder.freeze()
            log.info('Phase SSL: optical encoder frozen')
        elif phase == 'finetune':
            self.optical_encoder.unfreeze_last_n_layers(4)
            log.info('Phase finetune: optical last 4 layers unfrozen')
        elif phase == 'full':
            for param in self.parameters():
                param.requires_grad = True
            log.info('Phase full: all parameters trainable')

    def encode_optical(
        self, x: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return self.optical_encoder(x)

    def encode_sar(
        self, x: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return self.sar_encoder(x)

    def encode_lidar(
        self, x: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return self.lidar_encoder(x)

    def encode_thermal(
        self, x: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return self.thermal_encoder(x)

    def encode_osm(
        self,
        node_features: Optional[torch.Tensor],
        edge_index: Optional[torch.Tensor],
        batch: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        if node_features is None:
            return None
        return self.osm_encoder(node_features, edge_index, batch)

    def encode_iot(
        self, x: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return self.iot_encoder(x)

    def forward(
        self,
        optical: Optional[torch.Tensor] = None,
        sar: Optional[torch.Tensor] = None,
        lidar: Optional[torch.Tensor] = None,
        thermal: Optional[torch.Tensor] = None,
        osm_nodes: Optional[torch.Tensor] = None,
        osm_edges: Optional[torch.Tensor] = None,
        osm_batch: Optional[torch.Tensor] = None,
        iot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through full backbone.

        Args:
            optical:   [B, 6, H, W]  Sentinel-2
            sar:       [B, 2, H, W]  Sentinel-1
            lidar:     [B, 3, H, W]  DSM/DTM/nDSM
            thermal:   [B, 1, H, W]  LST
            osm_nodes: [N, 64]       OSM node features
            osm_edges: [2, E]        OSM edge index
            osm_batch: [N]           node-to-graph
            iot:       [B, T, 16]    IoT sensor sequence

        Returns:
            fused: [B, COMMON_DIM] unified representation
        """
        training = self.training

        # Encode each modality
        embeddings = {}

        if optical is not None:
            embeddings['optical'] = self.encode_optical(optical)
        if sar is not None:
            embeddings['sar'] = self.encode_sar(sar)
        if lidar is not None:
            embeddings['lidar'] = self.encode_lidar(lidar)
        if thermal is not None:
            embeddings['thermal'] = self.encode_thermal(thermal)
        if osm_nodes is not None:
            embeddings['osm'] = self.encode_osm(
                osm_nodes, osm_edges, osm_batch
            )
        if iot is not None:
            embeddings['iot'] = self.encode_iot(iot)

        if not embeddings:
            raise ValueError(
                'At least one modality must be provided'
            )

        # Project to common dimension
        projected = self.projection(embeddings)

        # Cross-modal fusion
        fused = self.fusion(projected, training=training)

        return fused  # [B, COMMON_DIM]


# ── Factory ────────────────────────────────────────────────────
def build_backbone(config_path: str) -> OmniGeoFusionBackbone:
    """Build backbone from config file."""
    cfg = yaml.safe_load(open(config_path))
    return OmniGeoFusionBackbone(cfg['model'])


# ── Main (test) ────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Test OmniGeoFusion backbone'
    )
    parser.add_argument(
        '--config', default='configs/model_config.yaml'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    log.info('Building OmniGeoFusion backbone...')
    backbone = build_backbone(args.config)
    backbone.eval()

    # Test with dummy data
    B = 2
    with torch.no_grad():
        fused = backbone(
            optical=torch.randn(B, 6, 224, 224),
            sar=torch.randn(B, 2, 224, 224),
            lidar=torch.randn(B, 3, 224, 224),
            thermal=torch.randn(B, 1, 224, 224),
            iot=torch.randn(B, 12, 16),
        )

    log.info(f'✅ Backbone test passed: output shape {fused.shape}')
    assert fused.shape == (B, COMMON_DIM), (
        f'Expected ({B}, {COMMON_DIM}), got {fused.shape}'
    )

    # Test with missing modalities
    with torch.no_grad():
        fused_partial = backbone(
            optical=torch.randn(B, 6, 224, 224),
            sar=torch.randn(B, 2, 224, 224),
            # lidar, thermal, osm, iot missing
        )

    log.info(
        f'✅ Partial modality test: output shape '
        f'{fused_partial.shape}'
    )
    print(f'\n✅ OmniGeoFusion backbone ready')
    print(f'   Output dimension: {COMMON_DIM}')
    print(f'   Supported modalities: {MODALITY_NAMES}')


if __name__ == '__main__':
    main()
