"""
GeoAI MLOps Project 1 — Siamese Prithvi Change Detection Model
===============================================================
Architecture:
    Two-tower Siamese network using Prithvi-EO-1.0-100M as backbone.
    Both towers share identical weights (true Siamese setup).
    Imagery embeddings are subtracted (encodes change in embedding space).
    Tabular features are concatenated (preserves baseline context).
    MLP fusion head produces a single change score.

Model flow per pair:
    patch_t1 [5,256,256] → Prithvi encoder → embedding_t1 [768]
    patch_t2 [5,256,256] → Prithvi encoder → embedding_t2 [768]
    embedding_diff = embedding_t2 - embedding_t1              [768]
    fusion = concat(embedding_diff, tabular_t1, tabular_t2, day_gap)
           = concat([768], [34], [34], [1])                   [837]
    output = MLP(fusion)                                       [1]

Encoder:
    Prithvi-EO-1.0-100M loaded via TerraTorch BACKBONE_REGISTRY.
    Accepts 6 bands — our 5-band input is zero-padded to 6.
    Output: mean pool of 196 patch tokens from last transformer layer → [768]

Two-phase training:
    Phase 1: freeze Prithvi encoder, train MLP head only
    Phase 2: unfreeze encoder, fine-tune end-to-end at low LR

Usage:
    from src.training.model import SiamesePrithviModel
    model = SiamesePrithviModel(
        prithvi_model_name='prithvi_eo_v1_100',
        n_tabular_features=34,
        mlp_hidden_1=256,
        mlp_hidden_2=64,
        dropout_1=0.3,
        dropout_2=0.2,
    )
    model.freeze_encoder()   # Phase 1
    model.unfreeze_encoder() # Phase 2
"""
import logging
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Expected embedding dimension from Prithvi-EO-1.0-100M
PRITHVI_EMBED_DIM = 768

# Our band count (5) vs Prithvi's original (6)
OUR_BANDS     = 5
PRITHVI_BANDS = 6


class AdaptedPrithviEncoder(nn.Module):
    """
    Prithvi-EO-1.0-100M encoder loaded via TerraTorch BACKBONE_REGISTRY.

    Accepts 5-band input by zero-padding to 6 bands before passing
    to Prithvi (which expects 6 HLS bands). The 6th band (SWIR2) is
    set to zeros — simpler and more stable than retraining the patch
    embedding from scratch.

    Forward pass:
        Input:  [B, 5, H, W]
        Resize: 256×256 → 224×224 (Prithvi native resolution)
        Pad:    [B, 5, 224, 224] → [B, 6, 224, 224]
        Encode: TerraTorch → 12 transformer layers × [B, 197, 768]
        Pool:   last layer, mean of 196 patch tokens → [B, 768]
        Output: [B, 768]
    """

    def __init__(self, prithvi_model_name='prithvi_eo_v1_100',
                 n_input_bands=OUR_BANDS):
        super().__init__()
        self.n_input_bands      = n_input_bands
        self.prithvi_model_name = prithvi_model_name
        self._load_encoder(prithvi_model_name)

    def _load_encoder(self, model_name):
        """Load Prithvi via TerraTorch BACKBONE_REGISTRY."""
        try:
            import os as _os
            _os.environ["MPLBACKEND"] = "agg"
            from terratorch.registry import BACKBONE_REGISTRY
            log.info(f"Loading {model_name} via TerraTorch...")
            self.encoder = BACKBONE_REGISTRY.build(
                model_name,
                pretrained=True,
                num_frames=1,
                in_chans=6,
            )
            log.info("✅ Prithvi-EO-1.0-100M loaded via TerraTorch")
        except Exception as e:
            log.error(f"Failed to load Prithvi via TerraTorch: {e}")
            log.info("Falling back to stub encoder for testing")
            self.encoder = _StubEncoder(PRITHVI_EMBED_DIM)

    def forward(self, x):
        """
        Forward pass through Prithvi encoder.

        Input:  x [B, 5, H, W]  (5 spectral bands)
        Output:   [B, 768]      (patch embedding)

        Steps:
          1. Resize to 224×224 (Prithvi native input size)
          2. Pad 5 bands → 6 bands (zero-pad SWIR2)
          3. Forward through encoder → 12 layer outputs
          4. Mean pool patch tokens from last layer (exclude CLS)
        """
        import torch.nn.functional as F

        # Step 1: Resize to Prithvi native size 224×224
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(
                x, size=(224, 224),
                mode='bilinear', align_corners=False
            )

        # Step 2: Pad 5 bands → 6 bands (zeros for missing SWIR2)
        zero = torch.zeros(
            x.shape[0], 1, 224, 224,
            device=x.device, dtype=x.dtype
        )
        x = torch.cat([x, zero], dim=1)  # [B, 6, 224, 224]

        # Step 3: Forward through encoder
        # Returns list of 12 tensors, each [B, 197, 768]
        # (197 = 196 patch tokens + 1 CLS token)
        out = self.encoder(x)

        # Step 4: Mean pool patch tokens from last layer
        # out[-1] = last transformer layer [B, 197, 768]
        # [:, 1:, :] = exclude CLS token (index 0)
        embedding = out[-1][:, 1:, :].mean(dim=1)  # [B, 768]

        return embedding


class _StubEncoder(nn.Module):
    """
    Lightweight stub encoder for local testing without downloading Prithvi.
    Mimics Prithvi's output shape [B, 768] using a simple CNN.
    Replace with real AdaptedPrithviEncoder on the training instance.
    """

    def __init__(self, embed_dim=PRITHVI_EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(OUR_BANDS, 32, kernel_size=8, stride=8),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
        )
        log.warning(
            "Using STUB encoder — replace with real Prithvi on training instance"
        )

    def forward(self, x):
        if x.dim() == 5:
            x = x.squeeze(2)  # remove time dim if present
        return self.net(x)


class MLPFusionHead(nn.Module):
    """
    MLP fusion head that combines imagery embeddings and tabular features.

    Input:
        fusion_vector [B, 837]
        = concat(embedding_diff [768], tabular_t1 [34], tabular_t2 [34], day_gap [1])

    Architecture:
        Linear(837→256) → LayerNorm → ReLU → Dropout(0.3)
        Linear(256→64)  → LayerNorm → ReLU → Dropout(0.2)
        Linear(64→1)    → raw scalar (no activation)

    LayerNorm instead of BatchNorm:
        More stable with variable/small batch sizes during fine-tuning.
        BatchNorm can behave poorly with batch_size=8 on ViT fine-tuning.
    """

    def __init__(
        self,
        embed_dim=PRITHVI_EMBED_DIM,
        n_tabular=34,
        hidden_1=256,
        hidden_2=64,
        dropout_1=0.3,
        dropout_2=0.2,
    ):
        super().__init__()
        # fusion_dim = embedding_diff + tabular_t1 + tabular_t2 + day_gap
        fusion_dim = embed_dim + n_tabular + n_tabular + 1

        self.net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout_1),
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.Linear(hidden_2, 1),
            # No activation — raw scalar output for regression
        )
        self._init_weights()
        log.info(
            f"MLPFusionHead: {fusion_dim}→{hidden_1}→{hidden_2}→1 "
            f"(dropout {dropout_1}/{dropout_2})"
        )

    def _init_weights(self):
        """Xavier uniform initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, fusion_vector):
        """
        Input:  fusion_vector [B, 837]
        Output: change_score  [B, 1]
        """
        return self.net(fusion_vector)


class SiamesePrithviModel(nn.Module):
    """
    Full Siamese Prithvi Change Detection Model.

    Combines:
        - AdaptedPrithviEncoder (shared weights, run twice — Siamese)
        - MLPFusionHead (fuses imagery change + tabular context)

    Forward pass:
        1. encode patch_t1 → embedding_t1 [B, 768]
        2. encode patch_t2 → embedding_t2 [B, 768]  (same encoder, Siamese)
        3. embedding_diff = embedding_t2 - embedding_t1 [B, 768]
        4. fusion = cat(embedding_diff, tabular_t1, tabular_t2, day_gap) [B, 837]
        5. output = MLP(fusion) → [B, 1]
    """

    def __init__(
        self,
        prithvi_model_name='prithvi_eo_v1_100',
        n_tabular_features=34,
        mlp_hidden_1=256,
        mlp_hidden_2=64,
        dropout_1=0.3,
        dropout_2=0.2,
    ):
        super().__init__()
        self.prithvi_model_name = prithvi_model_name
        self.n_tabular          = n_tabular_features

        # Shared Siamese encoder (run twice, same weights)
        self.encoder = AdaptedPrithviEncoder(prithvi_model_name)

        # MLP fusion head
        self.fusion_head = MLPFusionHead(
            embed_dim=PRITHVI_EMBED_DIM,
            n_tabular=n_tabular_features,
            hidden_1=mlp_hidden_1,
            hidden_2=mlp_hidden_2,
            dropout_1=dropout_1,
            dropout_2=dropout_2,
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable    = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"SiamesePrithviModel initialized")
        log.info(f"  Total params:     {total_params:,}")
        log.info(f"  Trainable params: {trainable:,}")

    def forward(self, patch_t1, patch_t2, tabular_t1, tabular_t2, day_gap):
        """
        Args:
            patch_t1:   [B, 5, 256, 256]  T1 imagery
            patch_t2:   [B, 5, 256, 256]  T2 imagery
            tabular_t1: [B, 34]           T1 tabular features (standardized)
            tabular_t2: [B, 34]           T2 tabular features (standardized)
            day_gap:    [B, 1]            days between T1 and T2 (standardized)
        Returns:
            change_score: [B, 1]  predicted change magnitude
        """
        # Encode both patches through shared Prithvi encoder (Siamese)
        embedding_t1 = self.encoder(patch_t1)  # [B, 768]
        embedding_t2 = self.encoder(patch_t2)  # [B, 768]

        # Subtraction encodes change in embedding space
        embedding_diff = embedding_t2 - embedding_t1  # [B, 768]

        # Concatenate all inputs to fusion head
        fusion = torch.cat(
            [embedding_diff, tabular_t1, tabular_t2, day_gap],
            dim=1
        )  # [B, 837]

        # MLP head → single change score
        change_score = self.fusion_head(fusion)  # [B, 1]
        return change_score

    # ── Training phase control ─────────────────────────────────────
    def freeze_encoder(self):
        """
        Phase 1: freeze Prithvi encoder, train MLP head only.
        Since we zero-pad bands (no new patch embedding layer),
        we freeze the entire encoder.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"Phase 1: encoder frozen | trainable params: {trainable:,}")

    def unfreeze_encoder(self):
        """
        Phase 2: unfreeze Prithvi encoder for full fine-tuning.
        Use very low LR for encoder (1e-5) to protect pretrained weights.
        MLP head uses higher LR (1e-4).
        """
        for param in self.encoder.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(f"Phase 2: encoder unfrozen | trainable params: {trainable:,}")

    def get_param_groups(self, encoder_lr=1e-5, head_lr=1e-4):
        """
        Return parameter groups with separate LRs for Phase 2.
        Encoder uses very low LR to protect pretrained weights.
        MLP head uses higher LR since it was trained from scratch.

        Usage:
            optimizer = AdamW(model.get_param_groups(
                encoder_lr=1e-5, head_lr=1e-4
            ))
        """
        return [
            {
                'params': self.encoder.parameters(),
                'lr':     encoder_lr,
                'name':   'prithvi_encoder',
            },
            {
                'params': self.fusion_head.parameters(),
                'lr':     head_lr,
                'name':   'mlp_fusion_head',
            },
        ]
