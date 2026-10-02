"""
GeoFoundationHub — Prithvi-EO-2.0-300M Adapter
=================================================
Wraps Prithvi-EO-2.0-300M for unified inference.
Uses TerraTorch backbone registry.
"""

import logging
import numpy as np
import torch
from typing import Optional, List
from models.base import BaseModelAdapter, InferenceInput, InferenceOutput

log = logging.getLogger(__name__)

# Normalization constants (HLS bands: B, G, R, NIR, SWIR1, SWIR2)
MEAN = np.array([0.05, 0.06, 0.07, 0.25, 0.20, 0.15], dtype=np.float32)
STD  = np.array([0.05, 0.05, 0.06, 0.10, 0.10, 0.08], dtype=np.float32)


class PrithviAdapter(BaseModelAdapter):
    """
    Adapter for Prithvi-EO-2.0-300M.
    Supports: embedding extraction for
    optical/multispectral imagery.
    """

    def __init__(self, model_id: str):
        super().__init__(
            model_id=model_id,
            model_name="Prithvi-EO-2.0-300M",
            version="2.0.0",
        )
        self.encoder = None
        self.device  = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        """Load Prithvi via TerraTorch."""
        import time
        t0 = time.time()
        try:
            from terratorch.registry import BACKBONE_REGISTRY
            log.info("Loading Prithvi-EO-2.0-300M via TerraTorch...")
            self.encoder = BACKBONE_REGISTRY.build(
                "prithvi_eo_v2_300",
                pretrained=True,
                num_frames=1,
                in_chans=6,
            )
            self.encoder.eval()
            self.encoder = self.encoder.to(self.device)
            self._loaded    = True
            self._load_time = round(time.time() - t0, 2)
            log.info(f"✅ Prithvi loaded in {self._load_time}s on {self.device}")
        except Exception as e:
            log.error(f"❌ Failed to load Prithvi: {e}")
            raise

    def _preprocess(self, inputs: InferenceInput) -> torch.Tensor:
        """Preprocess input image for Prithvi. Always outputs 6 channels."""
        img = np.array(inputs.image, dtype=np.float32)
        C   = inputs.channels
        H   = inputs.height
        W   = inputs.width
        img = img.reshape(C, H, W)

        # Prithvi requires exactly 6 channels
        if C < 6:
            # Pad with zeros to reach 6 channels
            pad = np.zeros((6 - C, H, W), dtype=np.float32)
            img = np.concatenate([img, pad], axis=0)
            C   = 6
        elif C > 6:
            img = img[:6]
            C   = 6

        # Normalize per channel
        img = (img - MEAN[:, None, None]) / (STD[:, None, None] + 1e-6)

        # Add batch + time dims: [1, 6, 1, H, W]
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(2)
        return tensor.to(self.device)

    def get_embedding(self, inputs: InferenceInput) -> List[float]:
        """Extract 1024d embedding from Prithvi."""
        tensor = self._preprocess(inputs)
        with torch.no_grad():
            out  = self.encoder(tensor)
            feat = out[-1] if isinstance(out, (list, tuple)) else out
            emb  = feat[:, 1:, :].mean(1) if feat.dim() == 3 \
                   else feat.mean([2, 3])
        return emb.squeeze(0).cpu().numpy().tolist()

    def predict(self, inputs: InferenceInput) -> InferenceOutput:
        """Run Prithvi inference — returns embedding."""
        embedding = self.get_embedding(inputs)
        return InferenceOutput(
            model_id=self.model_id,
            model_name=self.model_name,
            model_version=self.version,
            embedding=embedding,
            predictions={"embedding_dim": len(embedding)},
            patch_id=inputs.patch_id,
            metadata={"sensor": inputs.sensor_type},
        )
