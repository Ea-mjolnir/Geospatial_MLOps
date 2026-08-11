"""
OmniGeoFusion — SSL Pre-training Pipeline
==========================================
Stage 1: Self-supervised pre-training on unlabeled
Netherlands multimodal data.

Objectives:
  1. Cross-modal contrastive learning
     optical ↔ SAR, optical ↔ thermal, SAR ↔ LiDAR
  2. Temporal contrastive learning
     T1 ↔ T2 weighted by day gap

Normalization per modality:
  Sentinel-2: per-band z-score (mean/std from NL stats)
  Sentinel-1: clip to [-25, 0] dB → normalize to [-1, 1]
  LiDAR:      DSM/DTM clip [-10, 100]m, nDSM clip [0, 50]m
  Thermal:    clip [−20, 60]°C → normalize to [-1, 1]

Data loading:
  Patches cached to local disk before training
  → Fast random access during training (no S3 latency)

Checkpointing:
  Saves every epoch to GDrive
  Auto-resumes from ssl_latest.pt on restart
  Keeps best checkpoint as ssl_best.pt

Training:
  Batch size: 8 (T4 VRAM constraint)
  Epochs:     50
  LR:         1.5e-4 with cosine schedule
  Optical encoder: FROZEN throughout SSL
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import mlflow

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

log = logging.getLogger(__name__)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('s3transfer').setLevel(logging.WARNING)

# ── Normalization Stats ────────────────────────────────────────
# Per-band statistics for Netherlands (computed from data)
S2_MEAN = np.array([1200., 1100., 1050., 2800., 1800., 2400.],
                    dtype=np.float32)
S2_STD  = np.array([600.,  550.,  600.,  900.,  700.,  800.],
                    dtype=np.float32)

# SAR: VV_dB and VH_dB clipped to [-25, 0]
SAR_MIN, SAR_MAX = -25.0, 0.0

# LiDAR: DSM/DTM [-10, 100]m, nDSM [0, 50]m
LIDAR_MIN = np.array([-10., -10., 0.], dtype=np.float32)
LIDAR_MAX = np.array([100., 100., 50.], dtype=np.float32)

# Thermal: LST [-20, 60]°C
THERM_MIN, THERM_MAX = -20.0, 60.0


def normalize_s2(data: np.ndarray) -> np.ndarray:
    """Normalize Sentinel-2 [6,H,W] per band z-score."""
    data = data.copy().astype(np.float32)
    for b in range(min(6, data.shape[0])):
        data[b] = (data[b] - S2_MEAN[b]) / (S2_STD[b] + 1e-6)
    data = np.clip(data, -3, 3)
    return data


def normalize_sar(data: np.ndarray) -> np.ndarray:
    """Normalize SAR [2,H,W] VV/VH dB to [-1, 1]."""
    data = data.copy().astype(np.float32)
    data = np.clip(data, SAR_MIN, SAR_MAX)
    data = (data - SAR_MIN) / (SAR_MAX - SAR_MIN) * 2 - 1
    return data


def normalize_lidar(data: np.ndarray) -> np.ndarray:
    """Normalize LiDAR [3,H,W] DSM/DTM/nDSM to [-1, 1]."""
    data = data.copy().astype(np.float32)
    for b in range(min(3, data.shape[0])):
        data[b] = np.clip(data[b], LIDAR_MIN[b], LIDAR_MAX[b])
        data[b] = (data[b] - LIDAR_MIN[b]) / (
            LIDAR_MAX[b] - LIDAR_MIN[b]
        ) * 2 - 1
    return data


def normalize_thermal(data: np.ndarray) -> np.ndarray:
    """Normalize thermal [1,H,W] LST °C to [-1, 1]."""
    data = data.copy().astype(np.float32)
    data = np.clip(data, THERM_MIN, THERM_MAX)
    data = (data - THERM_MIN) / (THERM_MAX - THERM_MIN) * 2 - 1
    return data


def load_tif(
    path: Optional[str],
    n_bands: int,
    patch_size: int,
    normalize_fn=None
) -> torch.Tensor:
    """
    Load GeoTIFF patch from local disk.
    Handles nodata, resizing, normalization.
    Returns zeros if path missing or error.
    """
    import rasterio
    from torch.nn.functional import interpolate

    if path and os.path.exists(path):
        try:
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)

            # Replace nodata
            data[data == -9999] = 0.0
            data[~np.isfinite(data)] = 0.0

            # Resize if needed
            if data.shape[1] != patch_size or \
               data.shape[2] != patch_size:
                t = torch.from_numpy(data).unsqueeze(0)
                t = interpolate(
                    t,
                    size=(patch_size, patch_size),
                    mode='bilinear',
                    align_corners=False
                )
                data = t.squeeze(0).numpy()

            # Normalize
            if normalize_fn is not None:
                data = normalize_fn(data)

            return torch.from_numpy(data[:n_bands])

        except Exception as e:
            log.debug(f'Load error {path}: {e}')

    return torch.zeros(n_bands, patch_size, patch_size)


# ── SSL Dataset ───────────────────────────────────────────────
class SSLDataset(Dataset):
    """
    Dataset for SSL pre-training.
    Reads multimodal patches from local disk cache.
    Applies per-modality normalization.

    Each sample:
      optical_t1: [6, patch_size, patch_size] normalized
      optical_t2: [6, patch_size, patch_size] normalized
      sar_t1:     [2, patch_size, patch_size] normalized
      sar_t2:     [2, patch_size, patch_size] normalized
      lidar:      [3, patch_size, patch_size] normalized
      thermal:    [1, patch_size, patch_size] normalized
      iot:        [12, 16] zeros placeholder
      day_gap:    scalar
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        patch_size: int = 224,
        max_samples: Optional[int] = None,
    ):
        self.data_dir   = data_dir
        self.split      = split
        self.patch_size = patch_size

        index_path = os.path.join(
            data_dir, f'{split}_index.json'
        )
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.index = json.load(f)
        else:
            log.warning(
                f'No index at {index_path} — empty dataset'
            )
            self.index = []

        if max_samples:
            self.index = self.index[:max_samples]

        log.info(
            f'SSLDataset {split}: {len(self.index)} samples'
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        """Load one normalized multimodal sample."""
        item = self.index[idx]
        ps   = self.patch_size

        # Optical T1 + T2
        opt_t1 = load_tif(
            item.get('s2_t1_path'), 6, ps, normalize_s2
        )
        opt_t2 = load_tif(
            item.get('s2_t2_path', item.get('s2_t1_path')),
            6, ps, normalize_s2
        )

        # SAR T1 + T2
        sar_t1 = load_tif(
            item.get('sar_path'), 2, ps, normalize_sar
        )
        sar_t2 = load_tif(
            item.get('sar_t2_path', item.get('sar_path')),
            2, ps, normalize_sar
        )

        # LiDAR (static)
        lidar = load_tif(
            item.get('lidar_path'), 3, ps, normalize_lidar
        )

        # Thermal
        thermal = load_tif(
            item.get('thermal_path'), 1, ps, normalize_thermal
        )

        # Load OSM features from index (16 values)
        osm_raw = item.get('osm_features', [0.0] * 16)
        osm_arr = np.array(osm_raw[:16], dtype=np.float32)
        osm_max = np.array([
            10000, 2000, 50, 1,
            5000,  1000, 500, 1000,
            1000,  1000, 1000, 500,
            500,   10000, 100000, 1e7
        ], dtype=np.float32)
        osm_norm   = np.clip(osm_arr / (osm_max + 1e-6), 0, 1)
        osm_tensor = torch.from_numpy(osm_norm).unsqueeze(0)  # [1, 16]

        # Load IoT features from index (16 values)
        iot_raw = item.get('iot_features', [0.0] * 16)
        iot_arr = np.array(iot_raw[:16], dtype=np.float32)
        iot_max = np.array([
            100, 100, 100,
            100, 100, 100,
            200, 200, 200,
            75,  75,  75,
            1,   1,   1,  1
        ], dtype=np.float32)
        iot_norm   = np.clip(iot_arr / (iot_max + 1e-6), 0, 1)
        iot_tensor = torch.from_numpy(
            np.tile(iot_norm, (12, 1))
        ).float()  # [12, 16]

        return {
            'optical_t1': opt_t1,
            'optical_t2': opt_t2,
            'sar_t1':     sar_t1,
            'sar_t2':     sar_t2,
            'lidar':      lidar,
            'thermal':    thermal,
            'osm':        osm_tensor,
            'iot':        iot_tensor,
            'day_gap':    torch.tensor(
                item.get('day_gap', 90), dtype=torch.float32
            ),
            'patch_id':   item.get('patch_id', f'patch_{idx}'),
        }


# ── Contrastive Losses ────────────────────────────────────────
class CrossModalContrastiveLoss(nn.Module):
    """InfoNCE loss for cross-modal alignment."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, emb_a: torch.Tensor, emb_b: torch.Tensor
    ) -> torch.Tensor:
        emb_a = F.normalize(emb_a, dim=-1)
        emb_b = F.normalize(emb_b, dim=-1)
        sim   = torch.matmul(emb_a, emb_b.T) / self.temperature
        labels = torch.arange(sim.shape[0], device=sim.device)
        return (
            F.cross_entropy(sim, labels) +
            F.cross_entropy(sim.T, labels)
        ) / 2


class TemporalContrastiveLoss(nn.Module):
    """Temporal contrastive loss weighted by day gap."""

    def __init__(
        self,
        temperature: float = 0.07,
        max_positive_gap: int = 30,
    ):
        super().__init__()
        self.temperature      = temperature
        self.max_positive_gap = max_positive_gap

    def forward(
        self,
        emb_t1: torch.Tensor,
        emb_t2: torch.Tensor,
        day_gaps: torch.Tensor
    ) -> torch.Tensor:
        emb_t1 = F.normalize(emb_t1, dim=-1)
        emb_t2 = F.normalize(emb_t2, dim=-1)
        sim    = torch.matmul(emb_t1, emb_t2.T) / self.temperature
        gap_w  = torch.exp(
            -day_gaps.float() / self.max_positive_gap
        )
        labels = torch.arange(sim.shape[0], device=sim.device)
        return F.cross_entropy(
            sim * gap_w.unsqueeze(-1), labels
        )


# ── SSL Trainer ───────────────────────────────────────────────
class SSLTrainer:
    """
    SSL pre-training trainer.
    Objectives: cross-modal + temporal contrastive.
    Optical encoder is FROZEN throughout.
    Saves checkpoint every epoch to GDrive.
    Auto-resumes from ssl_latest.pt if exists.
    """

    def __init__(
        self,
        backbone,
        cfg: dict,
        device: torch.device,
    ):
        self.backbone = backbone
        self.cfg      = cfg
        self.device   = device
        ssl_cfg       = cfg['ssl']

        # Losses
        self.contrastive = CrossModalContrastiveLoss(
            temperature=ssl_cfg['contrastive']['temperature']
        )
        self.temporal = TemporalContrastiveLoss(
            temperature=ssl_cfg['contrastive']['temperature'],
            max_positive_gap=ssl_cfg['temporal'].get(
                'max_positive_gap_days', 30
            ),
        )

        # Loss weights
        self.w_contrastive = 0.7
        self.w_temporal    = 0.3

        # Optimizer (exclude frozen optical encoder)
        trainable = [
            p for p in backbone.parameters()
            if p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=ssl_cfg['learning_rate'],
            weight_decay=ssl_cfg['weight_decay'],
            betas=(0.9, 0.95)
        )
        self.scaler     = GradScaler()
        self.best_loss  = float('inf')

        # Checkpoint dir
        self.checkpoint_dir = cfg['checkpoints']['ssl_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        log.info(
            f'SSLTrainer: {len(trainable)} '
            f'trainable parameter tensors'
        )

    def _get_scheduler(self, total_steps: int):
        """Cosine LR schedule with warmup."""
        ssl_cfg = self.cfg['ssl']
        warmup  = ssl_cfg.get('warmup_epochs', 5)
        steps_per_epoch = total_steps // ssl_cfg['epochs']

        def lr_lambda(step):
            warmup_steps = warmup * steps_per_epoch
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(
                1, total_steps - warmup_steps
            )
            return 0.5 * (1 + np.cos(np.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """One SSL training step."""
        self.optimizer.zero_grad()

        optical_t1 = batch['optical_t1'].to(self.device)
        optical_t2 = batch['optical_t2'].to(self.device)
        sar_t1     = batch['sar_t1'].to(self.device)
        lidar      = batch['lidar'].to(self.device)
        thermal    = batch['thermal'].to(self.device)
        day_gaps   = batch['day_gap'].to(self.device)

        with autocast():
            # Encode all modalities
            emb_opt  = self.backbone.encode_optical(optical_t1)
            emb_sar  = self.backbone.encode_sar(sar_t1)
            emb_lid  = self.backbone.encode_lidar(lidar)
            emb_thm  = self.backbone.encode_thermal(thermal)
            emb_opt2 = self.backbone.encode_optical(optical_t2)

            # Project to common space
            proj = self.backbone.projection.projections
            p_opt  = proj['optical'](emb_opt)
            p_sar  = proj['sar'](emb_sar)
            p_lid  = proj['lidar'](emb_lid)
            p_thm  = proj['thermal'](emb_thm)
            p_opt2 = proj['optical'](emb_opt2)

            # Cross-modal contrastive
            loss_os = self.contrastive(p_opt, p_sar)
            loss_ot = self.contrastive(p_opt, p_thm)
            loss_sl = self.contrastive(p_sar, p_lid)
            loss_cm = (loss_os + loss_ot + loss_sl) / 3.0

            # Temporal contrastive
            loss_tp = self.temporal(p_opt, p_opt2, day_gaps)

            # Total loss
            total = (
                self.w_contrastive * loss_cm +
                self.w_temporal    * loss_tp
            )

        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.backbone.parameters(),
            self.cfg['ssl'].get('gradient_clip', 1.0)
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            'loss':             float(total),
            'contrastive_loss': float(loss_cm),
            'temporal_loss':    float(loss_tp),
        }

    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        scheduler,
    ) -> Dict[str, float]:
        """Train one SSL epoch — prints single summary at 100%."""
        import time
        self.backbone.train()
        self.backbone.optical_encoder.eval()

        totals     = {}
        ssl_epochs = self.cfg['ssl']['epochs']
        t_start    = time.time()

        for batch in loader:
            losses = self.train_step(batch)
            for k, v in losses.items():
                totals[k] = totals.get(k, 0) + v
            scheduler.step()

        n          = len(loader)
        avg        = {k: v / n for k, v in totals.items()}
        duration   = int(time.time() - t_start)
        mins, secs = divmod(duration, 60)

        print(
            f'Epoch {epoch:02d}/{ssl_epochs} | '
            f'100% ✅ | '
            f'Loss: {avg["loss"]:.4f} | '
            f'CM: {avg["contrastive_loss"]:.4f} | '
            f'TP: {avg["temporal_loss"]:.4f} | '
            f'{mins}m {secs}s',
            flush=True
        )
        return avg

    def save_checkpoint(
        self,
        epoch: int,
        losses: Dict[str, float],
        is_best: bool = False,
    ):
        """
        Save checkpoint strategy:
          - ssl_latest.pt: always overwrite (for auto-resume)
          - ssl_best.pt:   always overwrite (best loss)
          - ssl_epochXXX:  only every 10 epochs, delete previous
        Total max: 3 files (~6GB) to save GDrive space.
        """
        state = {
            'epoch':           epoch,
            'backbone_state':  self.backbone.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state':    self.scaler.state_dict(),
            'losses':          losses,
            'best_loss':       self.best_loss,
            'config':          self.cfg,
        }

        # Always save latest (overwrite — for auto-resume)
        latest = os.path.join(
            self.checkpoint_dir, 'ssl_latest.pt'
        )
        torch.save(state, latest)

        # Save epoch checkpoint only every 10 epochs
        if epoch % 10 == 0:
            epoch_path = os.path.join(
                self.checkpoint_dir,
                f'ssl_epoch{epoch:03d}_'
                f'loss{losses["loss"]:.4f}.pt'
            )
            torch.save(state, epoch_path)

            # Delete previous 10th epoch checkpoint
            prev_epoch = epoch - 10
            if prev_epoch > 0:
                import glob
                prev_pattern = os.path.join(
                    self.checkpoint_dir,
                    f'ssl_epoch{prev_epoch:03d}_*.pt'
                )
                for old_ckpt in glob.glob(prev_pattern):
                    os.remove(old_ckpt)
                    log.info(f'Deleted old checkpoint: {old_ckpt}')

            log.info(f'Checkpoint saved: {epoch_path}')

        # Always save best (overwrite)
        if is_best:
            best = os.path.join(
                self.checkpoint_dir, 'ssl_best.pt'
            )
            torch.save(state, best)
            log.info(f'✅ Best checkpoint: {best}')

    def auto_resume(self) -> int:
        """
        Auto-detect and resume from ssl_latest.pt.
        Returns start epoch (1 if no checkpoint found).
        """
        latest = os.path.join(
            self.checkpoint_dir, 'ssl_latest.pt'
        )
        if os.path.exists(latest):
            log.info(f'Auto-resuming from {latest}')
            ckpt = torch.load(
                latest, map_location=self.device
            )
            self.backbone.load_state_dict(
                ckpt['backbone_state']
            )
            self.optimizer.load_state_dict(
                ckpt['optimizer_state']
            )
            if 'scaler_state' in ckpt:
                self.scaler.load_state_dict(
                    ckpt['scaler_state']
                )
            self.best_loss = ckpt.get(
                'best_loss', float('inf')
            )
            start_epoch = ckpt['epoch'] + 1
            log.info(
                f'✅ Resumed from epoch {ckpt["epoch"]} '
                f'(best loss: {self.best_loss:.4f})'
            )
            return start_epoch
        return 1

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ):
        """
        Full SSL pre-training loop.
        Auto-resumes from latest checkpoint on Colab restart.
        """
        ssl_cfg = self.cfg['ssl']

        # Resume logic
        if resume_from and os.path.exists(resume_from):
            # Manual resume path specified
            log.info(f'Resuming from {resume_from}')
            ckpt = torch.load(
                resume_from, map_location=self.device
            )
            self.backbone.load_state_dict(
                ckpt['backbone_state']
            )
            self.optimizer.load_state_dict(
                ckpt['optimizer_state']
            )
            self.best_loss = ckpt.get(
                'best_loss', float('inf')
            )
            start_epoch = ckpt['epoch'] + 1
        else:
            # Auto-resume from latest
            start_epoch = self.auto_resume()

        total_steps = ssl_cfg['epochs'] * len(train_loader)
        scheduler   = self._get_scheduler(total_steps)

        # Fast-forward scheduler to current step
        if start_epoch > 1:
            steps_done = (start_epoch - 1) * len(train_loader)
            for _ in range(steps_done):
                scheduler.step()

        log.info(
            f'SSL pre-training: '
            f'epoch {start_epoch} → {ssl_cfg["epochs"]}, '
            f'{len(train_loader)} steps/epoch'
        )

        with mlflow.start_run(
            run_name='ssl_pretrain'
        ):
            mlflow.log_params({
                'epochs':      ssl_cfg['epochs'],
                'batch_size':  ssl_cfg['batch_size'],
                'lr':          ssl_cfg['learning_rate'],
                'start_epoch': start_epoch,
            })

            for epoch in range(
                start_epoch, ssl_cfg['epochs'] + 1
            ):
                t0 = time.time()

                train_losses = self.train_epoch(
                    train_loader, epoch, scheduler
                )
                duration = time.time() - t0

                print(
                    f'\nEpoch {epoch}/{ssl_cfg["epochs"]} | '
                    f'Loss: {train_losses["loss"]:.4f} | '
                    f'CM: {train_losses["contrastive_loss"]:.4f} | '
                    f'TP: {train_losses["temporal_loss"]:.4f} | '
                    f'{duration:.0f}s'
                )

                mlflow.log_metrics(
                    {
                        f'ssl_{k}': v
                        for k, v in train_losses.items()
                    },
                    step=epoch
                )

                is_best = train_losses['loss'] < self.best_loss
                if is_best:
                    self.best_loss = train_losses['loss']

                self.save_checkpoint(
                    epoch, train_losses, is_best
                )

        log.info(
            f'✅ SSL complete. '
            f'Best loss: {self.best_loss:.4f}'
        )
        log.info(
            f'Best ckpt: '
            f'{self.checkpoint_dir}/ssl_best.pt'
        )


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='OmniGeoFusion SSL pre-training'
    )
    parser.add_argument(
        '--model-config',
        default='configs/model_config.yaml'
    )
    parser.add_argument(
        '--train-config',
        default='configs/training_config.yaml'
    )
    parser.add_argument(
        '--data-dir',
        default='/content/omnigeofusion_data'
    )
    parser.add_argument(
        '--resume', default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--max-samples', type=int, default=None
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    model_cfg = yaml.safe_load(open(args.model_config))
    train_cfg = yaml.safe_load(open(args.train_config))
    cfg       = {**model_cfg, **train_cfg}

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    log.info(f'Device: {device}')

    sys.path.insert(0, os.path.abspath('.'))
    from src.fusion.backbone import OmniGeoFusionBackbone

    backbone = OmniGeoFusionBackbone(
        model_cfg['model']
    ).to(device)
    backbone.set_phase('ssl')

    train_ds = SSLDataset(
        args.data_dir, split='train',
        max_samples=args.max_samples
    )
    val_ds = SSLDataset(
        args.data_dir, split='val',
        max_samples=(
            args.max_samples // 5
            if args.max_samples else None
        )
    )

    ssl_cfg = train_cfg['ssl']
    hw_cfg  = cfg.get('hardware', {})

    train_loader = DataLoader(
        train_ds,
        batch_size=ssl_cfg['batch_size'],
        shuffle=True,
        num_workers=hw_cfg.get('num_workers', 2),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=ssl_cfg['batch_size'],
        shuffle=False,
        num_workers=2,
    )

    mlflow.set_tracking_uri(
        train_cfg.get('mlflow', {}).get(
            'tracking_uri', 'http://54.83.1.56:5000'
        )
    )
    mlflow.set_experiment(
        train_cfg.get('mlflow', {}).get(
            'experiment_name', 'omnigeofusion-ssl'
        )
    )

    trainer = SSLTrainer(backbone, cfg, device)
    trainer.train(
        train_loader,
        val_loader,
        resume_from=args.resume
    )


if __name__ == '__main__':
    main()
