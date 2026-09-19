"""
ThermalWatch — SSL Pre-training (InfoNCE v3)
=============================================
Improvements:
  1. Random init ThermalEncoder (forces real learning)
  2. InfoNCE with temperature=0.5
  3. LR warmup + cosine decay
  4. Early stopping on train_loss plateau
  5. Strong augmentations
  6. MixUp augmentation

SSL Objective: InfoNCE contrastive loss
  View 1: original thermal (augmented)
  View 2: differently augmented thermal
  Positive pairs: same patch
  Negative pairs: different patches in batch

Early stopping:
  Monitor:   train_loss plateau
  Patience:  10 epochs
  Min delta: 0.001
  Warmup:    10 epochs

Display:
  Epoch 001/100 | train=2.341 val=3.901 ✅ |
  align=0.312 | lr=0.000300 | 15m 23s | warmup
"""

import json
import time
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Normalization Constants ───────────────────────────────────
WILDFIRE_THERMAL_MEAN = 35.0
WILDFIRE_THERMAL_STD  = 15.0
SOLAR_THERMAL_MEAN    = 45.0
SOLAR_THERMAL_STD     = 20.0

OSM_MAX = np.array(
    [4000, 7000, 200, 200, 100, 200, 200],
    dtype=np.float32
)
S2_MEAN = np.array(
    [0.05, 0.06, 0.07, 0.25, 0.20, 0.15],
    dtype=np.float32
)
S2_STD = np.array(
    [0.05, 0.05, 0.06, 0.10, 0.10, 0.08],
    dtype=np.float32
)
WEATHER_MEAN = np.array(
    [290.0, 0.0, 0.0, 275.0, 2.0], dtype=np.float32
)
WEATHER_STD = np.array(
    [15.0, 3.0, 3.0, 15.0, 3.0], dtype=np.float32
)


def normalize_thermal(
    data: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    data = np.nan_to_num(data, nan=0.0)
    return (data - mean) / (std + 1e-6)


def load_patch(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Missing: {path}')
    data = np.load(path)
    if data.size == 0:
        raise ValueError(f'Empty: {path}')
    return data.astype(np.float32)


# ── SSL Dataset ───────────────────────────────────────────────
class ThermalSSLDataset(Dataset):
    """Combined wildfire + solar thermal dataset for SSL."""

    def __init__(
        self,
        wildfire_index: Optional[Path],
        solar_index:    Optional[Path],
        split:          str = 'train',
        patch_size:     int = 224,
    ):
        self.patch_size = patch_size
        self.samples    = []

        if wildfire_index and Path(wildfire_index).exists():
            wf_data = json.loads(
                Path(wildfire_index).read_text()
            )
            for entry in wf_data:
                entry['task']         = 'wildfire'
                entry['thermal_mean'] = WILDFIRE_THERMAL_MEAN
                entry['thermal_std']  = WILDFIRE_THERMAL_STD
                self.samples.append(entry)
            log.info(f'Wildfire {split}: {len(wf_data)}')

        if solar_index and Path(solar_index).exists():
            sol_data = json.loads(
                Path(solar_index).read_text()
            )
            for entry in sol_data:
                entry['task']         = 'solar'
                entry['thermal_mean'] = SOLAR_THERMAL_MEAN
                entry['thermal_std']  = SOLAR_THERMAL_STD
                self.samples.append(entry)
            log.info(f'Solar {split}: {len(sol_data)}')

        if not self.samples:
            raise RuntimeError(f'No samples for {split}')

        log.info(f'SSL {split}: {len(self.samples)} patches')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item    = self.samples[idx]
        thermal = load_patch(item['thermal_path'])
        thermal = normalize_thermal(
            thermal,
            item['thermal_mean'],
            item['thermal_std'],
        )
        if thermal.ndim == 2:
            thermal = thermal[np.newaxis]
        elif thermal.shape[0] != 1:
            thermal = thermal[0:1]

        if (thermal.shape[1] != self.patch_size or
                thermal.shape[2] != self.patch_size):
            t = torch.from_numpy(thermal).unsqueeze(0)
            t = F.interpolate(
                t,
                size=(self.patch_size, self.patch_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0)
            thermal = t.numpy()

        return {
            'thermal':  torch.from_numpy(thermal).float(),
            'task':     item['task'],
            'patch_id': item['patch_id'],
        }


# ── Augmentation ──────────────────────────────────────────────
def augment_thermal(
    thermal: torch.Tensor,
) -> torch.Tensor:
    """
    Strong thermal augmentation.
    - H/V flip
    - Random crop + resize (70-100%)
    - Gaussian noise
    - Temperature jitter
    - Random erasing
    """
    B, C, H, W = thermal.shape

    if torch.rand(1) > 0.5:
        thermal = torch.flip(thermal, dims=[-1])
    if torch.rand(1) > 0.5:
        thermal = torch.flip(thermal, dims=[-2])

    scale   = 0.7 + torch.rand(1).item() * 0.3
    crop_h  = int(H * scale)
    crop_w  = int(W * scale)
    top     = torch.randint(0, H-crop_h+1, (1,)).item()
    left    = torch.randint(0, W-crop_w+1, (1,)).item()
    thermal = thermal[:, :, top:top+crop_h, left:left+crop_w]
    thermal = F.interpolate(
        thermal, size=(H, W),
        mode='bilinear', align_corners=False,
    )

    thermal = thermal + torch.randn_like(thermal) * 0.2
    jitter  = 0.8 + torch.rand(1).item() * 0.4
    thermal = thermal * jitter

    if torch.rand(1) > 0.5:
        eh = int(H * (0.1 + torch.rand(1).item() * 0.2))
        ew = int(W * (0.1 + torch.rand(1).item() * 0.2))
        et = torch.randint(0, H-eh+1, (1,)).item()
        el = torch.randint(0, W-ew+1, (1,)).item()
        thermal[:, :, et:et+eh, el:el+ew] = 0.0

    return thermal


def mixup_thermal(
    x: torch.Tensor,
    alpha: float = 0.2,
) -> torch.Tensor:
    """MixUp: blend random pairs of patches."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1 - lam) * x[idx]


# ── InfoNCE Loss ──────────────────────────────────────────────
class InfoNCELoss(nn.Module):
    """
    InfoNCE with higher temperature (0.5) to prevent
    easy collapse.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        B  = z1.shape[0]
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        z  = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.temperature
        mask = torch.eye(2*B, device=z.device).bool()
        sim.masked_fill_(mask, float('-inf'))
        labels = torch.cat([
            torch.arange(B, 2*B, device=z.device),
            torch.arange(0, B,   device=z.device),
        ])
        return F.cross_entropy(sim, labels)


# ── LR Warmup + Cosine Schedule ──────────────────────────────
class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs:  int,
        lr_max:        float,
        lr_min:        float = 1e-6,
    ):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.lr_max        = lr_max
        self.lr_min        = lr_min

    def step(self, epoch: int) -> float:
        if epoch <= self.warmup_epochs:
            lr = self.lr_min + (
                self.lr_max - self.lr_min
            ) * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.lr_min + 0.5 * (
                self.lr_max - self.lr_min
            ) * (1 + np.cos(np.pi * progress))

        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ── SSL Trainer ───────────────────────────────────────────────
class SSLTrainer:
    """
    InfoNCE SSL trainer for ThermalEncoder.

    Key design:
      - Random init ThermalEncoder (no ImageNet shortcuts)
      - Two differently augmented views per patch
      - MixUp for additional regularization
      - InfoNCE temperature=0.5
      - LR warmup + cosine decay
      - Early stopping on train_loss plateau
    """

    def __init__(
        self,
        backbone,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          Dict,
        ckpt_dir:     Path,
        device:       torch.device,
    ):
        self.backbone     = backbone.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.ckpt_dir     = Path(ckpt_dir)
        self.device       = device
        self.best_loss    = float('inf')
        self.no_improve   = 0

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Projection head: 512 → 256 → 128
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
        ).to(device)

        self.loss_fn = InfoNCELoss(
            temperature=cfg.get('temperature', 0.5)
        )

        self.optimizer = torch.optim.AdamW(
            list(self.backbone.thermal_encoder.parameters()) +
            list(self.projector.parameters()),
            lr=float(cfg.get('lr_min', 1e-6)),
            weight_decay=float(cfg.get('weight_decay', 0.04)),
        )

        self.scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_epochs=cfg.get('warmup', 10),
            total_epochs=cfg.get('epochs', 100),
            lr_max=float(cfg.get('lr_max', 3e-4)),
            lr_min=float(cfg.get('lr_min', 1e-6)),
        )

    def _forward(
        self,
        thermal: torch.Tensor,
        augment: bool = True,
    ) -> torch.Tensor:
        if augment:
            thermal = augment_thermal(thermal)
        emb = self.backbone.thermal_encoder(thermal)
        return self.projector(emb)

    def _alignment(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> float:
        z1 = F.normalize(z1.detach(), dim=-1)
        z2 = F.normalize(z2.detach(), dim=-1)
        return float(
            F.cosine_similarity(z1, z2, dim=-1).mean()
        )

    def train_epoch(self) -> Tuple[float, float]:
        self.backbone.thermal_encoder.train()
        self.projector.train()
        total_loss  = 0.0
        total_align = 0.0

        for batch in self.train_loader:
            thermal = batch['thermal'].to(self.device)

            # Optional MixUp
            if (self.cfg.get('use_mixup', True) and
                    torch.rand(1).item() > 0.5):
                thermal = mixup_thermal(thermal)

            # Two differently augmented views
            z1 = self._forward(thermal, augment=True)
            z2 = self._forward(thermal, augment=True)

            loss = self.loss_fn(z1, z2)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.backbone.thermal_encoder.parameters()) +
                list(self.projector.parameters()),
                1.0,
            )
            self.optimizer.step()

            total_loss  += loss.item()
            total_align += self._alignment(z1, z2)

        n = len(self.train_loader)
        return total_loss / n, total_align / n

    def val_epoch(self) -> Tuple[float, float]:
        self.backbone.thermal_encoder.eval()
        self.projector.eval()
        total_loss  = 0.0
        total_align = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                thermal = batch['thermal'].to(self.device)
                z1 = self._forward(thermal, augment=True)
                z2 = self._forward(thermal, augment=True)
                loss = self.loss_fn(z1, z2)
                total_loss  += loss.item()
                total_align += self._alignment(z1, z2)

        n = len(self.val_loader)
        return total_loss / n, total_align / n

    def save_checkpoint(
        self,
        epoch:      int,
        train_loss: float,
        val_loss:   float,
    ) -> bool:
        state = {
            'epoch':           epoch,
            'backbone_state':  self.backbone.state_dict(),
            'projector_state': self.projector.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_loss':      train_loss,
            'val_loss':        val_loss,
            'best_loss':       self.best_loss,
            'config':          self.cfg,
        }

        is_best   = False
        min_delta = self.cfg.get('min_delta', 0.001)

        if train_loss < self.best_loss - min_delta:
            self.best_loss = train_loss
            torch.save(
                state, self.ckpt_dir / 'ssl_best.pt'
            )
            is_best = True

        if epoch % 10 == 0:
            ep_path = (
                self.ckpt_dir / f'ssl_epoch{epoch:03d}.pt'
            )
            torch.save(state, ep_path)
            prev = epoch - 10
            if prev > 0:
                old = (
                    self.ckpt_dir /
                    f'ssl_epoch{prev:03d}.pt'
                )
                if old.exists():
                    old.unlink()

        return is_best

    def _find_latest_checkpoint(self) -> Optional[Path]:
        ckpts = sorted(self.ckpt_dir.glob('ssl_epoch*.pt'))
        if ckpts:
            return ckpts[-1]
        best = self.ckpt_dir / 'ssl_best.pt'
        if best.exists():
            return best
        return None

    def train(
        self,
        resume_from: Optional[Path] = None,
    ):
        epochs    = self.cfg.get('epochs', 100)
        warmup    = self.cfg.get('warmup', 10)
        patience  = self.cfg.get('patience', 10)
        min_delta = self.cfg.get('min_delta', 0.001)
        start     = 1

        if resume_from is None:
            resume_from = self._find_latest_checkpoint()

        if resume_from and Path(resume_from).exists():
            ckpt = torch.load(
                resume_from, map_location=self.device
            )
            self.backbone.load_state_dict(
                ckpt['backbone_state']
            )
            self.projector.load_state_dict(
                ckpt['projector_state']
            )
            self.optimizer.load_state_dict(
                ckpt['optimizer_state']
            )
            self.best_loss = ckpt.get(
                'best_loss', float('inf')
            )
            start = ckpt['epoch'] + 1
            print(
                f'Resumed from epoch {ckpt["epoch"]} '
                f'(train={ckpt["train_loss"]:.4f})'
            )
        else:
            # Random init ThermalEncoder
            log.info(
                'Random init ThermalEncoder '
                '(no ImageNet shortcuts)'
            )
            for m in self.backbone.thermal_encoder.modules():
                if isinstance(m, (
                    torch.nn.Conv2d,
                    torch.nn.BatchNorm2d,
                    torch.nn.Linear,
                )):
                    m.reset_parameters()
            log.info('✅ ThermalEncoder reset')

        self.backbone.set_phase('ssl')
        print(
            f'InfoNCE SSL v3: epochs {start}→{epochs} | '
            f'warmup={warmup} patience={patience} '
            f'temp={self.cfg.get("temperature", 0.5)}'
        )
        print('-' * 72)

        for epoch in range(start, epochs + 1):
            t0  = time.time()
            lr  = self.scheduler.step(epoch)

            train_loss, train_align = self.train_epoch()
            val_loss, val_align     = self.val_epoch()

            is_best = self.save_checkpoint(
                epoch, train_loss, val_loss
            )

            # Early stopping on TRAIN loss plateau
            if train_loss < self.best_loss - min_delta:
                self.no_improve = 0
            elif epoch > warmup:
                self.no_improve += 1

            dur        = int(time.time() - t0)
            mins, secs = divmod(dur, 60)
            best_tag   = ' ✅' if is_best else ''
            pat_tag    = (
                f'patience: {self.no_improve}/{patience}'
                if epoch > warmup else 'warmup'
            )

            print(
                f'Epoch {epoch:03d}/{epochs} | '
                f'train={train_loss:.4f} '
                f'val={val_loss:.4f}{best_tag} | '
                f'align={train_align:.3f} | '
                f'lr={lr:.6f} | '
                f'{mins}m {secs}s | {pat_tag}',
                flush=True
            )

            if (epoch > warmup and
                    self.no_improve >= patience):
                print(
                    f'\nEarly stopping at epoch {epoch} '
                    f'(train_loss plateau {patience} epochs)'
                )
                break

        print(
            f'\n✅ SSL complete. '
            f'Best train_loss: {self.best_loss:.4f}'
        )


# ── Entry Point ───────────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    parser = argparse.ArgumentParser(
        description='ThermalWatch InfoNCE SSL v3'
    )
    parser.add_argument(
        '--index-dir',
        default='/content/thermalwatch/data/indexes'
    )
    parser.add_argument(
        '--ckpt-dir',
        default='/gdrive/MyDrive/thermalwatch/checkpoints/ssl'
    )
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--batch-size',  type=int,   default=128)
    parser.add_argument('--lr-max',      type=float, default=3e-4)
    parser.add_argument('--lr-min',      type=float, default=1e-6)
    parser.add_argument('--warmup',      type=int,   default=10)
    parser.add_argument('--patience',    type=int,   default=10)
    parser.add_argument('--temperature', type=float, default=0.5)
    args = parser.parse_args()

    device    = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    index_dir = Path(args.index_dir)

    train_ds = ThermalSSLDataset(
        wildfire_index=index_dir / 'wildfire_train_index.json',
        solar_index=index_dir / 'solar_train_index.json',
        split='train',
    )
    val_ds = ThermalSSLDataset(
        wildfire_index=index_dir / 'wildfire_val_index.json',
        solar_index=index_dir / 'solar_val_index.json',
        split='val',
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    from src.models.backbone.thermal_backbone import (
        ThermalWatchBackbone
    )
    backbone = ThermalWatchBackbone(
        pretrained_optical=True,
        freeze_optical=True,
    )

    cfg = {
        'epochs':       args.epochs,
        'lr_max':       args.lr_max,
        'lr_min':       args.lr_min,
        'weight_decay': 0.04,
        'temperature':  args.temperature,
        'warmup':       args.warmup,
        'patience':     args.patience,
        'min_delta':    0.001,
        'batch_size':   args.batch_size,
        'use_mixup':    True,
    }

    trainer = SSLTrainer(
        backbone=backbone,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        ckpt_dir=Path(args.ckpt_dir),
        device=device,
    )
    trainer.train()


if __name__ == '__main__':
    main()
