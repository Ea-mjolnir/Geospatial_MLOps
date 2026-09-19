"""
ThermalWatch — Fine-tuning
============================
Fine-tunes ThermalWatchBackbone for:
  - Wildfire risk prediction
  - Solar farm health monitoring

Two-phase training:
  Phase 1 (20 epochs): frozen backbone, train heads only
  Phase 2 (20 epochs max): unfreeze last 4 Prithvi layers
                           early stopping patience=5

Resume: automatically resumes from latest checkpoint
  → Phase 1 resumes from wildfire_best_phase1.pt
  → Phase 2 resumes from wildfire_epoch*_phase2.pt

Wildfire metrics:
  risk_score:     RMSE, R², MAE
  alert_level:    Accuracy, F1
  spread_prob:    RMSE, R²
  structure_risk: RMSE, R²

Solar metrics:
  efficiency_score:  RMSE, R², MAE
  hotspot_score:     RMSE, R²
  degradation_rate:  RMSE, R²
  maintenance_flag:  Accuracy, F1, Precision, Recall

Display (one line per epoch):
  P1 Ep 01/20 | train=1.234 val=1.123 ✅ |
  risk_rmse=0.21 risk_r2=0.67 alert_acc=72.3% | 3m 12s

Checkpoints:
  {task}_best_phase1.pt      → best val_loss phase 1
  {task}_best_phase2.pt      → best val_loss phase 2
  {task}_epoch{N}_phase2.pt  → every 10 epochs phase 2

Usage (Colab):
  python3 -m src.models.training.finetune \
    --task      wildfire \
    --index-dir /content/thermalwatch/data/indexes \
    --ckpt-dir  /gdrive/MyDrive/thermalwatch/checkpoints/finetune/wildfire
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


def load_patch(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Missing: {path}')
    return np.load(path).astype(np.float32)


# ── Wildfire Dataset ──────────────────────────────────────────
class WildfireDataset(Dataset):
    def __init__(self, index_path: Path, split: str = 'train'):
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f'Index not found: {index_path}')
        self.split   = split
        self.samples = json.loads(index_path.read_text())
        missing = [s for s in self.samples if s.get('risk_score') is None]
        if missing:
            raise RuntimeError(f'{len(missing)} patches missing labels.')
        with_s2 = sum(1 for s in self.samples if s.get('s2_path'))
        log.info(
            f'WildfireDataset {split}: {len(self.samples)} | '
            f'S2={with_s2/len(self.samples)*100:.1f}%'
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item    = self.samples[idx]
        thermal = load_patch(item['thermal_path'])
        thermal = np.nan_to_num(thermal, nan=0.0)
        if thermal.ndim == 2:
            thermal = thermal[np.newaxis]
        elif thermal.shape[0] != 1:
            thermal = thermal[0:1]
        thermal = (thermal - WILDFIRE_THERMAL_MEAN) / (WILDFIRE_THERMAL_STD + 1e-6)

        s2_path = item.get('s2_path')
        if s2_path and Path(s2_path).exists():
            try:
                optical = load_patch(s2_path)
                optical = np.nan_to_num(optical, nan=0.0)
                optical = (optical - S2_MEAN[:, None, None]) / (S2_STD[:, None, None] + 1e-6)
                has_s2  = True
            except Exception:
                optical = np.zeros((6, 224, 224), dtype=np.float32)
                has_s2  = False
        else:
            optical = np.zeros((6, 224, 224), dtype=np.float32)
            has_s2  = False

        weather_raw = item.get('weather_features')
        weather = (
            (np.array(weather_raw, dtype=np.float32) - WEATHER_MEAN) / (WEATHER_STD + 1e-6)
            if weather_raw and len(weather_raw) == 5
            else np.zeros(5, dtype=np.float32)
        )

        osm_raw = item.get('osm_features', [0.0]*7)
        osm     = np.clip(np.array(osm_raw[:7], dtype=np.float32) / (OSM_MAX + 1e-6), 0, 1)

        return {
            'thermal':  torch.from_numpy(thermal).float(),
            'optical':  torch.from_numpy(optical).float(),
            'weather':  torch.from_numpy(weather).float(),
            'osm':      torch.from_numpy(osm).float(),
            'has_s2':   torch.tensor(has_s2, dtype=torch.bool),
            'targets': {
                'risk_score':     float(item['risk_score']),
                'alert_level':    int(item['alert_level']),
                'spread_prob':    float(item['spread_prob']),
                'structure_risk': float(item['structure_risk']),
            },
            'patch_id': item['patch_id'],
        }


# ── Solar Dataset ─────────────────────────────────────────────
class SolarDataset(Dataset):
    def __init__(self, index_path: Path, split: str = 'train'):
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f'Index not found: {index_path}')
        self.split   = split
        self.samples = json.loads(index_path.read_text())
        missing = [s for s in self.samples if s.get('efficiency_score') is None]
        if missing:
            raise RuntimeError(f'{len(missing)} patches missing labels.')
        log.info(f'SolarDataset {split}: {len(self.samples)}')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        item    = self.samples[idx]
        thermal = load_patch(item['thermal_path'])
        thermal = np.nan_to_num(thermal, nan=0.0)
        if thermal.ndim == 2:
            thermal = thermal[np.newaxis]
        elif thermal.shape[0] != 1:
            thermal = thermal[0:1]
        thermal = (thermal - SOLAR_THERMAL_MEAN) / (SOLAR_THERMAL_STD + 1e-6)
        optical = np.zeros((6, 224, 224), dtype=np.float32)

        weather_raw = item.get('weather_features')
        weather = (
            (np.array(weather_raw, dtype=np.float32) - WEATHER_MEAN) / (WEATHER_STD + 1e-6)
            if weather_raw and len(weather_raw) == 5
            else np.zeros(5, dtype=np.float32)
        )
        osm = np.zeros(7, dtype=np.float32)

        return {
            'thermal':  torch.from_numpy(thermal).float(),
            'optical':  torch.from_numpy(optical).float(),
            'weather':  torch.from_numpy(weather).float(),
            'osm':      torch.from_numpy(osm).float(),
            'has_s2':   torch.tensor(False),
            'targets': {
                'efficiency_score': float(item['efficiency_score']),
                'hotspot_score':    float(item['hotspot_score']),
                'degradation_rate': float(item['degradation_rate']),
                'maintenance_flag': float(item['maintenance_flag']),
            },
            'patch_id': item['patch_id'],
        }


# ── Wildfire Head ─────────────────────────────────────────────
class WildfireHead(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.risk_head      = nn.Linear(128, 1)
        self.alert_head     = nn.Linear(128, 4)
        self.spread_head    = nn.Linear(128, 1)
        self.structure_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(x)
        return {
            'risk_score':     torch.sigmoid(self.risk_head(h)).squeeze(-1),
            'alert_level':    self.alert_head(h),
            'spread_prob':    torch.sigmoid(self.spread_head(h)).squeeze(-1),
            'structure_risk': torch.sigmoid(self.structure_head(h)).squeeze(-1),
        }


# ── Solar Head ────────────────────────────────────────────────
class SolarHead(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.LayerNorm(256),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.efficiency_head  = nn.Linear(128, 1)
        self.hotspot_head     = nn.Linear(128, 1)
        self.degradation_head = nn.Linear(128, 1)
        self.maintenance_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(x)
        return {
            'efficiency_score': torch.sigmoid(self.efficiency_head(h)).squeeze(-1),
            'hotspot_score':    torch.sigmoid(self.hotspot_head(h)).squeeze(-1),
            'degradation_rate': torch.sigmoid(self.degradation_head(h)).squeeze(-1),
            'maintenance_flag': self.maintenance_head(h).squeeze(-1),
        }


# ── Loss Functions ────────────────────────────────────────────
ALERT_WEIGHTS    = torch.tensor([0.351, 1.975, 2.297, 4.708])
MAINT_POS_WEIGHT = torch.tensor([50.0])


def focal_mse_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    mse    = (pred - target) ** 2
    weight = (1 - torch.exp(-torch.abs(pred - target))) ** gamma
    return (weight * mse).mean()


def wildfire_loss(preds: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
    """
    Simplified wildfire loss — standard MSE + CE.
    Natural distribution training for better generalization.
    """
    risk_loss   = F.mse_loss(preds['risk_score'], targets['risk_score'])
    alert_loss  = F.cross_entropy(
        preds['alert_level'], targets['alert_level'].long(),
    )
    spread_loss = F.mse_loss(preds['spread_prob'], targets['spread_prob'])
    struct_loss = F.mse_loss(preds['structure_risk'], targets['structure_risk'])
    total = risk_loss + alert_loss + spread_loss + struct_loss
    return total, {
        'risk': risk_loss.item(), 'alert': alert_loss.item(),
        'spread': spread_loss.item(), 'struct': struct_loss.item(),
    }


def solar_loss(preds: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
    device    = preds['efficiency_score'].device
    eff_loss  = focal_mse_loss(preds['efficiency_score'], targets['efficiency_score'])
    hot_loss  = focal_mse_loss(preds['hotspot_score'], targets['hotspot_score'])
    deg_loss  = F.mse_loss(preds['degradation_rate'], targets['degradation_rate'])
    maint_loss = F.binary_cross_entropy_with_logits(
        preds['maintenance_flag'], targets['maintenance_flag'],
        pos_weight=MAINT_POS_WEIGHT.to(device),
    )
    total = 3.0*eff_loss + 2.0*hot_loss + 2.0*deg_loss + 0.5*maint_loss
    return total, {
        'efficiency': eff_loss.item(), 'hotspot': hot_loss.item(),
        'degradation': deg_loss.item(), 'maintenance': maint_loss.item(),
    }


# ── Metrics ───────────────────────────────────────────────────
def compute_wildfire_metrics(all_preds: List[Dict], all_targets: List[Dict]) -> Dict:
    from sklearn.metrics import f1_score, accuracy_score

    def rmse(p, t): return float(np.sqrt(np.mean((p-t)**2)))
    def r2(p, t):
        ss_res = np.sum((t-p)**2); ss_tot = np.sum((t-t.mean())**2)
        return float(1 - ss_res/(ss_tot+1e-8))
    def mae(p, t): return float(np.mean(np.abs(p-t)))
    def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)

    risk      = np.concatenate([to_np(d['risk_score']) for d in all_preds])
    gt_risk   = np.concatenate([to_np(b['risk_score']) for b in all_targets])
    alert     = np.concatenate([
        to_np(d['alert_level'].argmax(-1) if isinstance(d['alert_level'], torch.Tensor)
              else np.argmax(d['alert_level'], axis=-1)) for d in all_preds
    ])
    gt_alert  = np.concatenate([to_np(b['alert_level']) for b in all_targets]).astype(int)
    spread    = np.concatenate([to_np(d['spread_prob']) for d in all_preds])
    gt_spread = np.concatenate([to_np(b['spread_prob']) for b in all_targets])
    struct    = np.concatenate([to_np(d['structure_risk']) for d in all_preds])
    gt_struct = np.concatenate([to_np(b['structure_risk']) for b in all_targets])

    return {
        'risk_rmse': rmse(risk, gt_risk), 'risk_r2': r2(risk, gt_risk), 'risk_mae': mae(risk, gt_risk),
        'alert_acc': accuracy_score(gt_alert, alert),
        'alert_f1':  f1_score(gt_alert, alert, average='macro', zero_division=0),
        'spread_rmse': rmse(spread, gt_spread), 'spread_r2': r2(spread, gt_spread),
        'struct_rmse': rmse(struct, gt_struct), 'struct_r2': r2(struct, gt_struct),
    }


def compute_solar_metrics(all_preds: List[Dict], all_targets: List[Dict]) -> Dict:
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    def rmse(p, t): return float(np.sqrt(np.mean((p-t)**2)))
    def r2(p, t):
        ss_res = np.sum((t-p)**2); ss_tot = np.sum((t-t.mean())**2)
        return float(1 - ss_res/(ss_tot+1e-8))
    def mae(p, t): return float(np.mean(np.abs(p-t)))
    def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)

    eff      = np.concatenate([to_np(d['efficiency_score']) for d in all_preds])
    gt_eff   = np.concatenate([to_np(b['efficiency_score']) for b in all_targets])
    hot      = np.concatenate([to_np(d['hotspot_score']) for d in all_preds])
    gt_hot   = np.concatenate([to_np(b['hotspot_score']) for b in all_targets])
    deg      = np.concatenate([to_np(d['degradation_rate']) for d in all_preds])
    gt_deg   = np.concatenate([to_np(b['degradation_rate']) for b in all_targets])
    maint    = np.concatenate([to_np(d['maintenance_flag']) for d in all_preds])
    gt_maint = np.concatenate([to_np(b['maintenance_flag']) for b in all_targets])
    maint_bin    = (torch.sigmoid(torch.tensor(maint)) > 0.5).numpy().astype(int)
    gt_maint_bin = gt_maint.astype(int)

    return {
        'eff_rmse': rmse(eff, gt_eff), 'eff_r2': r2(eff, gt_eff), 'eff_mae': mae(eff, gt_eff),
        'hot_rmse': rmse(hot, gt_hot), 'hot_r2': r2(hot, gt_hot),
        'deg_rmse': rmse(deg, gt_deg), 'deg_r2': r2(deg, gt_deg),
        'maint_acc':  accuracy_score(gt_maint_bin, maint_bin),
        'maint_f1':   f1_score(gt_maint_bin, maint_bin, zero_division=0),
        'maint_prec': precision_score(gt_maint_bin, maint_bin, zero_division=0),
        'maint_rec':  recall_score(gt_maint_bin, maint_bin, zero_division=0),
    }


# ── Fine-tuning Trainer ───────────────────────────────────────
class FinetuneTrainer:
    """
    Two-phase fine-tuning trainer with RESUME support.

    Phase 1 (20 epochs):
      Backbone frozen → heads only
      Auto-resumes from {task}_best_phase1.pt

    Phase 2 (20 epochs max):
      Last 4 Prithvi layers unfrozen
      batch_size reduced to 32 (OOM prevention)
      Early stopping: warmup=3, patience=5
      Auto-resumes from {task}_epoch*_phase2.pt
      Saves every 10 epochs + best
    """

    def __init__(
        self, backbone, head,
        train_loader: DataLoader, val_loader: DataLoader,
        loss_fn, metrics_fn, cfg: Dict,
        ckpt_dir: Path, device: torch.device, task: str,
    ):
        self.backbone     = backbone.to(device)
        self.head         = head.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.loss_fn      = loss_fn
        self.metrics_fn   = metrics_fn
        self.cfg          = cfg
        self.ckpt_dir     = Path(ckpt_dir)
        self.device       = device
        self.task         = task
        self.best_val     = float('inf')
        self.no_improve   = 0
        self.optimizer    = None
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _find_latest_checkpoint(self, phase: int) -> Optional[Path]:
        """Find most recent checkpoint for given phase.
        Prefers epoch checkpoints over best checkpoint
        since epoch checkpoints have exact epoch number.
        """
        # Epoch checkpoints (most specific)
        ckpts = sorted(
            self.ckpt_dir.glob(f'{self.task}_epoch*_phase{phase}.pt')
        )
        if ckpts:
            log.info(f'Found resume checkpoint: {ckpts[-1].name}')
            return ckpts[-1]
        # Best checkpoint
        best = self.ckpt_dir / f'{self.task}_best_phase{phase}.pt'
        if best.exists():
            log.info(f'Found best checkpoint: {best.name}')
            return best
        return None

    def _targets_to_device(self, targets) -> Dict:
        if isinstance(targets, dict):
            return {
                k: v.float().to(self.device) if isinstance(v, torch.Tensor)
                else torch.tensor(v, dtype=torch.float32).to(self.device)
                for k, v in targets.items()
            }
        keys = targets[0].keys()
        return {
            k: torch.tensor([t[k] for t in targets], dtype=torch.float32).to(self.device)
            for k in keys
        }

    def _forward(self, batch: Dict) -> Tuple[Dict, Dict]:
        thermal = batch['thermal'].to(self.device)
        optical = batch['optical'].to(self.device)
        weather = batch['weather'].to(self.device)
        osm     = batch['osm'].to(self.device)
        has_s2  = batch['has_s2']
        mask    = has_s2.to(self.device).float()
        optical = optical * mask.view(-1, 1, 1, 1)
        emb     = self.backbone(
            thermal=thermal,
            optical=optical if has_s2.any() else None,
            weather=weather, osm=osm,
        )
        preds   = self.head(emb)
        targets = self._targets_to_device(batch['targets'])
        return preds, targets

    def run_epoch(self, train: bool) -> Tuple[float, Dict, Dict]:
        if train:
            self.backbone.train(); self.head.train()
        else:
            self.backbone.eval(); self.head.eval()

        total_loss  = 0.0
        comp_losses = {}
        all_preds   = []
        all_targets = []
        loader      = self.train_loader if train else self.val_loader
        ctx         = torch.enable_grad() if train else torch.no_grad()

        with ctx:
            for batch in loader:
                preds, targets = self._forward(batch)
                loss, comps    = self.loss_fn(preds, targets)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.backbone.parameters()) +
                        list(self.head.parameters()), 1.0,
                    )
                    self.optimizer.step()
                total_loss += loss.item()
                for k, v in comps.items():
                    comp_losses[k] = comp_losses.get(k, 0) + v
                all_preds.append({k: v.detach() for k, v in preds.items()})
                all_targets.append({k: v.detach().cpu().numpy() for k, v in targets.items()})

        n = len(loader)
        return total_loss/n, {k: v/n for k, v in comp_losses.items()}, \
               self.metrics_fn(all_preds, all_targets)

    def _format_metrics(self, metrics: Dict, task: str) -> str:
        if task == 'wildfire':
            return (
                f'risk_rmse={metrics["risk_rmse"]:.3f} '
                f'risk_r2={metrics["risk_r2"]:.3f} '
                f'alert_acc={metrics["alert_acc"]*100:.1f}% '
                f'alert_f1={metrics["alert_f1"]:.3f}'
            )
        return (
            f'eff_rmse={metrics["eff_rmse"]:.3f} '
            f'eff_r2={metrics["eff_r2"]:.3f} '
            f'maint_acc={metrics["maint_acc"]*100:.1f}% '
            f'maint_f1={metrics["maint_f1"]:.3f}'
        )

    def save_checkpoint(self, epoch: int, phase: int, val_loss: float) -> bool:
        state = {
            'epoch':          epoch,
            'phase':          phase,
            'backbone_state': self.backbone.state_dict(),
            'head_state':     self.head.state_dict(),
            'optimizer_state':self.optimizer.state_dict(),
            'val_loss':       val_loss,
            'best_val':       self.best_val,
            'no_improve':     self.no_improve,
            'task':           self.task,
            'config':         self.cfg,
        }
        is_best   = False
        min_delta = self.cfg.get('min_delta', 0.001)
        if val_loss < self.best_val - min_delta:
            self.best_val = val_loss
            torch.save(state, self.ckpt_dir / f'{self.task}_best_phase{phase}.pt')
            is_best = True
        if phase == 2 and epoch % 5 == 0:
            ep_path = self.ckpt_dir / f'{self.task}_epoch{epoch:03d}_phase2.pt'
            torch.save(state, ep_path)
            prev = epoch - 5
            if prev > 0:
                old = self.ckpt_dir / f'{self.task}_epoch{prev:03d}_phase2.pt'
                if old.exists():
                    old.unlink()

        # Also save phase 1 every 5 epochs for resume
        if phase == 1 and epoch % 5 == 0:
            ep_path = self.ckpt_dir / f'{self.task}_epoch{epoch:03d}_phase1.pt'
            torch.save(state, ep_path)
            prev = epoch - 5
            if prev > 0:
                old = self.ckpt_dir / f'{self.task}_epoch{prev:03d}_phase1.pt'
                if old.exists():
                    old.unlink()
        return is_best

    def train_phase1(self, ssl_ckpt: Optional[Path]):
        """Phase 1: frozen backbone. Auto-resumes if checkpoint exists."""
        start_epoch = 1
        self.best_val = float('inf')

        # ── Try to resume Phase 1 ──────────────────────────
        resume = self._find_latest_checkpoint(phase=1)
        if resume:
            ckpt = torch.load(resume, map_location=self.device)
            self.backbone.load_state_dict(ckpt['backbone_state'])
            self.head.load_state_dict(ckpt['head_state'])
            self.best_val = ckpt.get('best_val', float('inf'))
            start_epoch   = ckpt.get('epoch', 0) + 1
            print(
                f'✅ Resumed Phase 1 from epoch {ckpt["epoch"]} '
                f'(val={ckpt["val_loss"]:.4f})'
            )
        elif ssl_ckpt and Path(ssl_ckpt).exists():
            ckpt = torch.load(ssl_ckpt, map_location=self.device)
            self.backbone.load_state_dict(ckpt['backbone_state'])
            print(f'✅ SSL checkpoint loaded: {ssl_ckpt}')
        else:
            print('⚠️  No checkpoint — using ImageNet pre-trained weights ✅')

        # Freeze backbone
        self.backbone.set_phase('ssl')
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=float(self.cfg.get('lr_head', 1e-3)),
            weight_decay=0.01,
        )

        epochs = self.cfg.get('phase1_epochs', 20)

        if start_epoch > epochs:
            print(f'Phase 1 already complete ({epochs} epochs) — skipping')
            return

        print(f'\n{"="*60}')
        print(f'PHASE 1: Frozen backbone — {self.task}')
        print(f'Epochs: {start_epoch}→{epochs} | lr_head={self.cfg.get("lr_head", 1e-3)}')
        print('='*60)

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            train_loss, _, _         = self.run_epoch(True)
            val_loss, _, val_metrics = self.run_epoch(False)
            is_best    = self.save_checkpoint(epoch, 1, val_loss)
            dur        = int(time.time() - t0)
            mins, secs = divmod(dur, 60)
            print(
                f'P1 Ep {epoch:02d}/{epochs} | '
                f'train={train_loss:.4f} val={val_loss:.4f}'
                f'{"✅" if is_best else ""} | '
                f'{self._format_metrics(val_metrics, self.task)} | '
                f'{mins}m {secs}s',
                flush=True
            )

        print(f'\nPhase 1 complete. Best val: {self.best_val:.4f}')

    def train_phase2(self):
        """Phase 2: unfreeze Prithvi. Auto-resumes if checkpoint exists."""
        # Aggressively free GPU memory from Phase 1
        import gc
        # Move models to CPU first to free GPU memory
        self.backbone.cpu()
        self.head.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            log.info(
                f'GPU memory after cache clear: '
                f'{torch.cuda.memory_allocated()/1e9:.2f}GB allocated'
            )
        # Move models back to GPU
        self.backbone = self.backbone.to(self.device)
        self.head     = self.head.to(self.device)

        # Unfreeze ThermalEncoder only (not Prithvi)
        # Prithvi is already trained on geospatial data
        # ThermalEncoder needs domain adaptation
        # Unfreezing Prithvi causes OOM on T4 14GB
        self.backbone.set_phase('ssl')  # keep Prithvi frozen
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze ThermalEncoder last 2 layers only
        thermal_layers = list(
            self.backbone.thermal_encoder.children()
        )
        for layer in thermal_layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Unfreeze fusion layers
        for p in self.backbone.projection.parameters():
            p.requires_grad = True
        for p in self.backbone.fusion.parameters():
            p.requires_grad = True

        trainable = sum(
            p.numel() for p in self.backbone.parameters()
            if p.requires_grad
        )
        log.info(
            f'Phase 2 trainable backbone params: '
            f'{trainable/1e6:.1f}M'
        )

        # Reduce batch size for Prithvi gradients
        phase2_batch = min(128, self.cfg.get('batch_size', 256))
        log.info(f'Phase 2 batch size: {phase2_batch}')
        self.train_loader = torch.utils.data.DataLoader(
            self.train_loader.dataset,
            batch_size=phase2_batch,
            sampler=self.train_loader.sampler,
            num_workers=2, pin_memory=True, drop_last=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_loader.dataset,
            batch_size=phase2_batch,
            shuffle=False, num_workers=2, pin_memory=True,
        )

        self.optimizer = torch.optim.AdamW([
            {'params': self.backbone.parameters(),
             'lr': float(self.cfg.get('lr_backbone', 1e-5))},
            {'params': self.head.parameters(),
             'lr': float(self.cfg.get('lr_head', 1e-3)) * 0.1},
        ], weight_decay=0.05)

        epochs      = self.cfg.get('phase2_epochs', 20)
        patience    = self.cfg.get('patience', 5)
        warmup      = self.cfg.get('phase2_warmup', 3)
        min_delta   = self.cfg.get('min_delta', 0.001)
        start_epoch = 1
        self.best_val   = float('inf')
        self.no_improve = 0

        # ── Try to resume Phase 2 ──────────────────────────
        resume = self._find_latest_checkpoint(phase=2)
        if resume:
            ckpt = torch.load(resume, map_location=self.device)
            self.backbone.load_state_dict(ckpt['backbone_state'])
            self.head.load_state_dict(ckpt['head_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.best_val   = ckpt.get('best_val', float('inf'))
            self.no_improve = ckpt.get('no_improve', 0)
            start_epoch     = ckpt.get('epoch', 0) + 1
            print(
                f'✅ Resumed Phase 2 from epoch {ckpt["epoch"]} '
                f'(val={ckpt["val_loss"]:.4f} '
                f'patience={self.no_improve}/{patience})'
            )
        else:
            # Load best phase 1 checkpoint as starting point
            phase1_best = self.ckpt_dir / f'{self.task}_best_phase1.pt'
            if phase1_best.exists():
                ckpt = torch.load(phase1_best, map_location=self.device)
                self.backbone.load_state_dict(ckpt['backbone_state'])
                self.head.load_state_dict(ckpt['head_state'])
                print(f'✅ Starting Phase 2 from best Phase 1 checkpoint')

        if start_epoch > epochs:
            print(f'Phase 2 already complete ({epochs} epochs) — skipping')
            return

        print(f'\n{"="*60}')
        print(f'PHASE 2: Partial unfreeze — {self.task}')
        print(
            f'Epochs: {start_epoch}→{epochs} | '
            f'lr_backbone={self.cfg.get("lr_backbone", 1e-5)} '
            f'lr_head={self.cfg.get("lr_head", 1e-3)} | '
            f'patience={patience} | batch={phase2_batch}'
        )
        print('='*60)

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            train_loss, _, _         = self.run_epoch(True)
            val_loss, _, val_metrics = self.run_epoch(False)
            is_best = self.save_checkpoint(epoch, 2, val_loss)

            if val_loss < self.best_val - min_delta:
                self.no_improve = 0
            elif epoch > warmup:
                self.no_improve += 1

            dur        = int(time.time() - t0)
            mins, secs = divmod(dur, 60)
            pat_tag    = (
                f'patience: {self.no_improve}/{patience}'
                if epoch > warmup else 'warmup'
            )
            print(
                f'P2 Ep {epoch:02d}/{epochs} | '
                f'train={train_loss:.4f} val={val_loss:.4f}'
                f'{"✅" if is_best else ""} | '
                f'{self._format_metrics(val_metrics, self.task)} | '
                f'{mins}m {secs}s | {pat_tag}',
                flush=True
            )

            if epoch > warmup and self.no_improve >= patience:
                print(f'\nEarly stopping at epoch {epoch}')
                break

        print(f'\nPhase 2 complete. Best val: {self.best_val:.4f}')

    def train(self, ssl_ckpt: Optional[Path] = None):
        """Full fine-tuning: phase1 → phase2."""
        self.train_phase1(ssl_ckpt)
        self.train_phase2()
        print(f'\n✅ Fine-tuning complete — {self.task}')


# ── Entry Point ───────────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    parser = argparse.ArgumentParser(description='ThermalWatch Fine-tuning')
    parser.add_argument('--task', choices=['wildfire', 'solar'], required=True)
    parser.add_argument('--index-dir', default='/content/thermalwatch/data/indexes')
    parser.add_argument('--ssl-ckpt',  default=None)
    parser.add_argument('--ckpt-dir',  default='/gdrive/MyDrive/thermalwatch/checkpoints/finetune')
    parser.add_argument('--batch-size',     type=int,   default=256)
    parser.add_argument('--phase1-epochs',  type=int,   default=20)
    parser.add_argument('--phase2-epochs',  type=int,   default=20)
    parser.add_argument('--lr-head',        type=float, default=1e-3)
    parser.add_argument('--lr-backbone',    type=float, default=1e-5)
    parser.add_argument('--patience',       type=int,   default=5)
    args = parser.parse_args()

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device} | Task: {args.task}')

    index_dir = Path(args.index_dir)
    ckpt_dir  = Path(args.ckpt_dir)

    if args.task == 'wildfire':
        train_ds   = WildfireDataset(index_dir / 'wildfire_train_index.json', 'train')
        val_ds     = WildfireDataset(index_dir / 'wildfire_val_index.json', 'val')
        head       = WildfireHead(512)
        loss_fn    = wildfire_loss
        metrics_fn = compute_wildfire_metrics
    else:
        train_ds   = SolarDataset(index_dir / 'solar_train_index.json', 'train')
        val_ds     = SolarDataset(index_dir / 'solar_val_index.json', 'val')
        head       = SolarHead(512)
        loss_fn    = solar_loss
        metrics_fn = compute_solar_metrics

    # WeightedRandomSampler for class imbalance
    from torch.utils.data import WeightedRandomSampler
    sample_weights = []
    for entry in train_ds.samples:
        alert = entry.get('alert_level', 0)
        w     = [0.351, 1.975, 2.297, 4.708]
        sample_weights.append(w[alert] if args.task == 'wildfire' else 1.0)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=2,
    )

    from src.models.backbone.thermal_backbone import ThermalWatchBackbone
    backbone = ThermalWatchBackbone(pretrained_optical=True, freeze_optical=True)

    cfg = {
        'phase1_epochs': args.phase1_epochs,
        'phase2_epochs': args.phase2_epochs,
        'lr_head':       args.lr_head,
        'lr_backbone':   args.lr_backbone,
        'patience':      args.patience,
        'phase2_warmup': 3,
        'min_delta':     0.001,
        'batch_size':    args.batch_size,
        'task':          args.task,
    }

    trainer = FinetuneTrainer(
        backbone=backbone, head=head,
        train_loader=train_loader, val_loader=val_loader,
        loss_fn=loss_fn, metrics_fn=metrics_fn,
        cfg=cfg, ckpt_dir=ckpt_dir,
        device=device, task=args.task,
    )
    trainer.train(ssl_ckpt=Path(args.ssl_ckpt) if args.ssl_ckpt else None)


if __name__ == '__main__':
    main()
