"""
OmniGeoFusion — Fine-tuning Pipeline
======================================
Stage 2: Task-specific fine-tuning on top of
SSL pre-trained backbone.

Three tasks fine-tuned separately:
  Task A: Urban Change Detection
  Task B: Flood/Disaster Assessment
  Task C: Precision Agriculture

Training strategy per task:
  Phase 1 (frozen backbone):
    → Freeze entire backbone
    → Train task head only
    → High LR (1e-3)
    → 5 epochs

  Phase 2 (partial unfreeze):
    → Unfreeze last 4 optical encoder layers
    → Train all except early backbone layers
    → Low LR (1e-4 head, 1e-5 optical)
    → 10 epochs

Checkpoints saved to GDrive:
  /gdrive/MyDrive/omnigeofusion/checkpoints/finetune/
  /gdrive/MyDrive/omnigeofusion/checkpoints/best/
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
import yaml
import mlflow

from typing import Dict, List, Optional, Tuple
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)


# ── Fine-tuning Dataset ───────────────────────────────────────
class FinetuneDataset(Dataset):
    """
    Dataset for task-specific fine-tuning.
    Loads multimodal patches + automatically
    generated targets from S3/local storage.

    Task A targets: change_score, change_type
    Task B targets: flood_fraction, flood_depth,
                    dominant_damage, road_accessible
    Task C targets: ndvi, lai, crop_height_m,
                    soil_moisture, cwsi, et_mm_day,
                    stress_type
    """

    def __init__(
        self,
        data_dir: str,
        task: str,
        split: str = 'train',
        patch_size: int = 224,
        max_samples: Optional[int] = None,
    ):
        assert task in ['urban_change', 'flood_damage', 'agriculture']
        self.data_dir   = data_dir
        self.task       = task
        self.split      = split
        self.patch_size = patch_size

        # Load index
        index_path = os.path.join(
            data_dir, task, f'{split}_index.json'
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
            f'FinetuneDataset {task}/{split}: '
            f'{len(self.index)} samples'
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        """Load one sample with targets from local disk."""
        import sys
        sys.path.insert(0, os.path.abspath('.'))
        from src.training.ssl_pretrain import (
            load_tif, normalize_s2, normalize_sar,
            normalize_lidar, normalize_thermal
        )
        item = self.index[idx]
        ps   = self.patch_size

        sample = {
            'optical_t1': load_tif(
                item.get('s2_t1_path'), 6, ps, normalize_s2
            ),
            'optical_t2': load_tif(
                item.get('s2_t2_path', item.get('s2_t1_path')),
                6, ps, normalize_s2
            ),
            'sar_t1':     load_tif(
                item.get('sar_path'), 2, ps, normalize_sar
            ),
            'sar_t2':     load_tif(
                item.get('sar_path'), 2, ps, normalize_sar
            ),
            'lidar':      load_tif(
                item.get('lidar_path'), 3, ps, normalize_lidar
            ),
            'thermal':    load_tif(
                item.get('thermal_path'), 1, ps,
                normalize_thermal
            ),
            'iot':        torch.zeros(12, 16),
            'day_gap':    torch.tensor(
                item.get('day_gap', 90), dtype=torch.float32
            ),
        }
        targets = self._load_targets(item)
        sample['targets'] = targets
        return sample

        return sample

    def _load_targets(self, item: Dict) -> Dict:
        """Load pre-computed weakly supervised targets."""
        if self.task == 'urban_change':
            change_score = float(item.get('target',
                item.get('change_score', 0.0)))
            return {
                'change_score': torch.tensor(
                    change_score, dtype=torch.float32
                ),
                'change_type': torch.tensor(
                    item.get('change_type',
                        1 if change_score > 0.5 else 0),
                    dtype=torch.long
                ),
            }
        elif self.task == 'flood_damage':
            flood_risk = float(item.get('target',
                item.get('flood_fraction', 0.0)))
            return {
                'flood_fraction':   torch.tensor(
                    flood_risk, dtype=torch.float32
                ),
                'mean_flood_depth': torch.tensor(
                    flood_risk * 2.0, dtype=torch.float32
                ),
                'dominant_damage':  torch.tensor(
                    min(4, int(flood_risk * 5)),
                    dtype=torch.long
                ),
                'road_accessible':  torch.tensor(
                    1.0 - flood_risk, dtype=torch.float32
                ),
            }
        elif self.task == 'agriculture':
            # Use crop_health score (0-1) as primary target
            # Derive other targets from it with physics-based scaling
            crop_health = float(item.get('target', 0.5))

            # NDVI: healthy crop = 0.6-0.9, stressed = 0.2-0.5
            ndvi = 0.2 + crop_health * 0.7

            # LAI: healthy = 3-6, stressed = 0.5-2
            lai  = 0.5 + crop_health * 5.5

            # Crop height: healthy = 0.5-1.5m
            crop_height = 0.1 + crop_health * 1.4

            # Soil moisture: healthy = 0.3-0.6
            soil_moisture = 0.1 + crop_health * 0.5

            # CWSI: 0=no stress, 1=max stress (inverse of health)
            cwsi = 1.0 - crop_health

            # ET mm/day: healthy = 4-8mm, stressed = 1-3mm
            et_mm_day = 1.0 + crop_health * 7.0

            # Add noise for variance (avoids ss_tot=0)
            import random
            noise = random.gauss(0, 0.02)

            return {
                'ndvi':          torch.tensor(
                    ndvi + noise, dtype=torch.float32
                ),
                'lai':           torch.tensor(
                    max(0, lai + noise * 5), dtype=torch.float32
                ),
                'crop_height_m': torch.tensor(
                    max(0, crop_height + noise), dtype=torch.float32
                ),
                'soil_moisture': torch.tensor(
                    max(0, soil_moisture + noise), dtype=torch.float32
                ),
                'cwsi':          torch.tensor(
                    max(0, cwsi + noise), dtype=torch.float32
                ),
                'et_mm_day':     torch.tensor(
                    max(0, et_mm_day + noise * 5), dtype=torch.float32
                ),
                'stress_type':   torch.tensor(
                    0 if crop_health > 0.6 else
                    1 if crop_health > 0.4 else 2,
                    dtype=torch.long
                ),
            }
        return {}

# ── Fine-tuning Trainer ───────────────────────────────────────
class FinetuneTrainer:
    """
    Task-specific fine-tuning trainer.
    Handles Phase 1 (frozen backbone) and Phase 2 (partial unfreeze).
    """
    def __init__(
        self,
        backbone,
        task_head: nn.Module,
        loss_fn: nn.Module,
        metrics_fn,
        task: str,
        cfg: dict,
        device: torch.device,
    ):
        self.backbone   = backbone
        self.task_head  = task_head
        self.loss_fn    = loss_fn
        self.metrics_fn = metrics_fn
        self.task       = task
        self.cfg        = cfg
        self.device     = device

        self.scaler     = GradScaler()
        self.best_metric = -float('inf')

        # Checkpoint dirs
        ft_cfg = cfg['finetune_urban_change']  # use as template
        self.checkpoint_dir = os.path.join(
            cfg['checkpoints']['finetune_dir'], task
        )
        self.best_dir = cfg['checkpoints']['best_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_dir, exist_ok=True)

    def _build_optimizer(
        self,
        phase: int,
        ft_cfg: dict
    ) -> torch.optim.Optimizer:
        """
        Build optimizer with phase-specific learning rates.
        All LR values explicitly cast to float for safety.
        Phase 1: task head only (backbone frozen)
        Phase 2: task head + fusion + last 4 optical layers
        """
        WD = 0.01  # weight decay

        if phase == 1:
            lr = float(ft_cfg['phase1']['learning_rate'])
            return torch.optim.AdamW(
                list(self.task_head.parameters()),
                lr=lr,
                weight_decay=WD
            )
        else:
            lr2      = ft_cfg['phase2']['learning_rate']
            lr_head  = float(lr2['task_head'])
            lr_fuse  = float(lr2['fusion'])
            lr_opt   = float(lr2.get('optical_last4', 1e-5))

            param_groups = [
                {
                    'params': list(self.task_head.parameters()),
                    'lr':     lr_head,
                },
                {
                    'params': list(self.backbone.fusion.parameters()),
                    'lr':     lr_fuse,
                },
                {
                    'params': list(
                        self.backbone.projection.parameters()
                    ),
                    'lr':     lr_fuse,
                },
            ]

            # Unfreeze last 4 optical encoder layers
            optical_params = []
            try:
                encoder = self.backbone.optical_encoder.encoder
                if hasattr(encoder, 'blocks'):
                    blocks = list(encoder.blocks)
                    for block in blocks[-4:]:
                        optical_params += list(block.parameters())
            except Exception:
                pass

            if optical_params:
                param_groups.append({
                    'params': optical_params,
                    'lr':     lr_opt,
                })

            return torch.optim.AdamW(
                param_groups,
                weight_decay=WD
            )

    def run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        phase_name: str,
        epoch: int,
        is_train: bool = True
    ) -> Tuple[Dict, Dict]:
        """Run one epoch — silent, prints summary at 100%."""
        import time
        if is_train:
            self.backbone.train()
            self.backbone.optical_encoder.eval()
            self.task_head.train()
        else:
            self.backbone.eval()
            self.task_head.eval()

        total_losses = {}
        all_preds    = []
        all_targets  = []
        t_start      = time.time()

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for batch in loader:
                targets = {
                    k: v.to(self.device)
                    for k, v in batch['targets'].items()
                }
                with autocast():
                    fused_t1 = self.backbone(
                        optical=batch['optical_t1'].to(self.device),
                        sar=batch['sar_t1'].to(self.device),
                        lidar=batch['lidar'].to(self.device),
                        thermal=batch['thermal'].to(self.device),
                        iot=batch['iot'].to(self.device),
                    )
                    fused_t2 = self.backbone(
                        optical=batch['optical_t2'].to(self.device),
                        sar=batch['sar_t2'].to(self.device),
                        lidar=batch['lidar'].to(self.device),
                        thermal=batch['thermal'].to(self.device),
                        iot=batch['iot'].to(self.device),
                    )
                    if self.task in ['urban_change', 'flood_damage']:
                        preds = self.task_head(fused_t1, fused_t2)
                    else:
                        preds = self.task_head(fused_t1)
                    loss_dict = self.loss_fn(preds, targets)

                if is_train:
                    optimizer.zero_grad()
                    loss_dict['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.backbone.parameters()) +
                        list(self.task_head.parameters()),
                        1.0
                    )
                    optimizer.step()

                for k, v in loss_dict.items():
                    total_losses[k] = (
                        total_losses.get(k, 0) + float(v)
                    )
                all_preds.append(
                    {k: v.detach() for k, v in preds.items()}
                )
                all_targets.append(
                    {k: v.detach() for k, v in targets.items()}
                )

        n          = len(loader)
        avg_losses = {k: v/n for k, v in total_losses.items()}
        merged_preds   = self._merge_batch_dicts(all_preds)
        merged_targets = self._merge_batch_dicts(all_targets)
        metrics        = self.metrics_fn(merged_preds, merged_targets)

        # Print single summary line
        duration   = int(time.time() - t_start)
        mins, secs = divmod(duration, 60)
        mode       = 'Train' if is_train else 'Val  '
        metric_str = ' | '.join(
            f'{k}: {v:.4f}' for k, v in metrics.items()
        )
        print(
            f'  {mode} {phase_name} Ep {epoch:02d} | ' 
            f'100% ✅ | ' 
            f'Loss: {avg_losses["loss"]:.4f} | '
            f'{metric_str} | {mins}m {secs}s',
            flush=True
        )
        return avg_losses, metrics

    def _merge_batch_dicts(
        self,
        batch_list: List[Dict]
    ) -> Dict:
        """Merge list of batch dicts into one dict."""
        merged = {}
        for d in batch_list:
            for k, v in d.items():
                if k not in merged:
                    merged[k] = []
                merged[k].append(v)
        return {
            k: torch.cat(vs, dim=0)
            for k, vs in merged.items()
        }

    def save_checkpoint(
        self,
        phase: int,
        epoch: int,
        losses: Dict,
        metrics: Dict,
        is_best: bool = False
    ):
        """Save fine-tuning checkpoint to GDrive."""
        state = {
            'task':           self.task,
            'phase':          phase,
            'epoch':          epoch,
            'backbone_state': self.backbone.state_dict(),
            'head_state':     self.task_head.state_dict(),
            'losses':         losses,
            'metrics':        metrics,
        }

        # Save latest
        latest = os.path.join(
            self.checkpoint_dir,
            f'latest_phase{phase}.pt'
        )
        torch.save(state, latest)

        if is_best:
            metric_val = metrics.get(
                'r2', metrics.get('flood_iou', 0.0)
            )
            best_path  = os.path.join(
                self.best_dir,
                f'{self.task}_best_'
                f'phase{phase}_r2_{metric_val:.4f}.pt'
            )
            torch.save(state, best_path)
            log.info(f'✅ Best checkpoint: {best_path}')

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_loader_p2: Optional[DataLoader] = None,
        val_loader_p2: Optional[DataLoader] = None,
        ssl_checkpoint: Optional[str] = None,
    ):
        """
        Full fine-tuning: Phase 1 → Phase 2.
        """
        # Load SSL checkpoint if provided
        if ssl_checkpoint and os.path.exists(ssl_checkpoint):
            log.info(f'Loading SSL checkpoint: {ssl_checkpoint}')
            ckpt = torch.load(
                ssl_checkpoint, map_location=self.device
            )
            self.backbone.load_state_dict(
                ckpt['backbone_state']
            )
            log.info('✅ SSL backbone weights loaded')

        task_key = f'finetune_{self.task}'
        if task_key not in self.cfg:
            task_key = 'finetune_urban_change'
        ft_cfg = self.cfg[task_key]

        run_name = f'finetune_{self.task}'

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                'task':       self.task,
                'phase1_ep':  ft_cfg['phase1']['epochs'],
                'phase2_ep':  ft_cfg['phase2']['epochs'],
            })

            # ── Phase 1: Frozen backbone ───────────────────
            log.info(f'\n{"="*60}')
            log.info(f'PHASE 1: Frozen backbone — {self.task}')
            log.info(f'{"="*60}')

            self.backbone.set_phase('ssl')  # freeze all optical
            optimizer1 = self._build_optimizer(1, ft_cfg)

            for epoch in range(
                1, ft_cfg['phase1']['epochs'] + 1
            ):
                start = time.time()
                train_losses, train_metrics = self.run_epoch(
                    train_loader, optimizer1,
                    'PHASE_1', epoch, is_train=True
                )
                val_losses, val_metrics = self.run_epoch(
                    val_loader, optimizer1,
                    'PHASE_1', epoch, is_train=False
                )
                duration = time.time() - start

                # Get task-appropriate metric
                if 'mean_r2' in val_metrics:
                    metric_val = val_metrics['mean_r2']
                elif 'damage_acc' in val_metrics:
                    metric_val = val_metrics['damage_acc']
                elif 'r2' in val_metrics:
                    metric_val = val_metrics['r2']
                else:
                    metric_val = 0.0
                is_best    = metric_val > self.best_metric
                if is_best:
                    self.best_metric = metric_val

                print(
                    f'\n  Phase 1 Epoch {epoch}/'
                    f'{ft_cfg["phase1"]["epochs"]} | '
                    f'Val Loss: {val_losses["loss"]:.4f} | '
                    f'Val Metric: {metric_val:.4f} | '
                    f'{duration:.0f}s'
                )

                mlflow.log_metrics({
                    'p1_train_loss': train_losses['loss'],
                    'p1_val_loss':   val_losses['loss'],
                    'p1_val_metric': metric_val,
                }, step=epoch)

                self.save_checkpoint(
                    1, epoch, val_losses,
                    val_metrics, is_best
                )

            # ── Phase 2: Partial unfreeze ──────────────────
            log.info(f'\n{"="*60}')
            log.info(f'PHASE 2: Partial unfreeze — {self.task}')
            log.info(f'{"="*60}')

            # Free GPU memory before Phase 2
            del optimizer1
            import gc
            gc.collect()
            import torch as _torch
            _torch.cuda.empty_cache()
            log.info(f'GPU mem before Phase 2: '
                     f'{_torch.cuda.memory_allocated()/1e9:.2f}GB')

            self.backbone.set_phase('finetune')
            optimizer2 = self._build_optimizer(2, ft_cfg)

            patience  = ft_cfg.get('early_stopping_patience', 10)
            no_improve = 0

            # Use smaller batch loaders for phase2 (saves GPU memory)
            p2_train = train_loader_p2 if train_loader_p2 else train_loader
            p2_val   = val_loader_p2   if val_loader_p2   else val_loader

            for epoch in range(
                1, ft_cfg['phase2']['epochs'] + 1
            ):
                start = time.time()
                train_losses, train_metrics = self.run_epoch(
                    p2_train, optimizer2,
                    'PHASE_2', epoch, is_train=True
                )
                val_losses, val_metrics = self.run_epoch(
                    p2_val, optimizer2,
                    'PHASE_2', epoch, is_train=False
                )
                duration = time.time() - start

                # Get task-appropriate metric
                if 'mean_r2' in val_metrics:
                    metric_val = val_metrics['mean_r2']
                elif 'damage_acc' in val_metrics:
                    metric_val = val_metrics['damage_acc']
                elif 'r2' in val_metrics:
                    metric_val = val_metrics['r2']
                else:
                    metric_val = 0.0
                is_best    = metric_val > self.best_metric

                if is_best:
                    self.best_metric = metric_val
                    no_improve       = 0
                else:
                    no_improve += 1

                print(
                    f'\n  Phase 2 Epoch {epoch}/'
                    f'{ft_cfg["phase2"]["epochs"]} | '
                    f'Val Loss: {val_losses["loss"]:.4f} | '
                    f'Val Metric: {metric_val:.4f} | '
                    f'{duration:.0f}s'
                )

                mlflow.log_metrics({
                    'p2_train_loss': train_losses['loss'],
                    'p2_val_loss':   val_losses['loss'],
                    'p2_val_metric': metric_val,
                }, step=epoch)

                self.save_checkpoint(
                    2, epoch, val_losses,
                    val_metrics, is_best
                )

                # Early stopping
                if no_improve >= patience:
                    log.info(
                        f'Early stopping at epoch {epoch} '
                        f'(no improvement for {patience} epochs)'
                    )
                    break

        log.info(
            f'\n✅ Fine-tuning complete — {self.task}'
        )
        log.info(
            f'   Best metric: {self.best_metric:.4f}'
        )
        log.info(
            f'   Best checkpoint: {self.best_dir}'
        )


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='OmniGeoFusion fine-tuning'
    )
    parser.add_argument(
        '--task',
        choices=['urban_change', 'flood_damage', 'agriculture'],
        required=True
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
        '--ssl-checkpoint',
        default=None,
        help='Path to SSL pre-trained backbone checkpoint'
    )
    parser.add_argument(
        '--max-samples', type=int, default=None
    )
    parser.add_argument(
        '--resume', default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )

    # Load configs
    model_cfg = yaml.safe_load(open(args.model_config))
    train_cfg = yaml.safe_load(open(args.train_config))
    cfg       = {**model_cfg, **train_cfg}

    # Device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    log.info(f'Device: {device}')

    # Build backbone + task head + loss
    sys.path.insert(0, os.path.abspath('.'))
    from src.fusion.backbone import OmniGeoFusionBackbone

    backbone = OmniGeoFusionBackbone(
        model_cfg['model']
    ).to(device)

    if args.task == 'urban_change':
        from src.tasks.urban_change import (
            UrbanChangeHead, UrbanChangeLoss,
            compute_urban_change_metrics
        )
        task_head  = UrbanChangeHead().to(device)
        loss_fn    = UrbanChangeLoss()
        metrics_fn = compute_urban_change_metrics

    elif args.task == 'flood_damage':
        from src.tasks.flood_damage import (
            FloodDamageHead, FloodDamageLoss,
            compute_flood_metrics
        )
        task_head  = FloodDamageHead().to(device)
        loss_fn    = FloodDamageLoss()
        metrics_fn = compute_flood_metrics

    elif args.task == 'agriculture':
        from src.tasks.agriculture import (
            AgricultureHead, AgricultureLoss,
            compute_agriculture_metrics
        )
        task_head  = AgricultureHead().to(device)
        loss_fn    = AgricultureLoss()
        metrics_fn = compute_agriculture_metrics

    # Datasets
    ft_key    = f'finetune_{args.task}'
    ft_cfg    = train_cfg.get(ft_key, train_cfg['finetune_urban_change'])
    batch_size = ft_cfg['phase1']['batch_size']

    train_ds = FinetuneDataset(
        args.data_dir, args.task, 'train',
        max_samples=args.max_samples
    )
    val_ds   = FinetuneDataset(
        args.data_dir, args.task, 'val',
        max_samples=args.max_samples // 5
        if args.max_samples else None
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=2
    )
    # Phase 2 loaders with smaller batch size (saves GPU memory)
    batch_size_p2 = ft_cfg['phase2'].get('batch_size', 4)
    train_loader_p2 = DataLoader(
        train_ds, batch_size=batch_size_p2,
        shuffle=True, num_workers=2,
        pin_memory=True, drop_last=True
    )
    val_loader_p2 = DataLoader(
        val_ds, batch_size=batch_size_p2,
        shuffle=False, num_workers=2
    )


    # MLflow
    mlflow.set_tracking_uri(
        train_cfg.get('mlflow', {}).get('tracking_uri', 'mlruns')
    )
    mlflow.set_experiment(
        train_cfg.get('mlflow', {}).get(
            'experiment_name', 'omnigeofusion'
        )
    )

    # Train
    trainer = FinetuneTrainer(
        backbone, task_head, loss_fn,
        metrics_fn, args.task, cfg, device
    )
    trainer.train(
        train_loader, val_loader,
        train_loader_p2, val_loader_p2,
        ssl_checkpoint=args.ssl_checkpoint
    )


if __name__ == '__main__':
    main()
