"""
GeoAI MLOps Project 1 — Training Script (Colab Version)
=========================================================
Changes from original:
  - DB credentials from config (not SSM) for Colab compatibility
  - Checkpoints saved to Google Drive (not S3/NVMe)
  - Best model saved to Google Drive
  - Pairs/stats paths are absolute (Google Drive paths)
  - patches_local_dir support for local/GDrive patch loading
  - Early stopping added to both training phases
  - autocast updated for newer PyTorch compatibility
"""

import os
import json
import math
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from scipy import stats as scipy_stats
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import sys
import boto3
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
log = logging.getLogger(__name__)


# ── Default config ─────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'project_dir':         '/content/geoai-mlops-p1',
    'train_pairs_csv':     '/gdrive/MyDrive/geoai_mlops/pairs/train_pairs.csv',
    'val_pairs_csv':       '/gdrive/MyDrive/geoai_mlops/pairs/val_pairs.csv',
    'test_pairs_csv':      '/gdrive/MyDrive/geoai_mlops/pairs/test_pairs.csv',
    'tabular_stats':       '/gdrive/MyDrive/geoai_mlops/stats/tabular_stats.json',
    'band_stats':          '/gdrive/MyDrive/geoai_mlops/stats/band_stats.json',
    'day_gap_stats':       '/gdrive/MyDrive/geoai_mlops/stats/day_gap_stats.json',
    'checkpoint_local_dir':'/gdrive/MyDrive/geoai_mlops/checkpoints/local',
    'checkpoint_best_dir': '/gdrive/MyDrive/geoai_mlops/checkpoints/best',
    'patches_local_dir':   '/gdrive/MyDrive/geoai_mlops/patches',
    'use_local_patches':   True,

    # Model
    'prithvi_model_name':  'prithvi_eo_v1_100',
    'n_tabular_features':  34,
    'mlp_hidden_1':        256,
    'mlp_hidden_2':        64,
    'dropout_1':           0.3,
    'dropout_2':           0.2,

    # Training
    'batch_size':          32,
    'phase1_epochs':       15,
    'phase2_epochs':       20,
    'phase1_lr':           1e-3,
    'phase2_encoder_lr':   1e-5,
    'phase2_head_lr':      1e-4,
    'weight_decay':        1e-4,
    'grad_clip':           1.0,
    'num_workers':         4,
    'early_stopping_patience': 10,

    # Checkpointing
    'checkpoint_local_steps':  50,
    'checkpoint_gdrive_steps': 200,

    # MLflow
    'mlflow_experiment':   'SiamesePrithviChangeDetection',
    'mlflow_tracking_uri': 'http://3.91.55.220:5000',

    # S3
    's3_bucket':           'geoai-mlops-p1-data-288528696055',
    's3_patches_prefix':   'processed/patches',

    # Promotion thresholds
    'min_val_r2':          0.5,
    'min_test_r2':         0.4,
    'min_test_spearman':   0.6,
}


def load_config(config_path=None):
    config = DEFAULT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            overrides = json.load(f)
        config.update(overrides)
        log.info(f"Config loaded from {config_path}")
    return config


def get_db_credentials(cfg):
    """Get DB credentials from config (Colab) or SSM (EC2)."""
    if cfg.get('db_host') and cfg.get('db_pass'):
        log.info("Using DB credentials from config")
        return cfg['db_host'], cfg.get('db_user', 'geoai_admin'), cfg['db_pass']
    log.info("Fetching DB credentials from SSM")
    ssm     = boto3.client('ssm', region_name='us-east-1')
    db_host = ssm.get_parameter(
        Name='/geoai-mlops-p1/db/host'
    )['Parameter']['Value']
    db_user = ssm.get_parameter(
        Name='/geoai-mlops-p1/db/user'
    )['Parameter']['Value']
    db_pass = ssm.get_parameter(
        Name='/geoai-mlops-p1/db/password', WithDecryption=True
    )['Parameter']['Value']
    return db_host, db_user, db_pass


def build_dataloaders(cfg, db_host, db_user, db_pass):
    """Build train, val, test DataLoaders."""
    from dataset import SentinelPairDataset

    # patches_local_dir — use absolute path from config
    patches_local = cfg.get('patches_local_dir', None) \
        if cfg.get('use_local_patches') else None

    train_ds = SentinelPairDataset(
        pairs_csv=cfg['train_pairs_csv'],
        tabular_stats_path=cfg['tabular_stats'],
        band_stats_path=cfg['band_stats'],
        day_gap_stats_path=cfg['day_gap_stats'],
        patches_s3_bucket=cfg['s3_bucket'],
        patches_s3_prefix=cfg['s3_patches_prefix'],
        db_host=db_host, db_user=db_user, db_pass=db_pass,
        augment=True,
        max_pairs=cfg.get('max_train_pairs', None),
        patches_local_dir=patches_local,
        features_cache=cfg.get("features_cache", None),
        pairs_offset=cfg.get("pairs_offset", 0),
    )
    val_ds = SentinelPairDataset(
        pairs_csv=cfg['val_pairs_csv'],
        tabular_stats_path=cfg['tabular_stats'],
        band_stats_path=cfg['band_stats'],
        day_gap_stats_path=cfg['day_gap_stats'],
        patches_s3_bucket=cfg['s3_bucket'],
        patches_s3_prefix=cfg['s3_patches_prefix'],
        db_host=db_host, db_user=db_user, db_pass=db_pass,
        augment=False,
        max_pairs=cfg.get('max_val_pairs', None),
        patches_local_dir=patches_local,
        features_cache=cfg.get("features_cache", None),
    )
    test_ds = SentinelPairDataset(
        pairs_csv=cfg['test_pairs_csv'],
        tabular_stats_path=cfg['tabular_stats'],
        band_stats_path=cfg['band_stats'],
        day_gap_stats_path=cfg['day_gap_stats'],
        patches_s3_bucket=cfg['s3_bucket'],
        patches_s3_prefix=cfg['s3_patches_prefix'],
        db_host=db_host, db_user=db_user, db_pass=db_pass,
        augment=False,
        max_pairs=cfg.get('max_test_pairs', None),
        patches_local_dir=patches_local,
        features_cache=cfg.get("features_cache", None),
    )

    log.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'],
        shuffle=True, num_workers=cfg['num_workers'],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'],
        shuffle=False, num_workers=cfg['num_workers'],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg['batch_size'],
        shuffle=False, num_workers=cfg['num_workers'],
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def compute_metrics(preds, targets):
    """Compute MAE, RMSE, R², Spearman."""
    preds   = np.array(preds).flatten()
    targets = np.array(targets).flatten()
    mae     = float(np.mean(np.abs(preds - targets)))
    rmse    = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_res  = np.sum((targets - preds) ** 2)
    ss_tot  = np.sum((targets - np.mean(targets)) ** 2)
    r2      = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    spearman = float(scipy_stats.spearmanr(preds, targets).correlation)
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'spearman': spearman}


def save_checkpoint_local(state, cfg, step, epoch, phase):
    """Save checkpoint to Google Drive every 50 steps."""
    ckpt_dir = cfg.get('checkpoint_local_dir',
                       '/gdrive/MyDrive/geoai_mlops/checkpoints/local')
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, 'latest.pt')
    torch.save(state, path)
    log.debug(f"  GDrive checkpoint saved: step {step} epoch {epoch}")


def save_checkpoint_gdrive(state, cfg, step, epoch, phase):
    """Save extra safety checkpoint to Google Drive every 200 steps."""
    ckpt_dir = cfg.get('checkpoint_local_dir',
                       '/gdrive/MyDrive/geoai_mlops/checkpoints/local')
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'checkpoint_step_{step}.pt')
    try:
        torch.save(state, path)
        log.info(f"  GDrive safety checkpoint: step {step} → {path}")
    except Exception as e:
        log.error(f"  GDrive checkpoint failed: {e}")


def save_best_model(model, cfg, metrics, run_id):
    """Save best model to Google Drive when val R² improves."""
    best_dir     = cfg.get('checkpoint_best_dir',
                       '/gdrive/MyDrive/geoai_mlops/checkpoints/best')
    chunk_number = cfg.get('chunk_number', 1)
    os.makedirs(best_dir, exist_ok=True)
    best_path = os.path.join(best_dir, f'best_model_chunk{chunk_number}_r2_{metrics["r2"]:.4f}.pt')
    try:
        # model_state_dict_or_direct: always save as dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_r2':           metrics['r2'],
            'chunk_number':     cfg.get('chunk_number', 1),
        }, best_path)
        log.info(
            f"  ✅ Best model → {best_path} "
            f"(val_r2={metrics['r2']:.4f})"
        )
    except Exception as e:
        log.error(f"  Best model save failed: {e}")
    return best_path


def load_checkpoint(resume_path, model, optimizer, scheduler, scaler, cfg):
    """Load checkpoint for resuming from interruption."""
    ckpt = torch.load(resume_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    log.info(
        f"✅ Resumed: epoch {ckpt['epoch']} step {ckpt['step']} "
        f"best_val_r2={ckpt['best_val_r2']:.4f}"
    )
    return ckpt['epoch'], ckpt['step'], ckpt['phase'], ckpt['best_val_r2']


def run_epoch(model, loader, optimizer, scaler, loss_fn,
              device, cfg, phase_name, epoch,
              global_step, best_val_r2,
              run_id, is_train=True, scheduler=None):
    """Run one epoch of training or evaluation."""
    model.train() if is_train else model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []
    n_batches   = len(loader)
    epoch_start = time.time()

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"  {'Train' if is_train else 'Val  '} epoch {epoch}",
            ncols=80,
            leave=True,
            file=sys.stdout,
        )
        for batch_idx, batch in pbar:

            patch_t1   = batch['patch_t1'].to(device)
            patch_t2   = batch['patch_t2'].to(device)
            tabular_t1 = batch['tabular_t1'].to(device)
            tabular_t2 = batch['tabular_t2'].to(device)
            day_gap    = batch['day_gap'].to(device)
            target     = batch['target'].to(device)

            if is_train:
                optimizer.zero_grad()

            # fp16 mixed precision
            with autocast():
                pred = model(patch_t1, patch_t2, tabular_t1, tabular_t2, day_gap)
                loss = loss_fn(pred, target)

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg['grad_clip']
                )
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

                # Checkpoint every 50 steps → GDrive
                if global_step % cfg['checkpoint_local_steps'] == 0:
                    state = {
                        'model_state_dict':     model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else {},
                        'scaler_state_dict':    scaler.state_dict(),
                        'epoch':                epoch,
                        'step':                 global_step,
                        'phase':                phase_name,
                        'best_val_r2':          best_val_r2,
                        'chunk_number':         cfg.get('chunk_number', 1),
                        'config':               cfg,
                    }
                    save_checkpoint_local(state, cfg, global_step, epoch, phase_name)

                # Extra checkpoint every 200 steps → GDrive
                if global_step % cfg.get('checkpoint_gdrive_steps', 200) == 0:
                    save_checkpoint_gdrive(state, cfg, global_step, epoch, phase_name)

            total_loss   += loss.item()
            all_preds.extend(pred.detach().cpu().numpy().flatten())
            all_targets.extend(target.detach().cpu().numpy().flatten())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    print()  # newline after progress bar
    avg_loss = total_loss / n_batches
    metrics  = compute_metrics(all_preds, all_targets)
    duration = time.time() - epoch_start

    prefix = 'Train' if is_train else 'Val  '
    print(
        f"  {prefix} Loss: {avg_loss:.4f} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"R2: {metrics['r2']:.4f} | "
        f"Spearman: {metrics['spearman']:.4f} | "
        f"{duration:.0f}s"
    )
    if not is_train:
        print(f"  {'─'*65}")
    return avg_loss, metrics, global_step, best_val_r2


def train_phase(model, train_loader, val_loader, optimizer, scheduler,
                scaler, loss_fn, device, cfg, phase_name, n_epochs,
                global_step, best_val_r2, best_model_path,
                run_id, start_epoch=1):
    """Run all epochs for one training phase with early stopping."""
    log.info(f"\n{'='*60}")
    log.info(f"STARTING {phase_name}")
    log.info(f"{'='*60}")

    patience      = cfg.get('early_stopping_patience', 5)
    no_improve    = 0
    phase_best_r2 = -float('inf')

    for epoch in range(start_epoch, n_epochs + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{n_epochs} — {phase_name}")

        # Training epoch
        train_loss, train_metrics, global_step, best_val_r2 = run_epoch(
            model, train_loader, optimizer, scaler, loss_fn,
            device, cfg, phase_name, epoch,
            global_step, best_val_r2,
            run_id, is_train=True, scheduler=scheduler
        )

        # Validation epoch
        val_loss, val_metrics, _, _ = run_epoch(
            model, val_loader, optimizer, scaler, loss_fn,
            device, cfg, phase_name, epoch,
            global_step, best_val_r2,
            run_id, is_train=False
        )

        scheduler.step()
        epoch_duration = time.time() - epoch_start

        # Log to MLflow
        mlflow.log_metrics({
            'train_loss':     train_loss,
            'train_mae':      train_metrics['mae'],
            'train_r2':       train_metrics['r2'],
            'val_loss':       val_loss,
            'val_mae':        val_metrics['mae'],
            'val_rmse':       val_metrics['rmse'],
            'val_r2':         val_metrics['r2'],
            'val_spearman':   val_metrics['spearman'],
            'learning_rate':  scheduler.get_last_lr()[0],
            'epoch_duration': epoch_duration,
        }, step=global_step)

        # Save best model if val R² improved
        if val_metrics['r2'] > best_val_r2:
            best_val_r2    = val_metrics['r2']
            best_model_path = save_best_model(model, cfg, val_metrics, run_id)
            mlflow.log_metric('best_val_r2', best_val_r2, step=global_step)

        # Early stopping
        if val_metrics['r2'] > phase_best_r2:
            phase_best_r2 = val_metrics['r2']
            no_improve    = 0
        else:
            no_improve += 1
            log.info(f"  No improvement {no_improve}/{patience} epochs")
            if no_improve >= patience:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    return global_step, best_val_r2, best_model_path


def evaluate_test(model, test_loader, loss_fn, device, cfg):
    """Final evaluation on Brandenburg test set."""
    log.info("\n" + "="*60)
    log.info("FINAL TEST EVALUATION (Brandenburg)")
    log.info("="*60)

    # TEST_EVAL_TQDM: clear GPU cache + show progress
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info("✅ GPU cache cleared before test evaluation")

    model.eval()
    all_preds   = []
    all_targets = []
    n_batches   = len(test_loader)

    print(f"\nTest evaluation: {n_batches} batches")
    with torch.no_grad():
        pbar = tqdm(test_loader, total=n_batches,
                    desc="  Test ", ncols=80,
                    leave=True, file=sys.stdout)
        for batch in pbar:
            patch_t1   = batch['patch_t1'].to(device)
            patch_t2   = batch['patch_t2'].to(device)
            tabular_t1 = batch['tabular_t1'].to(device)
            tabular_t2 = batch['tabular_t2'].to(device)
            day_gap    = batch['day_gap'].to(device)
            target     = batch['target'].to(device)

            with autocast():
                pred = model(patch_t1, patch_t2, tabular_t1, tabular_t2, day_gap)

            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    metrics = compute_metrics(all_preds, all_targets)
    log.info(
        f"TEST RESULTS:\n"
        f"  MAE:      {metrics['mae']:.4f}\n"
        f"  RMSE:     {metrics['rmse']:.4f}\n"
        f"  R²:       {metrics['r2']:.4f}\n"
        f"  Spearman: {metrics['spearman']:.4f}"
    )

    # Save predictions CSV
    import csv
    preds_path = '/gdrive/MyDrive/geoai_mlops/test_predictions.csv'
    with open(preds_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['predicted', 'actual'])
        w.writerows(zip(all_preds, all_targets))
    mlflow.log_artifact(preds_path, artifact_path='predictions')

    return metrics


def maybe_register_model(metrics, cfg, run_id):
    """Register model if metrics pass threshold."""
    passes = (
        metrics['r2']       >= cfg['min_test_r2'] and
        metrics['spearman'] >= cfg['min_test_spearman']
    )
    if passes:
        model_uri = f"runs:/{run_id}/best_model"
        mv = mlflow.register_model(
            model_uri=model_uri,
            name='SiamesePrithviChangeDetection',
        )
        log.info(
            f"✅ Model registered: v{mv.version} "
            f"(test_r2={metrics['r2']:.4f} spearman={metrics['spearman']:.4f})"
        )
    else:
        log.info(
            f"Model did not meet threshold "
            f"(test_r2={metrics['r2']:.4f} spearman={metrics['spearman']:.4f})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")
    if device.type == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    # S3 client (still needed for MLflow artifacts)
    session   = boto3.Session(region_name='us-east-1')
    s3_client = session.client('s3')

    # DB credentials
    db_host, db_user, db_pass = get_db_credentials(cfg)

    # Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, db_host, db_user, db_pass
    )

    # Build model
    from model import SiamesePrithviModel
    model = SiamesePrithviModel(
        prithvi_model_name=cfg['prithvi_model_name'],
        n_tabular_features=cfg['n_tabular_features'],
        mlp_hidden_1=cfg['mlp_hidden_1'],
        mlp_hidden_2=cfg['mlp_hidden_2'],
        dropout_1=cfg['dropout_1'],
        dropout_2=cfg['dropout_2'],
    ).to(device)

    loss_fn = nn.HuberLoss(delta=1.0)
    scaler  = GradScaler()

    # MLflow
    mlflow.set_tracking_uri(cfg['mlflow_tracking_uri'])
    mlflow.set_experiment(cfg['mlflow_experiment'])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        log.info(f"MLflow run ID: {run_id}")

        mlflow.log_params({
            'model':              cfg['prithvi_model_name'],
            'batch_size':         cfg['batch_size'],
            'phase1_epochs':      cfg['phase1_epochs'],
            'phase2_epochs':      cfg['phase2_epochs'],
            'phase1_lr':          cfg['phase1_lr'],
            'phase2_encoder_lr':  cfg['phase2_encoder_lr'],
            'phase2_head_lr':     cfg['phase2_head_lr'],
            'dropout_1':          cfg['dropout_1'],
            'dropout_2':          cfg['dropout_2'],
            'loss':               'huber',
            'optimizer':          'adamw',
            'early_stopping':     cfg.get('early_stopping_patience', 5),
            'train_aois':         'berlin,hamburg',
            'test_aoi':           'brandenburg',
        })

        global_step     = 0
        best_val_r2     = -float('inf')
        best_model_path = None
        start_phase     = 1
        start_epoch_p1  = 1
        start_epoch_p2  = 1
        chunk_number    = cfg.get('chunk_number', 1)

        # ── RESUME_FROM_CHECKPOINT ──────────────────────────────
        # Priority 1: crash recovery — latest.pt same chunk
        # Priority 2: cross-chunk — --resume best_model_chunkN.pt
        # Both handle old (direct state dict) and new (dict) formats

        ckpt_dir    = cfg.get('checkpoint_local_dir',
                         '/gdrive/MyDrive/geoai_mlops/checkpoints/local')
        latest_path = os.path.join(ckpt_dir, 'latest.pt')

        if os.path.exists(latest_path):
            try:
                ckpt       = torch.load(latest_path, map_location=device)
                ckpt_chunk = ckpt.get('chunk_number', chunk_number)
                if ckpt_chunk == chunk_number:
                    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                        model.load_state_dict(ckpt['model_state_dict'])
                    else:
                        model.load_state_dict(ckpt)
                    global_step = ckpt.get('step', 0)
                    best_val_r2 = ckpt.get('best_val_r2', -float('inf'))
                    saved_phase = ckpt.get('phase', 'PHASE_1')
                    saved_epoch = ckpt.get('epoch', 1)
                    if saved_phase == 'PHASE_1':
                        start_phase    = 1
                        start_epoch_p1 = saved_epoch + 1
                    elif saved_phase == 'PHASE_2':
                        start_phase    = 2
                        start_epoch_p2 = saved_epoch + 1
                    print(f'✅ Crash recovery: resuming {saved_phase} epoch {saved_epoch+1}')
                    log.info(f'✅ Crash recovery: {saved_phase} epoch {saved_epoch+1}')
                else:
                    log.info(f'latest.pt chunk {ckpt_chunk} != {chunk_number} — ignoring')
            except Exception as e:
                log.warning(f'Could not load latest.pt: {e} — starting fresh')
        elif args.resume and os.path.exists(args.resume):
            log.info(f'Loading weights from --resume: {args.resume}')
            ckpt = torch.load(args.resume, map_location=device)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
                best_val_r2 = ckpt.get('val_r2', ckpt.get('best_val_r2', -float('inf')))
            else:
                model.load_state_dict(ckpt)
                best_val_r2 = -float('inf')
            log.info(f'✅ Weights loaded — best_val_r2: {best_val_r2:.4f}')
        elif args.resume:
            log.warning(f'Checkpoint not found: {args.resume} — starting fresh')

        # ── PHASE 1: frozen encoder ────────────────────────────────
        if start_phase <= 1:
            model.freeze_encoder()
            phase1_optimizer = AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg['phase1_lr'],
                weight_decay=cfg['weight_decay']
            )
            phase1_scheduler = CosineAnnealingLR(
                phase1_optimizer, T_max=cfg['phase1_epochs']
            )
            global_step, best_val_r2, best_model_path = train_phase(
                model, train_loader, val_loader,
                phase1_optimizer, phase1_scheduler,
                scaler, loss_fn, device, cfg,
                'PHASE_1', cfg['phase1_epochs'],
                global_step, best_val_r2, best_model_path,
                run_id, start_epoch=start_epoch_p1
            )

        # ── PHASE 2: full fine-tuning ──────────────────────────────
        model.unfreeze_encoder()
        phase2_optimizer = AdamW(
            model.get_param_groups(
                encoder_lr=cfg['phase2_encoder_lr'],
                head_lr=cfg['phase2_head_lr']
            ),
            weight_decay=cfg['weight_decay']
        )
        phase2_scheduler = CosineAnnealingLR(
            phase2_optimizer, T_max=cfg['phase2_epochs']
        )
        global_step, best_val_r2, best_model_path = train_phase(
            model, train_loader, val_loader,
            phase2_optimizer, phase2_scheduler,
            scaler, loss_fn, device, cfg,
            'PHASE_2', cfg['phase2_epochs'],
            global_step, best_val_r2, best_model_path,
            run_id, start_epoch=start_epoch_p2
        )

        # Load best model for test evaluation
        # BEST_MODEL_LOAD_FIX: handle both dict and direct formats
        # BEST_MODEL_PATH_RECOVERY: find best model if path not set
        # Handles case where crash recovery skipped all training
        if not best_model_path or not os.path.exists(str(best_model_path)):
            import glob as _glob
            _best_dir  = cfg.get("checkpoint_best_dir",
                             "/gdrive/MyDrive/geoai_mlops/checkpoints/best")
            _chunk_num = cfg.get("chunk_number", 1)
            _pattern   = os.path.join(_best_dir,
                             f"best_model_chunk{_chunk_num}_r2_*.pt")
            _candidates = _glob.glob(_pattern)
            if _candidates:
                best_model_path = max(_candidates,
                    key=lambda x: float(
                        x.split("_r2_")[1].replace(".pt", "")))
                log.info(f"✅ Best model path recovered: {best_model_path}")
                print(f"✅ Best model path recovered: "
                      f"{os.path.basename(best_model_path)}")
            else:
                log.warning(
                    f"No best model found for chunk {_chunk_num} "
                    f"in {_best_dir} — test eval will use current weights")
        if best_model_path and os.path.exists(best_model_path):
            _ckpt = torch.load(best_model_path, map_location=device)
            if isinstance(_ckpt, dict) and 'model_state_dict' in _ckpt:
                model.load_state_dict(_ckpt['model_state_dict'])
            else:
                model.load_state_dict(_ckpt)
            log.info("✅ Best model loaded for test evaluation")

        # Final test evaluation
        test_metrics = evaluate_test(model, test_loader, loss_fn, device, cfg)
        mlflow.log_metrics({
            'test_mae':      test_metrics['mae'],
            'test_rmse':     test_metrics['rmse'],
            'test_r2':       test_metrics['r2'],
            'test_spearman': test_metrics['spearman'],
        })

        # Log best model + stats as MLflow artifacts
        if best_model_path and os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path, artifact_path='best_model')

        for stats_file in ['tabular_stats.json', 'band_stats.json', 'day_gap_stats.json']:
            path = cfg.get(stats_file.replace('.json','').replace('_stats','_stats'),
                          f'/gdrive/MyDrive/geoai_mlops/stats/{stats_file}')
            stats_path = f'/gdrive/MyDrive/geoai_mlops/stats/{stats_file}'
            if os.path.exists(stats_path):
                mlflow.log_artifact(stats_path, artifact_path='stats')

        maybe_register_model(test_metrics, cfg, run_id)

        log.info(f"\n✅ Training complete — MLflow run: {run_id}")
        log.info(f"   Best val R²:   {best_val_r2:.4f}")
        log.info(f"   Test R²:       {test_metrics['r2']:.4f}")
        log.info(f"   Test Spearman: {test_metrics['spearman']:.4f}")
        log.info(f"   View results:  http://3.91.55.220:5000")

        # Save metrics log for incremental training tracking
        import json as _json
        log_path     = os.path.join(
            cfg.get('checkpoint_best_dir',
                    '/gdrive/MyDrive/geoai_mlops/checkpoints/best'),
            'training_log.json'
        )
        chunk_number = cfg.get('chunk_number', 1)
        try:
            with open(log_path) as _f:
                training_log = _json.load(_f)
        except (FileNotFoundError, _json.JSONDecodeError):
            training_log = {}
        training_log[f'chunk_{chunk_number}'] = {
            'chunk_number':    chunk_number,
            'pairs_offset':    cfg.get('pairs_offset', 0),
            'max_train_pairs': cfg.get('max_train_pairs', 0),
            'best_val_r2':     round(best_val_r2, 4),
            'test_r2':         round(test_metrics['r2'], 4),
            'test_spearman':   round(test_metrics['spearman'], 4),
            'best_model':      os.path.basename(best_model_path) if best_model_path else None,
            'mlflow_run_id':   run_id,
            'total_chunks':    cfg.get('total_chunks', '?'),
            'is_last_chunk':   cfg.get('is_last_chunk', False),
        }
        with open(log_path, 'w') as _f:
            _json.dump(training_log, _f, indent=2)
        log.info(f'✅ Metrics log saved: {log_path}')
        print(f'\n✅ Chunk {chunk_number} / {cfg.get("total_chunks","?")} complete!')
        print(f'   Best val R2:   {best_val_r2:.4f}')
        print(f'   Test R2:       {test_metrics["r2"]:.4f}')
        print(f'   Test Spearman: {test_metrics["spearman"]:.4f}')
        print(f'   Best model:    {os.path.basename(best_model_path) if best_model_path else None}')
        if cfg.get('is_last_chunk', False):
            print('\n🎉 ALL CHUNKS COMPLETE — Final checkpoint saved!')
        else:
            print(f'   Next: set chunk_number={chunk_number+1} in Cell 5')


if __name__ == '__main__':
    main()
