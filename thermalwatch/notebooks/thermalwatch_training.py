# ThermalWatch — Colab Training Notebook
# =========================================
# Run each cell in order in Google Colab.
# Mount Google Drive first, then run cells.
#
# Structure expected in GDrive:
#   My Drive/thermalwatch/
#     project/thermalwatch.tar.gz
#     data/
#       wildfire/patches/*.tar + indexes/
#       solar/patches/*.tar + indexes/
#       indexes/ (train/val splits)
#     checkpoints/ssl/ finetune/wildfire/ finetune/solar/

# ── Cell 1: Mount GDrive + Install deps ──────────────────────
"""
from google.colab import drive
drive.mount('/gdrive')

import subprocess
subprocess.run([
    'pip', 'install', '-q',
    'torch', 'torchvision',
    'rasterio', 'geopandas',
    'pystac-client', 'planetary-computer',
    'pyproj', 'shapely',
    'terratorch',
], check=True)
print('✅ Dependencies installed')
"""

# ── Cell 2: Extract code ──────────────────────────────────────
"""
import subprocess, os

GDRIVE = '/gdrive/MyDrive/thermalwatch'
os.makedirs('/content/thermalwatch', exist_ok=True)

subprocess.run([
    'tar', '-xzf',
    f'{GDRIVE}/project/thermalwatch.tar.gz',
    '-C', '/content/',
], check=True)

os.chdir('/content/thermalwatch')
import sys
sys.path.insert(0, '/content/thermalwatch')
print('✅ Code extracted')
"""

# ── Cell 3: Extract patches ───────────────────────────────────
"""
import subprocess, os
from pathlib import Path

GDRIVE    = '/gdrive/MyDrive/thermalwatch'
LOCAL_WF  = Path('/content/data/wildfire/patches')
LOCAL_SOL = Path('/content/data/solar/patches')
LOCAL_WF.mkdir(parents=True, exist_ok=True)
LOCAL_SOL.mkdir(parents=True, exist_ok=True)

# Extract wildfire patches
wf_tars = sorted(
    Path(f'{GDRIVE}/data/wildfire/patches').glob('*.tar')
)
print(f'Extracting {len(wf_tars)} wildfire tarballs...')
for tar in wf_tars:
    subprocess.run([
        'tar', '-xf', str(tar),
        '-C', str(LOCAL_WF),
    ], check=True)
    print(f'  ✅ {tar.name}')

# Extract solar patches
sol_tars = sorted(
    Path(f'{GDRIVE}/data/solar/patches').glob('*.tar')
)
print(f'Extracting {len(sol_tars)} solar tarballs...')
for tar in sol_tars:
    subprocess.run([
        'tar', '-xf', str(tar),
        '-C', str(LOCAL_SOL),
    ], check=True)
    print(f'  ✅ {tar.name}')

# Copy indexes
import shutil
os.makedirs('/content/data/indexes', exist_ok=True)
for f in Path(f'{GDRIVE}/data/indexes').glob('*.json'):
    shutil.copy(f, '/content/data/indexes/')
    print(f'  ✅ {f.name}')

# Fix index paths to point to local /content/data
import json
for idx_path in Path('/content/data/indexes').glob('*.json'):
    idx = json.loads(idx_path.read_text())
    for entry in idx:
        if 'thermal_path' in entry:
            entry['thermal_path'] = entry[
                'thermal_path'
            ].replace(
                'data/wildfire',
                '/content/data/wildfire'
            ).replace(
                'data/solar',
                '/content/data/solar'
            )
        if entry.get('s2_path'):
            entry['s2_path'] = entry['s2_path'].replace(
                'data/wildfire',
                '/content/data/wildfire'
            ).replace(
                'data/solar',
                '/content/data/solar'
            )
    idx_path.write_text(json.dumps(idx))
    print(f'  ✅ Fixed paths: {idx_path.name}')

print('✅ All data ready')
"""

# ── Cell 4: SSL Pre-training ──────────────────────────────────
"""
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.backbone.thermal_backbone import (
    ThermalWatchBackbone
)
from src.models.training.ssl_pretrain import (
    ThermalSSLDataset, SSLTrainer
)

GDRIVE    = '/gdrive/MyDrive/thermalwatch'
INDEX_DIR = Path('/content/data/indexes')
CKPT_DIR  = Path(f'{GDRIVE}/checkpoints/ssl')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)
print(f'Device: {device}')

train_ds = ThermalSSLDataset(
    wildfire_index=INDEX_DIR / 'wildfire_train_index.json',
    solar_index=INDEX_DIR / 'solar_train_index.json',
    split='train',
)
val_ds = ThermalSSLDataset(
    wildfire_index=INDEX_DIR / 'wildfire_val_index.json',
    solar_index=INDEX_DIR / 'solar_val_index.json',
    split='val',
)

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=16,
    shuffle=False,
    num_workers=2,
)

backbone = ThermalWatchBackbone(
    pretrained_optical=True,
    freeze_optical=True,
)

cfg = {
    'epochs':       100,
    'lr':           3e-4,
    'weight_decay': 0.01,
    'temperature':  0.07,
    'batch_size':   16,
}

trainer = SSLTrainer(
    backbone=backbone,
    train_loader=train_loader,
    val_loader=val_loader,
    cfg=cfg,
    ckpt_dir=CKPT_DIR,
    device=device,
)

# Resume if checkpoint exists
resume = CKPT_DIR / 'ssl_latest.pt'
trainer.train(
    resume_from=resume if resume.exists() else None
)
"""

# ── Cell 5: Fine-tune Wildfire ────────────────────────────────
"""
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.backbone.thermal_backbone import (
    ThermalWatchBackbone
)
from src.models.wildfire.wildfire_head import WildfireHead
from src.models.training.finetune import (
    WildfireDataset, FinetuneTrainer, wildfire_loss
)

GDRIVE    = '/gdrive/MyDrive/thermalwatch'
INDEX_DIR = Path('/content/data/indexes')
SSL_CKPT  = Path(f'{GDRIVE}/checkpoints/ssl/ssl_best.pt')
CKPT_DIR  = Path(
    f'{GDRIVE}/checkpoints/finetune/wildfire'
)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

train_ds = WildfireDataset(
    INDEX_DIR / 'wildfire_train_index.json',
    split='train',
)
val_ds = WildfireDataset(
    INDEX_DIR / 'wildfire_val_index.json',
    split='val',
)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
)

backbone = ThermalWatchBackbone(
    pretrained_optical=True,
    freeze_optical=True,
)
head = WildfireHead(embed_dim=512)

cfg = {
    'phase1_epochs': 10,
    'phase2_epochs': 20,
    'lr_head':       1e-3,
    'lr_backbone':   1e-5,
    'patience':      5,
    'batch_size':    32,
}

trainer = FinetuneTrainer(
    backbone=backbone,
    head=head,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=wildfire_loss,
    cfg=cfg,
    ckpt_dir=CKPT_DIR,
    device=device,
    task='wildfire',
)
trainer.train(ssl_ckpt=SSL_CKPT)
"""

# ── Cell 6: Fine-tune Solar ───────────────────────────────────
"""
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.backbone.thermal_backbone import (
    ThermalWatchBackbone
)
from src.models.solar.solar_head import SolarHead
from src.models.training.finetune import (
    SolarDataset, FinetuneTrainer, solar_loss
)

GDRIVE    = '/gdrive/MyDrive/thermalwatch'
INDEX_DIR = Path('/content/data/indexes')
SSL_CKPT  = Path(f'{GDRIVE}/checkpoints/ssl/ssl_best.pt')
CKPT_DIR  = Path(
    f'{GDRIVE}/checkpoints/finetune/solar'
)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

train_ds = SolarDataset(
    INDEX_DIR / 'solar_train_index.json',
    split='train',
)
val_ds = SolarDataset(
    INDEX_DIR / 'solar_val_index.json',
    split='val',
)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
)

backbone = ThermalWatchBackbone(
    pretrained_optical=True,
    freeze_optical=True,
)
head = SolarHead(embed_dim=512)

cfg = {
    'phase1_epochs': 10,
    'phase2_epochs': 20,
    'lr_head':       1e-3,
    'lr_backbone':   1e-5,
    'patience':      5,
    'batch_size':    32,
}

trainer = FinetuneTrainer(
    backbone=backbone,
    head=head,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=solar_loss,
    cfg=cfg,
    ckpt_dir=CKPT_DIR,
    device=device,
    task='solar',
)
trainer.train(ssl_ckpt=SSL_CKPT)
"""

# ── Cell 7: Evaluate ─────────────────────────────────────────
"""
import torch, json, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.backbone.thermal_backbone import (
    ThermalWatchBackbone
)
from src.models.wildfire.wildfire_head import WildfireHead
from src.models.solar.solar_head import SolarHead
from src.models.training.finetune import (
    WildfireDataset, SolarDataset
)

GDRIVE = '/gdrive/MyDrive/thermalwatch'
device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

def evaluate(task, ckpt_path, dataset, loader):
    backbone = ThermalWatchBackbone(
        pretrained_optical=False
    ).to(device)
    head = (
        WildfireHead(512) if task == 'wildfire'
        else SolarHead(512)
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    backbone.load_state_dict(ckpt['backbone_state'])
    head.load_state_dict(ckpt['head_state'])
    backbone.eval()
    head.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            thermal = batch['thermal'].to(device)
            optical = batch['optical'].to(device)
            weather = batch['weather'].to(device)
            osm     = batch['osm'].to(device)
            has_s2  = batch['has_s2']
            mask    = has_s2.to(device).float()
            optical = optical * mask.view(-1, 1, 1, 1)

            emb   = backbone(
                thermal=thermal,
                optical=optical if has_s2.any() else None,
                weather=weather,
                osm=osm,
            )
            preds = head(emb)
            all_preds.append(preds)
            all_targets.append(batch['targets'])

    # Compute metrics
    if task == 'wildfire':
        risks   = torch.cat(
            [p['risk_score'] for p in all_preds]
        ).cpu().numpy()
        gt_risk = np.array(
            [t['risk_score'] for batch in all_targets
             for t in batch]
        )
        mse  = float(np.mean((risks - gt_risk)**2))
        rmse = float(np.sqrt(mse))
        r2   = float(
            1 - np.sum((risks - gt_risk)**2) /
            np.sum((gt_risk - gt_risk.mean())**2)
        )
        print(f'Wildfire Results:')
        print(f'  RMSE: {rmse:.4f}')
        print(f'  R²:   {r2:.4f}')

    else:
        eff   = torch.cat(
            [p['efficiency_score'] for p in all_preds]
        ).cpu().numpy()
        gt_eff = np.array(
            [t['efficiency_score'] for batch in all_targets
             for t in batch]
        )
        mse  = float(np.mean((eff - gt_eff)**2))
        rmse = float(np.sqrt(mse))
        r2   = float(
            1 - np.sum((eff - gt_eff)**2) /
            np.sum((gt_eff - gt_eff.mean())**2)
        )
        print(f'Solar Results:')
        print(f'  RMSE: {rmse:.4f}')
        print(f'  R²:   {r2:.4f}')

INDEX_DIR = Path('/content/data/indexes')

# Evaluate wildfire
wf_val = WildfireDataset(
    INDEX_DIR / 'wildfire_val_index.json', 'val'
)
wf_loader = DataLoader(wf_val, batch_size=32, num_workers=2)
evaluate(
    'wildfire',
    f'{GDRIVE}/checkpoints/finetune/wildfire/'
    'wildfire_best_phase2.pt',
    wf_val, wf_loader,
)

# Evaluate solar
sol_val = SolarDataset(
    INDEX_DIR / 'solar_val_index.json', 'val'
)
sol_loader = DataLoader(sol_val, batch_size=32, num_workers=2)
evaluate(
    'solar',
    f'{GDRIVE}/checkpoints/finetune/solar/'
    'solar_best_phase2.pt',
    sol_val, sol_loader,
)
"""
