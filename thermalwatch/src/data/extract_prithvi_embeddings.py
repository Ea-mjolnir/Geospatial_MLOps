"""
ThermalWatch — Pre-compute Prithvi Embeddings
===============================================
Runs Prithvi-EO-2.0-300M once on all S2 patches
and saves embeddings to disk.

This avoids running Prithvi during every training step.
Since Prithvi is frozen during SSL pre-training,
its output is deterministic — safe to pre-compute.

Output:
  data/wildfire/patches/{area_year}/prithvi_{patch_id}.npy
  Shape: [1024] float32

Index updated with:
  prithvi_path: path to embedding file

Usage:
  python3 -m src.data.extract_prithvi_embeddings \
    --task wildfire \
    --data-dir data/wildfire \
    --batch-size 32
"""

import json
import logging
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)

S2_MEAN = np.array(
    [0.05, 0.06, 0.07, 0.25, 0.20, 0.15],
    dtype=np.float32
)
S2_STD = np.array(
    [0.05, 0.05, 0.06, 0.10, 0.10, 0.08],
    dtype=np.float32
)


class S2PatchDataset(Dataset):
    """Dataset of S2 patches for Prithvi embedding extraction."""

    def __init__(self, entries: List[Dict]):
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict:
        entry   = self.entries[idx]
        s2_path = entry['s2_path']

        try:
            s2  = np.load(s2_path).astype(np.float32)
            s2  = np.nan_to_num(s2, nan=0.0)
            s2  = (s2 - S2_MEAN[:, None, None]) / (
                S2_STD[:, None, None] + 1e-6
            )
            ok  = True
        except Exception:
            s2  = np.zeros((6, 224, 224), dtype=np.float32)
            ok  = False

        return {
            's2':       torch.from_numpy(s2).float(),
            'patch_id': entry['patch_id'],
            'ok':       torch.tensor(ok, dtype=torch.bool),
        }


def extract_embeddings(
    task:       str,
    data_dir:   Path,
    batch_size: int = 32,
    device:     torch.device = None,
):
    """Extract Prithvi embeddings for all S2 patches."""
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    log.info(f'Device: {device}')

    # Load Prithvi
    log.info('Loading Prithvi-EO-2.0-300M...')
    from terratorch.registry import BACKBONE_REGISTRY
    encoder = BACKBONE_REGISTRY.build(
        'prithvi_eo_v2_300',
        pretrained=True,
        num_frames=1,
        in_chans=6,
    )
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    log.info('✅ Prithvi loaded')

    patch_dir = data_dir / 'patches'

    for idx_path in sorted(patch_dir.glob('*_index.json')):
        index = json.loads(idx_path.read_text())

        # Find patches with S2 but no prithvi embedding
        todo = [
            e for e in index
            if e.get('s2_path') and
            Path(e['s2_path']).exists() and
            not e.get('prithvi_path')
        ]

        if not todo:
            log.info(f'Skip {idx_path.stem} — all done')
            continue

        log.info(
            f'{idx_path.stem}: {len(todo)} patches to embed'
        )

        # Output directory
        area_year = idx_path.stem.replace('_index', '')
        out_dir   = patch_dir / area_year
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build lookup
        lookup = {e['patch_id']: e for e in index}

        # DataLoader
        ds     = S2PatchDataset(todo)
        loader = DataLoader(
            ds, batch_size=batch_size,
            shuffle=False, num_workers=4,
            pin_memory=True,
        )

        done = failed = 0

        with torch.no_grad():
            for batch in loader:
                s2       = batch['s2'].to(device)
                patch_ids = batch['patch_id']
                ok        = batch['ok']

                # Add time dimension [B,6,224,224] → [B,6,1,224,224]
                s2 = s2.unsqueeze(2)

                out  = encoder(s2)
                feat = out[-1] if isinstance(
                    out, (list, tuple)
                ) else out

                # Pool: [B, 197, 1024] → [B, 1024]
                if feat.dim() == 3:
                    emb = feat[:, 1:, :].mean(dim=1)
                else:
                    emb = feat.mean(dim=[2, 3])

                emb_np = emb.cpu().numpy()

                for i, pid in enumerate(patch_ids):
                    if not ok[i]:
                        failed += 1
                        continue

                    emb_path = out_dir / f'prithvi_{pid}.npy'
                    np.save(str(emb_path), emb_np[i])

                    if pid in lookup:
                        lookup[pid]['prithvi_path'] = str(
                            emb_path
                        )
                    done += 1

        # Save updated index
        idx_path.write_text(json.dumps(index, indent=2))
        log.info(
            f'✅ {idx_path.stem}: '
            f'done={done} failed={failed}'
        )

    log.info('✅ Prithvi embedding extraction complete')


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task', choices=['wildfire', 'solar'],
        required=True
    )
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    extract_embeddings(
        task=args.task,
        data_dir=Path(args.data_dir),
        batch_size=args.batch_size,
        device=device,
    )


if __name__ == '__main__':
    main()
