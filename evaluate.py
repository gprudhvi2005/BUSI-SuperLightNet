"""
evaluate.py — Inference + metrics + visualization for SuperLightNet2D on BUSI.

Usage:
    python evaluate.py \
        --data_dir /path/to/Dataset_BUSI_with_GT \
        --checkpoint checkpoints/busi_exp/best_model.pth \
        --output_dir eval_output

Outputs:
    eval_output/
        results.txt                     <- per-image and aggregate metrics
        visualizations/
            pred_0000.png  ...          <- 3-panel: input | GT | prediction
"""
import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import TrainConfig
from dataset import BUSIDataset
from metrics import (dice_coefficient, iou_score, hausdorff_distance_95,
                     sensitivity, specificity, precision)
from networks.superlightnet_2d import SuperLightNet2D

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, cfg: TrainConfig) -> nn.Module:
    model = SuperLightNet2D(
        init_channels=cfg.init_channels,
        n_channels=cfg.n_channels,
        class_nums=cfg.class_nums,
        depths_unidirectional=cfg.depths,
    )
    state = torch.load(checkpoint_path, map_location='cpu')
    # Strip DataParallel / DDP wrapper prefix
    if all(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Loaded: {checkpoint_path}  ({params_M:.3f} M params)')
    return model


# ── Visualization ─────────────────────────────────────────────────────────────

def save_panel(image_tensor, gt_mask, pred_mask, out_path: str):
    """Save 3-panel figure: original | ground truth overlay | prediction overlay."""
    img = image_tensor.squeeze().cpu().numpy()   # [H, W]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = ['Input', 'Ground Truth', 'Prediction']
    overlays = [None, gt_mask, pred_mask]
    cmaps    = [None, 'Reds', 'Blues']

    for ax, title, overlay, cmap in zip(axes, titles, overlays, cmaps):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        if overlay is not None:
            ax.imshow(overlay.astype(float), alpha=0.45, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Evaluation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, output_dir: str, save_vis: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    keys = ('DSC', 'IoU', 'HD95', 'Sens', 'Spec', 'Prec')
    per_image = {k: [] for k in keys}
    result_lines = ['idx,DSC,IoU,HD95,Sens,Spec,Prec']

    global_idx = 0
    for batch_idx, (images, masks) in enumerate(loader):
        images_gpu = images.to(DEVICE)
        logits     = model(images_gpu)
        pred_masks = (logits.argmax(dim=1) == 1).cpu().numpy()   # [B, H, W]
        target_np  = masks.numpy()

        for b in range(images.size(0)):
            p  = pred_masks[b]
            t  = target_np[b]
            m  = {
                'DSC':  dice_coefficient(p, t),
                'IoU':  iou_score(p, t),
                'HD95': hausdorff_distance_95(p, t),
                'Sens': sensitivity(p, t),
                'Spec': specificity(p, t),
                'Prec': precision(p, t),
            }
            for k in keys:
                per_image[k].append(m[k])

            result_lines.append(
                f"{global_idx},"
                + ','.join(f"{m[k]:.4f}" for k in keys)
            )

            if save_vis:
                # De-normalise image for display
                from dataset import BUSI_MEAN, BUSI_STD
                img_display = images[b] * BUSI_STD + BUSI_MEAN
                img_display = img_display.clamp(0, 1)
                save_panel(img_display, t, p,
                           os.path.join(vis_dir, f'pred_{global_idx:04d}.png'))
            global_idx += 1

        print(f'\r  Processed {batch_idx + 1}/{len(loader)} batches', end='', flush=True)

    print()

    # ── Aggregate results ──────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('EVALUATION RESULTS')
    print('=' * 60)
    agg = {}
    for k in keys:
        vals  = [v for v in per_image[k] if not np.isnan(v)]
        mean_ = float(np.mean(vals)) if vals else float('nan')
        std_  = float(np.std(vals))  if vals else float('nan')
        agg[k] = mean_
        print(f'  {k:5s}: {mean_:.4f} ± {std_:.4f}  (n={len(vals)})')
        result_lines.append(f'MEAN_{k},{mean_:.4f}')
    print('=' * 60)

    # Save result file
    results_path = os.path.join(output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write('\n'.join(result_lines) + '\n')
    print(f'\nResults saved → {results_path}')
    if save_vis:
        print(f'Visualizations → {vis_dir}/')

    return agg


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('SuperLightNet2D evaluation on BUSI')
    p.add_argument('--data_dir',   required=True, type=str)
    p.add_argument('--test_file',  default='test.txt', type=str)
    p.add_argument('--checkpoint', required=True, type=str)
    p.add_argument('--output_dir', default='eval_output', type=str)
    p.add_argument('--img_size',   default=256, type=int)
    p.add_argument('--batch_size', default=1,   type=int)
    p.add_argument('--n_channels', default=32,  type=int)
    p.add_argument('--depths',     default='small', type=str)
    p.add_argument('--gpus',       default='0', type=str)
    p.add_argument('--no_vis',     action='store_true',
                   help='Skip saving visualizations (faster)')
    return p.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    cfg = TrainConfig(n_channels=args.n_channels, depths=args.depths)
    model = load_model(args.checkpoint, cfg).to(DEVICE)

    test_ds = BUSIDataset(
        os.path.join(args.data_dir, args.test_file),
        args.data_dir, 'test', args.img_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    print(f'Test set: {len(test_ds)} images')

    evaluate(model, test_loader, args.output_dir, save_vis=not args.no_vis)


if __name__ == '__main__':
    main()

