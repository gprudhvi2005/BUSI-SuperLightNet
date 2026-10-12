"""
generate_superlightnet_plots.py
Extracts metrics from training logs and generates:
1. SuperLightNet-2D Training Curves (eval_output/busi_v1/training_curves.png, training_curves_superlightnet.png)
2. Comparison Training Curves (SuperLightNet-2D vs UNet Baseline)
3. SuperLightNet-2D Sample Predictions Grid (sample_predictions_superlightnet.png)
4. Direct visual prediction comparison: Input | GT | UNet | SuperLightNet-2D
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# -------------------------------------------------------------
# 1. Parse SuperLightNet-2D train.log
# -------------------------------------------------------------
sln_log_path = os.path.join('logs', 'busi_v1', 'train.log')
with open(sln_log_path, 'r') as f:
    sln_lines = f.readlines()

sln_epochs = []
sln_train_loss = []
sln_val_loss = []
sln_val_dsc = []
sln_val_iou = []

# Map epoch -> metrics to ensure correct alignment
sln_val_data = {}
sln_trn_data = {}

for line in sln_lines:
    m_val = re.search(r'\[Val\]\s+Ep\s+(\d+)\s+\|\s+Loss=([\d\.]+)\s+\|\s+DSC=([\d\.]+)\s+\|\s+IoU=([\d\.]+)', line)
    if m_val:
        ep = int(m_val.group(1))
        sln_val_data[ep] = {
            'loss': float(m_val.group(2)),
            'dsc': float(m_val.group(3)),
            'iou': float(m_val.group(4)),
        }
    
    m_trn = re.search(r'Ep\s+(\d+)/\d+\s+\|\s+LR=[^\s]+\s+\|\s+TrainLoss=([\d\.]+)', line)
    if m_trn:
        ep = int(m_trn.group(1))
        sln_trn_data[ep] = float(m_trn.group(2))

for ep in sorted(sln_val_data.keys()):
    if ep in sln_trn_data:
        sln_epochs.append(ep)
        sln_train_loss.append(sln_trn_data[ep])
        sln_val_loss.append(sln_val_data[ep]['loss'])
        sln_val_dsc.append(sln_val_data[ep]['dsc'])
        sln_val_iou.append(sln_val_data[ep]['iou'])

sln_best_dsc = max(sln_val_dsc)
best_ep = sln_epochs[sln_val_dsc.index(sln_best_dsc)]
print(f"SuperLightNet-2D: {len(sln_epochs)} epochs. Best Val DSC: {sln_best_dsc:.4f} at epoch {best_ep}")

# -------------------------------------------------------------
# 2. Parse UNet Baseline train.log
# -------------------------------------------------------------
unet_log_path = os.path.join('logs', 'unet_baseline', 'train.log')
unet_epochs = []
unet_train_loss = []
unet_val_loss = []
unet_val_dsc = []
unet_val_iou = []

if os.path.exists(unet_log_path):
    with open(unet_log_path, 'r') as f:
        unet_lines = f.readlines()
    for line in unet_lines:
        m = re.search(r'Ep\s+(\d+)/\d+\s+\|\s+LR=[^\s]+\s+\|\s+TrLoss=([\d\.]+)\s+\|\s+TrDice=[^\s]+\s+\|\s+ValLoss=([\d\.]+)\s+\|\s+ValDSC=([\d\.]+)\s+\|\s+ValIoU=([\d\.]+)', line)
        if m:
            unet_epochs.append(int(m.group(1)))
            unet_train_loss.append(float(m.group(2)))
            unet_val_loss.append(float(m.group(3)))
            unet_val_dsc.append(float(m.group(4)))
            unet_val_iou.append(float(m.group(5)))
    unet_best_dsc = max(unet_val_dsc) if unet_val_dsc else 0.0
    print(f"UNet Baseline: {len(unet_epochs)} epochs. Best Val DSC: {unet_best_dsc:.4f}")
else:
    unet_best_dsc = 0.0

# -------------------------------------------------------------
# 3. Plot SuperLightNet-2D Training Curves (Exact UNet style)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('SuperLightNet-2D — Training Curves', fontsize=14, fontweight='bold')

axes[0].plot(sln_epochs, sln_train_loss, label='Train', color='steelblue', lw=1.5)
axes[0].plot(sln_epochs, sln_val_loss,   label='Val',   color='tomato', lw=1.5)
axes[0].set_title('Loss', fontsize=12)
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(sln_epochs, sln_val_dsc, color='seagreen', lw=1.5)
axes[1].axhline(sln_best_dsc, color='gold', ls='--', label=f'Best={sln_best_dsc:.4f}')
axes[1].set_title('Val DSC', fontsize=12)
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(sln_epochs, sln_val_iou, color='mediumpurple', lw=1.5)
axes[2].set_title('Val IoU', fontsize=12)
axes[2].set_xlabel('Epoch')
axes[2].grid(alpha=0.3)

plt.tight_layout()

os.makedirs('eval_output/busi_v1', exist_ok=True)
sln_curve_path1 = os.path.join('eval_output', 'busi_v1', 'training_curves.png')
sln_curve_path2 = 'training_curves_superlightnet.png'
plt.savefig(sln_curve_path1, dpi=150, bbox_inches='tight')
plt.savefig(sln_curve_path2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved SuperLightNet training curves -> {sln_curve_path1} & {sln_curve_path2}")

# -------------------------------------------------------------
# 4. Plot Comparison Training Curves (Both models on same axes)
# -------------------------------------------------------------
if unet_epochs:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle('Training Dynamics Comparison: SuperLightNet-2D vs. UNet Baseline', fontsize=13, fontweight='bold')

    # Loss
    axes[0].plot(sln_epochs, sln_train_loss, label='SLN Train', color='#1f77b4', lw=1.5)
    axes[0].plot(sln_epochs, sln_val_loss,   label='SLN Val',   color='#aec7e8', lw=1.2, ls='--')
    axes[0].plot(unet_epochs, unet_train_loss, label='UNet Train', color='#d62728', lw=1.5)
    axes[0].plot(unet_epochs, unet_val_loss,   label='UNet Val',   color='#ff9896', lw=1.2, ls='--')
    axes[0].set_title('Training & Validation Loss', fontsize=11)
    axes[0].set_xlabel('Epoch')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Val DSC
    axes[1].plot(sln_epochs, sln_val_dsc, label=f'SuperLightNet-2D (Best={sln_best_dsc:.4f})', color='#2ca02c', lw=1.8)
    axes[1].plot(unet_epochs, unet_val_dsc, label=f'UNet Baseline (Best={unet_best_dsc:.4f})', color='#ff7f0e', lw=1.5, ls='--')
    axes[1].axhline(sln_best_dsc, color='#2ca02c', ls=':', alpha=0.7)
    axes[1].axhline(unet_best_dsc, color='#ff7f0e', ls=':', alpha=0.7)
    axes[1].set_title('Validation Dice (DSC)', fontsize=11)
    axes[1].set_xlabel('Epoch')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    # Val IoU
    axes[2].plot(sln_epochs, sln_val_iou, label='SuperLightNet-2D', color='#9467bd', lw=1.8)
    axes[2].plot(unet_epochs, unet_val_iou, label='UNet Baseline', color='#8c564b', lw=1.5, ls='--')
    axes[2].set_title('Validation IoU', fontsize=11)
    axes[2].set_xlabel('Epoch')
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    comp_path1 = 'training_curves_comparison.png'
    comp_path2 = os.path.join('eval_output', 'training_curves_comparison.png')
    plt.savefig(comp_path1, dpi=150, bbox_inches='tight')
    plt.savefig(comp_path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison curves -> {comp_path1}")

# -------------------------------------------------------------
# 5. SuperLightNet-2D Sample Predictions Grid (6 samples)
# -------------------------------------------------------------
sln_vis_dir = os.path.join('eval_output', 'busi_v1', 'visualizations')
sample_indices = [0, 1, 2, 3, 4, 6]  # 6 representative samples
loaded_imgs = []
for idx in sample_indices:
    p = os.path.join(sln_vis_dir, f'pred_{idx:04d}.png')
    if os.path.exists(p):
        loaded_imgs.append(mpimg.imread(p))

if loaded_imgs:
    fig, axes = plt.subplots(len(loaded_imgs), 1, figsize=(11, 3.2 * len(loaded_imgs)))
    fig.suptitle('SuperLightNet-2D — Sample Test Predictions (Input | Ground Truth | Prediction)', fontsize=13, fontweight='bold', y=0.995)
    for ax, img in zip(axes, loaded_imgs):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    sln_pred_grid_path = os.path.join('eval_output', 'busi_v1', 'sample_predictions.png')
    sln_pred_grid_root = 'sample_predictions_superlightnet.png'
    plt.savefig(sln_pred_grid_path, dpi=120, bbox_inches='tight')
    plt.savefig(sln_pred_grid_root, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved SuperLightNet sample predictions -> {sln_pred_grid_path}")

# -------------------------------------------------------------
# 6. Direct Side-by-Side Comparison: UNet vs SuperLightNet-2D
# -------------------------------------------------------------
unet_vis_dir = os.path.join('eval_output', 'unet_baseline', 'visualizations')
# We can create a 4-column figure: Input | GT | UNet Prediction | SuperLightNet-2D Prediction
# Notice each pred_XXXX.png has 3 panels: [Input, GT, Pred].
# Let's crop them and display side-by-side for 4 test cases!
comp_indices = [0, 1, 2, 4]
fig, axes = plt.subplots(len(comp_indices), 4, figsize=(14, 3.3 * len(comp_indices)))
col_titles = ['(a) Input Image', '(b) Ground Truth (Red)', '(c) UNet Baseline (Blue)', '(d) SuperLightNet-2D (Blue)']

for row_i, idx in enumerate(comp_indices):
    sln_p = os.path.join(sln_vis_dir, f'pred_{idx:04d}.png')
    unet_p = os.path.join(unet_vis_dir, f'pred_{idx:04d}.png')
    
    if os.path.exists(sln_p) and os.path.exists(unet_p):
        sln_full = mpimg.imread(sln_p)
        unet_full = mpimg.imread(unet_p)
        
        # In a 3-panel image saved by save_panel (width W, height H)
        # width is divided roughly into 3 thirds
        h, w = sln_full.shape[:2]
        w3 = w // 3
        
        # Sub-panel 0: Input
        inp_crop = sln_full[:, :w3]
        # Sub-panel 1: GT
        gt_crop = sln_full[:, w3:2*w3]
        # Sub-panel 2: SLN Pred
        sln_crop = sln_full[:, 2*w3:]
        # UNet Pred:
        unet_crop = unet_full[:, 2*w3:]
        
        crops = [inp_crop, gt_crop, unet_crop, sln_crop]
        for col_i, crop in enumerate(crops):
            ax = axes[row_i, col_i]
            ax.imshow(crop)
            if row_i == 0:
                ax.set_title(col_titles[col_i], fontsize=11, fontweight='semibold')
            ax.axis('off')

plt.suptitle('Qualitative Segmentation Comparison: UNet Baseline vs. SuperLightNet-2D', fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
side_by_side_path = 'qualitative_comparison_unet_vs_superlightnet.png'
plt.savefig(side_by_side_path, dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"Saved qualitative comparison -> {side_by_side_path}")
