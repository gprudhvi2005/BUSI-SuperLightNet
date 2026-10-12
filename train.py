"""
train.py — Full training script for SuperLightNet2D on BUSI.

Typical usage:
    python train.py \
        --data_dir /path/to/Dataset_BUSI_with_GT \
        --experiment busi_v1 \
        --epochs 200 \
        --batch_size 8 \
        --lr 1e-3 \
        --gpus 0

Splits (train.txt / val.txt) must already exist in data_dir.
Run split_dataset.py first if they do not.
"""
import os
import time
import random
import logging
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from config import TrainConfig
from dataset import BUSIDataset
from losses import BCEWithDiceLoss, FocalDiceLoss
from metrics import compute_all_metrics
from networks.superlightnet_2d import SuperLightNet2D

# ── device ────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Helpers ──────────────────────────────────────────────────────────────────

def seed_fix(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_optimizer(model, cfg: TrainConfig):
    if cfg.optimizer == 'AdamW':
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == 'Adam':
        return optim.Adam(model.parameters(),  lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == 'SGD':
        return optim.SGD(model.parameters(),   lr=cfg.lr, weight_decay=cfg.weight_decay,
                         momentum=0.99, nesterov=True)
    raise ValueError(f'Unknown optimizer: {cfg.optimizer}')


def build_scheduler(optimizer, cfg: TrainConfig):
    if cfg.lr_scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.t_max, eta_min=cfg.lr * cfg.eta_min_ratio)
    if cfg.lr_scheduler == 'poly':
        return optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda ep: (1 - ep / cfg.epochs) ** 0.9)
    raise ValueError(f'Unknown scheduler: {cfg.lr_scheduler}')


def build_criterion(cfg: TrainConfig):
    if cfg.criterion == 'BCEWithDiceLoss':
        return BCEWithDiceLoss()
    if cfg.criterion == 'FocalDiceLoss':
        return FocalDiceLoss()
    raise ValueError(f'Unknown criterion: {cfg.criterion}')


# ── Training & Validation ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer,
                    scheduler, scaler, cfg, epoch):
    model.train()
    total_loss = total_dice = 0.0

    for step, (images, masks) in enumerate(loader, 1):
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)
        optimizer.zero_grad()

        if cfg.amp:
            with autocast():
                logits = model(images)
                loss, dice = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss, dice = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_dice += dice.item()

        if step % 10 == 0 or step == len(loader):
            logging.info(
                f'[Train] Ep {epoch:03d} | {step:03d}/{len(loader)} | '
                f'Loss={loss.item():.4f} | Dice={dice.item():.4f}')

    scheduler.step()
    return total_loss / len(loader), total_dice / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, epoch):
    model.eval()
    total_loss = total_dice = 0.0
    all_m = {k: [] for k in ('DSC', 'IoU', 'HD95', 'Sens', 'Spec', 'Prec')}

    for images, masks in loader:
        images = images.to(DEVICE)
        masks  = masks.to(DEVICE)
        logits = model(images)
        loss, dice = criterion(logits, masks)
        total_loss += loss.item()
        total_dice += dice.item()
        m = compute_all_metrics(logits, masks)
        for k in all_m:
            all_m[k].append(m[k])

    avg = {k: float(np.nanmean(v)) for k, v in all_m.items()}
    logging.info(
        f'[Val]   Ep {epoch:03d} | '
        f'Loss={total_loss/len(loader):.4f} | '
        f'DSC={avg["DSC"]:.4f} | IoU={avg["IoU"]:.4f} | '
        f'HD95={avg["HD95"]:.1f} | Sens={avg["Sens"]:.4f}')
    return total_loss / len(loader), avg['DSC'], avg


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('SuperLightNet2D training on BUSI')
    p.add_argument('--data_dir',    required=True,  type=str)
    p.add_argument('--train_file',  default='train.txt', type=str)
    p.add_argument('--val_file',    default='val.txt',   type=str)
    p.add_argument('--experiment',  default='busi_exp',  type=str)
    p.add_argument('--epochs',      default=200, type=int)
    p.add_argument('--batch_size',  default=8,   type=int)
    p.add_argument('--lr',          default=1e-3, type=float)
    p.add_argument('--weight_decay',default=1e-2, type=float)
    p.add_argument('--n_channels',  default=32,  type=int)
    p.add_argument('--depths',      default='small', type=str,
                   choices=['small', 'medium', 'large'])
    p.add_argument('--criterion',   default='BCEWithDiceLoss', type=str)
    p.add_argument('--optimizer',   default='AdamW', type=str)
    p.add_argument('--lr_scheduler',default='cosine', type=str)
    p.add_argument('--amp',         action='store_true')
    p.add_argument('--img_size',    default=256, type=int)
    p.add_argument('--num_workers', default=4,   type=int)
    p.add_argument('--seed',        default=2024, type=int)
    p.add_argument('--gpus',        default='0',  type=str)
    p.add_argument('--save_freq',   default=10,  type=int)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    cfg = TrainConfig(**{k: v for k, v in vars(args).items()
                         if k in TrainConfig.__dataclass_fields__})

    seed_fix(cfg.seed)

    # ── Directories ──────────────────────────────────────────────────────────
    ckpt_dir = os.path.join('checkpoints', cfg.experiment)
    log_dir  = os.path.join('logs', cfg.experiment)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'train.log'))
        ])
    logging.info(f'Config: {cfg}')

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = BUSIDataset(
        os.path.join(cfg.data_dir, cfg.train_file), cfg.data_dir, 'train', cfg.img_size)
    val_ds   = BUSIDataset(
        os.path.join(cfg.data_dir, cfg.val_file),   cfg.data_dir, 'val',   cfg.img_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)
    logging.info(f'Train: {len(train_ds)} samples | Val: {len(val_ds)} samples')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SuperLightNet2D(
        init_channels=cfg.init_channels,
        n_channels=cfg.n_channels,
        class_nums=cfg.class_nums,
        depths_unidirectional=cfg.depths,
    ).to(DEVICE)
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    logging.info(f'Model parameters: {params_M:.3f} M')

    criterion = build_criterion(cfg).to(DEVICE)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = GradScaler(enabled=cfg.amp)

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_dsc  = 0.0
    t_start   = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t_ep = time.time()

        tr_loss, tr_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, epoch)
        val_loss, val_dsc, val_m = validate(model, val_loader, criterion, epoch)

        lr_now   = optimizer.param_groups[0]['lr']
        elapsed  = (time.time() - t_ep) / 60
        eta_h    = (cfg.epochs - epoch) * elapsed / 60
        logging.info(
            f'Ep {epoch:03d}/{cfg.epochs} | LR={lr_now:.2e} | '
            f'TrainLoss={tr_loss:.4f} | ValDSC={val_dsc:.4f} | '
            f'EpTime={elapsed:.1f}min | ETA={eta_h:.2f}h')

        # Save best
        if val_dsc > best_dsc:
            best_dsc = val_dsc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
            logging.info(f'  *** Best DSC={best_dsc:.4f} — saved best_model.pth ***')

        # Periodic checkpoint
        if epoch % cfg.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth'))

    total_h = (time.time() - t_start) / 3600
    logging.info(f'Training finished. Best Val DSC={best_dsc:.4f}. '
                 f'Total time={total_h:.2f} h')
    logging.info(f'Checkpoints saved in: {ckpt_dir}')


if __name__ == '__main__':
    main()

