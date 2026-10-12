"""
Loss functions for binary 2-D medical image segmentation.
Both losses return  (total_loss, dice_score)  where dice_score is detached.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithDiceLoss(nn.Module):
    """
    Cross-entropy + soft Dice loss.

    CE is applied on logits (stable); Dice is applied on softmax probabilities.
    Works for 2-class (binary) segmentation.

    Args:
        smooth      : Laplace smoothing to avoid division by zero.
        bce_weight  : Weight for the CE term.
        dice_weight : Weight for the Dice term.
    """

    def __init__(self, smooth: float = 1e-5,
                 bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.smooth      = smooth
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits  : [B, 2, H, W]  — raw (pre-softmax) output
        targets : [B, H, W]     — long tensor, values 0 or 1
        Returns : (loss, dice_score)
        """
        # ── CE ────────────────────────────────────────────────────────────
        ce = F.cross_entropy(logits, targets)

        # ── Soft Dice on foreground class (index 1) ────────────────────────
        prob = torch.softmax(logits, dim=1)[:, 1]       # [B, H, W]
        p    = prob.contiguous().view(-1)
        t    = targets.float().contiguous().view(-1)
        inter    = (p * t).sum()
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (
                          p.sum() + t.sum() + self.smooth)

        total = self.bce_weight * ce + self.dice_weight * dice_loss
        return total, (1.0 - dice_loss).detach()


class FocalDiceLoss(nn.Module):
    """
    Focal loss + soft Dice loss.
    Better suited for highly imbalanced foreground/background ratios
    (e.g. small tumors in normal-tissue-dominant BUSI images).

    Args:
        alpha  : Weighting factor for the focal loss.
        gamma  : Focusing parameter (0 = standard BCE).
        smooth : Laplace smoothing.
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, smooth: float = 1e-5):
        super().__init__()
        self.alpha  = alpha
        self.gamma  = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        prob = torch.softmax(logits, dim=1)[:, 1]       # [B, H, W]
        p    = prob.contiguous().view(-1)
        t    = targets.float().contiguous().view(-1)

        # ── Soft Dice ───────────────────────────────────────────────────────
        inter    = (p * t).sum()
        dice_loss = 1.0 - (2.0 * inter + self.smooth) / (
                          p.sum() + t.sum() + self.smooth)

        # ── Focal ───────────────────────────────────────────────────────────
        bce       = F.binary_cross_entropy(p, t, reduction='none')
        pt        = torch.exp(-bce)
        focal     = self.alpha * (1.0 - pt) ** self.gamma * bce

        total = focal.mean() + dice_loss
        return total, (1.0 - dice_loss).detach()
