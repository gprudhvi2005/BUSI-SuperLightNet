"""
Segmentation metrics for binary 2-D medical image segmentation.

All functions accept numpy boolean/uint8 arrays of shape [H, W].
compute_all_metrics() accepts batched torch tensors.
"""
import numpy as np
import torch
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree


# ── Per-image metrics ───────────────────────────────────────────────────────

def dice_coefficient(pred: np.ndarray, target: np.ndarray,
                     smooth: float = 1e-6) -> float:
    """Dice Similarity Coefficient (DSC)."""
    p = pred.astype(bool).ravel()
    t = target.astype(bool).ravel()
    inter = (p & t).sum()
    return float((2.0 * inter + smooth) / (p.sum() + t.sum() + smooth))


def iou_score(pred: np.ndarray, target: np.ndarray,
              smooth: float = 1e-6) -> float:
    """Intersection over Union (Jaccard index)."""
    p = pred.astype(bool).ravel()
    t = target.astype(bool).ravel()
    inter = (p & t).sum()
    union = (p | t).sum()
    return float((inter + smooth) / (union + smooth))


def sensitivity(pred: np.ndarray, target: np.ndarray,
                smooth: float = 1e-6) -> float:
    """Sensitivity (Recall / True Positive Rate)."""
    p = pred.astype(bool).ravel()
    t = target.astype(bool).ravel()
    tp = (p & t).sum()
    fn = (~p & t).sum()
    return float((tp + smooth) / (tp + fn + smooth))


def specificity(pred: np.ndarray, target: np.ndarray,
                smooth: float = 1e-6) -> float:
    """Specificity (True Negative Rate)."""
    p = pred.astype(bool).ravel()
    t = target.astype(bool).ravel()
    tn = (~p & ~t).sum()
    fp = (p & ~t).sum()
    return float((tn + smooth) / (tn + fp + smooth))


def precision(pred: np.ndarray, target: np.ndarray,
              smooth: float = 1e-6) -> float:
    """Precision (Positive Predictive Value)."""
    p = pred.astype(bool).ravel()
    t = target.astype(bool).ravel()
    tp = (p & t).sum()
    fp = (p & ~t).sum()
    return float((tp + smooth) / (tp + fp + smooth))


def _surface(mask: np.ndarray) -> np.ndarray:
    """Return surface-voxel coordinates of a binary mask."""
    border = mask.astype(bool) & ~binary_erosion(mask.astype(bool))
    return np.argwhere(border)


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    95th-percentile Hausdorff distance (HD95) between two binary masks.
    Returns np.nan if either mask is empty.
    """
    if pred.sum() == 0 or target.sum() == 0:
        return np.nan
    sp = _surface(pred.astype(bool))
    st = _surface(target.astype(bool))
    if len(sp) == 0 or len(st) == 0:
        return np.nan
    tree_t = cKDTree(st)
    tree_p = cKDTree(sp)
    d_pt, _ = tree_t.query(sp)   # pred → target distances
    d_tp, _ = tree_p.query(st)   # target → pred distances
    return float(np.percentile(np.concatenate([d_pt, d_tp]), 95))


# ── Batched helper ──────────────────────────────────────────────────────────

def compute_all_metrics(logits: torch.Tensor,
                        targets: torch.Tensor) -> dict:
    """
    Compute all metrics for a batch, averaged over batch dimension.

    Args:
        logits  : [B, 2, H, W] — raw model output (pre-softmax).
        targets : [B, H, W]    — long ground-truth masks (0/1).

    Returns:
        dict with keys 'DSC', 'IoU', 'HD95', 'Sens', 'Spec', 'Prec'.
    """
    pred_np   = (logits.argmax(dim=1) == 1).cpu().numpy()   # [B, H, W]
    target_np = targets.cpu().numpy()

    results = {k: [] for k in ('DSC', 'IoU', 'HD95', 'Sens', 'Spec', 'Prec')}
    for p, t in zip(pred_np, target_np):
        results['DSC'].append(dice_coefficient(p, t))
        results['IoU'].append(iou_score(p, t))
        results['HD95'].append(hausdorff_distance_95(p, t))
        results['Sens'].append(sensitivity(p, t))
        results['Spec'].append(specificity(p, t))
        results['Prec'].append(precision(p, t))

    return {k: float(np.nanmean(v)) for k, v in results.items()}
