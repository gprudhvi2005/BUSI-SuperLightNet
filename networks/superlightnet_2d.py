"""
SuperLightNet 2D  --  pure-PyTorch implementation for BUSI segmentation.
No MONAI dependency: all ops use standard nn.Conv2d / F.interpolate.
"""
from __future__ import annotations
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


# ── helpers ─────────────────────────────────────────────────────────────────

def _safe_groups(channels: int, target: int = 8) -> int:
    """Largest divisor of `channels` that is <= target."""
    for g in range(min(target, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class LayerNorm(nn.Module):
    """ConvNeXt-style LayerNorm (channels_first or channels_last)."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6,
                 data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps, self.data_format = eps, data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape,
                                self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MlpChannel2d(nn.Module):
    """Point-wise channel MLP (two 1x1 convs)."""

    def __init__(self, in_ch: int, expr: int = 1, out_ch: int = None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * expr, 1),
            nn.GELU(),
            nn.Conv2d(in_ch * expr, out_ch, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ── core attention (GHPA) ────────────────────────────────────────────────────

class GHPA(nn.Module):
    """
    Grouped Multi-Axis Hadamard Product Attention (2-D).
    Adapted from EGE-UNet; operates natively on [B, C, H, W].
    """

    def __init__(self, dim_in: int, x: int = 8, y: int = 8):
        super().__init__()
        c = dim_in // 4
        p = 1

        self.params_xy = nn.Parameter(torch.ones(1, c, x, y))
        self.conv_xy   = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=p, groups=c), nn.GELU(), nn.Conv2d(c, c, 1))

        self.params_zx = nn.Parameter(torch.ones(1, 1, c, x))
        self.conv_zx   = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=p, groups=c), nn.GELU(), nn.Conv1d(c, c, 1))

        self.params_zy = nn.Parameter(torch.ones(1, 1, c, y))
        self.conv_zy   = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=p, groups=c), nn.GELU(), nn.Conv1d(c, c, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c, c, 1), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1, groups=c))

        self.norm1 = LayerNorm(dim_in, data_format="channels_first")
        self.norm2 = LayerNorm(dim_in, data_format="channels_first")
        self.ldw   = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_in, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        # XY plane
        x1 = x1 * self.conv_xy(
            F.interpolate(self.params_xy, size=x1.shape[2:],
                          mode="bilinear", align_corners=True))
        # ZX plane (W treated as extra axis)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = x2 * self.conv_zx(
            F.interpolate(self.params_zx, size=x2.shape[2:],
                          mode="bilinear", align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        # ZY plane (H treated as extra axis)
        x3 = x3.permute(0, 2, 1, 3)
        x3 = x3 * self.conv_zy(
            F.interpolate(self.params_zy, size=x3.shape[2:],
                          mode="bilinear", align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        # DW branch
        x4 = self.dw(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.norm2(x)
        return self.ldw(x)


# ── encoder block ────────────────────────────────────────────────────────────

class THPAEncFR2(nn.Module):
    """
    2-D encoder block (adapted from THPAEncFR3).
    Splits channels: first half -> GHPA (fwd + h-flip); second half -> residual.
    """

    def __init__(self, in_ch: int, expr: int = 2):
        super().__init__()
        half = in_ch // 2
        self.norm1 = nn.InstanceNorm2d(half)
        self.ghpa  = GHPA(half)
        self.norm2 = nn.InstanceNorm2d(in_ch)
        self.mlp   = MlpChannel2d(in_ch, expr)

    def forward(self, x: Tensor) -> Tensor:
        x, res = torch.chunk(x, 2, dim=1)
        x = self.norm1(x)
        x = self.ghpa(x) + self.ghpa(x.flip([3])).flip([3])
        x = torch.cat((x, res), dim=1)
        x = self.norm2(x)
        return self.mlp(x)


class NormDownsample2d(nn.Module):
    """InstanceNorm2d + strided Conv2d(2x2)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_ch)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.norm(x))


# ── decoder block ────────────────────────────────────────────────────────────

class UpRepr2d(nn.Module):
    """
    2-D decoder block (adapted from Learnable_Res_Skip_UpRepr4).
    channel project -> bilinear x2 upsample -> learnable skip scale
    -> depthwise repr conv -> learnable residual scale.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        g = _safe_groups(out_ch, 8)

        self.upc  = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.repr = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1,
                      groups=g, bias=False),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
        )
        self.norm       = nn.InstanceNorm2d(out_ch)
        self.skip_scale = nn.Parameter(torch.ones(1, out_ch, 1, 1))
        self.res_scale  = nn.Parameter(torch.ones(1))

    def forward(self, inp_skip: tuple) -> Tensor:
        x, skip = inp_skip
        x   = F.interpolate(self.upc(x), scale_factor=2,
                             mode="bilinear", align_corners=False)
        x   = x + skip * self.skip_scale
        res = x
        x   = self.norm(x)
        return self.repr(x) + res * self.res_scale


# ── block factory ────────────────────────────────────────────────────────────

def _enc_stack(depth: int, ch: int) -> nn.Module:
    return nn.Sequential(*[THPAEncFR2(ch) for _ in range(depth)])


# ── encoder ──────────────────────────────────────────────────────────────────

class Encoder2d(nn.Module):
    _PRESETS = {
        "small":  [1, 1, 2, 2, 2],
        "medium": [3, 4, 4, 4, 4],
        "large":  [3, 4, 8, 8, 8],
    }

    def __init__(self, init_ch: int = 1, n_ch: int = 32,
                 depth_preset: str = "small"):
        super().__init__()
        d = self._PRESETS[depth_preset]
        C = n_ch
        self.stem = nn.Conv2d(init_ch, C, kernel_size=1)
        self.enc0, self.down0 = _enc_stack(d[0], C),    NormDownsample2d(C,    C*2)
        self.enc1, self.down1 = _enc_stack(d[1], C*2),  NormDownsample2d(C*2,  C*4)
        self.enc2, self.down2 = _enc_stack(d[2], C*4),  NormDownsample2d(C*4,  C*8)
        self.enc3, self.down3 = _enc_stack(d[3], C*8),  NormDownsample2d(C*8,  C*16)
        self.bottleneck       = _enc_stack(d[4], C*16)

    def forward(self, x: Tensor):
        x = self.stem(x)
        skips = []
        for enc, down in [(self.enc0, self.down0), (self.enc1, self.down1),
                          (self.enc2, self.down2), (self.enc3, self.down3)]:
            s = enc(x); skips.append(s); x = down(s); del s
        return self.bottleneck(x), tuple(skips)


# ── decoder ──────────────────────────────────────────────────────────────────

class Decoder2d(nn.Module):
    def __init__(self, n_ch: int = 32):
        super().__init__()
        C = n_ch
        self.up3 = UpRepr2d(C*16, C*8)
        self.up2 = UpRepr2d(C*8,  C*4)
        self.up1 = UpRepr2d(C*4,  C*2)
        self.up0 = UpRepr2d(C*2,  C)

    def forward(self, hidden: Tensor, skips: tuple) -> Tensor:
        x = self.up3((hidden,  skips[3]))
        x = self.up2((x,       skips[2]))
        x = self.up1((x,       skips[1]))
        x = self.up0((x,       skips[0]))
        return x


# ── top-level model ───────────────────────────────────────────────────────────

class SuperLightNet2D(nn.Module):
    """
    SuperLightNet adapted for 2-D BUSI segmentation (pure-PyTorch, no MONAI).

    Input:  [B, init_channels, H, W]   -- default [B, 1, 256, 256]
    Output: [B, class_nums, H, W]      -- default [B, 2, 256, 256]
    """

    def __init__(self, init_channels: int = 1, n_channels: int = 32,
                 class_nums: int = 2, depths_unidirectional: str = "small"):
        super().__init__()
        self.encoder = Encoder2d(init_channels, n_channels, depths_unidirectional)
        self.decoder = Decoder2d(n_channels)
        self.norm    = nn.GroupNorm(n_channels, n_channels)
        self.head    = MlpChannel2d(n_channels, expr=2, out_ch=class_nums)

    def forward(self, x: Tensor) -> Tensor:
        hidden, skips = self.encoder(x)
        out = self.decoder(hidden, skips)
        return self.head(self.norm(out))


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(2, 1, 256, 256, device=device)
    m = SuperLightNet2D().to(device)
    o = m(x)
    p = sum(v.numel() for v in m.parameters()) / 1e6
    print(f"Input : {list(x.shape)}")
    print(f"Output: {list(o.shape)}")
    print(f"Params: {p:.3f} M")
