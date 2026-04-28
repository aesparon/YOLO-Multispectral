"""
patch_backbone_with_attention.py (FIXED)

Stable Conv-level patcher for Ultralytics YOLOv11-seg (and forks).

Goals
-----
1) Add optional CBAM / ECA / DropBlock / GroupNorm / Spectral mixing WITHOUT breaking the Ultralytics graph.
2) Never break segmentation head/proto shapes (skip head by default).
3) Keep state_dict keys stable by reusing existing conv/bn/act modules.
4) Avoid the spectral channel-mismatch bug by applying spectral mixing on *post-conv* features
   (shape [B, c_out, H, W]) so in_channels == out_channels == c_out.

This file MUST export: `patch_backbone_with_attention`
so `from patch_backbone_with_attention import patch_backbone_with_attention` works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Best-effort import of Ultralytics Conv
# ----------------------------
def _try_import_ultralytics_conv_types() -> Tuple[type, ...]:
    candidates = []
    # Common Ultralytics paths
    for modpath in (
        "ultralytics.nn.modules",
        "ultralytics.nn.modules.conv",
        "ultralytics_MS.ultralytics.nn.modules",
        "ultralytics_MS.ultralytics.nn.modules.conv",
    ):
        try:
            mod = __import__(modpath, fromlist=["Conv"])
            candidates.append(getattr(mod, "Conv"))
        except Exception:
            pass
    # De-dup
    uniq = []
    for c in candidates:
        if c is not None and c not in uniq:
            uniq.append(c)
    return tuple(uniq)


_CONV_TYPES = _try_import_ultralytics_conv_types()


# ----------------------------
# Attention blocks
# ----------------------------
class _ChannelGate(nn.Module):
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        hidden = max(1, ch // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.mlp(self.avg(x))
        m = self.mlp(self.max(x))
        w = self.act(a + m)
        return x * w


class _SpatialGate(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        p = (k - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=p, bias=False)
        self.act = nn.Sigmoid()

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     avg = torch.mean(x, dim=1, keepdim=True)
    #     mx, _ = torch.max(x, dim=1, keepdim=True)
    #     w = self.act(self.conv(torch.cat([avg, mx], dim=1)))
    #     return x * w
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)

        conv = getattr(self, "conv", None)
        if conv is None:
            # Extremely defensive fallback (shouldn't happen)
            return x

        y = conv(torch.cat([avg, mx], dim=1))

        # Legacy-safe: some old checkpoints may not have self.act
        act = getattr(self, "act", None)
        if act is not None:
            w = act(y)
        else:
            # Common alternatives
            sigmoid = getattr(self, "sigmoid", None)
            if sigmoid is not None:
                w = sigmoid(y)
            else:
                w = torch.sigmoid(y)

        return x * w


class CBAM(nn.Module):
    def __init__(self, ch: int, r: int = 16, k: int = 7):
        super().__init__()
        self.channel = _ChannelGate(ch, r)
        self.spatial = _SpatialGate(k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Legacy-safe: older checkpoints may not have 'spatial' or may name it differently
        channel = getattr(self, "channel", None)
        if channel is not None:
            x = channel(x)

        spatial = getattr(self, "spatial", None)

        # Common legacy alternative attribute names (try them if 'spatial' missing)
        if spatial is None:
            spatial = getattr(self, "spatial_attention", None)
        if spatial is None:
            spatial = getattr(self, "spatial_attn", None)
        if spatial is None:
            spatial = getattr(self, "sa", None)

        if spatial is not None:
            x = spatial(x)

        return x


class ECA(nn.Module):
    """Efficient Channel Attention."""
    def __init__(self, ch: int, k: int = 3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     y = self.avg(x)  # (B,C,1,1)
    #     y = y.squeeze(-1).transpose(1, 2)  # (B,1,C)
    #     y = self.conv1d(y)
    #     y = self.act(y).transpose(1, 2).unsqueeze(-1)  # (B,C,1,1)
    #     return x * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Legacy-safe: older checkpoints may not have 'avg', 'conv1d', or 'act'
        avg = getattr(self, "avg", None)
        if avg is None:
            avg = getattr(self, "avg_pool", None)  # common alt name
        if avg is None:
            avg = nn.AdaptiveAvgPool2d(1)  # fallback

        conv1d = getattr(self, "conv1d", None)
        if conv1d is None:
            conv1d = getattr(self, "conv", None)  # common alt name

        act = getattr(self, "act", None)
        if act is None:
            act = getattr(self, "sigmoid", None)

        y = avg(x)  # (B,C,1,1)

        # If conv1d missing, fall back to simple sigmoid gate (still valid + stable)
        if conv1d is not None:
            y = y.squeeze(-1).transpose(1, 2)  # (B,1,C)
            y = conv1d(y)
            if act is not None:
                y = act(y)
            else:
                y = torch.sigmoid(y)
            y = y.transpose(1, 2).unsqueeze(-1)  # (B,C,1,1)
            return x * y

        # Fallback: no conv1d available -> identity-ish gate from mean activation
        y = torch.sigmoid(y)
        return x * y




# ----------------------------
# DropBlock (simple)
# ----------------------------
class DropBlock2D(nn.Module):
    def __init__(self, drop_prob: float = 0.1, block_size: int = 7) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.block_size = int(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.drop_prob <= 0.0:
            return x
        b, c, h, w = x.shape
        bs = min(self.block_size, h, w)
        if bs <= 1:
            return x

        gamma = (
            self.drop_prob * (h * w)
            / ((bs**2) * max(1, (h - bs + 1)) * max(1, (w - bs + 1)) + 1e-8)
        )

        mask = (torch.rand(b, c, h - bs + 1, w - bs + 1, device=x.device) < gamma).float()
        mask = F.pad(mask, [bs // 2] * 4)
        mask = 1 - F.max_pool2d(mask, kernel_size=bs, stride=1, padding=bs // 2)
        keep = mask.sum().clamp(min=1.0)
        return x * mask * (mask.numel() / keep)


# ----------------------------
# Spectral mixing (SAFE: post-conv)
# ----------------------------
class SpectralMix(nn.Module):
    """A simple channel-mixing conv applied on post-conv features (C -> C)."""
    def __init__(self, channels: int):
        super().__init__()
        # 1x1 mixing keeps spatial dims, preserves C
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(x)

class ConvWithExtras(nn.Module):
    """
    Wrap an Ultralytics Conv-like block without changing conv/bn/act parameter names.
    - `conv`, `bn`, `act` are reused from the original block.
    - extras (spectral/drop/attn) are applied AFTER conv->bn->act by default.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        spectral: Optional[nn.Module] = None,
        drop: Optional[nn.Module] = None,
        attn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv = getattr(base, "conv", None)
        self.bn = getattr(base, "bn", None)
        self.act = getattr(base, "act", None)
        if self.conv is None:
            raise TypeError("ConvWithExtras expects base to have a .conv module")

        # Always define these (new models)
        self.spectral = spectral
        self.drop = drop
        self.attn = attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)

        # Legacy-safe attribute access (older pickles may not have these attrs)
        spectral = getattr(self, "spectral", None)
        if spectral is not None:
            x = spectral(x)

        drop = getattr(self, "drop", None)
        if drop is not None:
            x = drop(x)

        attn = getattr(self, "attn", None)
        if attn is not None:
            x = attn(x)

        return x




@dataclass
class PatchOptions:
    use_cbam: bool = False
    use_eca: bool = False
    use_spectral: bool = False
    use_dropblock: bool = False
    drop_prob: float = 0.1
    use_groupnorm: bool = False
    gn_groups: int = 4



# # ------------------------------------------------------------------
# # Backward-compat for old checkpoints
# # Older .pt files expect patch_backbone_with_attention.ChannelAttention
# # to exist at module scope.
# # ------------------------------------------------------------------

# try:
#     ChannelAttention  # type: ignore
# except NameError:
#     ChannelAttention = _ChannelGate  # type: ignore


# ------------------------------------------------------------------
# Backward-compat for old checkpoints
# Older .pt files expect patch_backbone_with_attention.ChannelAttention
# and patch_backbone_with_attention.SpatialAttention to exist.
# ------------------------------------------------------------------

try:
    ChannelAttention  # type: ignore
except NameError:
    ChannelAttention = _ChannelGate  # type: ignore

try:
    SpatialAttention  # type: ignore
except NameError:
    SpatialAttention = _SpatialGate  # type: ignore


def _preserve_ultralytics_graph_attrs(src: nn.Module, dst: nn.Module) -> None:
    """Copy Ultralytics graph metadata if present (prevents AttributeError on .f/.i)."""
    for attr in ("i", "f", "type", "np"):
        if hasattr(src, attr):
            try:
                setattr(dst, attr, getattr(src, attr))
            except Exception:
                pass


def _replace_bn_with_gn(module: nn.Module, groups: int) -> None:
    def _gn(ch: int) -> nn.Module:
        g = max(1, min(groups, ch))
        # make divisible if possible
        while ch % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, ch)

    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, _gn(child.num_features))
        else:
            _replace_bn_with_gn(child, groups)


def patch_backbone_with_attention(
    model_nn: nn.Module,
    use_cbam: bool = False,
    use_eca: bool = False,
    use_spectral: bool = False,
    use_dropblock: bool = False,
    drop_prob: float = 0.1,
    use_groupnorm: bool = False,
    gn_groups: int = 4,
    *,
    skip_head: bool = True,
    skip_head_from_model_index: Optional[int] = None,
) -> None:
    """
    Patch Conv blocks in the YOLO backbone.

    Parameters
    ----------
    model_nn:
        The Ultralytics task model (e.g., yolo.model from a YOLO instance).
    skip_head:
        If True, skip patching the last top-level module in `model_nn.model` (recommended for seg).
    skip_head_from_model_index:
        Optional explicit boundary index (patch indices < this). If None, derived from `skip_head`.
    """
    opts = PatchOptions(
        use_cbam=bool(use_cbam),
        use_eca=bool(use_eca),
        use_spectral=bool(use_spectral),
        use_dropblock=bool(use_dropblock),
        drop_prob=float(drop_prob),
        use_groupnorm=bool(use_groupnorm),
        gn_groups=int(gn_groups),
    )

    if opts.use_groupnorm:
        _replace_bn_with_gn(model_nn, opts.gn_groups)

    if not _CONV_TYPES:
        raise ImportError(
            "Could not import Ultralytics Conv class. "
            "Update _try_import_ultralytics_conv_types() paths for your fork."
        )

    def _make_attn(c_out: int) -> Optional[nn.Module]:
        blocks = []
        if opts.use_dropblock and opts.drop_prob > 0:
            blocks.append(DropBlock2D(drop_prob=opts.drop_prob))
        if opts.use_cbam:
            blocks.append(CBAM(c_out))
        if opts.use_eca:
            blocks.append(ECA(c_out))
        if not blocks:
            return None
        return nn.Sequential(*blocks) if len(blocks) > 1 else blocks[0]

    def _make_spectral(c_out: int) -> Optional[nn.Module]:
        return SpectralMix(c_out) if opts.use_spectral else None

    def _patch_module(parent: nn.Module, name: str, child: nn.Module) -> None:
        # Identify Ultralytics Conv-like modules
        if isinstance(child, _CONV_TYPES):
            conv = getattr(child, "conv", None)
            if isinstance(conv, nn.Conv2d):
                c_out = conv.out_channels
                spectral = _make_spectral(c_out)
                attn = _make_attn(c_out)

                if spectral is None and attn is None:
                    return  # nothing to do

                wrapped = ConvWithExtras(child, spectral=spectral, drop=None, attn=attn)
                _preserve_ultralytics_graph_attrs(child, wrapped)
                setattr(parent, name, wrapped)
        else:
            # recurse
            for n2, c2 in list(child.named_children()):
                _patch_module(child, n2, c2)

    # Top-level patching: preserve graph attributes especially here
    if hasattr(model_nn, "model") and isinstance(getattr(model_nn, "model"), nn.Sequential):
        seq: nn.Sequential = getattr(model_nn, "model")
        n = len(seq)
        if skip_head_from_model_index is None:
            skip_head_from_model_index = (n - 1) if skip_head else n

        for i in range(int(skip_head_from_model_index)):
            m = seq[i]
            # patch direct child if it's a Conv block
            if isinstance(m, _CONV_TYPES):
                # use a temporary parent to reuse same logic
                tmp = nn.Module()
                setattr(tmp, "x", m)
                _patch_module(tmp, "x", m)
                seq[i] = getattr(tmp, "x")
            else:
                for n2, c2 in list(m.named_children()):
                    _patch_module(m, n2, c2)
    else:
        for n1, c1 in list(model_nn.named_children()):
            _patch_module(model_nn, n1, c1)


__all__ = [
    "CBAM",
    "ECA",
    "DropBlock2D",
    "SpectralMix",
    "ConvWithExtras",
    "patch_backbone_with_attention",
]
