"""
yolo11_seg_transformer_patch.py

Minimal patch to attach TransformerBlocks to selected C2f layers
in a YOLO11-seg model (e.g. yolo11x-seg.pt) with minimal disruption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.transformer import TransformerBlock


# -------------------------
# 1) Wrapper module
# -------------------------

class C2fWithTransformer(nn.Module):
    """
    Thin wrapper: C2f -> TransformerBlock.
    Keeps input/output shape identical to original C2f.

    x: [B, Cin, H, W]  ->  C2f ->  [B, C2, H, W]
                         -> TransformerBlock(C2->C2) -> [B, C2, H, W]
    """

    def __init__(self, c2f: C2f, num_heads: int = 4, num_layers: int = 1):
        super().__init__()
        self.c2f = c2f
        # Infer output channels of C2f from its last conv
        c2 = c2f.cv2.conv.out_channels
        self.tr = TransformerBlock(c1=c2, c2=c2, num_heads=num_heads, num_layers=num_layers)

    def forward(self, x):
        x = self.c2f(x)
        x = self.tr(x)
        return x

    @classmethod
    def from_existing(cls, c2f: C2f, num_heads: int = 4, num_layers: int = 1) -> "C2fWithTransformer":
        """Convenience constructor that clones an existing C2f."""
        return cls(c2f, num_heads=num_heads, num_layers=num_layers)


# -------------------------
# 2) Patch function
# -------------------------

@dataclass
class TransformerPatchSpec:
    """Which C2f indices to patch, and transformer hyper-params."""
    indices: List[int]
    num_heads: int = 4
    num_layers: int = 1
    label: str = "tr"


def patch_yolo11_seg_with_transformers(
    model: YOLO,
    patch_specs: Iterable[TransformerPatchSpec],
    verbose: bool = True,
) -> None:
    """
    In-place patch: wraps selected C2f layers in C2fWithTransformer.

    Args:
        model: ultralytics.YOLO object loaded with a *-seg.pt model.
        patch_specs: iterable of TransformerPatchSpec.
        verbose: print what got patched.
    """
    core = model.model          # SegmentationModel (DetectionModel subclass) :contentReference[oaicite:1]{index=1}
    layers = core.model         # nn.ModuleList of backbone+neck+head layers

    for spec in patch_specs:
        for idx in spec.indices:
            m = layers[idx]
            if not isinstance(m, C2f):
                raise TypeError(
                    f"Layer at index {idx} is {type(m).__name__}, not C2f. "
                    f"Re-check indices from model.summary()."
                )
            wrapped = C2fWithTransformer.from_existing(
                m,
                num_heads=spec.num_heads,
                num_layers=spec.num_layers,
            )
            layers[idx] = wrapped
            if verbose:
                print(
                    f"[TransformerPatch] Wrapped C2f at idx={idx} "
                    f"with heads={spec.num_heads} layers={spec.num_layers} "
                    f"({spec.label})"
                )


# -------------------------
# 3) Helper to inspect indices
# -------------------------

def print_yolo11_layer_summary(model: YOLO, max_rows: int = 120):
    """
    Quick text summary to help choose C2f indices to patch.

    Usage:
        y = YOLO("yolo11x-seg.pt")
        print_yolo11_layer_summary(y)
    """
    core = model.model
    print(core)  # full structure
    print("=" * 80)
    print("Indexed layer list (first N rows):")
    for m in list(core.model)[:max_rows]:
        print(f"i={m.i:3d}  f={m.f!s:>4}  type={m.type:25}  n={getattr(m, 'n', '')}")
