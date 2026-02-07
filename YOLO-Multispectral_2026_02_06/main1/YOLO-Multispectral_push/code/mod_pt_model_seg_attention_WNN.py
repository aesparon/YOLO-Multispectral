# -*- coding: utf-8 -*-
"""
mod_pt_model_seg_attention_WNN.py

Extended patcher for YOLOv11-seg .pt checkpoints:
- Applies optional attention modules / DropBlock / SpectralConv / GroupNorm
  using patch_backbone_with_attention.py
- Applies optional Wavelet residual wrapper (ResidualWaveletFromConv) at stage-1 only
  using wavelet_yolo_backbone_patch.py
- Saves a new patched .pt checkpoint + sidecar metadata JSON

Design goals:
- DO NOT modify RSL baseline patcher (mod_pt_model_seg.py)
- Apply-and-save for reproducibility and sharing
- .pt-only (preserves transfer learning workflows)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn

def patch_yolo_seg_ckpt_extended(
    input_pt: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    # Attention / regularisation toggles:
    use_cbam: bool = False,
    use_eca: bool = False,
    use_spectral: bool = False,
    use_dropblock: bool = False,
    drop_prob: float = 0.1,
    use_groupnorm: bool = False,
    gn_groups: int = 4,
    # Wavelet toggles (default off):
    use_wavelet_residual: bool = False,
    wavelet_inject_stage: int = 1,     # stage-1 only by default
    wavelet_alpha_init: float = 0.0,   # identity-safe
    # Provenance:
    save_tag: Optional[str] = None,
    yolo_source_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Load a YOLO .pt checkpoint, apply optional backbone enhancements + wavelet residual wrapper,
    and save a new .pt checkpoint.

    Returns: path to saved patched checkpoint.
    """
    input_pt = str(input_pt)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allow optional sys.path injection by caller before importing ultralytics
    if yolo_source_path is not None:
        # Optional: if you have the same tasks.py guard patcher you use in mod_pt_model_seg.py,
        # you can import and call it here as well. But for .pt-only use it’s typically not needed.
        pass

    from ultralytics import YOLO  # type: ignore

    print(f"📦 Loading YOLO checkpoint (.pt): {input_pt}")
    yolo = YOLO(input_pt)

    # ----------------------------
    # A) Attention / DropBlock / GN / SpectralConv patches
    # ----------------------------
    if any([use_cbam, use_eca, use_spectral, use_dropblock, use_groupnorm]):
        from patch_backbone_with_attention import patch_backbone_with_attention  # type: ignore

        print("🧠 Applying backbone attention/regularisation patches...")
        # patch_backbone_with_attention(
        #     model_nn=yolo.model,
        #     use_cbam=use_cbam,
        #     use_eca=use_eca,
        #     use_spectral=use_spectral,
        #     use_dropblock=use_dropblock,
        #     drop_prob=float(drop_prob),
        #     use_groupnorm=use_groupnorm,
        #     gn_groups=int(gn_groups),
        # )
        patch_backbone_with_attention(
            model_nn=yolo.model,
            use_cbam=use_cbam,
            use_eca=use_eca,
            use_spectral=use_spectral,
            use_dropblock=use_dropblock,
            drop_prob=float(drop_prob),
            use_groupnorm=use_groupnorm,
            gn_groups=int(gn_groups),
            skip_head=True,   # ✅ critical
        )

        # ✅ verify injection occurred (debug + provenance)
        cbam_n = sum(1 for m in yolo.model.modules() if m.__class__.__name__ == "CBAM")
        eca_n = sum(1 for m in yolo.model.modules() if m.__class__.__name__ == "ECA")
        db_n = sum(1 for m in yolo.model.modules() if m.__class__.__name__ == "DropBlock2D")
        #gn_n = sum(1 for m in yolo.model.modules() if isinstance(m, torch.nn.GroupNorm))
        gn_n = sum(1 for m in yolo.model.modules() if isinstance(m, nn.GroupNorm))
        print(f"[PATCH_VERIFY] CBAM={cbam_n} ECA={eca_n} DropBlock={db_n} GroupNorm={gn_n}")

        # Fail-fast: if user requested a patch but nothing was injected
        if use_cbam and cbam_n == 0:
            raise RuntimeError("[PATCH_VERIFY] use_cbam=True but CBAM count is 0 (patch did not apply).")
        if use_eca and eca_n == 0:
            raise RuntimeError("[PATCH_VERIFY] use_eca=True but ECA count is 0 (patch did not apply).")
        if use_dropblock and db_n == 0:
            raise RuntimeError("[PATCH_VERIFY] use_dropblock=True but DropBlock2D count is 0 (patch did not apply).")
        if use_groupnorm and gn_n == 0:
            raise RuntimeError("[PATCH_VERIFY] use_groupnorm=True but no GroupNorm modules found (patch did not apply).")


    # ----------------------------
    # B) Wavelet residual patch (ResidualWaveletFromConv), stage-1 only by default
    # ----------------------------
    if use_wavelet_residual:
        from wavelet_yolo_backbone_patch import patch_input_with_residual_wavelet  # type: ignore

        print("🌊 Applying ResidualWaveletFromConv patch...")
        patch_input_with_residual_wavelet(
            model=yolo.model,
            alpha_init=float(wavelet_alpha_init),
            inject_stage=int(wavelet_inject_stage),
        )

    # ----------------------------
    # Save patched model
    # ----------------------------
    stem = Path(input_pt).stem
    tag = save_tag or "extended"
    out_path = out_dir / f"{stem}_{tag}.pt"

    # print(f"💾 Saving patched model -> {out_path}")
    # yolo.save(str(out_path))

    print(f"💾 Saving patched model -> {out_path}")

    # ✅ Save as a proper Ultralytics training checkpoint dict so architecture is preserved
    ckpt = {
        "model": yolo.model,  # <-- patched nn.Module graph (contains ConvWithExtras/CBAM/etc.)
        "ema": None,
        "optimizer": None,
        "train_args": {
            "task": "segment",
        },
        "epoch": 0,
        "best_fitness": 0.0,
        "updates": 0,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "ultralytics-ms",
    }

    torch.save(ckpt, str(out_path))


    # Sidecar metadata
    # meta: Dict[str, Any] = {
    #     "input_pt": input_pt,
    #     "output_pt": str(out_path),
    #     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    #     "patches": {
    #         "use_cbam": bool(use_cbam),
    #         "use_eca": bool(use_eca),
    #         "use_spectral": bool(use_spectral),
    #         "use_dropblock": bool(use_dropblock),
    #         "drop_prob": float(drop_prob),
    #         "use_groupnorm": bool(use_groupnorm),
    #         "gn_groups": int(gn_groups),
    #         "use_wavelet_residual": bool(use_wavelet_residual),
    #         "wavelet_inject_stage": int(wavelet_inject_stage),
    #         "wavelet_alpha_init": float(wavelet_alpha_init),
    #     },
    # }
    # (out_dir / f"{out_path.stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    meta: Dict[str, Any] = {
        "input_pt": input_pt,
        "output_pt": str(out_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patches": {
            "use_cbam": bool(use_cbam),
            "use_eca": bool(use_eca),
            "use_spectral": bool(use_spectral),
            "use_dropblock": bool(use_dropblock),
            "drop_prob": float(drop_prob),
            "use_groupnorm": bool(use_groupnorm),
            "gn_groups": int(gn_groups),
            "use_wavelet_residual": bool(use_wavelet_residual),
            "wavelet_inject_stage": int(wavelet_inject_stage),
            "wavelet_alpha_init": float(wavelet_alpha_init),
        },
    }

    # ✅ add this block HERE
    meta["verify"] = {
        "cbam_modules": int(cbam_n) if "cbam_n" in locals() else 0,
        "eca_modules": int(eca_n) if "eca_n" in locals() else 0,
        "dropblock_modules": int(db_n) if "db_n" in locals() else 0,
        "groupnorm_modules": int(gn_n) if "gn_n" in locals() else 0,
    }

    (out_dir / f"{out_path.stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"🧾 Wrote metadata -> {out_dir / f'{out_path.stem}.meta.json'}")

    return str(out_path)


def _build_save_tag(args: argparse.Namespace) -> str:
    parts = []
    parts.append("cbam1" if args.use_cbam else "cbam0")
    parts.append("eca1" if args.use_eca else "eca0")
    parts.append("spec1" if args.use_spectral else "spec0")
    parts.append("db1" if args.use_dropblock else "db0")
    if args.use_dropblock:
        parts.append(f"dbp{str(args.drop_prob).replace('.', '_')}")
    parts.append("gn1" if args.use_groupnorm else "gn0")
    if args.use_groupnorm:
        parts.append(f"gng{args.gn_groups}")
    parts.append("wnnS1" if args.use_wavelet_residual else "wnn0")
    if args.use_wavelet_residual:
        parts.append(f"a{str(args.wavelet_alpha_init).replace('.', '_')}")
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_pt", type=str, required=True, help="Path to input .pt checkpoint")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for patched .pt")

    ap.add_argument("--use_cbam", action="store_true")
    ap.add_argument("--use_eca", action="store_true")
    ap.add_argument("--use_spectral", action="store_true")
    ap.add_argument("--use_dropblock", action="store_true")
    ap.add_argument("--drop_prob", type=float, default=0.1)
    ap.add_argument("--use_groupnorm", action="store_true")
    ap.add_argument("--gn_groups", type=int, default=4)

    ap.add_argument("--use_wavelet_residual", action="store_true")
    ap.add_argument("--wavelet_inject_stage", type=int, default=1)
    ap.add_argument("--wavelet_alpha_init", type=float, default=0.0)

    ap.add_argument("--save_tag", type=str, default=None)
    args = ap.parse_args()

    tag = args.save_tag or _build_save_tag(args)

    patch_yolo_seg_ckpt_extended(
        input_pt=args.input_pt,
        out_dir=args.out_dir,
        use_cbam=args.use_cbam,
        use_eca=args.use_eca,
        use_spectral=args.use_spectral,
        use_dropblock=args.use_dropblock,
        drop_prob=args.drop_prob,
        use_groupnorm=args.use_groupnorm,
        gn_groups=args.gn_groups,
        use_wavelet_residual=args.use_wavelet_residual,
        wavelet_inject_stage=args.wavelet_inject_stage,
        wavelet_alpha_init=args.wavelet_alpha_init,
        save_tag=tag,
    )


if __name__ == "__main__":
    main()
