# -*- coding: utf-8 -*-
"""
run_transfer_vs_scratch_extended.py

Extended runner:
- Uses mod_pt_model_seg.patch_yolo_seg_ckpt (baseline) to create multispectral TL or scratch .pt
- Then uses mod_pt_model_seg_attention_WNN.patch_yolo_seg_ckpt_extended to add attention/DropBlock/GN/SpectralConv + wavelet residual
- Trains YOLO on the resulting extended checkpoint

Does NOT touch the RSL baseline runner.
"""

import os
import sys
import time
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Paths
CODE_PATH = Path(__file__).resolve().parent
CODE_PARENT = CODE_PATH.parent

YOLO_SOURCE_PATH = CODE_PATH / "ultralytics_MS"
if str(YOLO_SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(YOLO_SOURCE_PATH))

from ultralytics import YOLO  # type: ignore

from mod_pt_model_seg import patch_yolo_seg_ckpt  # baseline patcher
from mod_pt_model_seg_attention_WNN import patch_yolo_seg_ckpt_extended  # extended patcher

# -----------------------------
# User settings
# -----------------------------
EPOCHS = 600
IMGSZ = 600  # Weeds_galore 600 by 600
BATCH = 8
WORKERS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 100
NC = 5

MODEL_PT = CODE_PATH / "yolo11x-seg.pt"
MODEL_YAML = CODE_PATH / "yolo11x-seg.yaml"

RUNS_ROOT = CODE_PATH / "runs_transfer_vs_scratch_EXTENDED"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

# Defaults (used only if you call run_one without variant_tag/toggles)
USE_CBAM = False
USE_ECA = False
USE_SPECTRAL = False
USE_DROPBLOCK = False
DROP_PROB = 0.10
USE_GROUPNORM = False
GN_GROUPS = 4

USE_WAVELET_RESIDUAL = False
WAVELET_INJECT_STAGE = 1
WAVELET_ALPHA_INIT = 0.0  # identity-safe

OPTIMIZER = "AdamW"
LR0 = 0.001111
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# -----------------------------
# NEW: Variant loop (minimal reinstatement)
# -----------------------------
ATTENTION_MODS = [
    "BASE",          # no architectural extensions
    "CBAM",          # attention
    "ECA",           # lightweight attention
    "WAVELET_S1",    # Residual wavelet, stage-1 only
    "CBAM_WAVELET",  # combination

    "SPECTRAL",      # spectral mixing module
    "DROPBLOCK",     # regularisation
    "GROUPNORM",     # normalisation
]


# -----------------------------
@dataclass
class Cfg:
    exp_id: str
    dataset_id: str
    dataset_yaml: str
    in_channels: int


@dataclass
class ToggleConfig:
    use_cbam: bool = False
    use_eca: bool = False
    use_spectral: bool = False
    use_dropblock: bool = False
    drop_prob: float = 0.10
    use_groupnorm: bool = False
    gn_groups: int = 4

    use_wavelet_residual: bool = False
    wavelet_inject_stage: int = 1
    wavelet_alpha_init: float = 0.0  # identity-safe


@dataclass
class TrainConfig:
    epochs: int = 600
    imgsz: int = 600
    batch: int = 8
    workers: int = 0
    patience: int = 100
    device: str = "cuda"
    deterministic: bool = True

    optimizer: Optional[str] = None
    lr0: Optional[float] = None
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None


EXPS: List[Cfg] = [
    Cfg(
        exp_id="rgb3_extended",
        dataset_id="rgb3",
        dataset_yaml=str(CODE_PARENT / "datasets/weeds_galore_processed/RGB/outputs/data_RGB.yaml"),
        in_channels=3,
    ),
    Cfg(
        exp_id="rgb5_extended",
        dataset_id="rgb5",
        dataset_yaml=str(CODE_PARENT / "datasets/weeds_galore_processed/RGBRN/outputs/data_RGBRN.yaml"),
        in_channels=5,
    ),
]

INIT_TYPES = ["pt", "yaml"]  # "pt" = TL, "yaml" = scratch rebuilt -> pt then trained
CHANNEL_INIT_MODES = ["avg"]
SEEDS = [0]


def write_json(path: Path, obj: Dict[str, Any]):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def toggles_from_variant(variant_tag: str) -> ToggleConfig:
    """Minimal mapping from ATTENTION_MODS -> ToggleConfig (no re-architecture)."""
    v = variant_tag.upper()

    # start from all-off defaults
    t = ToggleConfig(
        use_cbam=False,
        use_eca=False,
        use_spectral=False,
        use_dropblock=False,
        drop_prob=float(DROP_PROB),
        use_groupnorm=False,
        gn_groups=int(GN_GROUPS),
        use_wavelet_residual=False,
        wavelet_inject_stage=1,
        wavelet_alpha_init=float(WAVELET_ALPHA_INIT),
    )

    if v == "BASE":
        return t
    if v == "CBAM":
        t.use_cbam = True
        return t
    if v == "ECA":
        t.use_eca = True
        return t
    if v == "SPECTRAL":
        t.use_spectral = True
        return t
    if v == "DROPBLOCK":
        t.use_dropblock = True
        t.drop_prob = float(DROP_PROB)
        return t
    if v == "GROUPNORM":
        t.use_groupnorm = True
        t.gn_groups = int(GN_GROUPS)
        return t
    if v == "WAVELET_S1":
        t.use_wavelet_residual = True
        t.wavelet_inject_stage = 1
        t.wavelet_alpha_init = float(WAVELET_ALPHA_INIT)
        return t
    if v == "CBAM_WAVELET":
        t.use_cbam = True
        t.use_wavelet_residual = True
        t.wavelet_inject_stage = 1
        t.wavelet_alpha_init = float(WAVELET_ALPHA_INIT)
        return t

    raise ValueError(f"Unknown variant_tag='{variant_tag}'. Must be one of: {ATTENTION_MODS}")


def run_one(
    cfg: Cfg,
    init_type: str,
    channel_init_mode: str,
    seed: int,
    variant_tag: str = "BASE",
    *,
    toggles: Optional[ToggleConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    skip_if_done: bool = True,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment with explicit toggles + training parameters."""
    if toggles is None:
        # Backwards-compatible default behavior
        toggles = ToggleConfig(
            use_cbam=USE_CBAM,
            use_eca=USE_ECA,
            use_spectral=USE_SPECTRAL,
            use_dropblock=USE_DROPBLOCK,
            drop_prob=float(DROP_PROB),
            use_groupnorm=USE_GROUPNORM,
            gn_groups=int(GN_GROUPS),
            use_wavelet_residual=USE_WAVELET_RESIDUAL,
            wavelet_inject_stage=int(WAVELET_INJECT_STAGE),
            wavelet_alpha_init=float(WAVELET_ALPHA_INIT),
        )

    if train_cfg is None:
        train_cfg = TrainConfig(
            epochs=int(EPOCHS),
            imgsz=int(IMGSZ),
            batch=int(BATCH),
            workers=int(WORKERS),
            patience=int(PATIENCE),
            device=str(DEVICE),
            deterministic=True,
            optimizer=OPTIMIZER,
            lr0=LR0,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )

    # ✅ include variant_tag in run name (so BASE/CBAM/etc don't collide)
    run_name = f"{cfg.exp_id}__{init_type}__{channel_init_mode}__{variant_tag}__seed{seed}"
    run_dir = RUNS_ROOT / run_name

    results_csv = run_dir / "train" / "results.csv"
    if results_csv.exists() and skip_if_done and not force_rerun:
        print(f"⏭️  Skipping completed run (results.csv exists): {run_name}")
        return {
            "status": "skipped",
            "run_dir": str(run_dir),
            "reason": "results.csv exists",
            "exp_id": cfg.exp_id,
            "dataset_id": cfg.dataset_id,
            "dataset_yaml": cfg.dataset_yaml,
            "in_channels": cfg.in_channels,
            "seed": seed,
            "init_type": init_type,
            "channel_init_mode": channel_init_mode,
            "variant_tag": variant_tag,
            "toggles": asdict(toggles),
            "train_cfg": asdict(train_cfg),
        }

    if run_dir.exists() and force_rerun:
        print(f"♻️  Re-running (forced): {run_name}")
        shutil.rmtree(run_dir, ignore_errors=True)

    run_dir.mkdir(parents=True, exist_ok=True)

    base = str(MODEL_PT if init_type == "pt" else MODEL_YAML)

    # ---- A) Baseline patch: build the multispectral / scratch pt ----
    model_for_train = base
    effective_init_mode = "na"
    if (init_type == "yaml") or (cfg.in_channels != 3):
        patched_path, effective_init_mode = patch_yolo_seg_ckpt(
            input_ckpt=base,
            out_dir=str(run_dir / "mod_model_baseline"),
            in_channels=int(cfg.in_channels),
            channel_init_mode=str(channel_init_mode),
            nc=int(NC),
            yolo_source_path=str(YOLO_SOURCE_PATH / "ultralytics"),
            save_tag=f"{init_type}_{channel_init_mode}_seed{seed}",
        )
        model_for_train = patched_path

    # ---- B) Extended patch: attention + wavelet -> save another pt ----
    ext_tag = (
        f"EXT_cbam{int(toggles.use_cbam)}_eca{int(toggles.use_eca)}_spec{int(toggles.use_spectral)}_"
        f"db{int(toggles.use_dropblock)}_gn{int(toggles.use_groupnorm)}_"
        f"wnnS{int(toggles.wavelet_inject_stage) if toggles.use_wavelet_residual else 0}"
    )

    extended_pt = patch_yolo_seg_ckpt_extended(
        input_pt=model_for_train,
        out_dir=str(run_dir / "mod_model_extended"),
        use_cbam=bool(toggles.use_cbam),
        use_eca=bool(toggles.use_eca),
        use_spectral=bool(toggles.use_spectral),
        use_dropblock=bool(toggles.use_dropblock),
        drop_prob=float(toggles.drop_prob),
        use_groupnorm=bool(toggles.use_groupnorm),
        gn_groups=int(toggles.gn_groups),
        use_wavelet_residual=bool(toggles.use_wavelet_residual),
        wavelet_inject_stage=int(toggles.wavelet_inject_stage),
        wavelet_alpha_init=float(toggles.wavelet_alpha_init),
        save_tag=ext_tag,
    )

    meta = {
        "exp_id": cfg.exp_id,
        "dataset_id": cfg.dataset_id,
        "dataset_yaml": cfg.dataset_yaml,
        "in_channels": cfg.in_channels,
        "seed": seed,
        "init_type": init_type,
        "requested_channel_init_mode": channel_init_mode,
        "effective_init_mode": effective_init_mode,
        "variant_tag": variant_tag,
        "base_model": base,
        "baseline_pt": model_for_train,
        "extended_pt": extended_pt,
        "train_cfg": asdict(train_cfg),
        "toggles": asdict(toggles),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patches": {
            "use_cbam": bool(toggles.use_cbam),
            "use_eca": bool(toggles.use_eca),
            "use_spectral": bool(toggles.use_spectral),
            "use_dropblock": bool(toggles.use_dropblock),
            "use_groupnorm": bool(toggles.use_groupnorm),
            "use_wavelet_residual": bool(toggles.use_wavelet_residual),
            "wavelet_inject_stage": int(toggles.wavelet_inject_stage),
        },
    }
    write_json(run_dir / "run_meta.json", meta)

    print(f"\n=== TRAIN {run_dir.name} ===")
    print(meta)

    model = YOLO(extended_pt)

    cbam_count = sum(1 for m in model.model.modules() if m.__class__.__name__ == "CBAM")
    print(f"[TRAIN_LOAD_VERIFY] cbam_count={cbam_count} | pt={extended_pt}")


    def _count(model, name):
        return sum(1 for m in model.model.modules() if m.__class__.__name__ == name)

    print(
        f"[TRAIN_LOAD_VERIFY] pt={extended_pt} | "
        f"CBAM={_count(model, 'CBAM')} "
        f"ECA={_count(model, 'ECA')} "
        f"ConvWithExtras={_count(model, 'ConvWithExtras')}"
    )


    # Force Trainer to use the already-loaded model object, not rebuild from path
    model.overrides["model"] = None
    model.overrides["pretrained"] = False

    def _count_cls(mm, name):
        return sum(1 for m in mm.modules() if m.__class__.__name__ == name)

    def on_train_start(trainer):
        m = trainer.model
        print(
            f"[TRAINER_MODEL_VERIFY] "
            f"CBAM={_count_cls(m, 'CBAM')} "
            f"ECA={_count_cls(m, 'ECA')} "
            f"ConvWithExtras={_count_cls(m, 'ConvWithExtras')}"
        )

    model.add_callback("on_train_start", on_train_start)


    train_kwargs = dict(
        task="segment",
        data=cfg.dataset_yaml,
        epochs=int(train_cfg.epochs),
        imgsz=int(train_cfg.imgsz),
        batch=int(train_cfg.batch),
        device=str(train_cfg.device),
        workers=int(train_cfg.workers),
        patience=int(train_cfg.patience),
        seed=int(seed),
        deterministic=bool(train_cfg.deterministic),
        project=str(run_dir),
        name="train",
        exist_ok=True,
    )

    if train_cfg.optimizer is not None:
        train_kwargs["optimizer"] = str(train_cfg.optimizer)
    if train_cfg.lr0 is not None:
        train_kwargs["lr0"] = float(train_cfg.lr0)
    if train_cfg.momentum is not None:
        train_kwargs["momentum"] = float(train_cfg.momentum)
    if train_cfg.weight_decay is not None:
        train_kwargs["weight_decay"] = float(train_cfg.weight_decay)

    model.train(**train_kwargs)

    return {"status": "done", "run_dir": str(run_dir), **meta}


def main():
    train_cfg = TrainConfig(
        epochs=int(EPOCHS),
        imgsz=int(IMGSZ),
        batch=int(BATCH),
        workers=int(WORKERS),
        patience=int(PATIENCE),
        device=str(DEVICE),
        deterministic=True,
        optimizer=OPTIMIZER,
        lr0=LR0,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    results = []
    for cfg in EXPS:
        for init_type in INIT_TYPES:
            for channel_init_mode in CHANNEL_INIT_MODES:
                for variant_tag in ATTENTION_MODS:
                    # ✅ per-variant toggles
                    toggles = toggles_from_variant(variant_tag)

                    for seed in SEEDS:
                        results.append(
                            run_one(
                                cfg,
                                init_type,
                                channel_init_mode,
                                seed,
                                variant_tag=variant_tag,
                                toggles=toggles,
                                train_cfg=train_cfg,
                                skip_if_done=True,
                                force_rerun=False,
                            )
                        )

    out = RUNS_ROOT / "summary_extended.json"
    write_json(out, {"runs": results})
    print(f"\n✅ Summary -> {out}")


if __name__ == "__main__":
    main()
