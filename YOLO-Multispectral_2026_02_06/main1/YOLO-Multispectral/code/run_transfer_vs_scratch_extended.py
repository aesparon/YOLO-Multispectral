# -*- coding: utf-8 -*-
"""
run_transfer_vs_scratch_extended.py

Extended runner:
- Uses mod_pt_model_seg.patch_yolo_seg_ckpt (baseline) to create multispectral TL or scratch .pt
- Then uses mod_pt_model_seg_attention_WNN.patch_yolo_seg_ckpt_extended to add attention/DropBlock/GN/SpectralConv + wavelet residual
- Trains YOLO on the resulting extended checkpoint

NEW (Summary Export):
- After all runs, scans RUNS_ROOT/**/train/results.csv
- Extracts best mask mAP50 from metrics/mAP50(M) and the epoch it occurred
- Writes:
    1) summary_extended_runs_mask.csv   (one row per run)
    2) summary_table_s2_mask.csv       (Table S2-style wide format: scratch vs TL side-by-side)
"""

import os
import sys
import time
import json
import shutil
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import csv

import torch
import pandas as pd

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
EPOCHS = 3000   #  600
IMGSZ = 600  # Weeds_galore 600 by 600  ############################CHANGE###############################
BATCH = 8
WORKERS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 500  # 100                  ############################CHANGE###############################  
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
# Variant loop (minimal reinstatement)
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
# METRIC COLUMN (LOCKED)
# -----------------------------
MASK_MAP50_COL = "metrics/mAP50(M)"   # instance segmentation mAP50 (mask)
EPOCH_COL = "epoch"

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
SEEDS = [0,1,2]


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


# -----------------------------
# Summary helpers (NEW)
# -----------------------------
def _bands_label(dataset_id: str) -> str:
    d = (dataset_id or "").lower()
    if d == "rgb3":
        return "RGB"
    if d == "rgb5":
        return "RGB+NIR+RE"
    return dataset_id


def _backbone_label(variant_tag: str) -> str:
    """Map internal variant names to Table S2-friendly backbone modification labels."""
    v = (variant_tag or "").upper()
    if v == "BASE":
        return "None"
    if v == "CBAM":
        return "CBAM"
    if v == "ECA":
        return "ECA"
    if v == "WAVELET_S1":
        return "Stage-1 WNN"
    if v == "CBAM_WAVELET":
        return "CBAM + Stage-1 WNN"
    if v == "SPECTRAL":
        return "Spectral"
    if v == "DROPBLOCK":
        return "DropBlock"
    if v == "GROUPNORM":
        return "GroupNorm"
    return variant_tag


def _init_label(init_type: str) -> str:
    # Locked interpretation
    return "Transfer learning" if init_type == "pt" else "Training from scratch"


def _best_metric_and_epoch(results_csv: Path) -> Optional[Tuple[float, int]]:
    """
    Reads results.csv and returns (best_mask_map50, epoch_at_best).
    Uses locked mask metric column: metrics/mAP50(M)
    """
    try:
        df = pd.read_csv(results_csv)
    except Exception as e:
        print(f"[WARN] Could not read {results_csv}: {e}")
        return None

    if df.empty:
        return None

    if EPOCH_COL not in df.columns or MASK_MAP50_COL not in df.columns:
        print(f"[WARN] Missing required columns in {results_csv}: "
              f"need '{EPOCH_COL}' and '{MASK_MAP50_COL}'")
        return None

    # Ensure numeric
    df[EPOCH_COL] = pd.to_numeric(df[EPOCH_COL], errors="coerce")
    df[MASK_MAP50_COL] = pd.to_numeric(df[MASK_MAP50_COL], errors="coerce")

    # Drop NaNs in metric
    dd = df.dropna(subset=[MASK_MAP50_COL])
    if dd.empty:
        return None

    idx = dd[MASK_MAP50_COL].idxmax()
    best_val = float(dd.loc[idx, MASK_MAP50_COL])
    best_epoch = dd.loc[idx, EPOCH_COL]
    if pd.isna(best_epoch):
        best_epoch = int(idx)
    else:
        best_epoch = int(best_epoch)

    return best_val, best_epoch


def evaluate_on_test_and_save_csv(run_dir: Path, data_yaml: str) -> Optional[Path]:
    """
    Evaluate best trained weights on the TEST split and write key metrics to results_test.csv.

    Assumes Ultralytics writes best weights to:
      {run_dir}/train/weights/best.pt
    """
    best_pt = run_dir / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[WARN] No best.pt found for test eval: {best_pt}")
        return None

    print(f"\n=== TEST EVAL {run_dir.name} ===")
    model_test = YOLO(str(best_pt))

    # Evaluate on TEST split (requires 'test:' in YAML)
    r = model_test.val(
        data=str(data_yaml),
        split="test",
        task="segment",
        project=str(run_dir),
        name="test",
        exist_ok=True,
        verbose=False,
    )

    # Try to extract common Ultralytics metrics robustly
    # r.box and r.seg are typically present for segmentation tasks
    def _get(attr, default=None):
        try:
            return getattr(r, attr)
        except Exception:
            return default

    box = _get("box", None)
    seg = _get("seg", None)

    row = {
        "run_name": run_dir.name,
        "best_pt": str(best_pt),
        "data_yaml": str(data_yaml),
        "split": "test",
    }

    # Box metrics
    if box is not None:
        row.update({
            "precision(B)": getattr(box, "mp", None),   # mean precision
            "recall(B)": getattr(box, "mr", None),      # mean recall
            "mAP50(B)": getattr(box, "map50", None),
            "mAP50-95(B)": getattr(box, "map", None),
        })

    # Mask/seg metrics
    if seg is not None:
        row.update({
            "precision(M)": getattr(seg, "mp", None),
            "recall(M)": getattr(seg, "mr", None),
            "mAP50(M)": getattr(seg, "map50", None),
            "mAP50-95(M)": getattr(seg, "map", None),
        })

    out_csv = run_dir / "results_test.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    print(f"✅ Wrote test metrics -> {out_csv}")
    return out_csv



def export_summaries(runs_root: Path) -> None:
    """
    Writes:
      1) summary_extended_runs_mask.csv
      2) summary_table_s2_mask.csv (wide: scratch vs TL columns)
    """
    rows: List[Dict[str, Any]] = []

    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        results_csv = run_dir / "train" / "results.csv"
        meta_path = run_dir / "run_meta.json"
        if not results_csv.exists() or not meta_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

        best = _best_metric_and_epoch(results_csv)
        if best is None:
            continue
        best_map50, best_epoch = best

        dataset_id = str(meta.get("dataset_id", ""))
        variant_tag = str(meta.get("variant_tag", ""))
        init_type = str(meta.get("init_type", ""))
        seed = meta.get("seed", None)

        rows.append({
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "bands": _bands_label(dataset_id),
            "dataset_id": dataset_id,
            "backbone_mod": _backbone_label(variant_tag),
            "variant_tag": variant_tag,
            "initialisation": _init_label(init_type),
            "init_type": init_type,
            "seed": seed,
            "mask_map50_best": best_map50,
            "epoch_best": best_epoch,
            "metric_col": MASK_MAP50_COL,
        })

    if not rows:
        print("[WARN] No runs found for summary export (no results.csv + run_meta.json).")
        return

    df = pd.DataFrame(rows)

    # 1) Per-run summary
    out_runs = runs_root / "summary_extended_runs_mask.csv"
    df.sort_values(["bands", "initialisation", "backbone_mod", "seed", "run_name"], inplace=True)
    df.to_csv(out_runs, index=False)
    print(f"✅ Wrote per-run summary -> {out_runs}")

    # 2) Table S2-style aggregation (mean ± std across seeds)
    g = df.groupby(["bands", "backbone_mod", "initialisation"], dropna=False)
    agg = g.agg(
        n=("mask_map50_best", "count"),
        map50_mean=("mask_map50_best", "mean"),
        map50_std=("mask_map50_best", "std"),
        epoch_mean=("epoch_best", "mean"),
        epoch_std=("epoch_best", "std"),
    ).reset_index()

    def fmt_mean_std(m, s) -> str:
        if m is None or (isinstance(m, float) and math.isnan(m)):
            return ""
        if s is None or (isinstance(s, float) and math.isnan(s)):
            return f"{m:.1f}"
        return f"{m:.1f} ± {s:.1f}"

    agg["mAP50_mask"] = [fmt_mean_std(m, s) for m, s in zip(agg["map50_mean"], agg["map50_std"])]
    agg["Epoch"] = [fmt_mean_std(m, s) for m, s in zip(agg["epoch_mean"], agg["epoch_std"])]

    # Wide format to match your Table S2 layout (scratch vs TL side-by-side)
    # Columns:
    # Bands | Backbone modifications | (Scratch mAP50, Scratch Epoch) | (TL mAP50, TL Epoch)
    wide = agg.pivot_table(
        index=["bands", "backbone_mod"],
        columns="initialisation",
        values=["mAP50_mask", "Epoch"],
        aggfunc="first"
    )

    # Flatten columns
    wide.columns = [f"{init}__{val}" for val, init in wide.columns]
    wide.reset_index(inplace=True)

    # Ensure expected column ordering (create empty if missing)
    for c in [
        "Training from scratch__mAP50_mask",
        "Training from scratch__Epoch",
        "Transfer learning__mAP50_mask",
        "Transfer learning__Epoch",
    ]:
        if c not in wide.columns:
            wide[c] = ""

    wide = wide[[
        "bands",
        "backbone_mod",
        "Training from scratch__mAP50_mask",
        "Training from scratch__Epoch",
        "Transfer learning__mAP50_mask",
        "Transfer learning__Epoch",
    ]].rename(columns={
        "bands": "Bands",
        "backbone_mod": "Backbone modifications",
        "Training from scratch__mAP50_mask": "Training from scratch mAP50",
        "Training from scratch__Epoch": "Training from scratch Epoch",
        "Transfer learning__mAP50_mask": "Transfer learning mAP50",
        "Transfer learning__Epoch": "Transfer learning Epoch",
    })

    out_s2 = runs_root / "summary_table_s2_mask.csv"
    wide.sort_values(["Bands", "Backbone modifications"], inplace=True)
    wide.to_csv(out_s2, index=False)
    print(f"✅ Wrote Table S2-style summary -> {out_s2}")


# -----------------------------
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

    # results_csv = run_dir / "train" / "results.csv"
    # if results_csv.exists() and skip_if_done and not force_rerun:
    #     print(f"⏭️  Skipping completed run (results.csv exists): {run_name}")
    #     return {
    #         "status": "skipped",
    #         "run_dir": str(run_dir),
    #         "reason": "results.csv exists",
    #         "exp_id": cfg.exp_id,
    #         "dataset_id": cfg.dataset_id,
    #         "dataset_yaml": cfg.dataset_yaml,
    #         "in_channels": cfg.in_channels,
    #         "seed": seed,
    #         "init_type": init_type,
    #         "channel_init_mode": channel_init_mode,
    #         "variant_tag": variant_tag,
    #         "toggles": asdict(toggles),
    #         "train_cfg": asdict(train_cfg),
    #     }

    results_csv = run_dir / "train" / "results.csv"
    test_csv = run_dir / "results_test.csv"

    skip_train = results_csv.exists() and skip_if_done and not force_rerun
    skip_test = test_csv.exists() and skip_if_done and not force_rerun

    if skip_train:
        print(f"⏭️  Training already completed: {run_name}")


    test_csv = run_dir / "results_test.csv"
    if test_csv.exists() and skip_if_done and not force_rerun:
        print(f"⏭️  Skipping test eval (results_test.csv exists): {run_name}")

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

    def _count(model_obj, name):
        return sum(1 for m in model_obj.model.modules() if m.__class__.__name__ == name)

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

    #model.train(**train_kwargs)

    if not skip_train:
        model.train(**train_kwargs)
    else:
        print(f"➡️  Using existing trained model for test evaluation")


    # NEW: evaluate on TEST split and write results_test.csv
    #evaluate_on_test_and_save_csv(run_dir, cfg.dataset_yaml)

    if not skip_test:
        evaluate_on_test_and_save_csv(run_dir, cfg.dataset_yaml)
    else:
        print(f"⏭️  Test evaluation already exists: {run_name}")


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

    # NEW: Export CSV summaries after all runs
    export_summaries(RUNS_ROOT)


if __name__ == "__main__":
    main()
