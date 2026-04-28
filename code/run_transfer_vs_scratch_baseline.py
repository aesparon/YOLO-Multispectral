import os
import sys
import csv
import json
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

#########################  
# Workaround for Intel OpenMP runtime duplication crash on Windows.
# NOTE: Safer long-term fix is to ensure only one OpenMP runtime is loaded,
# but this keeps experiments moving.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# -----------------------------------------------------------------------------
# IMPORTANT: point to your modified Ultralytics fork (kept exactly as you want)
# -----------------------------------------------------------------------------
from pathlib import Path
code_path = Path(__file__).resolve().parent
# get parent path
code_parent_path = code_path.parent
YOLO_SOURCE_PATH =  code_path / 'ultralytics_MS'  
if str(YOLO_SOURCE_PATH) not in sys.path:
    sys.path.insert(0, str(YOLO_SOURCE_PATH))
from ultralytics import YOLO  # type: ignore



# Local project imports (your original mod file)
from mod_pt_model_seg import patch_yolo_seg_ckpt  # type: ignore

import torch



# -----------------------------------------------------------------------------
# USER SETTINGS (restore your intended defaults)
# -----------------------------------------------------------------------------
EPOCHS = 600
IMGSZ = 600
BATCH = 8
WORKERS = 4
#DEVICE = "cuda"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATIENCE =  100   #   50  # set to what you want (e.g., 50, 100, 200). 0 disables early stop.

NC = 5  # your dataset classes

# Model sources
MODEL_PT = code_path / 'yolo11x-seg.pt'       
MODEL_YAML =  code_path / 'yolo11x-seg.yaml'

# MODEL_PT = code_path / 'yolov8x-seg.pt'       
# MODEL_YAML =  code_path / 'yolov8x-seg.yaml'

# Where runs go
#RUNS_ROOT =  code_path / 'runs_transfer_vs_scratch_V8'
RUNS_ROOT =  code_path / 'runs_transfer_vs_scratch_V11x_3_seed'


RUNS_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Experiment config
# -----------------------------------------------------------------------------
@dataclass
class BaselineCfg:
    exp_id: str
    dataset_id: str
    dataset_yaml: str
    in_channels: int


# Edit these to match your 2 datasets
EXPS: List[BaselineCfg] = [
    BaselineCfg(
        exp_id="rgb3_baseline",
        dataset_id="rgb3",
        dataset_yaml= code_parent_path / 'datasets/weeds_galore_processed/RGB/outputs/data_RGB.yaml',
        in_channels=3,
    ),
    BaselineCfg(
        exp_id="rgb5_baseline",
        dataset_id="rgb5",
        dataset_yaml= code_parent_path / 'datasets/weeds_galore_processed/RGBRN/outputs/data_RGBRN.yaml',
        in_channels=5,
    ),
]

# init_type: "pt" (transfer) or "yaml" (yaml->pt rebuilt then trained)
INIT_TYPES = ["pt", "yaml"]

# channel init modes used when in_channels != 3 or when doing yaml->pt rebuild
#CHANNEL_INIT_MODES = ["avg", "random", "copy_g"]  # add copy_r/copy_b/repeat_rgb if you like
#CHANNEL_INIT_MODES = ["na" , "avg", "random", "copy_g"] 

# all
CHANNEL_INIT_MODES = ["na", "avg", "random", "copy_g", "copy_r", "copy_b", "repeat_rgb"]

SEEDS = [0,1,2]  # extend to [0,1,2] etc


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def run_dir_name(cfg: BaselineCfg, init_type: str, channel_init_mode: str, seed: int) -> str:
    # match your existing naming style
    bb = "yolo11x"
    #bb = "yolov8x"
    ch = channel_init_mode if cfg.in_channels != 3 or init_type == "yaml" else "na"
    return f"{cfg.exp_id}_{bb}_{init_type}_{ch}_seed{seed}"


def already_done(run_dir: Path) -> bool:
    p = run_dir / "results.csv"
    return p.exists() and p.stat().st_size > 0


def json_safe(x: Any) -> Any:
    # Fix: sets -> lists, Paths -> str, etc (prevents "set is not JSON serializable")
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, set):
        return sorted(list(x))
    if isinstance(x, tuple):
        return [json_safe(v) for v in x]
    if isinstance(x, list):
        return [json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2)


def read_last_metrics(results_csv: Path) -> Dict[str, Any]:
    """
    Reads final row from Ultralytics results.csv and returns key metrics.
    Works for both detect/seg results formats; returns whatever columns exist.
    """
    if not results_csv.exists() or results_csv.stat().st_size == 0:
        return {}

    # Use csv to avoid pandas dependency issues
    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return {}
        last = rows[-1]

    # Keep a useful subset if present; else return all.
    keep_keys = []
    for k in last.keys():
        lk = k.lower()
        if ("map" in lk) or ("precision" in lk) or ("recall" in lk) or ("metrics/" in lk):
            keep_keys.append(k)

    out = {}
    if keep_keys:
        for k in keep_keys:
            out[k] = last.get(k)
    else:
        out = dict(last)

    return out


# -----------------------------------------------------------------------------
# Core run function (stable)
# -----------------------------------------------------------------------------
def run_single(cfg: BaselineCfg, seed: int, init_type: str, channel_init_mode: str) -> Dict:
    run_dir = RUNS_ROOT / run_dir_name(cfg, init_type, channel_init_mode, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ✅ MUST be here: before any patching / YOLO() / train()
    if already_done(run_dir):
        print(f"[SKIP] Found existing results.csv -> {run_dir}")
        # still enrich summary with metrics:
        metrics = read_last_metrics(run_dir / "results.csv")
        return {
            "status": "skipped",
            "run_dir": str(run_dir),
            "exp_id": cfg.exp_id,
            "dataset_id": cfg.dataset_id,
            "dataset_yaml": cfg.dataset_yaml,
            "in_channels": cfg.in_channels,
            "seed": seed,
            "init_type": init_type,
            "channel_init_mode": channel_init_mode,
            "effective_init_mode": "na" if (cfg.in_channels == 3 and init_type == "pt") else channel_init_mode,
            **metrics,
        }

    # If folder exists but no results.csv, treat as FAILED/INCOMPLETE and restart clean
    if run_dir.exists():
        print(f"[CLEAN] Incomplete run detected (no results.csv). Removing -> {run_dir}")
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    base = MODEL_PT if init_type == "pt" else MODEL_YAML
    model_for_train = str(base)
    effective_init_mode = "na"

    # For YAML init or any non-3ch input, build a patched .pt first
    if (init_type == "yaml") or (cfg.in_channels != 3):
        patched_path, effective_init_mode = patch_yolo_seg_ckpt(
            input_ckpt=str(base),
            out_dir=str(run_dir / "mod_model"),
            in_channels=int(cfg.in_channels),
            channel_init_mode=str(channel_init_mode),
            nc=int(NC),
            yolo_source_path=str(YOLO_SOURCE_PATH / "ultralytics"),  # IMPORTANT: correct package root
            save_tag=f"{init_type}_{channel_init_mode}_seed{seed}",
        )
        model_for_train = str(patched_path)

    meta = {
        "exp_id": cfg.exp_id,
        "dataset_id": cfg.dataset_id,
        "dataset_yaml": cfg.dataset_yaml,
        "in_channels": cfg.in_channels,
        "seed": seed,
        "init_type": init_type,
        "requested_channel_init_mode": channel_init_mode,
        "effective_init_mode": effective_init_mode,
        "base_model": str(base),
        "model_for_train": model_for_train,
        "epochs": EPOCHS,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "device": DEVICE,
        "workers": WORKERS,
        "patience": PATIENCE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    write_json(run_dir / "run_meta.json", meta)

    print(f"\n=== RUN {run_dir.name} ===")
    print(meta)

    # ---- EXACT place model.train() is called ----
    model = YOLO(model_for_train)
    model.train(
        task="segment",
        data=cfg.dataset_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        patience=PATIENCE,
        project=str(RUNS_ROOT),
        name=run_dir.name,
        exist_ok=True,
        plots=False,
    )

    # Ultralytics writes results.csv into RUNS_ROOT/run_dir.name already
    # Do NOT copy it (prevents SameFileError)
    results_csv = run_dir / "results.csv"
    metrics = read_last_metrics(results_csv)

    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "exp_id": cfg.exp_id,
        "dataset_id": cfg.dataset_id,
        "dataset_yaml": cfg.dataset_yaml,
        "in_channels": cfg.in_channels,
        "seed": seed,
        "init_type": init_type,
        "channel_init_mode": channel_init_mode,
        "effective_init_mode": effective_init_mode,
        **metrics,
    }


def write_summary_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # union of keys
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def main() -> None:
    print("\n[SANITY] exp_id -> (dataset_id, in_channels, yaml)")
    print("-" * 110)
    print(f"{'exp_id':25s} {'dataset_id':10s} {'in_ch':5s} yaml")
    print("-" * 110)
    for c in EXPS:
        print(f"{c.exp_id:25s} {c.dataset_id:10s} {str(c.in_channels):5s} {c.dataset_yaml}")
    print("-" * 110)

    # jobs = []
    # for cfg in EXPS:
    #     for seed in SEEDS:
    #         for init_type in INIT_TYPES:
    #             if cfg.in_channels == 3 and init_type == "pt":
    #                 jobs.append((cfg, seed, init_type, "na"))
    #             else:
    #                 for ch in CHANNEL_INIT_MODES:
    #                     jobs.append((cfg, seed, init_type, ch))

    # # Build jobs
    # jobs = []
    # for cfg in EXPS:
    #     for seed in SEEDS:
    #         for init_type in INIT_TYPES:
    #             for ch_mode in CHANNEL_INIT_MODES:

    #                 # ✅ Skip pointless channel init sweeps for scratch YAML
    #                 if init_type == "yaml":
    #                     if cfg.in_channels == 3:
    #                         # only one meaningful mode for 3ch scratch
    #                         if ch_mode != "na":
    #                             continue
    #                     else:
    #                         # for >3ch scratch, only random is meaningful (your preference)
    #                         if ch_mode != "random":
    #                             continue

    #                 jobs.append((cfg, seed, init_type, ch_mode))

    # Build jobs
    jobs = []
    for cfg in EXPS:
        for seed in SEEDS:
            for init_type in INIT_TYPES:

                # ----------------------------
                # RULES YOU WANT:
                # 1) pt + 3ch: only "na"
                # 2) pt + >3ch: sweep all non-"na" modes (or include "na" if you want)
                # 3) yaml + 3ch (scratch): only "na"
                # 4) yaml + >3ch (scratch): only "random" (your preference)
                # ----------------------------

                if init_type == "pt" and cfg.in_channels == 3:
                    jobs.append((cfg, seed, init_type, "na"))
                    continue

                if init_type == "yaml" and cfg.in_channels == 3:
                    jobs.append((cfg, seed, init_type, "na"))
                    continue

                if init_type == "yaml" and cfg.in_channels != 3:
                    jobs.append((cfg, seed, init_type, "random"))
                    continue

                # pt + >3ch : sweep init modes that matter
                for ch_mode in CHANNEL_INIT_MODES:
                    if ch_mode == "na":
                        continue
                    jobs.append((cfg, seed, init_type, ch_mode))


    rows: List[Dict[str, Any]] = []
    total = len(jobs)

    for i, (cfg, seed, init_type, ch_mode) in enumerate(jobs, start=1):
        print(f"\n[JOB {i}/{total}] exp_id={cfg.exp_id} dataset={cfg.dataset_id} init_type={init_type} ch_mode={ch_mode} seed={seed}")
        r = run_single(cfg=cfg, seed=seed, init_type=init_type, channel_init_mode=ch_mode)
        rows.append(r)

    # JSONL summary (always)
    jsonl_path = RUNS_ROOT / "summary_runs.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(json_safe(r)) + "\n")
    print(f"\n[SUMMARY] wrote {len(rows)} rows to: {jsonl_path}")

    # CSV summary (what you want)
    csv_path = RUNS_ROOT / "summary_runs.csv"
    write_summary_csv(rows, csv_path)
    print(f"[SUMMARY] wrote CSV to: {csv_path}")


if __name__ == "__main__":
    main()
