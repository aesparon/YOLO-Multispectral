"""
run_yolo11_seg_transformer_experiments.py

Small experiment runner to compare baseline YOLO11-seg vs.
Transformer-augmented variants on Weeds Galore RGB3 / RGB5.

It:
- runs multiple seeds
- reads metrics/mAP50(B) from results.csv
- prints a leaderboard sorted by mean mAP50.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ultralytics import YOLO
from yolo11_seg_transformer_patch import (
    TransformerPatchSpec,
    patch_yolo11_seg_with_transformers,
)


# -------------------------
# Config
# -------------------------

# Update these paths to your actual YAMLs
DATASET_RGB3_YAML = r"D:\path\to\weeds_galore_RGB3.yaml"
DATASET_RGB5_YAML = r"D:\path\to\weeds_galore_RGB5.yaml"

BASE_MODEL = "yolo11x-seg.pt"   # or yolo11m-seg.pt etc.

# Typical small sweep
SEEDS = [0, 1, 2]

EPOCHS = 100
BATCH_SIZE = 16
IMGSZ = 640

PROJECT_ROOT = Path(
    r"D:\PD\Publications\Yolo_mod\experiments\yolo11_seg_transformer"
)


@dataclass
class ExpConfig:
    exp_id: str
    data_yaml: str
    use_transformer: bool
    # indices are for C2f layers you decided on via print_yolo11_layer_summary
    transformer_indices: List[int] | None = None
    num_heads: int = 4
    num_layers: int = 1


EXPERIMENTS: List[ExpConfig] = [
    # --- RGB3 ---
    ExpConfig(
        exp_id="rgb3_baseline",
        data_yaml=DATASET_RGB3_YAML,
        use_transformer=False,
    ),
    ExpConfig(
        exp_id="rgb3_tr_backbone_neck",
        data_yaml=DATASET_RGB3_YAML,
        use_transformer=True,
        transformer_indices=[6, 13],  # <-- adjust to your C2f indices
        num_heads=4,
        num_layers=1,
    ),

    # --- RGB5 ---
    ExpConfig(
        exp_id="rgb5_baseline",
        data_yaml=DATASET_RGB5_YAML,
        use_transformer=False,
    ),
    ExpConfig(
        exp_id="rgb5_tr_backbone_neck",
        data_yaml=DATASET_RGB5_YAML,
        use_transformer=True,
        transformer_indices=[6, 13],  # <-- adjust to your C2f indices
        num_heads=4,
        num_layers=1,
    ),
]


# -------------------------
# Helpers
# -------------------------

def train_one_run(cfg: ExpConfig, seed: int) -> Path:
    """
    Train one experiment config + seed, return the run directory containing results.csv.
    """
    name = f"{cfg.exp_id}_s{seed}"
    print(f"\n=== TRAIN {name} ===")

    model = YOLO(BASE_MODEL)

    if cfg.use_transformer:
        if not cfg.transformer_indices:
            raise ValueError(f"Transformer indices not set for {cfg.exp_id}")
        patch_specs = [
            TransformerPatchSpec(
                indices=cfg.transformer_indices,
                num_heads=cfg.num_heads,
                num_layers=cfg.num_layers,
                label=cfg.exp_id,
            )
        ]
        patch_yolo11_seg_with_transformers(model, patch_specs, verbose=True)

    # Ultralytics train() returns a results object with save_dir attribute :contentReference[oaicite:3]{index=3}
    results = model.train(
        data=cfg.data_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        seed=seed,
        project=str(PROJECT_ROOT),
        name=name,
        # you can also pass device, lr0, etc. here as needed
    )

    run_dir = Path(results.save_dir)
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found at {csv_path}")
    return run_dir


def get_best_map50(results_csv: Path) -> float:
    """
    Read Ultralytics results.csv and return the best metrics/mAP50(B) value.
    """
    df = pd.read_csv(results_csv)
    # Some versions store it as 'metrics/mAP50(B)' or similar
    candidates = [c for c in df.columns if "metrics/mAP50" in c]
    if not candidates:
        raise KeyError(
            f"No metrics/mAP50 column found in {results_csv}. "
            f"Columns available: {list(df.columns)}"
        )
    col = candidates[0]
    return float(df[col].max())


# -------------------------
# Main loop
# -------------------------

def main():
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, float]] = {}
    per_run: List[Dict] = []

    for cfg in EXPERIMENTS:
        scores: List[float] = []

        for seed in SEEDS:
            run_dir = train_one_run(cfg, seed)
            csv_path = run_dir / "results.csv"
            best_map50 = get_best_map50(csv_path)
            scores.append(best_map50)

            per_run.append(
                dict(
                    exp_id=cgfg.exp_id if (cgfg := cfg) else cfg.exp_id,  # PyCharm-friendly trick
                    seed=seed,
                    run_dir=str(run_dir),
                    best_map50=best_map50,
                )
            )

            print(f"[RESULT] {cfg.exp_id} seed={seed} best mAP50={best_map50:.4f}")

        import numpy as np

        mean_map50 = float(np.mean(scores))
        std_map50 = float(np.std(scores))
        summary[cfg.exp_id] = dict(mean_map50=mean_map50, std_map50=std_map50)

    # Leaderboard sorted by mean mAP50
    print("\n================ LEADERBOARD (by mean mAP50) ================")
    rows = [
        (exp_id, vals["mean_map50"], vals["std_map50"])
        for exp_id, vals in summary.items()
    ]
    rows = sorted(rows, key=lambda x: x[1], reverse=True)

    for rank, (exp_id, mean_map50, std_map50) in enumerate(rows, start=1):
        print(
            f"{rank:2d}. {exp_id:30s}  mean mAP50={mean_map50:.4f}  ±{std_map50:.4f}"
        )

    # Save JSON summaries for later paper plots / tables
    (PROJECT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    (PROJECT_ROOT / "per_run.json").write_text(json.dumps(per_run, indent=2))


if __name__ == "__main__":
    main()
