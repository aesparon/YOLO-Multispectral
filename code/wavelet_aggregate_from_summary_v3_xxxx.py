#!/usr/bin/env python
"""
Post-hoc aggregation of wavelet experiments from an existing summary CSV.

Input:
  - A CSV like summary_wavelet_final_20251224_195839.csv
    containing one row per (dataset, cfg_id, aug_profile, seed)
    with columns:
      best_map50, best_map5095, etc.

Output:
  - summary_wavelet_runs_AGG_<timestamp>.csv     (original rows + outlier flag)
  - summary_wavelet_agg_AGG_<timestamp>.csv      (multi-seed aggregation)
  - summary_wavelet_wavelet_deltas_AGG_<timestamp>.csv  (Won vs Woff deltas)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd


# =============================================================================
# 1. CONFIG: where is your existing summary CSV?
# =============================================================================

# TODO: set this to your actual file
# e.g. r"D:/PD/Publications/.../summary_wavelet_final_20251224_195839.csv"
INPUT_SUMMARY = Path(
    r"D:\PD\Publications\Yolo_mod\github\YOLO-Multispectral_2025_12_12"
    r"\examples\notebooks\YOLO-Multispectral\runs_wavelet_final_sweep\summary_wavelet_final_20251224_195839.csv"
)

# Where to drop the new aggregated CSVs
OUT_DIR = INPUT_SUMMARY.parent / "summaries_posthoc"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. OUTLIER FLAGGING
# =============================================================================

def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark outlier seeds at config-level:
    outlier if best_map5095 outside (μ ± 2σ) for that config
    (where config is defined by dataset+cfg_id+aug_profile+wavelet params).
    """
    group_cols = [
        "dataset", "cfg_id", "aug_profile", "wavelet_on",
        "wavelet_inject_stage", "alpha_max", "alpha_ramp",
        "lr0", "epochs", "imgsz", "batch",
    ]

    def _flag(sub: pd.DataFrame) -> pd.DataFrame:
        mu = sub["best_map5095"].mean()
        sigma = sub["best_map5095"].std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            sub["is_outlier"] = False
            return sub
        lo = mu - 2 * sigma
        hi = mu + 2 * sigma
        sub["is_outlier"] = (sub["best_map5095"] < lo) | (sub["best_map5095"] > hi)
        return sub

    # FutureWarning-safe: include_groups=False in newer pandas, but this is fine
    df_out = df.groupby(group_cols, group_keys=False).apply(_flag)
    return df_out


# =============================================================================
# 3. AGGREGATION & ROBUST SCORE
# =============================================================================

def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset", "cfg_id", "aug_profile", "wavelet_on",
        "wavelet_inject_stage", "alpha_max", "alpha_ramp",
        "lr0", "epochs", "imgsz", "batch",
    ]

    agg = (
        df.groupby(group_cols)
          .agg(
              n_seeds=("seed", "nunique"),
              mean_best_map50=("best_map50", "mean"),
              std_best_map50=("best_map50", "std"),
              mean_best_map5095=("best_map5095", "mean"),
              std_best_map5095=("best_map5095", "std"),
          )
          .reset_index()
    )

    # robust_score = mean - 0.5 * std
    agg["robust_score_map5095"] = agg["mean_best_map5095"] - 0.5 * agg["std_best_map5095"]

    agg = agg.sort_values("robust_score_map5095", ascending=False)
    return agg


# =============================================================================
# 4. WAVELET DELTAS (Won vs Woff)
# =============================================================================

def compute_wavelet_deltas(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Build a wavelet_on vs wavelet_off delta table for configs that share
    (dataset, cfg_id, aug_profile).
    """
    key_cols = ["dataset", "cfg_id", "aug_profile"]
    cols_keep = key_cols + [
        "wavelet_on",
        "mean_best_map5095",
        "std_best_map5095",
        "robust_score_map5095",
        "n_seeds",
    ]
    sub = agg[cols_keep].copy()

    piv = sub.pivot_table(
        index=key_cols,
        columns="wavelet_on",
        values=["mean_best_map5095", "std_best_map5095", "robust_score_map5095"],
    )

    # Flatten MultiIndex columns: ("mean_best_map5095", True) -> "mean_best_map5095_waveletTrue"
    piv.columns = [f"{metric}_wavelet{flag}" for metric, flag in piv.columns]
    piv = piv.reset_index()

    # Deltas where both on/off exist
    if "mean_best_map5095_waveletTrue" in piv.columns and "mean_best_map5095_waveletFalse" in piv.columns:
        piv["delta_mean_map5095_Won_minus_Woff"] = (
            piv["mean_best_map5095_waveletTrue"] - piv["mean_best_map5095_waveletFalse"]
        )

    if "robust_score_map5095_waveletTrue" in piv.columns and "robust_score_map5095_waveletFalse" in piv.columns:
        piv["delta_robust_score_map5095_Won_minus_Woff"] = (
            piv["robust_score_map5095_waveletTrue"] - piv["robust_score_map5095_waveletFalse"]
        )

    return piv


# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    if not INPUT_SUMMARY.exists():
        raise FileNotFoundError(f"Input summary CSV not found: {INPUT_SUMMARY}")

    print(f"[LOAD] {INPUT_SUMMARY}")
    df = pd.read_csv(INPUT_SUMMARY)

    # Sanity check: required columns
    required_cols = {
        "dataset", "cfg_id", "aug_profile", "seed",
        "wavelet_on", "wavelet_inject_stage",
        "alpha_max", "alpha_ramp",
        "lr0", "epochs", "imgsz", "batch",
        "best_map50", "best_map5095",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input summary: {missing}")

    # 1) Flag outliers
    df_flagged = flag_outliers(df)

    # 2) Aggregate multi-seed
    df_agg = aggregate_results(df_flagged)

    # 3) Wavelet deltas
    df_delta = compute_wavelet_deltas(df_agg)

    # 4) Save outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_runs = OUT_DIR / f"summary_wavelet_runs_AGG_{ts}.csv"
    out_agg = OUT_DIR / f"summary_wavelet_agg_AGG_{ts}.csv"
    out_delta = OUT_DIR / f"summary_wavelet_wavelet_deltas_AGG_{ts}.csv"

    df_flagged.to_csv(out_runs, index=False)
    df_agg.to_csv(out_agg, index=False)
    df_delta.to_csv(out_delta, index=False)

    print("\n[SUMMARY FILES]")
    print(f"Per-run (with outlier flag): {out_runs}")
    print(f"Aggregated:                  {out_agg}")
    print(f"Wavelet deltas:              {out_delta}")

    # 5) Print a nice leaderboard
    print("\n[TOP CONFIGS by robust_score_map5095]")
    cols_show = [
        "dataset", "cfg_id", "aug_profile", "wavelet_on",
        "mean_best_map5095", "std_best_map5095",
        "robust_score_map5095", "n_seeds",
    ]
    print(df_agg[cols_show].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
