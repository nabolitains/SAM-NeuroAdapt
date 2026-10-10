"""Paired SOTA comparison statistics for SAM-NeuroAdapt result tables."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd


def _wilcoxon_pvalue(diff: pd.Series) -> float:
    values = pd.to_numeric(diff, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size < 2:
        return math.nan
    if np.allclose(values, 0.0, atol=1e-12, rtol=0.0):
        return 1.0
    try:
        from scipy import stats

        return float(stats.wilcoxon(values, zero_method="wilcox").pvalue)
    except Exception:
        return math.nan


def add_sam_paired_stats(
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    baseline_method: str = "SAM-NeuroAdapt",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add paired Dice deltas and Wilcoxon p-values versus SAM-NeuroAdapt.

    ``long_df`` receives subject-level ``sam_dice``, ``delta_dice_vs_sam`` and
    method-level ``p_wilcoxon_dice`` columns. ``summary_df`` receives
    ``paired_with_sam``, ``delta_dice_vs_sam`` and ``p_wilcoxon_dice``.
    Gold masks are used only because these tables are post-inference metrics.
    """

    long_df = long_df.copy()
    summary_df = summary_df.copy()
    if "method" not in long_df.columns or "subject" not in long_df.columns or "dice" not in long_df.columns:
        summary_df["paired_with_sam"] = 0
        summary_df["delta_dice_vs_sam"] = np.nan
        summary_df["p_wilcoxon_dice"] = np.nan
        return long_df, summary_df

    long_df["dice"] = pd.to_numeric(long_df["dice"], errors="coerce")
    error = (
        long_df["error"].fillna("").astype(str).str.strip()
        if "error" in long_df.columns
        else pd.Series([""] * len(long_df), index=long_df.index)
    )
    valid = long_df[error.eq("") & long_df["dice"].notna()].copy()
    sam = (
        valid[valid["method"].astype(str).eq(baseline_method)][["subject", "dice"]]
        .rename(columns={"dice": "sam_dice"})
        .dropna(subset=["sam_dice"])
        .drop_duplicates(subset=["subject"], keep="last")
    )

    if "sam_dice" in long_df.columns:
        long_df = long_df.drop(columns=["sam_dice"])
    long_df = long_df.merge(sam, on="subject", how="left")
    long_df["delta_dice_vs_sam"] = pd.to_numeric(long_df["dice"], errors="coerce") - pd.to_numeric(
        long_df["sam_dice"], errors="coerce"
    )

    stats_rows = []
    for method in summary_df.get("method", pd.Series(dtype=str)).astype(str):
        method_valid = valid[valid["method"].astype(str).eq(method)].copy()
        paired = method_valid.merge(sam, on="subject", how="inner").dropna(subset=["dice", "sam_dice"])
        diff = paired["dice"] - paired["sam_dice"]
        delta = float(diff.mean()) if len(paired) else math.nan
        if method == baseline_method and len(paired):
            delta = 0.0
        p_value = math.nan if method == baseline_method else _wilcoxon_pvalue(diff)
        stats_rows.append(
            {
                "method": method,
                "paired_with_sam": int(len(paired)),
                "delta_dice_vs_sam": delta,
                "p_wilcoxon_dice": p_value,
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        for col in ["paired_with_sam", "delta_dice_vs_sam", "p_wilcoxon_dice"]:
            if col in summary_df.columns:
                summary_df = summary_df.drop(columns=[col])
        summary_df = summary_df.merge(stats_df, on="method", how="left")
        p_by_method = stats_df.set_index("method")["p_wilcoxon_dice"].to_dict()
        long_df["p_wilcoxon_dice"] = long_df["method"].astype(str).map(p_by_method)
    else:
        summary_df["paired_with_sam"] = 0
        summary_df["delta_dice_vs_sam"] = np.nan
        summary_df["p_wilcoxon_dice"] = np.nan
        long_df["p_wilcoxon_dice"] = np.nan

    return long_df, summary_df
