"""Recalcule ASSD et HD95 avec la définition STANDARD surface-à-surface.

Motif (2026-07-29) : le harnais d'évaluation calcule la distance des voxels de bord
prédits vers le *volume* de référence rempli (`distance_transform_edt(~gs)`), et non vers
sa *surface*. Tout voxel de bord prédit situé à l'intérieur de la référence compte donc
pour une distance nulle, ce qui SOUS-ESTIME systématiquement ASSD et HD95 (mesuré :
0.70 mm au lieu de 2.90 mm sur IBSR_1_24). Le biais est identique pour toutes les
méthodes comparées — les comparaisons restent valides — mais les valeurs absolues ne le
sont pas, et un relecteur qui recalcule avec la définition usuelle trouverait 2 à 4×.

Ce script recalcule, pour les 3 benchmarks équitables et les 3 méthodes, les distances
surface-à-surface en millimètres réels (spacing anisotrope respecté), et écrit
`outputs/ablation_2026-07/tables/surface_metrics_corrected.csv`.

    .venv/bin/python scripts/recompute_surface_metrics.py
"""
from __future__ import annotations

import glob
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, ".")
from scripts.make_fig3_error_maps import bobs_manifest, ibsr18_manifest, mbg_manifest

OUT = "outputs/ablation_2026-07/tables"
os.makedirs(OUT, exist_ok=True)


def _surface(m: np.ndarray) -> np.ndarray:
    return m & ~ndi.binary_erosion(m, iterations=1, border_value=0)


def surface_metrics(pred: np.ndarray, ref: np.ndarray, spacing) -> tuple[float, float]:
    """(ASSD, HD95) en mm — distances SURFACE-à-SURFACE, spacing réel respecté."""
    ps, rs = _surface(pred), _surface(ref)
    if not ps.any() or not rs.any():
        return float("nan"), float("nan")
    dt_ref = ndi.distance_transform_edt(~rs, sampling=spacing)   # vers la SURFACE de réf.
    dt_pred = ndi.distance_transform_edt(~ps, sampling=spacing)
    d_p2r = dt_ref[ps]
    d_r2p = dt_pred[rs]
    assd = float((d_p2r.mean() + d_r2p.mean()) / 2.0)            # moyenne des deux sens
    hd95 = float(max(np.percentile(d_p2r, 95), np.percentile(d_r2p, 95)))
    return assd, hd95


def rows_for(bench: str, manifest) -> list[dict]:
    out = []
    for i, (_b, sub, gtp, masks) in enumerate(manifest, 1):
        gt_img = nib.load(gtp)
        gt = np.asarray(gt_img.dataobj) > 0
        sp = gt_img.header.get_zooms()[:3]
        for method, path in masks.items():
            if not (path and os.path.exists(str(path))):
                continue
            pr = np.asarray(nib.load(str(path)).dataobj) > 0
            if pr.shape != gt.shape:
                print(f"  !! shape mismatch {bench}/{sub}/{method}")
                continue
            s = pr.sum() + gt.sum()
            dice = float(2 * (pr & gt).sum() / s) if s else 0.0
            assd, hd95 = surface_metrics(pr, gt, sp)
            out.append(dict(benchmark=bench, subject=sub, method=method,
                            dice=dice, assd=assd, hd95=hd95))
        if i % 20 == 0:
            print(f"  ... {bench} {i}/{len(manifest)}", flush=True)
    return out


def main() -> int:
    rows: list[dict] = []
    for bench, mf in (("BOBs", bobs_manifest), ("IBSR18", ibsr18_manifest),
                      ("Mindboggle-101", mbg_manifest)):
        print(f"=== {bench} ===", flush=True)
        rows += rows_for(bench, mf())
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/surface_metrics_corrected.csv", index=False)

    print("\n=== ASSD / HD95 corrigés (surface-à-surface, mm) ===")
    agg = df.groupby(["benchmark", "method"]).agg(
        n=("dice", "size"), dice=("dice", "mean"),
        assd=("assd", "mean"), assd_sd=("assd", "std"),
        hd95=("hd95", "mean"), hd95_sd=("hd95", "std")).round(3)
    print(agg.to_string())
    print(f"\nécrit : {OUT}/surface_metrics_corrected.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
