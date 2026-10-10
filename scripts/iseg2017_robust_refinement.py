"""iSeg2017-specific robust atlas scoring and pediatric T1/T2 refinement."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .neuro_utils import (
    largest_component,
    load_data,
    load_img,
    load_mask,
    mask_overlap_stats,
    mask_volume_cm3,
    save_float,
    save_mask,
    update_metrics_json,
)


def _ball(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    grid = np.ogrid[tuple(slice(-radius, radius + 1) for _ in range(3))]
    return sum(g.astype(float) ** 2 for g in grid) <= radius**2


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    iterations = max(0, int(iterations))
    if iterations == 0:
        return np.asarray(mask, dtype=bool)
    return ndi.binary_dilation(mask, structure=ndi.generate_binary_structure(3, 1), iterations=iterations)


def robust_atlas_image_alignment_score(atlas_path: str | Path, image_path: str | Path) -> tuple[float, dict[str, Any]]:
    """Robust inside/outside intensity score for infant T1/T2 atlas alignment."""

    try:
        atlas_img = load_img(atlas_path)
        image_img = load_img(image_path)
        atlas = atlas_img.get_fdata(dtype=np.float32) > 0.5
        data = image_img.get_fdata(dtype=np.float32)
        if atlas.shape != data.shape or atlas.sum() < 1000:
            return 0.0, {"intensity_ratio": None, "reason": "shape_mismatch_or_empty"}
        shell = ndi.binary_dilation(atlas, structure=_ball(3)) & (~atlas)
        if shell.sum() < 1000:
            return 0.0, {"intensity_ratio": None, "reason": "empty_shell"}
        inside = float(np.nanmedian(data[atlas]))
        outside = float(np.nanmedian(data[shell]))
        diff = inside - outside
        spread = float(np.nanpercentile(data[atlas | shell], 95) - np.nanpercentile(data[atlas | shell], 5) + 1e-6)
        score = float(np.clip((diff / spread + 0.20) / 0.45, 0.0, 1.0))
        ratio = inside / (outside + 1e-6)
        return score, {"intensity_ratio": float(ratio), "inside_median": inside, "shell_median": outside}
    except Exception as exc:
        return 0.0, {"intensity_ratio": None, "error": str(exc)}


def apply_pediatric_t2_refinement(
    mask_path: str | Path,
    brain_path: str | Path,
    input_image_path: str | Path,
    t2_path: str | Path | None,
    atlas_path: str | Path,
    metrics_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recover bounded peripheral tissue for iSeg using atlas shell + T1/T2."""

    cfg = {
        "enabled": True,
        "shell_width": 20,
        "atlas_dilation": 15,
        "t1_percentile": 3.0,
        "t2_percentile": 3.0,
        "max_added_cm3": 500.0,
        "max_volume_cm3": 1100.0,
        "min_atlas_precision": 0.75,
        "allow_atlas_dice_decrease_on_undercoverage": False,
        "target_min_atlas_volume_ratio": 0.93,
        "min_recovery_added_cm3": 25.0,
        "max_atlas_dice_drop": 0.12,
        "backup_suffix": "_pre-pediatric-t2-refine",
    }
    if config:
        cfg.update(config)
    result: dict[str, Any] = {
        "pediatric_refinement_used": False,
        "pediatric_refinement_reason": "disabled" if not cfg.get("enabled", True) else "not_improved",
    }
    if not cfg.get("enabled", True):
        return result

    mask_path = Path(mask_path)
    atlas_path = Path(atlas_path)
    if not mask_path.exists() or not atlas_path.exists():
        result["pediatric_refinement_reason"] = "missing_mask_or_atlas"
        return result

    img = load_img(input_image_path)
    t1 = img.get_fdata(dtype=np.float32)
    t2 = load_data(t2_path, np.float32) if t2_path and Path(t2_path).exists() else None
    mask = load_mask(mask_path)
    atlas = load_mask(atlas_path)
    if mask.shape != atlas.shape or mask.shape != t1.shape or (t2 is not None and t2.shape != mask.shape):
        result["pediatric_refinement_reason"] = "shape_mismatch"
        return result

    vox = float(np.prod(img.header.get_zooms()[:3]) / 1000.0)
    before = mask_overlap_stats(mask, atlas, img)
    before_vol = mask_volume_cm3(mask, img)
    atlas_vol = mask_volume_cm3(atlas, img)
    atlas_env = _dilate(atlas, int(cfg["atlas_dilation"]))
    shell = atlas_env & (~mask) & _dilate(mask, int(cfg["shell_width"]))
    proposed = np.zeros(mask.shape, dtype=bool)
    if shell.any() and mask.any():
        t1_thr = float(np.percentile(t1[mask], float(cfg["t1_percentile"])))
        proposed = shell & (t1 >= t1_thr)
        result["pediatric_refinement_t1_threshold"] = t1_thr
        if t2 is not None:
            t2_thr = float(np.percentile(t2[mask], float(cfg["t2_percentile"])))
            proposed = proposed & (t2 >= t2_thr)
            result["pediatric_refinement_t2_threshold"] = t2_thr
        max_vox = int(float(cfg["max_added_cm3"]) / max(vox, 1e-9))
        if proposed.sum() > max_vox:
            vals = t1[proposed]
            cutoff = np.partition(vals, max(0, vals.size - max_vox))[max(0, vals.size - max_vox)]
            proposed = proposed & (t1 >= cutoff)

    candidate = largest_component(mask | proposed)
    after = mask_overlap_stats(candidate, atlas, img)
    after_vol = mask_volume_cm3(candidate, img)
    result.update(
        {
            "pediatric_refinement_proposed_added_cm3": float(proposed.sum() * vox),
            "pediatric_refinement_proposed_volume_after_cm3": after_vol,
            "pediatric_refinement_proposed_atlas_precision_after": after["precision"],
            "pediatric_refinement_proposed_atlas_coverage_after": after["coverage"],
            "pediatric_refinement_volume_before_cm3": before_vol,
            "pediatric_refinement_volume_after_cm3": after_vol,
            "pediatric_refinement_volume_ratio": after_vol / max(before_vol, 1e-6),
            "pediatric_refinement_atlas_precision_after": after["precision"],
            "pediatric_refinement_atlas_coverage_after": after["coverage"],
            "pediatric_refinement_atlas_dice_before": before["dice"],
            "pediatric_refinement_atlas_dice_after": after["dice"],
            "pediatric_refinement_atlas_volume_cm3": atlas_vol,
        }
    )
    if not proposed.any():
        result["pediatric_refinement_reason"] = "no_candidate_voxels"
        return result
    if after_vol > float(cfg["max_volume_cm3"]):
        result["pediatric_refinement_reason"] = f"volume_too_large {after_vol:.1f}"
        return result
    if after["precision"] < float(cfg["min_atlas_precision"]):
        result["pediatric_refinement_reason"] = f"atlas_precision_too_low {after['precision']:.3f}"
        return result

    atlas_dice_drop = float(before["dice"] - after["dice"])
    added_cm3 = float(max(0.0, after_vol - before_vol))
    target_ratio = float(cfg.get("target_min_atlas_volume_ratio", 0.93))
    undercoverage = before_vol < target_ratio * max(atlas_vol, 1e-6)
    meaningful_recovery = added_cm3 >= float(cfg.get("min_recovery_added_cm3", 25.0))
    allow_recall_recovery = (
        bool(cfg.get("allow_atlas_dice_decrease_on_undercoverage", False))
        and undercoverage
        and meaningful_recovery
        and atlas_dice_drop <= float(cfg.get("max_atlas_dice_drop", 0.12))
    )
    result.update(
        {
            "pediatric_refinement_atlas_dice_drop": atlas_dice_drop,
            "pediatric_refinement_undercoverage_before": bool(undercoverage),
            "pediatric_refinement_meaningful_recovery": bool(meaningful_recovery),
            "pediatric_refinement_allow_recall_recovery": bool(allow_recall_recovery),
        }
    )
    if after["dice"] + 1e-6 < before["dice"] and not allow_recall_recovery:
        result["pediatric_refinement_reason"] = f"atlas_dice_degraded {before['dice']:.4f}->{after['dice']:.4f}"
        return result

    backup = mask_path.with_name(mask_path.name.replace(".nii.gz", f"{cfg['backup_suffix']}.nii.gz"))
    if not backup.exists():
        shutil.copyfile(mask_path, backup)
    save_mask(candidate, img, mask_path)
    save_float(t1 * candidate, img, brain_path)
    result.update(
        {
            "pediatric_refinement_used": True,
            "pediatric_refinement_reason": "atlas_t1_t2_shell_recovery_recall" if allow_recall_recovery else "atlas_t1_t2_shell_recovery",
            "pediatric_refinement_added_cm3": float((candidate.sum() - mask.sum()) * vox),
        }
    )
    if metrics_path:
        update_metrics_json(metrics_path, result)
    return result
