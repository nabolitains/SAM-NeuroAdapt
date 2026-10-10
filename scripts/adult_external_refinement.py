"""Conservative atlas/T1 post-processing for adult external datasets."""

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
    dist = sum(g.astype(float) ** 2 for g in grid)
    return dist <= radius**2


def _quality(mask: np.ndarray, atlas: np.ndarray, ref_img) -> dict[str, float]:
    s = mask_overlap_stats(mask, atlas, ref_img)
    s["volume_cm3"] = mask_volume_cm3(mask, ref_img)
    return s


def apply_adult_external_refinement(
    mask_path: str | Path,
    brain_path: str | Path,
    input_image_path: str | Path,
    atlas_path: str | Path,
    metrics_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refine a SAM mask inside an atlas-guided local shell.

    The method is publication-safe: it does not read the gold mask.  It prunes
    clear atlas-inconsistent false positives and optionally recovers a bounded
    amount of local tissue in the atlas shell using only T1 intensity.
    """

    cfg = {
        "enabled": True,
        "min_volume_cm3": 650.0,
        "max_volume_cm3": 1900.0,
        "prune_enabled": True,
        "prune_atlas_dilation": 1,
        "prune_if_atlas_precision_below": 0.92,
        "prune_if_volume_ratio_above": 1.10,
        "prune_min_keep_ratio": 0.88,
        "recover_enabled": True,
        "recover_shell_width": 2,
        "recover_atlas_dilation": 2,
        "recover_intensity_percentile": 8.0,
        "recover_max_added_cm3": 120.0,
        "recover_max_volume_ratio": 1.08,
        "recover_min_atlas_precision": 0.88,
        "recover_min_initial_volume_ratio_to_atlas": 0.70,
        "recover_max_initial_volume_ratio_to_atlas": 1.14,
        "min_atlas_dice_gain": 0.002,
        "reference_atlas_trust_score": 0.0,
        "min_reference_trust_for_refinement": 0.12,
        "low_trust_keep_current_if_hard_ok": True,
        "backup_suffix": "_pre-adult-refine",
    }
    if config:
        cfg.update(config)

    result: dict[str, Any] = {
        "adult_refinement_used": False,
        "adult_refinement_reason": "disabled" if not cfg.get("enabled", True) else "not_improved",
    }
    if not cfg.get("enabled", True):
        return result

    mask_path = Path(mask_path)
    atlas_path = Path(atlas_path)
    if not mask_path.exists() or not atlas_path.exists():
        result["adult_refinement_reason"] = "missing_mask_or_atlas"
        return result

    img = load_img(input_image_path)
    image = img.get_fdata(dtype=np.float32)
    mask = load_mask(mask_path)
    atlas = load_mask(atlas_path)
    if mask.shape != atlas.shape or mask.shape != image.shape:
        result["adult_refinement_reason"] = "shape_mismatch"
        return result

    vox = float(np.prod(img.header.get_zooms()[:3]) / 1000.0)
    before = _quality(mask, atlas, img)
    reference_trust = float(cfg.get("reference_atlas_trust_score", 0.0) or 0.0)
    current_hard_ok = float(cfg["min_volume_cm3"]) <= float(before["volume_cm3"]) <= float(cfg["max_volume_cm3"])
    low_trust_skip = (
        reference_trust < float(cfg.get("min_reference_trust_for_refinement", 0.12))
        and (
            not bool(cfg.get("low_trust_keep_current_if_hard_ok", True))
            or current_hard_ok
        )
    )
    if low_trust_skip:
        result.update(
            {
                "adult_refinement_reason": "insufficient_external_atlas_support_low_trust",
                "adult_refinement_reference_atlas_trust_score": float(reference_trust),
                "adult_refinement_volume_before_cm3": before["volume_cm3"],
                "adult_refinement_current_hard_volume_ok": bool(current_hard_ok),
                "adult_refinement_atlas_precision_before": before["precision"],
                "adult_refinement_atlas_coverage_before": before["coverage"],
                "adult_refinement_atlas_dice_before": before["dice"],
            }
        )
        if metrics_path:
            update_metrics_json(metrics_path, result)
        return result
    refined = mask.copy()
    changed = False
    reasons: list[str] = []

    atlas_dil = ndi.binary_dilation(atlas, structure=_ball(int(cfg["prune_atlas_dilation"])))
    if cfg.get("prune_enabled", True):
        volume_ratio = before["volume_cm3"] / max(mask_volume_cm3(atlas, img), 1e-6)
        if before["precision"] < float(cfg["prune_if_atlas_precision_below"]) or volume_ratio > float(cfg["prune_if_volume_ratio_above"]):
            candidate = refined & atlas_dil
            keep_ratio = candidate.sum() / max(refined.sum(), 1)
            if keep_ratio >= float(cfg["prune_min_keep_ratio"]):
                refined = candidate
                changed = True
                reasons.append("prune_outside_atlas")

    atlas_volume = mask_volume_cm3(atlas, img)
    initial_volume_ratio_to_atlas = before["volume_cm3"] / max(atlas_volume, 1e-6)
    recover_ratio_ok = (
        float(cfg["recover_min_initial_volume_ratio_to_atlas"])
        <= initial_volume_ratio_to_atlas
        <= float(cfg["recover_max_initial_volume_ratio_to_atlas"])
    )

    if cfg.get("recover_enabled", True) and recover_ratio_ok:
        shell_width = int(cfg["recover_shell_width"])
        atlas_rec = ndi.binary_dilation(atlas, structure=_ball(int(cfg["recover_atlas_dilation"])))
        shell = atlas_rec & (~refined)
        shell = shell & ndi.binary_dilation(refined, structure=_ball(shell_width))
        if shell.any():
            inside = image[refined]
            if inside.size:
                threshold = float(np.percentile(inside, float(cfg["recover_intensity_percentile"])))
                proposed = shell & (image >= threshold)
                max_vox = int(float(cfg["recover_max_added_cm3"]) / max(vox, 1e-9))
                if proposed.sum() > max_vox:
                    vals = image[proposed]
                    cutoff = np.partition(vals, max(0, vals.size - max_vox))[max(0, vals.size - max_vox)]
                    proposed = proposed & (image >= cutoff)
                candidate = refined | proposed
                after_candidate = _quality(candidate, atlas, img)
                if (
                    proposed.any()
                    and after_candidate["volume_cm3"] <= float(cfg["max_volume_cm3"])
                    and after_candidate["volume_cm3"] <= before["volume_cm3"] * float(cfg["recover_max_volume_ratio"])
                    and after_candidate["precision"] >= float(cfg["recover_min_atlas_precision"])
                    and after_candidate["dice"] >= before["dice"] + float(cfg["min_atlas_dice_gain"])
                ):
                    refined = candidate
                    changed = True
                    reasons.append("recover_local_shell")

    refined = largest_component(refined)
    after = _quality(refined, atlas, img)
    if not changed or after["dice"] < before["dice"] + float(cfg["min_atlas_dice_gain"]):
        result.update(
            {
                "adult_refinement_reason": f"no_atlas_dice_gain {before['dice']:.4f}->{after['dice']:.4f}",
                "adult_refinement_volume_before_cm3": before["volume_cm3"],
                "adult_refinement_volume_after_cm3": after["volume_cm3"],
                "adult_refinement_atlas_precision_after": after["precision"],
                "adult_refinement_atlas_coverage_after": after["coverage"],
                "adult_refinement_atlas_dice_after": after["dice"],
            }
        )
        return result

    backup = mask_path.with_name(mask_path.name.replace(".nii.gz", f"{cfg['backup_suffix']}.nii.gz"))
    if not backup.exists():
        shutil.copyfile(mask_path, backup)
    save_mask(refined, img, mask_path)
    brain_data = image * refined
    save_float(brain_data, img, brain_path)

    result.update(
        {
            "adult_refinement_used": True,
            "adult_refinement_reason": "+".join(reasons) or "atlas_guided",
            "adult_refinement_volume_before_cm3": before["volume_cm3"],
            "adult_refinement_volume_after_cm3": after["volume_cm3"],
            "adult_refinement_atlas_precision_after": after["precision"],
            "adult_refinement_atlas_coverage_after": after["coverage"],
            "adult_refinement_atlas_dice_after": after["dice"],
            "adult_refinement_backup_mask": str(backup),
        }
    )
    if metrics_path:
        update_metrics_json(metrics_path, result)
    return result
