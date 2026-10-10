"""Strict no-gold adult T1 boundary candidate/refinement stage."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi

from .neuro_utils import largest_component, mask_overlap_stats, mask_stats, save_float, save_mask, update_metrics_json


def _cfg(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = {
        "enabled": True,
        "min_volume_cm3": 650.0,
        "soft_min_volume_cm3": 850.0,
        "soft_max_volume_cm3": 1300.0,
        "max_volume_cm3": 1900.0,
        "max_components": 4,
        "atlas_dilation": 2,
        "boundary_width": 2,
        "shell_width": 7,
        "trim_percentiles": [1.0, 2.0, 4.0],
        "recover_percentiles": [2.0, 5.0, 8.0],
        "recover_max_added_cm3": 90.0,
        "max_apply_change_ratio": 0.18,
        "max_candidate_change_ratio": 0.32,
        "min_score_gain_to_apply": 0.018,
        "max_atlas_dice_drop": 0.025,
        "boundary_trim_bonus": 0.030,
        "over_reference_ratio_for_trim_bonus": 1.04,
        "over_reference_trim_bonus": 0.020,
        "current_over_reference_penalty": 0.020,
        "reference_atlas_trust_score": 0.0,
        "min_reference_trust_for_apply": 0.12,
        "low_trust_keep_current_if_hard_ok": True,
        "candidate_suffix": "_adult-boundary-candidate",
        "backup_suffix": "_pre-adult-boundary-refine",
    }
    if config:
        cfg.update(dict(config))
    cfg["trim_percentiles"] = [float(v) for v in cfg.get("trim_percentiles", [])]
    cfg["recover_percentiles"] = [float(v) for v in cfg.get("recover_percentiles", [])]
    return cfg


def _as_ref_img(path: str | Path, ref_img=None, order: int = 0):
    img = nib.load(str(path))
    if img.ndim > 3:
        data = np.asarray(img.dataobj)[..., 0]
        img = nib.Nifti1Image(data, img.affine, img.header)
    if ref_img is not None and (
        img.shape[:3] != ref_img.shape[:3] or not np.allclose(img.affine, ref_img.affine, atol=1e-3)
    ):
        img = resample_from_to(img, ref_img, order=order, cval=0)
    return img


def _read_mask(path: str | Path, ref_img) -> np.ndarray:
    return _as_ref_img(path, ref_img, order=0).get_fdata(dtype=np.float32) > 0.5


def _unique_existing(paths: Iterable[str | Path | None]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for item in paths:
        if not item:
            continue
        path = Path(item)
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    iterations = max(0, int(iterations))
    if iterations == 0:
        return np.asarray(mask, dtype=bool)
    return ndi.binary_dilation(mask, structure=ndi.generate_binary_structure(3, 1), iterations=iterations)


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    iterations = max(0, int(iterations))
    if iterations == 0:
        return np.asarray(mask, dtype=bool)
    return ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1), iterations=iterations)


def _clean(mask: np.ndarray) -> np.ndarray:
    mask = ndi.binary_fill_holes(np.asarray(mask, dtype=bool)).astype(bool)
    return largest_component(mask)


def _atlas_vote(atlas_paths: Iterable[str | Path | None], ref_img) -> tuple[np.ndarray | None, list[str]]:
    masks = []
    used = []
    for path in _unique_existing(atlas_paths):
        try:
            mask = _read_mask(path, ref_img)
        except Exception:
            continue
        if mask.any():
            masks.append(mask)
            used.append(str(path))
    if not masks:
        return None, []
    stack = np.stack([m.astype(np.uint8) for m in masks], axis=0)
    return stack.sum(axis=0) >= max(1, int(np.ceil(0.5 * len(masks)))), used


def _volume_score(volume_cm3: float, cfg: Mapping[str, Any]) -> float:
    lo = float(cfg["min_volume_cm3"])
    soft_lo = float(cfg["soft_min_volume_cm3"])
    soft_hi = float(cfg["soft_max_volume_cm3"])
    hi = float(cfg["max_volume_cm3"])
    if volume_cm3 < lo or volume_cm3 > hi:
        return 0.0
    if soft_lo <= volume_cm3 <= soft_hi:
        return 1.0
    if volume_cm3 < soft_lo:
        return float(np.clip((volume_cm3 - lo) / max(soft_lo - lo, 1e-6), 0.0, 1.0))
    return float(np.clip((hi - volume_cm3) / max(hi - soft_hi, 1e-6), 0.0, 1.0))


def _intensity_score(mask: np.ndarray, image: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    finite = np.isfinite(image)
    inside = image[mask & finite]
    shell = _dilate(mask, 3) & ~mask
    outside = image[shell & finite]
    if inside.size < 100 or outside.size < 100:
        return 0.5
    inside_med = float(np.median(inside[:: max(1, int(np.ceil(inside.size / 200_000)))]))
    outside_med = float(np.median(outside[:: max(1, int(np.ceil(outside.size / 200_000)))]))
    return float(np.clip((inside_med / max(outside_med, 1e-6) - 0.92) / 0.95, 0.0, 1.0))


def _score_candidate(
    name: str,
    mask: np.ndarray,
    current: np.ndarray,
    atlas: np.ndarray | None,
    image: np.ndarray,
    ref_img,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    stats = mask_stats(mask, ref_img)
    overlap = mask_overlap_stats(mask, current, ref_img)
    atlas_metrics = mask_overlap_stats(mask, atlas, ref_img) if atlas is not None and atlas.any() else None
    change_ratio = float(np.logical_xor(mask, current).sum() / max(current.sum(), 1))
    score = (
        0.28 * _volume_score(float(stats["volume_cm3"]), cfg)
        + 0.24 * _intensity_score(mask, image)
        + 0.18 * float(overlap["dice"])
        + 0.18 * (0.5 if atlas_metrics is None else float(atlas_metrics["dice"]))
        + 0.12 * (1.0 if int(stats["components"]) <= int(cfg["max_components"]) else 0.0)
    )
    if name == "current":
        score += 0.02
    if change_ratio > float(cfg["max_candidate_change_ratio"]):
        score -= min(0.50, (change_ratio - float(cfg["max_candidate_change_ratio"])) * 1.7)
    return {
        "name": name,
        "volume_cm3": float(stats["volume_cm3"]),
        "components": int(stats["components"]),
        "extent_ratio": float(stats.get("extent_ratio", 0.0)),
        "current_dice": float(overlap["dice"]),
        "atlas_dice": None if atlas_metrics is None else float(atlas_metrics["dice"]),
        "atlas_precision": None if atlas_metrics is None else float(atlas_metrics["precision"]),
        "atlas_coverage": None if atlas_metrics is None else float(atlas_metrics["coverage"]),
        "change_ratio": change_ratio,
        "volume_score": _volume_score(float(stats["volume_cm3"]), cfg),
        "intensity_score": _intensity_score(mask, image),
        "score": float(score),
        "mask": mask,
    }


def apply_adult_t1_boundary_refinement(
    mask_path: str | Path,
    brain_path: str | Path,
    input_image_path: str | Path,
    *,
    reference_atlas_path: str | Path | None = None,
    atlas_paths: Iterable[str | Path | None] = (),
    metrics_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build/apply a strict local boundary candidate using no gold labels."""

    cfg = _cfg(config)
    result: dict[str, Any] = {
        "boundary_refinement_enabled": bool(cfg.get("enabled", True)),
        "boundary_refinement_used": False,
        "boundary_refinement_reason": "disabled" if not cfg.get("enabled", True) else "not_improved",
    }
    if not cfg.get("enabled", True):
        return result

    mask_path = Path(mask_path)
    brain_path = Path(brain_path)
    input_image_path = Path(input_image_path)
    if not mask_path.exists() or not input_image_path.exists():
        result["boundary_refinement_reason"] = "missing_mask_or_input"
        return result

    ref_img = _as_ref_img(input_image_path, order=1)
    image = np.nan_to_num(ref_img.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    current = _read_mask(mask_path, ref_img)
    current = _clean(current)
    if not current.any():
        result["boundary_refinement_reason"] = "empty_current"
        return result

    reference = None
    if reference_atlas_path and Path(reference_atlas_path).exists():
        reference = _read_mask(reference_atlas_path, ref_img)
    atlas_vote, used_atlases = _atlas_vote([reference_atlas_path, *list(atlas_paths)], ref_img)
    if reference is None:
        reference = atlas_vote

    vox_cm3 = float(np.prod(ref_img.header.get_zooms()[:3]) / 1000.0)
    candidates: dict[str, np.ndarray] = {"current": current}
    if reference is not None and reference.any():
        atlas_env = _dilate(reference, int(cfg["atlas_dilation"]))
        pruned = current & atlas_env
        if pruned.sum() / max(current.sum(), 1) >= 0.70:
            candidates["current_pruned_to_adaptive_atlas"] = _clean(pruned)

    inside = image[current & np.isfinite(image)]
    if inside.size >= 100:
        core = _erode(current, int(cfg["boundary_width"]))
        boundary = current & ~core
        for pct in cfg["trim_percentiles"]:
            thr = float(np.percentile(inside, pct))
            candidates[f"low_intensity_boundary_trim_p{pct:g}"] = _clean(core | (boundary & (image >= thr)))

        if reference is not None and reference.any():
            shell = _dilate(current, int(cfg["shell_width"])) & ~current & _dilate(reference, int(cfg["atlas_dilation"]))
            max_vox = int(float(cfg["recover_max_added_cm3"]) / max(vox_cm3, 1e-9))
            for pct in cfg["recover_percentiles"]:
                thr = float(np.percentile(inside, pct))
                recovery = shell & (image >= thr)
                if recovery.sum() > max_vox > 0:
                    vals = image[recovery]
                    cutoff = np.partition(vals, max(0, vals.size - max_vox))[max(0, vals.size - max_vox)]
                    recovery &= image >= cutoff
                candidates[f"atlas_shell_recovery_p{pct:g}"] = _clean(current | recovery)

    scored = [
        _score_candidate(name, mask, current, reference, image, ref_img, cfg)
        for name, mask in candidates.items()
        if mask.any()
    ]
    if not scored:
        result["boundary_refinement_reason"] = "no_candidates"
        return result
    current_volume = float(mask_stats(current, ref_img)["volume_cm3"])
    reference_volume = float(mask_stats(reference, ref_img)["volume_cm3"]) if reference is not None and reference.any() else 0.0
    current_reference_ratio = current_volume / max(reference_volume, 1e-6) if reference_volume > 0 else 1.0
    for item in scored:
        name = str(item.get("name", ""))
        volume = float(item.get("volume_cm3", current_volume))
        if name.startswith("low_intensity_boundary_trim"):
            item["score"] = float(item["score"]) + float(cfg.get("boundary_trim_bonus", 0.030))
            if current_reference_ratio > float(cfg.get("over_reference_ratio_for_trim_bonus", 1.04)) and volume <= current_volume:
                item["score"] = float(item["score"]) + float(cfg.get("over_reference_trim_bonus", 0.020))
                item["boundary_trim_over_reference_bonus_applied"] = True
        if name == "current" and current_reference_ratio > float(cfg.get("over_reference_ratio_for_trim_bonus", 1.04)):
            item["score"] = float(item["score"]) - float(cfg.get("current_over_reference_penalty", 0.020))
            item["current_over_reference_penalty_applied"] = True

    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    current_item = next((item for item in scored if item["name"] == "current"), scored[0])
    best = scored[0]

    candidate_path = mask_path.with_name(mask_path.name.replace(".nii.gz", f"{cfg['candidate_suffix']}.nii.gz"))
    save_mask(best["mask"], ref_img, candidate_path)

    hard_ok = float(cfg["min_volume_cm3"]) <= float(best["volume_cm3"]) <= float(cfg["max_volume_cm3"])
    current_hard_ok = float(cfg["min_volume_cm3"]) <= current_volume <= float(cfg["max_volume_cm3"])
    reference_trust = float(cfg.get("reference_atlas_trust_score", 0.0) or 0.0)
    low_trust_apply_blocked = (
        best["name"] != "current"
        and reference_trust < float(cfg.get("min_reference_trust_for_apply", 0.12))
        and (
            not bool(cfg.get("low_trust_keep_current_if_hard_ok", True))
            or current_hard_ok
        )
    )
    ref_drop_ok = True
    if reference is not None and reference.any() and best["atlas_dice"] is not None and current_item["atlas_dice"] is not None:
        ref_drop_ok = float(best["atlas_dice"]) >= float(current_item["atlas_dice"]) - float(cfg["max_atlas_dice_drop"])
    should_apply = (
        best["name"] != "current"
        and hard_ok
        and int(best["components"]) <= int(cfg["max_components"])
        and float(best["change_ratio"]) <= float(cfg["max_apply_change_ratio"])
        and float(best["score"]) >= float(current_item["score"]) + float(cfg["min_score_gain_to_apply"])
        and ref_drop_ok
        and not low_trust_apply_blocked
    )

    result.update(
        {
            "boundary_refinement_candidate_path": str(candidate_path),
            "boundary_refinement_selected_candidate": str(best["name"]),
            "boundary_refinement_selected_score": float(best["score"]),
            "boundary_refinement_current_score": float(current_item["score"]),
            "boundary_refinement_candidate_count": int(len(scored)),
            "boundary_refinement_candidate_volume_cm3": float(best["volume_cm3"]),
            "boundary_refinement_change_ratio": float(best["change_ratio"]),
            "boundary_refinement_current_reference_ratio": float(current_reference_ratio),
            "boundary_refinement_reference_atlas_trust_score": float(reference_trust),
            "boundary_refinement_low_trust_apply_blocked": bool(low_trust_apply_blocked),
            "boundary_refinement_current_hard_volume_ok": bool(current_hard_ok),
            "boundary_refinement_used_atlases": used_atlases,
            "boundary_refinement_candidates": [{k: v for k, v in item.items() if k != "mask"} for item in scored],
        }
    )

    if not should_apply:
        result["boundary_refinement_reason"] = (
            "insufficient_external_atlas_support_low_trust"
            if low_trust_apply_blocked
            else "candidate_saved_not_applied"
        )
        _write_qc(mask_path, result)
        if metrics_path:
            update_metrics_json(metrics_path, result)
        return result

    backup = mask_path.with_name(mask_path.name.replace(".nii.gz", f"{cfg['backup_suffix']}.nii.gz"))
    if not backup.exists():
        shutil.copyfile(mask_path, backup)
    save_mask(best["mask"], ref_img, mask_path)
    save_float(image * best["mask"].astype(np.float32), ref_img, brain_path)
    result.update(
        {
            "boundary_refinement_used": True,
            "boundary_refinement_reason": "strict_candidate_applied",
            "boundary_refinement_backup_mask": str(backup),
            "boundary_refinement_volume_after_cm3": float(best["volume_cm3"]),
        }
    )
    _write_qc(mask_path, result)
    if metrics_path:
        update_metrics_json(metrics_path, result)
    return result


def _write_qc(mask_path: Path, result: dict[str, Any]) -> None:
    qc_path = mask_path.with_name(mask_path.name.replace(".nii.gz", "_adult-boundary-refine_qc.json"))
    qc_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    result["boundary_refinement_qc_path"] = str(qc_path)
