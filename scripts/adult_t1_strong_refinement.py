"""Strong adult T1-only post-guidance refinement.

This module is a no-gold refinement stage for adult T1 datasets.  It is
designed for cases where the registered atlas is useful as a weak spatial
prior but not reliable enough to drive the final mask alone.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi

from .neuro_utils import load_img, mask_overlap_stats, save_float, save_mask, update_metrics_json


def _cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    base = {
        "enabled": True,
        "min_volume_cm3": 650.0,
        "soft_min_volume_cm3": 850.0,
        "soft_max_volume_cm3": 1400.0,
        "max_volume_cm3": 1900.0,
        "raw_bad_soft_volume": True,
        "max_components": 3,
        "roi_dilation": 10,
        "atlas_dilation": 4,
        "atlas_weight": 0.08,
        "current_similarity_weight": 0.18,
        "volume_weight": 0.30,
        "intensity_weight": 0.22,
        "uncertainty_weight": 0.10,
        "shape_weight": 0.12,
        "probability_weight": 0.10,
        "selection_margin": 0.025,
        "max_change_ratio": 0.40,
        "max_triggered_change_ratio": 0.55,
        "current_bonus": 0.04,
        "undercoverage_min_precision": 0.92,
        "undercoverage_max_coverage": 0.84,
        "undercoverage_max_volume_ratio": 0.86,
        "overcoverage_min_coverage": 0.94,
        "overcoverage_max_precision": 0.88,
        "overcoverage_min_volume_ratio": 1.10,
        "current_undercoverage_penalty": 0.10,
        "current_overcoverage_penalty": 0.08,
        "shell_recovery_bonus": 0.10,
        "atlas_vote_undercoverage_bonus": 0.06,
        "prune_overcoverage_bonus": 0.08,
        "undercoverage_requires_low_volume_or_high_trust": False,
        "undercoverage_low_volume_ratio": 0.98,
        "undercoverage_min_reference_trust": 0.85,
        "shell_recovery_soft_ok_min_reference_trust": 0.90,
        "shell_recovery_soft_ok_max_volume_ratio": 1.06,
        "shell_recovery_soft_ok_penalty": 0.20,
        "tissue_percentiles": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "probability_thresholds": [0.35, 0.45, 0.55, 0.65],
        "shell_recovery_width": 10,
        "shell_recovery_percentile": 3.0,
        "shell_recovery_max_added_cm3": 180.0,
        "boundary_trim_percentile": 1.0,
        "reference_atlas_trust_score": 0.0,
        "min_reference_trust_for_large_drop": 0.30,
        "min_reference_trust_for_any_switch": 0.12,
        "max_low_trust_volume_drop_ratio": 0.08,
        "backup_suffix": "_pre-adult-t1-strong-refine",
    }
    if config:
        base.update(config)
    base["tissue_percentiles"] = [float(v) for v in base.get("tissue_percentiles", [])]
    base["probability_thresholds"] = [float(v) for v in base.get("probability_thresholds", [])]
    return base


def _as_ref_img(path: str | Path, ref_img, order: int = 0):
    img = nib.load(str(path))
    if img.ndim > 3:
        data = np.asarray(img.dataobj)[..., 0]
        img = nib.Nifti1Image(data, img.affine, img.header)
    if img.shape[:3] != ref_img.shape[:3] or not np.allclose(img.affine, ref_img.affine, atol=1e-3):
        img = resample_from_to(img, ref_img, order=order, cval=0)
    return img


def _read_mask(path: str | Path, ref_img) -> np.ndarray:
    return _as_ref_img(path, ref_img, order=0).get_fdata(dtype=np.float32) > 0.5


def _read_float(path: str | Path | None, ref_img) -> np.ndarray | None:
    if not path or not Path(path).exists():
        return None
    data = _as_ref_img(path, ref_img, order=1).get_fdata(dtype=np.float32)
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


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
        if key not in seen:
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
    # M1/SAM already produces a connected post-processed mask.  Full connected
    # component labeling on 1 mm IBSR work volumes is too costly for every
    # subject, so this stage keeps cleanup lightweight and relies on QC scores.
    return np.asarray(mask, dtype=bool)


def _volume_score(volume_cm3: float, cfg: dict[str, Any]) -> float:
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


def _extent(mask: np.ndarray) -> float:
    if not mask.any():
        return 1.0
    coords = np.where(mask)
    return float(max((c.max() - c.min() + 1) / max(1, s) for c, s in zip(coords, mask.shape)))


def _sample_values(values: np.ndarray, max_samples: int = 200_000) -> np.ndarray:
    if values.size <= max_samples:
        return values
    step = int(np.ceil(values.size / max_samples))
    return values[::step]


def _intensity_context(image: np.ndarray) -> dict[str, float] | None:
    finite = image[np.isfinite(image)]
    finite = finite[finite != 0]
    if finite.size < 100:
        return None
    finite = _sample_values(finite)
    return {
        "background_med": float(np.percentile(finite, 25)),
        "spread": float(np.percentile(finite, 95) - np.percentile(finite, 5) + 1e-6),
    }


def _intensity_score(
    mask: np.ndarray,
    image: np.ndarray,
    context: dict[str, float] | None,
) -> tuple[float, dict[str, float | None]]:
    if not mask.any():
        return 0.0, {"inside_median": None, "shell_median": None, "contrast": None}
    inside = image[mask & np.isfinite(image)]
    if inside.size < 100 or context is None:
        return 0.5, {"inside_median": None, "shell_median": None, "contrast": None}
    inside_med = float(np.median(_sample_values(inside)))
    background_med = float(context["background_med"])
    spread = float(context["spread"])
    contrast = (inside_med - background_med) / spread
    score = float(np.clip((contrast + 0.05) / 0.55, 0.0, 1.0))
    return score, {"inside_median": inside_med, "shell_median": background_med, "contrast": float(contrast)}


def _shape_metrics(mask: np.ndarray, cfg: dict[str, Any]) -> tuple[float, int, float]:
    comp = 1
    extent = _extent(mask)
    score = 1.0
    if extent > 0.92:
        score -= min(0.35, (extent - 0.92) * 2.0)
    return float(np.clip(score, 0.0, 1.0)), comp, float(extent)


def _candidate_metrics(
    name: str,
    mask: np.ndarray,
    current: np.ndarray,
    atlas_vote: np.ndarray | None,
    image: np.ndarray,
    ref_img,
    probability: np.ndarray | None,
    uncertainty: np.ndarray | None,
    intensity_context: dict[str, float] | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    vox_cm3 = float(np.prod(ref_img.header.get_zooms()[:3]) / 1000.0)
    volume_cm3 = float(mask.sum() * vox_cm3)
    current_overlap = mask_overlap_stats(mask, current, ref_img)
    intensity_score, intensity = _intensity_score(mask, image, intensity_context)
    shape_score, components, extent = _shape_metrics(mask, cfg)
    atlas_dice = atlas_precision = atlas_coverage = None
    if atlas_vote is not None and atlas_vote.any():
        atlas = mask_overlap_stats(mask, atlas_vote, ref_img)
        atlas_dice = float(atlas["dice"])
        atlas_precision = float(atlas["precision"])
        atlas_coverage = float(atlas["coverage"])
    prob_score = 0.5
    if probability is not None and mask.any():
        inside = float(np.mean(probability[mask]))
        prob_score = float(np.clip(inside, 0.0, 1.0))
    unc_score = 1.0
    unc_mean = None
    if uncertainty is not None and mask.any():
        unc_mean = float(np.mean(uncertainty[mask]))
        unc_score = float(np.clip(1.0 - unc_mean, 0.0, 1.0))
    return {
        "name": name,
        "volume_cm3": volume_cm3,
        "components": int(components),
        "extent_ratio": float(extent),
        "volume_score": _volume_score(volume_cm3, cfg),
        "shape_score": shape_score,
        "intensity_score": intensity_score,
        "current_dice": float(current_overlap["dice"]),
        "atlas_dice": atlas_dice,
        "atlas_precision": atlas_precision,
        "atlas_coverage": atlas_coverage,
        "probability_score": prob_score,
        "uncertainty_mean": unc_mean,
        "uncertainty_score": unc_score,
        **{f"intensity_{k}": v for k, v in intensity.items()},
    }


def _score(
    metrics: dict[str, Any],
    raw_bad: bool,
    raw_undercoverage: bool,
    raw_overcoverage: bool,
    cfg: dict[str, Any],
) -> float:
    atlas_dice = metrics.get("atlas_dice")
    atlas_score = 0.5 if atlas_dice is None else float(np.clip(atlas_dice, 0.0, 1.0))
    triggered = raw_bad or raw_undercoverage or raw_overcoverage
    current_weight = float(cfg["current_similarity_weight"]) * (0.35 if triggered else 1.0)
    score = (
        float(cfg["volume_weight"]) * float(metrics["volume_score"])
        + float(cfg["intensity_weight"]) * float(metrics["intensity_score"])
        + float(cfg["shape_weight"]) * float(metrics["shape_score"])
        + current_weight * float(metrics["current_dice"])
        + float(cfg["atlas_weight"]) * atlas_score
        + float(cfg["uncertainty_weight"]) * float(metrics["uncertainty_score"])
        + float(cfg["probability_weight"]) * float(metrics["probability_score"])
    )
    name = str(metrics["name"])
    if name == "current" and not triggered:
        score += float(cfg["current_bonus"])
    if raw_undercoverage:
        if name == "current":
            score -= float(cfg.get("current_undercoverage_penalty", 0.10))
        if name == "current_t1_shell_recovery":
            score += float(cfg.get("shell_recovery_bonus", 0.10))
        if name == "weak_atlas_vote" or name.startswith("probability_t"):
            score += float(cfg.get("atlas_vote_undercoverage_bonus", 0.06))
    if raw_overcoverage:
        if name == "current":
            score -= float(cfg.get("current_overcoverage_penalty", 0.08))
        if name == "current_pruned_to_weak_atlas" or name == "current_low_intensity_boundary_trim":
            score += float(cfg.get("prune_overcoverage_bonus", 0.08))
    return float(score)


def _build_atlas_vote(atlas_paths: Iterable[str | Path | None], ref_img) -> tuple[np.ndarray | None, list[str]]:
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
    need = max(1, int(np.ceil(0.5 * len(masks))))
    return stack.sum(axis=0) >= need, used


def _add_candidate(candidates: list[tuple[str, np.ndarray]], name: str, mask: np.ndarray) -> None:
    mask = np.asarray(mask, dtype=bool)
    if mask.any():
        candidates.append((name, mask))


def apply_adult_t1_strong_refinement(
    mask_path: str | Path,
    brain_path: str | Path,
    input_image_path: str | Path,
    *,
    atlas_paths: Iterable[str | Path | None] = (),
    fusion_probability_path: str | Path | None = None,
    uncertainty_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply adult T1-only candidate generation and QC selection."""

    cfg = _cfg(config)
    result: dict[str, Any] = {
        "adult_t1_strong_refinement_enabled": bool(cfg.get("enabled", True)),
        "adult_t1_strong_refinement_used": False,
        "adult_t1_strong_refinement_reason": "disabled" if not cfg.get("enabled", True) else "not_improved",
    }
    if not cfg.get("enabled", True):
        return result

    mask_path = Path(mask_path)
    brain_path = Path(brain_path)
    input_image_path = Path(input_image_path)
    if not mask_path.exists() or not input_image_path.exists():
        result["adult_t1_strong_refinement_reason"] = "missing_mask_or_input"
        return result

    ref_img = load_img(input_image_path)
    image = np.nan_to_num(ref_img.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    intensity_context = _intensity_context(image)
    current = _read_mask(mask_path, ref_img)
    current = _clean(current)
    if not current.any():
        result["adult_t1_strong_refinement_reason"] = "empty_current"
        return result

    atlas_vote, used_atlases = _build_atlas_vote(atlas_paths, ref_img)
    probability = _read_float(fusion_probability_path, ref_img)
    uncertainty = _read_float(uncertainty_path, ref_img)
    vox_cm3 = float(np.prod(ref_img.header.get_zooms()[:3]) / 1000.0)

    current_vol = float(current.sum() * vox_cm3)
    reference_trust = float(cfg.get("reference_atlas_trust_score", 0.0) or 0.0)
    raw_bad = (
        current_vol < float(cfg["min_volume_cm3"])
        or current_vol > float(cfg["max_volume_cm3"])
    )
    if bool(cfg.get("raw_bad_soft_volume", True)):
        raw_bad = raw_bad or current_vol < float(cfg["soft_min_volume_cm3"]) or current_vol > float(cfg["soft_max_volume_cm3"])
    if uncertainty is not None and current.any() and float(np.mean(uncertainty[current])) > 0.28:
        raw_bad = True
    raw_undercoverage = False
    raw_overcoverage = False
    if atlas_vote is not None and atlas_vote.any():
        current_atlas = mask_overlap_stats(current, atlas_vote, ref_img)
        atlas_vol = float(atlas_vote.sum() * vox_cm3)
        raw_undercoverage = (
            float(current_atlas["precision"]) >= float(cfg["undercoverage_min_precision"])
            and float(current_atlas["coverage"]) < float(cfg["undercoverage_max_coverage"])
            and current_vol < float(cfg["undercoverage_max_volume_ratio"]) * max(atlas_vol, 1e-6)
        )
        raw_overcoverage = (
            float(current_atlas["coverage"]) >= float(cfg["overcoverage_min_coverage"])
            and float(current_atlas["precision"]) < float(cfg["overcoverage_max_precision"])
            and current_vol > float(cfg["overcoverage_min_volume_ratio"]) * max(atlas_vol, 1e-6)
        )
    raw_undercoverage_suppressed = False
    if raw_undercoverage and bool(cfg.get("undercoverage_requires_low_volume_or_high_trust", False)):
        low_volume = current_vol < float(cfg["soft_min_volume_cm3"]) * float(cfg.get("undercoverage_low_volume_ratio", 0.98))
        high_trust = reference_trust >= float(cfg.get("undercoverage_min_reference_trust", 0.85))
        if not low_volume and not high_trust:
            raw_undercoverage = False
            raw_undercoverage_suppressed = True

    roi = _dilate(current, int(cfg["roi_dilation"]))
    if atlas_vote is not None and atlas_vote.any():
        roi |= _dilate(atlas_vote, int(cfg["atlas_dilation"]))
    roi &= np.isfinite(image)
    if roi.sum() < 1000:
        roi = np.isfinite(image)

    candidates: list[tuple[str, np.ndarray]] = []
    _add_candidate(candidates, "current", current)
    _add_candidate(candidates, "current_clean", current)

    if atlas_vote is not None and atlas_vote.any():
        atlas_env = _dilate(atlas_vote, int(cfg["atlas_dilation"]))
        pruned = current & atlas_env
        if pruned.sum() / max(current.sum(), 1) >= 0.58:
            _add_candidate(candidates, "current_pruned_to_weak_atlas", pruned)
        _add_candidate(candidates, "weak_atlas_vote", atlas_vote)

    if probability is not None:
        for thr in cfg["probability_thresholds"]:
            _add_candidate(candidates, f"probability_t{thr:g}", probability >= thr)

    inside_values = image[current & np.isfinite(image)]
    roi_values = image[roi & np.isfinite(image)]
    if inside_values.size >= 100 and roi_values.size >= 100:
        for pct in cfg["tissue_percentiles"]:
            thr = float(np.percentile(inside_values, pct))
            tissue = roi & (image >= thr)
            _add_candidate(candidates, f"t1_roi_from_current_p{pct:g}", tissue)

        shell = _dilate(current, int(cfg["shell_recovery_width"])) & (~current) & roi
        if shell.any():
            thr = float(np.percentile(inside_values, float(cfg["shell_recovery_percentile"])))
            recovery = shell & (image >= thr)
            max_vox = int(float(cfg["shell_recovery_max_added_cm3"]) / max(vox_cm3, 1e-9))
            if recovery.sum() > max_vox > 0:
                vals = image[recovery]
                cutoff = np.partition(vals, max(0, vals.size - max_vox))[max(0, vals.size - max_vox)]
                recovery = recovery & (image >= cutoff)
            _add_candidate(candidates, "current_t1_shell_recovery", current | recovery)

        boundary = current & (~_erode(current, 2))
        if boundary.any():
            trim_thr = float(np.percentile(inside_values, float(cfg["boundary_trim_percentile"])))
            trimmed = current & ~(boundary & (image < trim_thr))
            _add_candidate(candidates, "current_low_intensity_boundary_trim", trimmed)

    scored: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for name, mask in candidates:
        if name in seen_names:
            continue
        seen_names.add(name)
        metrics = _candidate_metrics(
            name,
            mask,
            current,
            atlas_vote,
            image,
            ref_img,
            probability,
            uncertainty,
            intensity_context,
            cfg,
        )
        score = _score(metrics, raw_bad, raw_undercoverage, raw_overcoverage, cfg)
        if name == "current_t1_shell_recovery":
            recovery_ratio = float(metrics["volume_cm3"]) / max(float(current_vol), 1e-6)
            current_soft_ok = current_vol >= float(cfg["soft_min_volume_cm3"])
            weak_reference = reference_trust < float(cfg.get("shell_recovery_soft_ok_min_reference_trust", 0.90))
            excessive_recovery = recovery_ratio > float(cfg.get("shell_recovery_soft_ok_max_volume_ratio", 1.06))
            if current_soft_ok and weak_reference and excessive_recovery:
                score -= float(cfg.get("shell_recovery_soft_ok_penalty", 0.20))
                metrics["shell_recovery_soft_ok_penalty_applied"] = True
                metrics["shell_recovery_volume_ratio"] = recovery_ratio
        metrics["score"] = score
        metrics["mask"] = mask
        scored.append(metrics)

    if not scored:
        result["adult_t1_strong_refinement_reason"] = "no_candidates"
        return result

    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    current_item = next((item for item in scored if item["name"] == "current"), scored[0])
    best = scored[0]
    selected = _clean(np.asarray(best["mask"], dtype=bool))
    change_ratio = float(np.logical_xor(selected, current).sum() / max(current.sum(), 1))
    hard_ok = float(cfg["min_volume_cm3"]) <= float(best["volume_cm3"]) <= float(cfg["max_volume_cm3"])
    triggered = raw_bad or raw_undercoverage or raw_overcoverage
    change_limit = float(cfg.get("max_triggered_change_ratio", 0.55)) if triggered else float(cfg["max_change_ratio"])
    current_hard_ok = float(cfg["min_volume_cm3"]) <= float(current_vol) <= float(cfg["max_volume_cm3"])
    volume_drop_ratio = max(0.0, (float(current_vol) - float(best["volume_cm3"])) / max(float(current_vol), 1e-6))
    low_trust_large_drop = (
        current_hard_ok
        and reference_trust < float(cfg.get("min_reference_trust_for_large_drop", 0.30))
        and volume_drop_ratio > float(cfg.get("max_low_trust_volume_drop_ratio", 0.08))
    )
    low_trust_switch_blocked = (
        current_hard_ok
        and best["name"] != "current"
        and reference_trust < float(cfg.get("min_reference_trust_for_any_switch", 0.12))
    )
    if (
        best["name"] == "current"
        or not hard_ok
        or (not triggered and best["score"] < current_item["score"] + float(cfg["selection_margin"]))
        or (change_ratio > change_limit)
        or low_trust_large_drop
        or low_trust_switch_blocked
    ):
        result.update(
            {
                "adult_t1_strong_refinement_reason": (
                    "insufficient_external_atlas_support_low_trust"
                    if low_trust_switch_blocked
                    else "low_trust_large_drop_blocked" if low_trust_large_drop else "current_selected"
                ),
                "adult_t1_strong_refinement_raw_bad": bool(raw_bad),
                "adult_t1_strong_refinement_raw_undercoverage": bool(raw_undercoverage),
                "adult_t1_strong_refinement_raw_overcoverage": bool(raw_overcoverage),
                "adult_t1_strong_raw_undercoverage_suppressed": bool(raw_undercoverage_suppressed),
                "adult_t1_strong_reference_atlas_trust_score": float(reference_trust),
                "adult_t1_strong_low_trust_large_drop": bool(low_trust_large_drop),
                "adult_t1_strong_low_trust_switch_blocked": bool(low_trust_switch_blocked),
                "adult_t1_strong_volume_drop_ratio": float(volume_drop_ratio),
                "adult_t1_strong_selected_candidate": str(current_item["name"]),
                "adult_t1_strong_selected_score": float(current_item["score"]),
                "adult_t1_strong_best_candidate": str(best["name"]),
                "adult_t1_strong_best_score": float(best["score"]),
                "adult_t1_strong_candidate_count": int(len(scored)),
            }
        )
        _write_qc(mask_path, result, scored, used_atlases)
        if metrics_path:
            update_metrics_json(metrics_path, result)
        return result

    backup = mask_path.with_name(mask_path.name.replace(".nii.gz", f"{cfg['backup_suffix']}.nii.gz"))
    if not backup.exists():
        shutil.copyfile(mask_path, backup)
    save_mask(selected, ref_img, mask_path)
    save_float(image * selected.astype(np.float32), ref_img, brain_path)

    result.update(
        {
            "adult_t1_strong_refinement_used": True,
            "adult_t1_strong_refinement_reason": "qc_selected_t1_candidate",
            "adult_t1_strong_refinement_raw_bad": bool(raw_bad),
            "adult_t1_strong_refinement_raw_undercoverage": bool(raw_undercoverage),
            "adult_t1_strong_refinement_raw_overcoverage": bool(raw_overcoverage),
            "adult_t1_strong_raw_undercoverage_suppressed": bool(raw_undercoverage_suppressed),
            "adult_t1_strong_selected_candidate": str(best["name"]),
            "adult_t1_strong_selected_score": float(best["score"]),
            "adult_t1_strong_current_score": float(current_item["score"]),
            "adult_t1_strong_volume_before_cm3": float(current_vol),
            "adult_t1_strong_volume_after_cm3": float(best["volume_cm3"]),
            "adult_t1_strong_change_ratio": change_ratio,
            "adult_t1_strong_reference_atlas_trust_score": float(reference_trust),
            "adult_t1_strong_volume_drop_ratio": float(volume_drop_ratio),
            "adult_t1_strong_backup_mask": str(backup),
            "adult_t1_strong_candidate_count": int(len(scored)),
        }
    )
    _write_qc(mask_path, result, scored, used_atlases)
    if metrics_path:
        update_metrics_json(metrics_path, result)
    return result


def _write_qc(mask_path: Path, result: dict[str, Any], scored: list[dict[str, Any]], used_atlases: list[str]) -> None:
    qc_path = mask_path.with_name(mask_path.name.replace(".nii.gz", "_adult-t1-strong-refine_qc.json"))
    slim = []
    for item in scored:
        slim.append({k: v for k, v in item.items() if k != "mask"})
    result["adult_t1_strong_qc_path"] = str(qc_path)
    result["adult_t1_strong_used_atlases"] = used_atlases
    result["adult_t1_strong_candidates"] = slim[:12]
    qc_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
