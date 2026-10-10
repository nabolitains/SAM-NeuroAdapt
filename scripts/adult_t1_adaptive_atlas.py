"""Adult T1 adaptive external atlas builder without gold labels or LOO data."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi

from .neuro_utils import largest_component, mask_overlap_stats, mask_stats, mask_volume_cm3, save_float, save_mask


def _cfg(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = {
        "enabled": True,
        "min_volume_cm3": 650.0,
        "soft_min_volume_cm3": 850.0,
        "soft_max_volume_cm3": 1300.0,
        "max_volume_cm3": 1900.0,
        "max_components": 6,
        "max_extent_ratio": 0.99,
        "envelope_dilation": 2,
        "boundary_width": 2,
        "shell_width": 8,
        "max_added_cm3": 140.0,
        "max_change_ratio": 0.36,
        "external_volume_floor_enabled": True,
        "external_volume_floor_ratio": 0.88,
        "external_volume_floor_min_cm3": 1250.0,
        "external_volume_floor_max_score_drop": 0.12,
        "selection_margin": 0.012,
        "tie_break_margin": 0.004,
        "tie_break_prefer_lower_volume": True,
        "trim_percentiles": [1.0, 2.0, 4.0, 6.0],
        "tissue_percentiles": [2.0, 5.0, 8.0, 12.0],
        "probability_thresholds": [0.45, 0.50, 0.55, 0.60],
        "save_top_candidates": 6,
        "intensity_weight": 0.32,
        "volume_weight": 0.24,
        "agreement_weight": 0.21,
        "shape_weight": 0.13,
        "seed_weight": 0.10,
    }
    if config:
        cfg.update(dict(config))
    for key in ("trim_percentiles", "tissue_percentiles", "probability_thresholds"):
        cfg[key] = [float(v) for v in cfg.get(key, [])]
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


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")[:64] or "candidate"


def _volume_score(volume_cm3: float, cfg: Mapping[str, Any]) -> float:
    lo = float(cfg["min_volume_cm3"])
    soft_lo = float(cfg["soft_min_volume_cm3"])
    soft_hi = float(cfg["soft_max_volume_cm3"])
    hi = float(cfg["max_volume_cm3"])
    if not np.isfinite(volume_cm3) or volume_cm3 < lo or volume_cm3 > hi:
        return 0.0
    if soft_lo <= volume_cm3 <= soft_hi:
        return 1.0
    if volume_cm3 < soft_lo:
        return float(np.clip((volume_cm3 - lo) / max(soft_lo - lo, 1e-6), 0.0, 1.0))
    return float(np.clip((hi - volume_cm3) / max(hi - soft_hi, 1e-6), 0.0, 1.0))


def _sample(values: np.ndarray, max_samples: int = 200_000) -> np.ndarray:
    if values.size <= max_samples:
        return values
    step = int(np.ceil(values.size / max_samples))
    return values[::step]


def _intensity_score(mask: np.ndarray, image: np.ndarray) -> tuple[float, dict[str, float | None]]:
    if not mask.any():
        return 0.0, {"inside_median": None, "shell_median": None, "contrast_ratio": None}
    finite = np.isfinite(image)
    inside = _sample(image[mask & finite])
    shell = _dilate(mask, 3) & ~mask
    outside = _sample(image[shell & finite])
    if inside.size < 100 or outside.size < 100:
        return 0.5, {"inside_median": None, "shell_median": None, "contrast_ratio": None}
    inside_med = float(np.median(inside))
    outside_med = float(np.median(outside))
    ratio = inside_med / max(outside_med, 1e-6)
    return float(np.clip((ratio - 0.92) / 0.95, 0.0, 1.0)), {
        "inside_median": inside_med,
        "shell_median": outside_med,
        "contrast_ratio": float(ratio),
    }


def _shape_score(mask: np.ndarray, stats: Mapping[str, Any], cfg: Mapping[str, Any]) -> float:
    score = 1.0
    components = int(stats.get("components", 1))
    extent = float(stats.get("extent_ratio", 0.0))
    if components > int(cfg["max_components"]):
        score -= min(0.55, 0.18 * (components - int(cfg["max_components"])))
    if extent > float(cfg["max_extent_ratio"]):
        score -= min(0.45, (extent - float(cfg["max_extent_ratio"])) * 12.0)
    if not mask.any():
        score = 0.0
    return float(np.clip(score, 0.0, 1.0))


def _candidate_metrics(
    name: str,
    mask: np.ndarray,
    seed: np.ndarray,
    external_masks: list[np.ndarray],
    image: np.ndarray,
    ref_img,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    stats = mask_stats(mask, ref_img)
    volume = float(stats["volume_cm3"])
    agreements = [mask_overlap_stats(mask, ext, ref_img)["dice"] for ext in external_masks if ext.any()]
    agreement = float(np.mean(agreements)) if agreements else 0.0
    seed_dice = float(mask_overlap_stats(mask, seed, ref_img)["dice"]) if seed.any() else 0.0
    intensity, intensity_stats = _intensity_score(mask, image)
    change_ratio = float(np.logical_xor(mask, seed).sum() / max(seed.sum(), 1))
    score = (
        float(cfg["intensity_weight"]) * intensity
        + float(cfg["volume_weight"]) * _volume_score(volume, cfg)
        + float(cfg["agreement_weight"]) * agreement
        + float(cfg["shape_weight"]) * _shape_score(mask, stats, cfg)
        + float(cfg["seed_weight"]) * seed_dice
    )
    if change_ratio > float(cfg["max_change_ratio"]):
        score -= min(0.45, (change_ratio - float(cfg["max_change_ratio"])) * 1.5)
    return {
        "name": name,
        "volume_cm3": volume,
        "components": int(stats["components"]),
        "extent_ratio": float(stats.get("extent_ratio", 0.0)),
        "voxels": int(stats.get("voxels", int(mask.sum()))),
        "agreement_score": agreement,
        "seed_dice": seed_dice,
        "volume_score": _volume_score(volume, cfg),
        "shape_score": _shape_score(mask, stats, cfg),
        "intensity_score": intensity,
        "change_ratio": change_ratio,
        "score": float(score),
        **{f"intensity_{k}": v for k, v in intensity_stats.items()},
    }


def _add_candidate(candidates: dict[str, np.ndarray], name: str, mask: np.ndarray, cfg: Mapping[str, Any], ref_img) -> None:
    clean = _clean(mask)
    if not clean.any():
        return
    stats = mask_stats(clean, ref_img)
    if float(stats["volume_cm3"]) < float(cfg["min_volume_cm3"]) or float(stats["volume_cm3"]) > float(cfg["max_volume_cm3"]):
        return
    candidates.setdefault(name, clean)


def _read_external_masks(registered_paths: Mapping[str, Path], ref_img, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for name, path in sorted(registered_paths.items()):
        try:
            mask = _clean(_read_mask(path, ref_img))
            stats = mask_stats(mask, ref_img)
        except Exception as exc:
            items.append({"name": name, "path": str(path), "error": str(exc)})
            continue
        plausible = (
            mask.any()
            and float(cfg["min_volume_cm3"]) <= float(stats["volume_cm3"]) <= float(cfg["max_volume_cm3"])
            and int(stats["components"]) <= int(cfg["max_components"])
            and float(stats.get("extent_ratio", 0.0)) <= float(cfg["max_extent_ratio"])
        )
        item = {"name": name, "path": str(path), "mask": mask, "plausible": plausible, **stats}
        items.append(item)
    return items


def build_adult_t1_adaptive_guidance_atlas(
    registered_paths: Mapping[str, Path],
    input_image_path: str | Path,
    *,
    output_path: str | Path,
    seed_atlas_path: str | Path | None = None,
    consensus_atlas_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build a subject-adaptive guidance atlas from external masks and T1 only."""

    cfg = _cfg(config)
    result: dict[str, Any] = {
        "adaptive_atlas_enabled": bool(cfg.get("enabled", True)),
        "adaptive_atlas_used": False,
        "adaptive_atlas_reason": "disabled" if not cfg.get("enabled", True) else "not_improved",
    }
    if not cfg.get("enabled", True):
        return result

    ref_img = _as_ref_img(input_image_path, order=1)
    image = np.nan_to_num(ref_img.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    output_path = Path(output_path)
    qc_path = output_path.with_name(output_path.name.replace(".nii.gz", "_qc.json"))
    if output_path.exists() and qc_path.exists() and not force:
        try:
            cached = json.loads(qc_path.read_text())
            cached["used_cache"] = True
            return cached
        except Exception:
            pass

    external_items = _read_external_masks(registered_paths, ref_img, cfg)
    external_masks = [np.asarray(item["mask"], dtype=bool) for item in external_items if item.get("plausible") and "mask" in item]
    if consensus_atlas_path and Path(consensus_atlas_path).exists():
        try:
            external_masks.insert(0, _read_mask(consensus_atlas_path, ref_img))
        except Exception:
            pass
    if not external_masks:
        result["adaptive_atlas_reason"] = "no_plausible_external_masks"
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    if seed_atlas_path and Path(seed_atlas_path).exists():
        seed = _read_mask(seed_atlas_path, ref_img)
    elif consensus_atlas_path and Path(consensus_atlas_path).exists():
        seed = _read_mask(consensus_atlas_path, ref_img)
    else:
        seed = external_masks[0]
    seed = _clean(seed)
    stack = np.stack([m.astype(np.float32) for m in external_masks], axis=0)
    probability = np.clip(stack.mean(axis=0), 0.0, 1.0).astype(np.float32)
    envelope = _dilate(np.any(stack >= 0.5, axis=0), int(cfg["envelope_dilation"]))
    envelope = ndi.binary_fill_holes(envelope).astype(bool)
    external_volumes = [mask_volume_cm3(mask, ref_img) for mask in external_masks if mask.any()]
    external_reference_volume = float(np.median(external_volumes)) if external_volumes else 0.0

    candidates: dict[str, np.ndarray] = {}
    _add_candidate(candidates, "seed", seed, cfg, ref_img)
    for thr in cfg["probability_thresholds"]:
        _add_candidate(candidates, f"external_probability_t{thr:g}", probability >= thr, cfg, ref_img)
    for item in external_items:
        if item.get("plausible") and "mask" in item:
            _add_candidate(candidates, f"external_{item['name']}", item["mask"], cfg, ref_img)

    seed_values = image[seed & np.isfinite(image)]
    if seed_values.size >= 100:
        core = _erode(seed, int(cfg["boundary_width"]))
        boundary = seed & ~core
        for pct in cfg["trim_percentiles"]:
            thr = float(np.percentile(seed_values, pct))
            trimmed = core | (boundary & (image >= thr)) | (seed & (probability >= 0.75))
            _add_candidate(candidates, f"seed_boundary_trim_p{pct:g}", trimmed, cfg, ref_img)

        shell = _dilate(seed, int(cfg["shell_width"])) & ~seed & envelope
        max_vox = int(float(cfg["max_added_cm3"]) / max(float(np.prod(ref_img.header.get_zooms()[:3]) / 1000.0), 1e-9))
        for pct in cfg["tissue_percentiles"]:
            thr = float(np.percentile(seed_values, pct))
            tissue = envelope & (image >= thr) & (probability >= 0.35)
            _add_candidate(candidates, f"envelope_tissue_p{pct:g}", tissue, cfg, ref_img)
            recovery = shell & (image >= thr) & (probability >= 0.30)
            if recovery.sum() > max_vox > 0:
                vals = image[recovery]
                cutoff = np.partition(vals, max(0, vals.size - max_vox))[max(0, vals.size - max_vox)]
                recovery &= image >= cutoff
            _add_candidate(candidates, f"seed_shell_recovery_p{pct:g}", seed | recovery, cfg, ref_img)

    scored: list[dict[str, Any]] = []
    for name, mask in candidates.items():
        metrics = _candidate_metrics(name, mask, seed, external_masks, image, ref_img, cfg)
        metrics["mask"] = mask
        scored.append(metrics)
    if not scored:
        result["adaptive_atlas_reason"] = "no_valid_candidates"
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    seed_item = next((item for item in scored if item["name"] == "seed"), scored[0])
    best = scored[0]
    tie_break_selected = False
    if bool(cfg.get("tie_break_prefer_lower_volume", True)):
        near = [
            item
            for item in scored
            if float(item["score"]) >= float(best["score"]) - float(cfg.get("tie_break_margin", 0.004))
            and float(item["volume_cm3"]) >= float(cfg["soft_min_volume_cm3"])
            and float(item["change_ratio"]) <= float(cfg["max_change_ratio"])
        ]
        if len(near) > 1:
            best = min(near, key=lambda item: (float(item["volume_cm3"]), -float(item["score"])))
            tie_break_selected = best["name"] != scored[0]["name"]
    if (
        best["name"] != "seed"
        and not tie_break_selected
        and float(best["score"]) < float(seed_item["score"]) + float(cfg["selection_margin"])
    ):
        best = seed_item
    volume_floor_selected = False
    if (
        bool(cfg.get("external_volume_floor_enabled", True))
        and external_reference_volume >= float(cfg.get("external_volume_floor_min_cm3", 1250.0))
        and float(best["volume_cm3"]) < external_reference_volume * float(cfg.get("external_volume_floor_ratio", 0.88))
    ):
        floor_volume = external_reference_volume * float(cfg.get("external_volume_floor_ratio", 0.88))
        floor_candidates = [
            item
            for item in scored
            if float(item["volume_cm3"]) >= floor_volume
            and float(item["volume_score"]) > 0.0
            and float(item["shape_score"]) >= 0.70
            and float(item["change_ratio"]) <= float(cfg["max_change_ratio"])
        ]
        if floor_candidates:
            floor_best = max(floor_candidates, key=lambda item: float(item["score"]))
            if float(floor_best["score"]) >= float(best["score"]) - float(cfg.get("external_volume_floor_max_score_drop", 0.12)):
                best = floor_best
                volume_floor_selected = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_mask(best["mask"], ref_img, output_path)
    prob_path = output_path.with_name(output_path.name.replace(".nii.gz", "_probability.nii.gz"))
    save_float(probability, ref_img, prob_path)

    candidate_paths: list[str] = []
    for item in scored[: max(1, int(cfg["save_top_candidates"]))]:
        cand_path = output_path.with_name(output_path.name.replace(".nii.gz", f"_candidate-{_safe_name(item['name'])}.nii.gz"))
        save_mask(item["mask"], ref_img, cand_path)
        candidate_paths.append(str(cand_path))

    public_scores = [{k: v for k, v in item.items() if k != "mask"} for item in scored]
    result.update(
        {
            "adaptive_atlas_used": True,
            "adaptive_atlas_reason": "selected_adaptive_candidate" if best["name"] != "seed" else "seed_selected",
            "output_path": str(output_path),
            "adaptive_atlas_path": str(output_path),
            "adaptive_atlas_probability_path": str(prob_path),
            "adaptive_atlas_candidate_paths": candidate_paths,
            "adaptive_atlas_selected_candidate": str(best["name"]),
            "adaptive_atlas_selected_score": float(best["score"]),
            "adaptive_atlas_seed_score": float(seed_item["score"]),
            "adaptive_atlas_volume_cm3": float(best["volume_cm3"]),
            "adaptive_atlas_change_ratio": float(best["change_ratio"]),
            "adaptive_atlas_candidate_count": int(len(scored)),
            "adaptive_atlas_scores": public_scores,
            "adaptive_atlas_external_count": int(len(external_masks)),
            "adaptive_atlas_external_reference_volume_cm3": external_reference_volume,
            "adaptive_atlas_volume_floor_selected": bool(volume_floor_selected),
        }
    )
    qc_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    result["adaptive_atlas_qc_path"] = str(qc_path)
    return result
