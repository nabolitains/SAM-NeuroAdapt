"""Final no-gold adult mask candidate adjudication.

This stage is deliberately limited to masks already produced by the M1
external-atlas pipeline. It never reads SOTA baseline outputs and never reads
gold labels.
"""

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


FORBIDDEN_PATH_TOKENS = (
    "hdbet",
    "hd-bet",
    "fsl",
    "bet_mask",
    "sota_eval",
    "_sota_",
    "/loo",
    "_loo",
    "loo_",
)
SOTA_TOOL_PATH_TOKENS = (
    "mri_synthstrip",
    "synthstrip_standalone",
    "synthstrip.1.pt",
    "synthstrip_results",
    "synthstrip_mask",
    "synthstrip_brain",
    "hdbet_results",
    "hd-bet_results",
    "hd_bet_results",
    "fsl_bet_results",
)


def _cfg(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "enabled": True,
        "min_volume_cm3": 650.0,
        "soft_min_volume_cm3": 850.0,
        "soft_max_volume_cm3": 1700.0,
        "max_volume_cm3": 1900.0,
        "max_components": 6,
        "max_extent_ratio": 0.992,
        "high_uncertainty_threshold": 0.30,
        "soft_volume_trigger": True,
        "atlas_mismatch_min_trust": 0.45,
        "atlas_mismatch_max_dice": 0.90,
        "overcoverage_min_coverage": 0.94,
        "overcoverage_max_precision": 0.90,
        "overcoverage_min_volume_ratio": 1.08,
        "undercoverage_min_precision": 0.94,
        "undercoverage_max_coverage": 0.86,
        "undercoverage_max_volume_ratio": 0.88,
        "mild_overcoverage_min_coverage": 0.985,
        "mild_overcoverage_max_precision": 0.935,
        "mild_overcoverage_min_volume_ratio": 1.04,
        "surface_low_trust_threshold": 0.12,
        "surface_low_trust_min_volume_ratio_to_softmax": 0.92,
        "surface_low_trust_max_atlas_dice": 0.94,
        "surface_boundary_uncertainty_threshold": 0.18,
        "surface_boundary_low_probability_threshold": 0.30,
        "surface_boundary_low_probability_fraction": 0.20,
        "surface_signal_max_reference_trust": 0.40,
        "refinement_regression_risk_enabled": True,
        "refinement_regression_min_volume_change_ratio": 0.03,
        "refinement_regression_max_precision_after": 0.94,
        "refinement_regression_min_coverage_after": 0.94,
        "refinement_regression_max_reference_trust": 0.75,
        "refinement_regression_backup_bonus": 0.34,
        "refinement_regression_pre_refine_bonus": 0.08,
        "refinement_regression_current_prune_penalty": 0.10,
        "refinement_regression_backup_rescue_min_current_dice": 0.90,
        "refinement_regression_backup_rescue_min_atlas_coverage": 0.94,
        "refinement_regression_backup_rescue_min_probability_score": 0.80,
        "refinement_regression_backup_rescue_min_uncertainty_score": 0.85,
        "refinement_regression_backup_min_median_closeness_gain": 0.01,
        "overcoverage_precision_dominant_min_precision": 0.985,
        "overcoverage_precision_dominant_max_coverage": 0.93,
        "overcoverage_precision_dominant_bonus": 0.08,
        "overcoverage_boundary_min_precision": 0.965,
        "overcoverage_boundary_max_current_volume_ratio": 0.98,
        "overcoverage_boundary_penalty": 0.08,
        "high_uncertainty_tissue_recovery_bonus": 0.065,
        "high_uncertainty_tissue_recovery_min_atlas_coverage": 0.70,
        "high_uncertainty_tissue_recovery_min_current_dice": 0.60,
        "high_uncertainty_tissue_recovery_min_uncertainty_score": 0.90,
        "high_uncertainty_tissue_recovery_max_volume_cm3": 1850.0,
        "atlas_rescue_enabled": True,
        "atlas_rescue_min_current_dice": 0.55,
        "atlas_rescue_undercoverage_min_current_dice": 0.40,
        "atlas_rescue_min_precision": 0.985,
        "atlas_rescue_min_coverage": 0.80,
        "atlas_rescue_undercoverage_min_precision": 0.84,
        "atlas_rescue_undercoverage_min_coverage": 0.64,
        "atlas_rescue_relaxed_min_precision": 0.70,
        "atlas_rescue_relaxed_min_coverage": 0.90,
        "atlas_rescue_min_volume_score": 0.45,
        "atlas_rescue_min_intensity_score": 0.45,
        "atlas_rescue_min_probability_score": 0.75,
        "atlas_rescue_undercoverage_min_probability_score": 0.45,
        "atlas_rescue_min_uncertainty_score": 0.75,
        "atlas_rescue_undercoverage_min_uncertainty_score": 0.45,
        "atlas_rescue_overcoverage_max_current_volume_ratio": 0.93,
        "atlas_rescue_bonus": 0.18,
        "external_family_rescue_enabled": True,
        "external_family_rescue_min_pairwise_dice": 0.88,
        "external_family_rescue_min_families": 2,
        "external_family_rescue_min_current_dice": 0.45,
        "external_family_rescue_min_volume_score": 0.70,
        "external_family_rescue_min_intensity_score": 0.45,
        "external_family_rescue_min_probability_score": 0.45,
        "external_family_rescue_min_uncertainty_score": 0.74,
        "external_family_rescue_bonus": 0.30,
        "external_family_rescue_current_prune_penalty": 0.16,
        "stable_external_rescue_enabled": False,
        "stable_external_rescue_min_current_dice": 0.88,
        "stable_external_rescue_min_volume_score": 0.90,
        "stable_external_rescue_min_intensity_score": 0.80,
        "stable_external_rescue_min_probability_score": 0.82,
        "stable_external_rescue_min_uncertainty_score": 0.84,
        "stable_external_rescue_min_atlas_precision": 0.92,
        "stable_external_rescue_min_atlas_coverage": 0.78,
        "stable_external_rescue_max_current_atlas_dice": 0.96,
        "stable_external_rescue_min_volume_ratio_to_current": 0.88,
        "stable_external_rescue_max_volume_ratio_to_current": 0.99,
        "stable_external_rescue_score_drop_tolerance": 0.12,
        "stable_external_rescue_bonus": 0.18,
        "nomask_pair_min_pairwise_dice": 0.88,
        "nomask_pair_min_families": 2,
        "nomask_rescue_min_probability": 0.78,
        "nomask_rescue_min_uncertainty_score": 0.84,
        "nomask_rescue_min_volume_score": 0.70,
        "nomask_pair_strong_pairwise_dice": 0.92,
        "nomask_pair_strong_min_uncertainty_score": 0.93,
        "nomask_pair_strong_min_intensity_score": 0.90,
        "nomask_pair_strong_min_volume_score": 0.95,
        "nomask_pair_min_intensity_score": 0.75,
        "nomask_pair_max_current_volume_ratio": 1.00,
        "nomask_pair_undercoverage_max_current_volume_ratio": 1.35,
        "nomask_pair_max_median_ratio": 1.02,
        "nomask_pair_undercoverage_max_median_ratio": 1.25,
        "nomask_pair_bonus": 0.52,
        "nomask_pair_non_nomask_penalty": 0.38,
        "single_nomask_boundary_bonus": 0.72,
        "single_nomask_boundary_min_probability": 0.86,
        "single_nomask_boundary_min_uncertainty_score": 0.92,
        "single_nomask_boundary_min_intensity_score": 0.90,
        "single_nomask_boundary_min_volume_score": 0.95,
        "single_nomask_boundary_min_rank_score": 0.92,
        "single_nomask_boundary_max_current_volume_ratio": 0.82,
        "single_nomask_boundary_max_median_ratio": 0.94,
        "single_nomask_boundary_score_margin": 0.0,
        "single_nomask_non_nomask_penalty": 0.24,
        "atlas_switch_min_current_dice": 0.86,
        "masked_nomask_conflict_max_pairwise_dice": 0.75,
        "masked_nomask_conflict_penalty": 0.32,
        "compact_atlas_min_probability_score": 0.78,
        "compact_atlas_low_probability_penalty": 0.16,
        "compact_atlas_large_change_current_dice": 0.90,
        "compact_atlas_large_change_penalty": 0.08,
        "compact_consensus_large_change_penalty": 0.22,
        "family_consensus_raw_bad_penalty": 0.08,
        "current_prune_rescue_enabled": True,
        "current_prune_rescue_min_current_dice": 0.965,
        "current_prune_rescue_min_probability_score": 0.84,
        "current_prune_rescue_min_uncertainty_score": 0.84,
        "current_prune_rescue_max_volume_ratio": 0.995,
        "current_prune_rescue_bonus": 0.12,
        "current_prune_rescue_probability_bonus": 0.06,
        "current_prune_rescue_reduction_target": 0.16,
        "current_prune_rescue_reduction_bonus": 0.04,
        "atlas_vote_family_balanced": True,
        "atlas_vote_compact_fraction": 0.55,
        "stability_enabled": True,
        "stability_atlas_disagreement_threshold": 0.30,
        "stability_volume_iqr_ratio_threshold": 0.22,
        "stability_disagreement_safe_ratio_low": 0.88,
        "stability_disagreement_safe_ratio_high": 1.12,
        "stability_overseg_current_to_median_ratio": 1.12,
        "stability_current_atlas_precision_max": 0.94,
        "stability_current_atlas_coverage_min": 0.94,
        "stability_low_trust_reference_threshold": 0.12,
        "stability_surface_volume_ratio_to_softmax": 0.92,
        "stability_current_penalty": 0.18,
        "stability_compact_candidate_bonus": 0.14,
        "stability_prune_candidate_bonus": 0.10,
        "stability_candidate_score_drop_tolerance": 0.04,
        "stability_force_switch_score_drop_tolerance": 0.08,
        "stability_min_noncurrent_score": 0.80,
        "stability_inconclusive_selection_margin": 0.08,
        "min_inconclusive_candidate_current_dice": 0.94,
        "save_generated_candidates": True,
        "generated_candidate_save_limit": 32,
        "current_pruning_enabled": True,
        "atlas_envelope_prune_dilation": 1,
        "current_prune_atlas_dilations": (0, 1, 2, 3),
        "current_prune_probability_quantiles": (10, 20, 30),
        "current_prune_uncertainty_quantiles": (70, 80, 90),
        "current_prune_t1_boundary_quantiles": (5, 10, 15, 20, 25, 35),
        "current_prune_min_volume_ratio": 0.74,
        "current_prune_max_volume_ratio": 0.995,
        "current_prune_bonus": 0.06,
        "current_prune_good_reduction_bonus": 0.05,
        "current_prune_precision_gain_bonus": 0.05,
        "boundary_energy_enabled": True,
        "boundary_energy_inner_width": 2,
        "boundary_energy_outer_width": 3,
        "boundary_energy_candidate_quantiles": (15, 25, 35),
        "boundary_energy_recover_quantiles": (75, 85),
        "boundary_energy_min_volume_ratio": 0.72,
        "boundary_energy_max_volume_ratio": 1.05,
        "boundary_energy_atlas_support_weight": 0.25,
        "surface_current_penalty": 0.06,
        "min_mild_overcoverage_candidate_current_dice": 0.955,
        "min_surface_candidate_current_dice": 0.985,
        "min_current_prune_triggered_candidate_current_dice": 0.90,
        "stable_selection_margin": 0.05,
        "triggered_selection_margin": 0.015,
        "current_stable_bonus": 0.08,
        "current_trigger_penalty": 0.14,
        "high_uncertainty_current_penalty": 0.22,
        "min_stable_candidate_current_dice": 0.98,
        "min_triggered_candidate_current_dice": 0.74,
        "min_high_uncertainty_candidate_current_dice": 0.45,
        "min_raw_bad_candidate_current_dice": 0.80,
        "min_candidate_intensity_score": 0.05,
        "candidate_globs": (
            "sub-*desc-brain_mask.nii.gz",
            "sub-*desc-brain_mask_pre-*.nii.gz",
            "sub-*adult-boundary-candidate.nii.gz",
            "atlases_advanced/external_adult_t1_adaptive_atlas*.nii.gz",
            "atlases_advanced/sub-*brain_mask*.nii.gz",
            "atlases_advanced/_cleaned_atlases/*.nii.gz",
        ),
        "backup_suffix": "_pre-adult-candidate-adjudicator",
    }
    if config:
        cfg.update(dict(config))
    cfg["candidate_globs"] = tuple(str(v) for v in cfg.get("candidate_globs", ()))
    for key in (
        "current_prune_atlas_dilations",
        "current_prune_probability_quantiles",
        "current_prune_uncertainty_quantiles",
        "current_prune_t1_boundary_quantiles",
        "boundary_energy_candidate_quantiles",
        "boundary_energy_recover_quantiles",
    ):
        cfg[key] = tuple(cfg.get(key, ()))
    return cfg


def _is_truthy(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(np.isnan(value)):  # type: ignore[arg-type]
            return False
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ok", "oui"}


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return default if np.isnan(out) else out
    except Exception:
        return default


def _safe_name(path_or_name: str | Path) -> str:
    text = Path(path_or_name).name if not isinstance(path_or_name, str) or "/" in str(path_or_name) else str(path_or_name)
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")[:96] or "candidate"


def _forbidden(path: str | Path | None) -> bool:
    if not path:
        return False
    text = str(path).replace("\\", "/").lower()
    parts = [part for part in text.split("/") if part]
    if any(part in {"hdbet", "hd-bet", "hd_bet", "fsl", "loo"} for part in parts):
        return True
    if any(token in text for token in SOTA_TOOL_PATH_TOKENS):
        return True
    return any(token in text for token in FORBIDDEN_PATH_TOKENS)


def _looks_like_mask_candidate(path: Path) -> bool:
    name = path.name.lower()
    text = str(path).replace("\\", "/").lower()
    if _forbidden(path):
        return False
    blocked_always = (
        "uncertainty",
        "warp",
        "inversewarp",
        "transform",
        "desc-n4",
        "desc-ws",
        "desc-brain_t1w",
        "_brain_t1w",
        ".json",
    )
    if any(token in name for token in blocked_always):
        return False
    if "_t1w" in name and "mask" not in name:
        return False
    if "adaptive-fusion_probability" in name:
        return False
    if "external_adult_t1_adaptive_atlas_probability" in name:
        return False
    return path.suffix == ".gz" or path.suffix == ".nii" or text.endswith(".nii.gz")


def _as_ref_img(path: str | Path, ref_img=None, order: int = 0) -> nib.Nifti1Image:
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
    img = _as_ref_img(path, ref_img, order=0)
    return img.get_fdata(dtype=np.float32) > 0.5


def _read_float(path: str | Path | None, ref_img) -> np.ndarray | None:
    if not path or _forbidden(path) or not Path(path).exists():
        return None
    img = _as_ref_img(path, ref_img, order=1)
    return np.nan_to_num(img.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _clean(mask: np.ndarray) -> np.ndarray:
    mask = ndi.binary_fill_holes(np.asarray(mask, dtype=bool)).astype(bool)
    return largest_component(mask)


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    iterations = max(0, int(iterations))
    if iterations <= 0:
        return np.asarray(mask, dtype=bool)
    return ndi.binary_dilation(mask, structure=ndi.generate_binary_structure(3, 1), iterations=iterations)


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    iterations = max(0, int(iterations))
    if iterations <= 0:
        return np.asarray(mask, dtype=bool)
    return ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1), iterations=iterations)


def _inner_boundary(mask: np.ndarray, width: int = 1) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask & ~_erode(mask, max(1, int(width)))


def _sample(values: np.ndarray, max_samples: int = 200_000) -> np.ndarray:
    if values.size <= max_samples:
        return values
    return values[:: max(1, int(np.ceil(values.size / max_samples)))]


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
    shell_med = float(np.median(outside))
    ratio = inside_med / max(shell_med, 1e-6)
    return float(np.clip((ratio - 0.98) / 0.20, 0.0, 1.0)), {
        "inside_median": inside_med,
        "shell_median": shell_med,
        "contrast_ratio": float(ratio),
    }


def _candidate_kind(name: str, path: str | Path | None) -> str:
    text = f"{name} {path or ''}".lower()
    if "desc-brain_mask.nii" in text and "pre-" not in text:
        return "current"
    if "pre-" in text or "_pre_" in text:
        return "backup"
    if "boundary_energy_prune" in text:
        return "current_prune"
    if "boundary_energy" in text:
        return "boundary"
    if "family_consensus_compact" in text:
        return "atlas"
    if "atlas_envelope_prune" in text:
        return "current_prune"
    if "boundary_trim" in text or "boundary-candidate" in text:
        return "boundary"
    if "current_" in text and ("prune" in text or "trim" in text or "smooth" in text):
        return "current_prune"
    if "tissue" in text:
        return "tissue"
    if "cleaned_atlases" in text or "brain_mask_" in text:
        return "atlas"
    return "candidate"


def _collect_path_candidates(
    candidate_paths: Iterable[str | Path | None],
    candidate_search_dirs: Iterable[str | Path | None],
    cfg: Mapping[str, Any],
) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()

    def add(path_like: str | Path | None) -> None:
        if not path_like:
            return
        path = Path(path_like)
        if not path.exists() or not path.is_file() or not _looks_like_mask_candidate(path):
            return
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)

    for item in candidate_paths:
        add(item)
    for root_like in candidate_search_dirs:
        if not root_like:
            continue
        root = Path(root_like)
        if not root.exists() or not root.is_dir():
            continue
        for pattern in cfg.get("candidate_globs", ()):
            for path in root.glob(str(pattern)):
                add(path)
    return out


def _atlas_family(path: str | Path) -> str:
    text = str(path).replace("\\", "/").lower()
    name = Path(path).name.lower()
    if "external_atlas_consensus" in name:
        return "consensus"
    if "mni152" in text:
        return "mni152"
    if "oasis30ants" in text:
        return "oasis30ants"
    if "sri24" in text:
        return "sri24"
    if "colin27" in text:
        return "colin27"
    if "icbm" in text:
        return "icbm"
    return "other_external"


def _atlas_variant(name_or_path: str | Path | None) -> str:
    text = str(name_or_path or "").replace("\\", "/").lower()
    name = Path(text).name
    if "family_consensus_compact" in text or "external_atlas_consensus" in name:
        return "consensus"
    if "tissue_p" in text or "_tissue_" in text:
        return "tissue"
    if "nomask" in text:
        return "nomask"
    if any(token in text for token in ("mni152", "oasis30ants", "sri24", "colin27", "icbm")):
        return "masked"
    return "other"


def _candidate_family_variant(item: Mapping[str, Any]) -> tuple[str, str]:
    path = item.get("path")
    path_name = "" if path is None else Path(str(path)).name
    text = f"{item.get('name', '')} {path_name}"
    family = _atlas_family(text)
    variant = _atlas_variant(text)
    return family, variant


def _is_external_atlas_item(item: Mapping[str, Any]) -> bool:
    if bool(item.get("is_current")):
        return False
    path = item.get("path")
    path_name = "" if path is None else Path(str(path)).name
    text = f"{item.get('name', '')} {path_name}".lower()
    kind = str(item.get("kind", "")).lower()
    if kind == "atlas" or "family_consensus_compact" in text:
        return True
    return any(
        token in text
        for token in (
            "external_atlas",
            "external_adult_t1_adaptive_atlas",
            "adaptive_atlas",
            "mni152",
            "oasis30ants",
            "sri24",
            "colin27",
            "icbm",
        )
    )


def _pairwise_mask_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    if denom <= 0:
        return 0.0
    return float(2.0 * int(np.logical_and(a, b).sum()) / denom)


def _vote_path_allowed(path: str | Path) -> bool:
    """Keep the QC atlas vote independent from current-mask derivatives."""

    text = str(path).replace("\\", "/").lower()
    name = Path(path).name.lower()
    if _forbidden(path) or "/atlases_advanced/" not in text:
        return False
    if "external_atlas_consensus" in name:
        return True
    if "/_cleaned_atlases/" in text and "brain_mask" in name:
        return True
    if "sub-" in name and "brain_mask" in name and any(
        token in name for token in ("mni152", "oasis30ants", "sri24", "colin27", "icbm")
    ):
        return True
    return False


def _mask_quality_score(
    mask: np.ndarray,
    ref_img,
    image: np.ndarray | None,
    probability: np.ndarray | None,
    uncertainty: np.ndarray | None,
    cfg: Mapping[str, Any],
) -> tuple[float, dict[str, float | None]]:
    stats = mask_stats(mask, ref_img)
    volume_score = _volume_score(float(stats["volume_cm3"]), cfg)
    intensity_score = 0.5
    if image is not None:
        intensity_score, _ = _intensity_score(mask, image)
    probability_score = 0.5
    if probability is not None and mask.any():
        probability_score = float(np.clip(np.mean(probability[mask]), 0.0, 1.0))
    uncertainty_score = 1.0
    if uncertainty is not None and mask.any():
        uncertainty_score = float(np.clip(1.0 - float(np.mean(uncertainty[mask])), 0.0, 1.0))
    shape_score = (
        1.0
        if int(stats["components"]) <= int(cfg["max_components"])
        and float(stats.get("extent_ratio", 0.0)) <= float(cfg["max_extent_ratio"])
        else 0.25
    )
    score = (
        0.34 * volume_score
        + 0.24 * intensity_score
        + 0.18 * probability_score
        + 0.14 * uncertainty_score
        + 0.10 * shape_score
    )
    return float(score), {
        "volume_cm3": float(stats["volume_cm3"]),
        "volume_score": float(volume_score),
        "intensity_score": float(intensity_score),
        "probability_score": float(probability_score),
        "uncertainty_score": float(uncertainty_score),
        "shape_score": float(shape_score),
    }


def _atlas_vote(
    atlas_paths: Iterable[str | Path | None],
    ref_img,
    image: np.ndarray | None = None,
    probability: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray | None, list[str], dict[str, Any]]:
    cfg = _cfg(cfg)
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in atlas_paths:
        if not item or not _vote_path_allowed(item):
            continue
        path = Path(item)
        if not path.exists() or not path.is_file():
            continue
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            mask = _read_mask(path, ref_img)
        except Exception:
            continue
        if mask.any():
            score, quality = _mask_quality_score(mask, ref_img, image, probability, uncertainty, cfg)
            if "nomask" in path.name.lower():
                score -= 0.02
            if "external_atlas_consensus" in path.name.lower():
                score += 0.03
            items.append(
                {
                    "path": str(path),
                    "family": _atlas_family(path),
                    "mask": mask,
                    "score": float(score),
                    **quality,
                }
            )
    if not items:
        return None, [], {
            "stability_reference_policy": "no_external_vote",
            "atlas_vote_count": 0,
            "atlas_vote_family_count": 0,
        }

    selected_items: list[dict[str, Any]] = []
    if bool(cfg.get("atlas_vote_family_balanced", True)):
        by_family: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            by_family.setdefault(str(item["family"]), []).append(item)
        for family_items in by_family.values():
            selected_items.append(max(family_items, key=lambda value: float(value["score"])))
    else:
        selected_items = items
    selected_items = sorted(selected_items, key=lambda value: float(value["score"]), reverse=True)
    masks = [np.asarray(item["mask"], dtype=bool) for item in selected_items]
    used = [str(item["path"]) for item in selected_items]
    stack = np.stack([m.astype(np.uint8) for m in masks], axis=0)
    compact_fraction = float(cfg.get("atlas_vote_compact_fraction", 0.55))
    need = max(1, int(np.ceil(compact_fraction * len(masks))))
    vote = stack.sum(axis=0) >= need
    if not vote.any():
        vote = np.any(stack > 0, axis=0)

    volumes = np.asarray([float(item["volume_cm3"]) for item in selected_items], dtype=np.float64)
    median = float(np.median(volumes)) if volumes.size else float("nan")
    q25 = float(np.percentile(volumes, 25)) if volumes.size else float("nan")
    q75 = float(np.percentile(volumes, 75)) if volumes.size else float("nan")
    iqr = float(q75 - q25) if volumes.size else float("nan")
    summary = {
        "stability_reference_policy": "family_balanced_compact_vote"
        if bool(cfg.get("atlas_vote_family_balanced", True))
        else "compact_vote",
        "atlas_vote_count": int(len(items)),
        "atlas_vote_family_count": int(len({str(item["family"]) for item in selected_items})),
        "stability_atlas_volume_median_cm3": None if not np.isfinite(median) else median,
        "stability_atlas_volume_iqr": None if not np.isfinite(iqr) else iqr,
        "stability_atlas_volume_iqr_ratio": None
        if not np.isfinite(iqr) or not np.isfinite(median) or median <= 0.0
        else float(iqr / median),
        "stability_atlas_vote_volume_cm3": float(mask_stats(vote, ref_img)["volume_cm3"]),
        "stability_atlas_vote_used": used,
        "stability_atlas_vote_items": [
            {k: v for k, v in item.items() if k != "mask"} for item in selected_items
        ],
    }
    return vote, used, summary


def _generated_current_pruning_candidates(
    current: np.ndarray,
    atlas_vote: np.ndarray | None,
    image: np.ndarray,
    probability: np.ndarray | None,
    uncertainty: np.ndarray | None,
    cfg: Mapping[str, Any],
) -> list[tuple[str, np.ndarray]]:
    if not bool(cfg.get("current_pruning_enabled", True)) or not current.any():
        return []

    current = np.asarray(current, dtype=bool)
    min_ratio = float(cfg.get("current_prune_min_volume_ratio", 0.74))
    max_ratio = float(cfg.get("current_prune_max_volume_ratio", 0.995))
    boundary1 = _inner_boundary(current, 1)
    boundary2 = _inner_boundary(current, 2)
    candidates: list[tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()

    def add(name: str, mask: np.ndarray) -> None:
        mask = _clean(mask)
        if not mask.any():
            return
        ratio = float(mask.sum() / max(current.sum(), 1))
        if ratio < min_ratio or ratio > max_ratio:
            return
        key = np.packbits(mask.ravel()).tobytes()
        if key in seen:
            return
        seen.add(key)
        candidates.append((name, mask))

    if atlas_vote is not None and atlas_vote.any():
        envelope_pad = int(cfg.get("atlas_envelope_prune_dilation", 1))
        add(f"atlas_envelope_prune_d{envelope_pad}", current & _dilate(atlas_vote, envelope_pad))
        for pad in cfg.get("current_prune_atlas_dilations", ()):
            pad_i = int(pad)
            support = _dilate(atlas_vote, pad_i)
            add(f"current_atlas_prune_d{pad_i}", current & support)

    if probability is not None and boundary1.any():
        vals = probability[boundary2 if boundary2.any() else boundary1]
        vals = vals[np.isfinite(vals)]
        if vals.size >= 100:
            for q in cfg.get("current_prune_probability_quantiles", ()):
                qf = float(q)
                threshold = float(np.percentile(vals, qf))
                suspect = boundary2 & (probability <= threshold)
                if atlas_vote is not None and atlas_vote.any():
                    suspect |= boundary1 & ~_dilate(atlas_vote, 1) & (probability <= np.percentile(vals, min(75.0, qf + 35.0)))
                add(f"current_probability_prune_q{qf:g}", current & ~suspect)

    if uncertainty is not None and boundary1.any():
        vals = uncertainty[boundary2 if boundary2.any() else boundary1]
        vals = vals[np.isfinite(vals)]
        if vals.size >= 100:
            for q in cfg.get("current_prune_uncertainty_quantiles", ()):
                qf = float(q)
                threshold = float(np.percentile(vals, qf))
                suspect = boundary2 & (uncertainty >= threshold)
                if atlas_vote is not None and atlas_vote.any():
                    suspect |= boundary1 & ~_dilate(atlas_vote, 1) & (uncertainty >= np.percentile(vals, max(25.0, qf - 35.0)))
                add(f"current_uncertainty_prune_q{qf:g}", current & ~suspect)

    finite = np.isfinite(image)
    if boundary1.any() and np.any(boundary2 & finite):
        vals = image[(boundary2 if boundary2.any() else boundary1) & finite]
        vals = vals[np.isfinite(vals)]
        if vals.size >= 100:
            for q in cfg.get("current_prune_t1_boundary_quantiles", ()):
                qf = float(q)
                threshold = float(np.percentile(vals, qf))
                suspect = boundary2 & finite & (image <= threshold)
                if atlas_vote is not None and atlas_vote.any():
                    suspect |= boundary1 & ~_dilate(atlas_vote, 1) & finite & (image <= np.percentile(vals, min(60.0, qf + 25.0)))
                add(f"current_t1_prune_q{qf:g}", current & ~suspect)

    opened = ndi.binary_opening(current, structure=ndi.generate_binary_structure(3, 1), iterations=1)
    add("current_surface_smooth_open1", opened)
    return candidates


def _generated_boundary_energy_candidates(
    current: np.ndarray,
    atlas_vote: np.ndarray | None,
    image: np.ndarray,
    probability: np.ndarray | None,
    uncertainty: np.ndarray | None,
    cfg: Mapping[str, Any],
) -> list[tuple[str, np.ndarray]]:
    """Create no-gold boundary candidates from local image/support evidence.

    The candidate generator stays inside the existing M1 workflow.  It only
    uses the current mask, external atlas vote, SAM probability/uncertainty
    maps, and the preprocessed T1 image.
    """

    if not bool(cfg.get("boundary_energy_enabled", True)) or not current.any():
        return []

    current = np.asarray(current, dtype=bool)
    inner_width = max(1, int(cfg.get("boundary_energy_inner_width", 2)))
    outer_width = max(1, int(cfg.get("boundary_energy_outer_width", 3)))
    min_ratio = float(cfg.get("boundary_energy_min_volume_ratio", 0.72))
    max_ratio = float(cfg.get("boundary_energy_max_volume_ratio", 1.05))
    inner = _inner_boundary(current, inner_width)
    outside_shell = _dilate(current, outer_width) & ~current
    if not inner.any() and not outside_shell.any():
        return []

    finite = np.isfinite(image)
    roi = (_dilate(current, max(outer_width + 2, 4)) | current) & finite
    vals = image[roi]
    vals = vals[np.isfinite(vals)]
    if vals.size >= 100:
        lo, hi = np.percentile(_sample(vals), [2.0, 98.0])
        intensity = np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        intensity = np.zeros_like(image, dtype=np.float32) + 0.5

    support = 0.42 * intensity.astype(np.float32)
    if probability is not None:
        support += 0.23 * np.clip(probability.astype(np.float32), 0.0, 1.0)
    else:
        support += 0.23 * 0.5
    if uncertainty is not None:
        support += 0.18 * np.clip(1.0 - uncertainty.astype(np.float32), 0.0, 1.0)
    else:
        support += 0.18 * 0.9
    if atlas_vote is not None and atlas_vote.any():
        atlas_support = _dilate(atlas_vote, 1).astype(np.float32)
        support += float(cfg.get("boundary_energy_atlas_support_weight", 0.25)) * atlas_support
    support = np.nan_to_num(support, nan=0.0, posinf=0.0, neginf=0.0)

    candidates: list[tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()

    def add(name: str, mask: np.ndarray) -> None:
        mask = _clean(mask)
        if not mask.any():
            return
        ratio = float(mask.sum() / max(current.sum(), 1))
        if ratio < min_ratio or ratio > max_ratio:
            return
        key = np.packbits(mask.ravel()).tobytes()
        if key in seen:
            return
        seen.add(key)
        candidates.append((name, mask))

    inner_vals = support[inner]
    inner_vals = inner_vals[np.isfinite(inner_vals)]
    if inner.any() and inner_vals.size >= 100:
        for q in cfg.get("boundary_energy_candidate_quantiles", ()):
            qf = float(q)
            threshold = float(np.percentile(inner_vals, qf))
            suspect = inner & (support <= threshold)
            if atlas_vote is not None and atlas_vote.any():
                atlas_shell_miss = inner & ~_dilate(atlas_vote, 1)
                suspect |= atlas_shell_miss & (support <= float(np.percentile(inner_vals, min(80.0, qf + 40.0))))
            add(f"boundary_energy_prune_q{qf:g}", current & ~suspect)

    shell_vals = support[outside_shell]
    shell_vals = shell_vals[np.isfinite(shell_vals)]
    if outside_shell.any() and shell_vals.size >= 100:
        atlas_shell = _dilate(atlas_vote, 1) if atlas_vote is not None and atlas_vote.any() else None
        for q in cfg.get("boundary_energy_recover_quantiles", ()):
            qf = float(q)
            threshold = float(np.percentile(shell_vals, qf))
            recover = outside_shell & (support >= threshold)
            if atlas_shell is not None:
                recover &= atlas_shell
            add(f"boundary_energy_recover_q{qf:g}", current | recover)
            if inner_vals.size >= 100:
                prune_threshold = float(np.percentile(inner_vals, max(5.0, 100.0 - qf)))
                suspect = inner & (support <= prune_threshold)
                if atlas_shell is not None:
                    suspect |= inner & ~atlas_shell & (support <= float(np.percentile(inner_vals, 60.0)))
                add(f"boundary_energy_balanced_q{qf:g}", (current & ~suspect) | recover)

    return candidates


def _trigger_state(
    current: np.ndarray,
    ref_img,
    atlas_vote: np.ndarray | None,
    uncertainty: np.ndarray | None,
    context: Mapping[str, Any],
    atlas_summary: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    atlas_summary = dict(atlas_summary or {})
    vox_cm3 = float(np.prod(ref_img.header.get_zooms()[:3]) / 1000.0)
    current_vol = float(current.sum() * vox_cm3)
    atlas_vol = float(atlas_vote.sum() * vox_cm3) if atlas_vote is not None and atlas_vote.any() else float("nan")
    atlas_volume_median = _to_float(atlas_summary.get("stability_atlas_volume_median_cm3"))
    atlas_volume_iqr = _to_float(atlas_summary.get("stability_atlas_volume_iqr"))
    atlas_volume_iqr_ratio = _to_float(atlas_summary.get("stability_atlas_volume_iqr_ratio"))
    current_volume_ratio_to_median = (
        float(current_vol / max(atlas_volume_median, 1e-6))
        if np.isfinite(atlas_volume_median) and atlas_volume_median > 0.0
        else float("nan")
    )
    current_unc = _to_float(context.get("adaptive_fusion_current_uncertainty_mean"))
    if not np.isfinite(current_unc) and uncertainty is not None and current.any():
        current_unc = float(np.mean(uncertainty[current]))

    atlas_dice = _to_float(context.get("adaptive_fusion_current_atlas_dice"))
    atlas_precision = _to_float(context.get("adaptive_fusion_current_atlas_precision"))
    atlas_coverage = _to_float(context.get("adaptive_fusion_current_atlas_coverage"))
    vote_atlas_dice = float("nan")
    vote_atlas_precision = float("nan")
    vote_atlas_coverage = float("nan")
    if atlas_vote is not None and atlas_vote.any():
        overlap = mask_overlap_stats(current, atlas_vote, ref_img)
        vote_atlas_dice = float(overlap["dice"])
        vote_atlas_precision = float(overlap["precision"])
        vote_atlas_coverage = float(overlap["coverage"])
        if not np.isfinite(atlas_dice):
            atlas_dice = vote_atlas_dice
        if not np.isfinite(atlas_precision):
            atlas_precision = vote_atlas_precision
        if not np.isfinite(atlas_coverage):
            atlas_coverage = vote_atlas_coverage

    raw_bad = _is_truthy(context.get("adaptive_fusion_raw_bad")) or _is_truthy(
        context.get("adult_t1_strong_refinement_raw_bad")
    )
    raw_under = _is_truthy(context.get("adaptive_fusion_raw_undercoverage")) or _is_truthy(
        context.get("adult_t1_strong_refinement_raw_undercoverage")
    )
    raw_over = _is_truthy(context.get("adaptive_fusion_raw_overcoverage")) or _is_truthy(
        context.get("adult_t1_strong_refinement_raw_overcoverage")
    )
    raw_mismatch = _is_truthy(context.get("adaptive_fusion_raw_atlas_mismatch"))

    volume_hard_bad = current_vol < float(cfg["min_volume_cm3"]) or current_vol > float(cfg["max_volume_cm3"])
    volume_soft_bad = bool(cfg.get("soft_volume_trigger", True)) and (
        current_vol < float(cfg["soft_min_volume_cm3"]) or current_vol > float(cfg["soft_max_volume_cm3"])
    )
    high_uncertainty = np.isfinite(current_unc) and current_unc >= float(cfg["high_uncertainty_threshold"])
    reference_trust = _to_float(context.get("adaptive_fusion_reference_atlas_trust_score"), 0.0)
    mismatch_values = [value for value in (atlas_dice,) if np.isfinite(value)]
    if (
        not raw_mismatch
        and mismatch_values
        and reference_trust >= float(cfg.get("atlas_mismatch_min_trust", 0.45))
        and min(mismatch_values) < float(cfg.get("atlas_mismatch_max_dice", 0.90))
    ):
        raw_mismatch = True

    overcoverage_pairs = [
        (atlas_coverage, atlas_precision),
        (vote_atlas_coverage, vote_atlas_precision),
    ]
    if not raw_over and atlas_vote is not None and np.isfinite(atlas_vol):
        raw_over = any(
            np.isfinite(cov)
            and np.isfinite(prec)
            and cov >= float(cfg["overcoverage_min_coverage"])
            and prec < float(cfg["overcoverage_max_precision"])
            and current_vol > float(cfg["overcoverage_min_volume_ratio"]) * max(atlas_vol, 1e-6)
            for cov, prec in overcoverage_pairs
        )

    undercoverage_pairs = [
        (atlas_coverage, atlas_precision),
        (vote_atlas_coverage, vote_atlas_precision),
    ]
    if not raw_under and atlas_vote is not None and np.isfinite(atlas_vol):
        raw_under = any(
            np.isfinite(cov)
            and np.isfinite(prec)
            and prec >= float(cfg["undercoverage_min_precision"])
            and cov < float(cfg["undercoverage_max_coverage"])
            and current_vol < float(cfg["undercoverage_max_volume_ratio"]) * max(atlas_vol, 1e-6)
            for cov, prec in undercoverage_pairs
        )

    mild_overcoverage = False
    if atlas_vote is not None and np.isfinite(atlas_vol):
        mild_overcoverage = any(
            np.isfinite(cov)
            and np.isfinite(prec)
            and cov >= float(cfg.get("mild_overcoverage_min_coverage", 0.985))
            and prec <= float(cfg.get("mild_overcoverage_max_precision", 0.935))
            and current_vol > float(cfg.get("mild_overcoverage_min_volume_ratio", 1.04)) * max(atlas_vol, 1e-6)
            for cov, prec in overcoverage_pairs
        )

    boundary_unc = float("nan")
    boundary_low_probability_fraction = float("nan")
    boundary = _inner_boundary(current, 2)
    if boundary.any() and uncertainty is not None:
        vals = uncertainty[boundary]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            boundary_unc = float(np.mean(vals))
    probability = context.get("_candidate_adjudicator_probability")
    if boundary.any() and isinstance(probability, np.ndarray):
        vals = probability[boundary]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            boundary_low_probability_fraction = float(
                np.mean(vals <= float(cfg.get("surface_boundary_low_probability_threshold", 0.30)))
            )
    low_trust_surface = (
        reference_trust <= float(cfg.get("surface_low_trust_threshold", 0.12))
        and (
            current_vol
            >= float(cfg.get("surface_low_trust_min_volume_ratio_to_softmax", 0.92))
            * float(cfg["soft_max_volume_cm3"])
            or (np.isfinite(atlas_dice) and atlas_dice <= float(cfg.get("surface_low_trust_max_atlas_dice", 0.94)))
            or (
                np.isfinite(vote_atlas_dice)
                and vote_atlas_dice <= float(cfg.get("surface_low_trust_max_atlas_dice", 0.94))
            )
        )
    )
    uncertain_surface = np.isfinite(boundary_unc) and boundary_unc >= float(
        cfg.get("surface_boundary_uncertainty_threshold", 0.18)
    )
    low_probability_surface = np.isfinite(boundary_low_probability_fraction) and boundary_low_probability_fraction >= float(
        cfg.get("surface_boundary_low_probability_fraction", 0.20)
    )
    refine_before = _to_float(context.get("adult_refinement_volume_before_cm3"))
    refine_after = _to_float(context.get("adult_refinement_volume_after_cm3"))
    refine_precision_after = _to_float(context.get("adult_refinement_atlas_precision_after"))
    refine_coverage_after = _to_float(context.get("adult_refinement_atlas_coverage_after"))
    refine_change_ratio = (
        abs(refine_after - refine_before) / max(refine_before, 1e-6)
        if np.isfinite(refine_before) and np.isfinite(refine_after) and refine_before > 0.0
        else float("nan")
    )
    refinement_regression_risk = (
        bool(cfg.get("refinement_regression_risk_enabled", True))
        and _is_truthy(context.get("adult_refinement_used"))
        and np.isfinite(refine_change_ratio)
        and refine_change_ratio >= float(cfg.get("refinement_regression_min_volume_change_ratio", 0.03))
        and np.isfinite(refine_precision_after)
        and refine_precision_after <= float(cfg.get("refinement_regression_max_precision_after", 0.94))
        and np.isfinite(refine_coverage_after)
        and refine_coverage_after >= float(cfg.get("refinement_regression_min_coverage_after", 0.94))
        and reference_trust <= float(cfg.get("refinement_regression_max_reference_trust", 0.75))
    )
    surface_signal = uncertain_surface or low_probability_surface
    surface_context_ok = (
        mild_overcoverage
        or raw_over
        or raw_under
        or raw_mismatch
        or reference_trust <= float(cfg.get("surface_signal_max_reference_trust", 0.40))
    )
    surface_outlier_risk = bool(low_trust_surface or (surface_signal and surface_context_ok))

    run_disagreement = _to_float(context.get("atlas_disagreement_score"))
    atlas_disagreement_risk = False
    if bool(cfg.get("stability_enabled", True)):
        atlas_disagreement_risk = (
            (np.isfinite(run_disagreement) and run_disagreement >= float(cfg.get("stability_atlas_disagreement_threshold", 0.30)))
            or (
                np.isfinite(atlas_volume_iqr_ratio)
                and atlas_volume_iqr_ratio >= float(cfg.get("stability_volume_iqr_ratio_threshold", 0.22))
            )
        )
    current_precision_values = [value for value in (atlas_precision, vote_atlas_precision) if np.isfinite(value)]
    current_coverage_values = [value for value in (atlas_coverage, vote_atlas_coverage) if np.isfinite(value)]
    current_precision_min = min(current_precision_values) if current_precision_values else float("nan")
    current_coverage_max = max(current_coverage_values) if current_coverage_values else float("nan")
    stability_overseg_risk = (
        bool(cfg.get("stability_enabled", True))
        and np.isfinite(current_volume_ratio_to_median)
        and current_volume_ratio_to_median >= float(cfg.get("stability_overseg_current_to_median_ratio", 1.12))
        and (
            atlas_disagreement_risk
            or (
                np.isfinite(current_precision_min)
                and current_precision_min <= float(cfg.get("stability_current_atlas_precision_max", 0.94))
                and np.isfinite(current_coverage_max)
                and current_coverage_max >= float(cfg.get("stability_current_atlas_coverage_min", 0.94))
            )
        )
    )
    stability_low_trust_surface = (
        bool(cfg.get("stability_enabled", True))
        and reference_trust <= float(cfg.get("stability_low_trust_reference_threshold", 0.12))
        and (
            surface_outlier_risk
            or current_vol >= float(cfg.get("stability_surface_volume_ratio_to_softmax", 0.92)) * float(cfg["soft_max_volume_cm3"])
        )
    )
    stability_ratio_close = (
        np.isfinite(current_volume_ratio_to_median)
        and float(cfg.get("stability_disagreement_safe_ratio_low", 0.88))
        <= current_volume_ratio_to_median
        <= float(cfg.get("stability_disagreement_safe_ratio_high", 1.12))
    )
    if stability_overseg_risk:
        stability_risk_class = "overseg_risk"
    elif stability_low_trust_surface and not stability_ratio_close:
        stability_risk_class = "low_trust_surface"
    elif atlas_disagreement_risk and not stability_ratio_close:
        stability_risk_class = "atlas_disagreement"
    elif atlas_disagreement_risk or stability_low_trust_surface:
        stability_risk_class = "inconclusive"
    elif bool(cfg.get("stability_enabled", True)) and surface_outlier_risk:
        stability_risk_class = "inconclusive"
    else:
        stability_risk_class = "stable"

    labels = []
    if high_uncertainty:
        labels.append("high_uncertainty")
    if raw_bad or volume_hard_bad:
        labels.append("raw_bad")
    elif volume_soft_bad:
        labels.append("soft_volume_outlier")
    if raw_mismatch:
        labels.append("atlas_mismatch")
    if raw_over:
        labels.append("overcoverage")
    if raw_under:
        labels.append("undercoverage")
    if mild_overcoverage and "overcoverage" not in labels:
        labels.append("mild_overcoverage")
    if surface_outlier_risk:
        labels.append("surface_outlier_risk")
    if refinement_regression_risk:
        labels.append("refinement_regression_risk")
    if stability_risk_class != "stable" and stability_risk_class not in labels:
        labels.append(stability_risk_class)
    return {
        "triggered": bool(labels),
        "labels": labels,
        "trigger": "|".join(labels) if labels else "stable",
        "high_uncertainty": bool(high_uncertainty),
        "raw_bad": bool(raw_bad or volume_hard_bad),
        "soft_volume_bad": bool(volume_soft_bad),
        "raw_mismatch": bool(raw_mismatch),
        "raw_overcoverage": bool(raw_over),
        "raw_undercoverage": bool(raw_under),
        "mild_overcoverage": bool(mild_overcoverage),
        "surface_outlier_risk": bool(surface_outlier_risk),
        "surface_low_trust": bool(low_trust_surface),
        "surface_boundary_uncertainty": None if not np.isfinite(boundary_unc) else float(boundary_unc),
        "surface_boundary_low_probability_fraction": None
        if not np.isfinite(boundary_low_probability_fraction)
        else float(boundary_low_probability_fraction),
        "refinement_regression_risk": bool(refinement_regression_risk),
        "refinement_regression_change_ratio": None
        if not np.isfinite(refine_change_ratio)
        else float(refine_change_ratio),
        "current_uncertainty_mean": None if not np.isfinite(current_unc) else float(current_unc),
        "current_volume_cm3": float(current_vol),
        "atlas_volume_cm3": None if not np.isfinite(atlas_vol) else float(atlas_vol),
        "current_atlas_dice": None if not np.isfinite(atlas_dice) else float(atlas_dice),
        "current_atlas_precision": None if not np.isfinite(atlas_precision) else float(atlas_precision),
        "current_atlas_coverage": None if not np.isfinite(atlas_coverage) else float(atlas_coverage),
        "vote_current_atlas_dice": None if not np.isfinite(vote_atlas_dice) else float(vote_atlas_dice),
        "vote_current_atlas_precision": None if not np.isfinite(vote_atlas_precision) else float(vote_atlas_precision),
        "vote_current_atlas_coverage": None if not np.isfinite(vote_atlas_coverage) else float(vote_atlas_coverage),
        "reference_trust_score": float(reference_trust),
        "stability_risk_class": stability_risk_class,
        "stability_reference_policy": atlas_summary.get("stability_reference_policy", "legacy_vote"),
        "stability_atlas_volume_iqr": None if not np.isfinite(atlas_volume_iqr) else float(atlas_volume_iqr),
        "stability_atlas_volume_iqr_ratio": None
        if not np.isfinite(atlas_volume_iqr_ratio)
        else float(atlas_volume_iqr_ratio),
        "stability_atlas_volume_median_cm3": None
        if not np.isfinite(atlas_volume_median)
        else float(atlas_volume_median),
        "stability_current_volume_ratio_to_median": None
        if not np.isfinite(current_volume_ratio_to_median)
        else float(current_volume_ratio_to_median),
        "stability_atlas_disagreement": None if not np.isfinite(run_disagreement) else float(run_disagreement),
        "stability_atlas_disagreement_risk": bool(atlas_disagreement_risk),
        "stability_overseg_risk": bool(stability_overseg_risk),
        "stability_low_trust_surface": bool(stability_low_trust_surface),
        "stability_ratio_close_to_atlas_median": bool(stability_ratio_close),
    }


def _annotate_atlas_support(
    candidates: list[dict[str, Any]],
    trigger: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Add no-gold atlas-family support fields used by scoring and QC."""

    for item in candidates:
        family, variant = _candidate_family_variant(item)
        item["atlas_family"] = family
        item["atlas_variant"] = variant
        item["atlas_external_like"] = bool(_is_external_atlas_item(item))
        item["nomask_pair_consensus_context"] = False
        item["single_nomask_boundary_rescue_context"] = False
        item["masked_nomask_conflict"] = False
        item["nomask_pair_global_context"] = False
        item["single_nomask_global_context"] = False
        item["nomask_pair_volume_ok"] = False
        item["nomask_pair_support_families"] = 0
        item["nomask_pair_support_best_dice"] = None
        item["same_family_nomask_dice"] = None
        item["nomask_rescue_rank_score"] = None
        item["external_family_rescue_context"] = False
        item["external_family_rescue_global_context"] = False
        item["stable_external_rescue_context"] = False

    trigger_active = bool(
        trigger.get("triggered")
        or trigger.get("raw_bad")
        or trigger.get("soft_volume_bad")
        or trigger.get("raw_mismatch")
        or trigger.get("raw_overcoverage")
        or trigger.get("raw_undercoverage")
        or trigger.get("surface_outlier_risk")
        or str(trigger.get("stability_risk_class", "stable")) != "stable"
    )
    current_volume = _to_float(trigger.get("current_volume_cm3"), 0.0)
    atlas_median = _to_float(trigger.get("stability_atlas_volume_median_cm3"), 0.0)
    raw_undercoverage = bool(trigger.get("raw_undercoverage"))
    nomask_items = [
        item
        for item in candidates
        if bool(item.get("atlas_external_like"))
        and str(item.get("atlas_variant")) == "nomask"
        and np.asarray(item.get("mask"), dtype=bool).any()
        and float(item.get("volume_score", 0.0)) >= float(cfg.get("nomask_rescue_min_volume_score", 0.70))
        and int(item.get("components", 99)) <= int(cfg.get("max_components", 6))
        and float(item.get("extent_ratio", 0.0)) <= float(cfg.get("max_extent_ratio", 0.992))
    ]
    min_pair = float(cfg.get("nomask_pair_min_pairwise_dice", 0.88))
    min_families = int(cfg.get("nomask_pair_min_families", 2))
    max_current_ratio = float(
        cfg.get(
            "nomask_pair_undercoverage_max_current_volume_ratio"
            if raw_undercoverage
            else "nomask_pair_max_current_volume_ratio",
            1.35 if raw_undercoverage else 1.00,
        )
    )
    max_median_ratio = float(
        cfg.get(
            "nomask_pair_undercoverage_max_median_ratio" if raw_undercoverage else "nomask_pair_max_median_ratio",
            1.25 if raw_undercoverage else 1.02,
        )
    )
    pair_edges: list[dict[str, Any]] = []
    support_by_item: dict[int, set[str]] = {id(item): {str(item.get("atlas_family", ""))} for item in nomask_items}
    best_pair_by_item: dict[int, float] = {id(item): 0.0 for item in nomask_items}

    for i, left in enumerate(nomask_items):
        for right in nomask_items[i + 1 :]:
            if str(left.get("atlas_family")) == str(right.get("atlas_family")):
                continue
            dice = _pairwise_mask_dice(left["mask"], right["mask"])
            if dice >= min_pair:
                support_by_item[id(left)].add(str(right.get("atlas_family")))
                support_by_item[id(right)].add(str(left.get("atlas_family")))
                best_pair_by_item[id(left)] = max(best_pair_by_item[id(left)], dice)
                best_pair_by_item[id(right)] = max(best_pair_by_item[id(right)], dice)
                pair_edges.append(
                    {
                        "left": str(left.get("name")),
                        "right": str(right.get("name")),
                        "left_family": str(left.get("atlas_family")),
                        "right_family": str(right.get("atlas_family")),
                        "dice": float(dice),
                    }
                )

    pair_context_count = 0
    external_family_context_count = 0
    for item in nomask_items:
        support_families = support_by_item.get(id(item), set())
        best_pair = best_pair_by_item.get(id(item), 0.0)
        volume = _to_float(item.get("volume_cm3"), 0.0)
        volume_pair_ok = volume > 0.0
        if current_volume > 0.0:
            volume_pair_ok = volume_pair_ok and volume / current_volume <= max_current_ratio
        if atlas_median > 0.0:
            volume_pair_ok = volume_pair_ok and volume / atlas_median <= max_median_ratio
        uncertainty_score = _to_float(item.get("uncertainty_score"), 0.0)
        intensity_score = _to_float(item.get("intensity_score"), 0.0)
        volume_score = _to_float(item.get("volume_score"), 0.0)
        probability_score = _to_float(item.get("probability_score"), 0.0)
        strong_pair_ok = (
            best_pair >= float(cfg.get("nomask_pair_strong_pairwise_dice", 0.92))
            and uncertainty_score >= float(cfg.get("nomask_pair_strong_min_uncertainty_score", 0.93))
            and intensity_score >= float(cfg.get("nomask_pair_strong_min_intensity_score", 0.90))
            and volume_score >= float(cfg.get("nomask_pair_strong_min_volume_score", 0.95))
        )
        item["nomask_pair_support_families"] = int(len(support_families))
        item["nomask_pair_support_best_dice"] = None if best_pair <= 0.0 else float(best_pair)
        item["nomask_pair_volume_ok"] = bool(volume_pair_ok)
        external_family_ok = (
            trigger_active
            and len(support_families) >= int(cfg.get("external_family_rescue_min_families", min_families))
            and best_pair >= float(cfg.get("external_family_rescue_min_pairwise_dice", min_pair))
            and _to_float(item.get("current_dice"), 0.0)
            >= float(cfg.get("external_family_rescue_min_current_dice", 0.45))
            and volume_score >= float(cfg.get("external_family_rescue_min_volume_score", 0.70))
            and intensity_score >= float(cfg.get("external_family_rescue_min_intensity_score", 0.45))
            and probability_score >= float(cfg.get("external_family_rescue_min_probability_score", 0.45))
            and uncertainty_score >= float(cfg.get("external_family_rescue_min_uncertainty_score", 0.74))
        )
        if external_family_ok:
            item["external_family_rescue_context"] = True
            external_family_context_count += 1
        if (
            trigger_active
            and len(support_families) >= min_families
            and volume_pair_ok
            and volume_score >= float(cfg.get("nomask_rescue_min_volume_score", 0.70))
            and intensity_score >= float(cfg.get("nomask_pair_min_intensity_score", 0.75))
            and uncertainty_score >= float(cfg.get("nomask_rescue_min_uncertainty_score", 0.84))
            and (probability_score >= float(cfg.get("nomask_rescue_min_probability", 0.78)) or strong_pair_ok)
        ):
            item["nomask_pair_consensus_context"] = True
            pair_context_count += 1

    pair_global_context = bool(pair_context_count > 0)
    external_family_global_context = bool(external_family_context_count > 0)
    for item in candidates:
        item["nomask_pair_global_context"] = pair_global_context
        item["external_family_rescue_global_context"] = external_family_global_context

    nomask_by_family: dict[str, list[dict[str, Any]]] = {}
    for item in nomask_items:
        nomask_by_family.setdefault(str(item.get("atlas_family")), []).append(item)

    conflict_count = 0
    max_conflict_dice = float(cfg.get("masked_nomask_conflict_max_pairwise_dice", 0.75))
    for item in candidates:
        if not bool(item.get("atlas_external_like")):
            continue
        if str(item.get("atlas_variant")) not in {"masked", "tissue", "consensus"}:
            continue
        same_family_nomasks = nomask_by_family.get(str(item.get("atlas_family")), [])
        if not same_family_nomasks:
            continue
        best_nomask = max(
            same_family_nomasks,
            key=lambda value: (
                bool(value.get("nomask_pair_consensus_context")),
                _to_float(value.get("probability_score"), 0.0),
                _to_float(value.get("uncertainty_score"), 0.0),
                _to_float(value.get("volume_score"), 0.0),
            ),
        )
        dice = _pairwise_mask_dice(item["mask"], best_nomask["mask"])
        item["same_family_nomask_dice"] = float(dice)
        if dice <= max_conflict_dice and (
            bool(best_nomask.get("nomask_pair_consensus_context"))
            or (
                int(best_nomask.get("nomask_pair_support_families", 0)) >= min_families
                and bool(best_nomask.get("nomask_pair_volume_ok"))
            )
        ):
            item["masked_nomask_conflict"] = True
            conflict_count += 1

    overseg_or_mismatch = bool(
        trigger.get("raw_mismatch")
        or trigger.get("raw_overcoverage")
        or trigger.get("mild_overcoverage")
        or trigger.get("surface_outlier_risk")
        or str(trigger.get("stability_risk_class", "stable")) in {"overseg_risk", "atlas_disagreement", "inconclusive"}
    )
    current_prune_best = max(
        (
            _to_float(item.get("score"), -1e9)
            for item in candidates
            if str(item.get("kind", "")).lower() == "current_prune"
        ),
        default=-1e9,
    )
    single_context_count = 0
    for item in nomask_items:
        if bool(item.get("nomask_pair_consensus_context")):
            continue
        volume = _to_float(item.get("volume_cm3"), 0.0)
        if not overseg_or_mismatch or current_volume <= 0.0 or volume <= 0.0:
            continue
        if volume / current_volume > float(cfg.get("single_nomask_boundary_max_current_volume_ratio", 0.98)):
            continue
        if atlas_median > 0.0 and volume / atlas_median > float(cfg.get("single_nomask_boundary_max_median_ratio", 0.94)):
            continue
        if _to_float(item.get("volume_score"), 0.0) < float(cfg.get("single_nomask_boundary_min_volume_score", 0.80)):
            continue
        if _to_float(item.get("probability_score"), 0.0) < float(
            cfg.get("single_nomask_boundary_min_probability", 0.78)
        ):
            continue
        if _to_float(item.get("uncertainty_score"), 0.0) < float(
            cfg.get("single_nomask_boundary_min_uncertainty_score", 0.84)
        ):
            continue
        if _to_float(item.get("intensity_score"), 0.0) < float(
            cfg.get("single_nomask_boundary_min_intensity_score", 0.90)
        ):
            continue
        rescue_rank = (
            0.28 * _to_float(item.get("volume_score"), 0.0)
            + 0.24 * _to_float(item.get("intensity_score"), 0.0)
            + 0.24 * _to_float(item.get("probability_score"), 0.0)
            + 0.18 * _to_float(item.get("uncertainty_score"), 0.0)
            + 0.06 * _to_float(item.get("shape_score"), 0.0)
        )
        item["nomask_rescue_rank_score"] = float(rescue_rank)
        if rescue_rank < float(cfg.get("single_nomask_boundary_min_rank_score", 0.92)):
            continue
        item["single_nomask_boundary_rescue_context"] = True
        single_context_count += 1

    single_global_context = bool(single_context_count > 0)
    for item in candidates:
        item["single_nomask_global_context"] = single_global_context

    return {
        "nomask_pair_consensus_context": bool(pair_context_count > 0),
        "nomask_pair_consensus_candidate_count": int(pair_context_count),
        "nomask_pair_consensus_edges": pair_edges,
        "external_family_rescue_context": bool(external_family_context_count > 0),
        "external_family_rescue_candidate_count": int(external_family_context_count),
        "single_nomask_boundary_rescue_context": bool(single_context_count > 0),
        "single_nomask_boundary_rescue_candidate_count": int(single_context_count),
        "masked_nomask_conflict_count": int(conflict_count),
    }


def _score_candidate(metrics: dict[str, Any], trigger: Mapping[str, Any], cfg: Mapping[str, Any]) -> float:
    name = str(metrics["name"])
    kind = str(metrics["kind"])
    triggered = bool(trigger["triggered"])
    high_unc = bool(trigger["high_uncertainty"])
    raw_bad = bool(trigger["raw_bad"] or trigger["soft_volume_bad"])
    raw_mismatch = bool(trigger["raw_mismatch"])
    raw_overcoverage = bool(trigger["raw_overcoverage"])
    raw_undercoverage = bool(trigger["raw_undercoverage"])
    mild_overcoverage = bool(trigger.get("mild_overcoverage"))
    surface_outlier = bool(trigger.get("surface_outlier_risk"))
    stability_risk = str(trigger.get("stability_risk_class", "stable"))
    mismatch = bool(raw_mismatch or raw_overcoverage or raw_undercoverage)
    surface_mode = bool((mild_overcoverage or surface_outlier) and not (high_unc or raw_bad or raw_mismatch))

    vol_score = float(metrics["volume_score"])
    volume_cm3 = float(metrics["volume_cm3"])
    intensity = float(metrics["intensity_score"])
    unc_score = float(metrics["uncertainty_score"])
    prob = float(metrics["probability_score"])
    current_dice = float(metrics["current_dice"])
    atlas_dice = float(metrics.get("atlas_dice") if metrics.get("atlas_dice") is not None else 0.5)
    atlas_precision = float(metrics.get("atlas_precision") if metrics.get("atlas_precision") is not None else atlas_dice)
    atlas_coverage = float(metrics.get("atlas_coverage") if metrics.get("atlas_coverage") is not None else atlas_dice)
    shape = float(metrics["shape_score"])

    if surface_mode:
        score = (
            0.24 * current_dice
            + 0.20 * unc_score
            + 0.18 * prob
            + 0.14 * atlas_precision
            + 0.10 * vol_score
            + 0.08 * intensity
            + 0.04 * shape
            + 0.02 * atlas_dice
        )
    elif high_unc:
        score = (
            0.26 * intensity
            + 0.24 * unc_score
            + 0.16 * vol_score
            + 0.12 * prob
            + 0.10 * atlas_precision
            + 0.06 * atlas_coverage
            + 0.04 * shape
            + 0.02 * current_dice
        )
    elif raw_bad:
        score = (
            0.22 * vol_score
            + 0.18 * prob
            + 0.18 * unc_score
            + 0.14 * intensity
            + 0.12 * atlas_precision
            + 0.08 * atlas_coverage
            + 0.04 * current_dice
            + 0.04 * shape
        )
    elif mismatch:
        score = (
            0.18 * vol_score
            + 0.18 * intensity
            + 0.18 * unc_score
            + 0.14 * prob
            + 0.14 * atlas_precision
            + 0.08 * atlas_coverage
            + 0.06 * current_dice
            + 0.04 * shape
        )
    else:
        score = (
            0.20 * vol_score
            + 0.18 * intensity
            + 0.16 * unc_score
            + 0.14 * prob
            + 0.12 * atlas_dice
            + 0.12 * current_dice
            + 0.08 * shape
        )

    over_or_mismatch = raw_overcoverage or raw_mismatch
    current_volume = _to_float(trigger.get("current_volume_cm3"), 0.0)
    if over_or_mismatch:
        if (
            atlas_precision >= float(cfg.get("overcoverage_precision_dominant_min_precision", 0.985))
            and atlas_coverage <= float(cfg.get("overcoverage_precision_dominant_max_coverage", 0.93))
        ):
            score += float(cfg.get("overcoverage_precision_dominant_bonus", 0.08))
            if kind == "atlas":
                score += 0.02
        if atlas_coverage >= 0.98 and atlas_precision < float(cfg.get("overcoverage_boundary_min_precision", 0.965)):
            score -= 0.04

    if kind == "boundary":
        if (
            over_or_mismatch
            and current_volume > 0
            and volume_cm3 >= float(cfg.get("overcoverage_boundary_max_current_volume_ratio", 0.98)) * current_volume
            and atlas_precision < float(cfg.get("overcoverage_boundary_min_precision", 0.965))
        ):
            score -= float(cfg.get("overcoverage_boundary_penalty", 0.08))
        else:
            score += 0.06 if triggered else 0.015
    elif kind == "tissue":
        score += 0.05 if high_unc or raw_bad else 0.010
        if (
            high_unc
            and float(cfg["soft_max_volume_cm3"]) < volume_cm3 <= float(cfg["max_volume_cm3"])
            and volume_cm3 <= float(cfg.get("high_uncertainty_tissue_recovery_max_volume_cm3", 1850.0))
            and atlas_coverage >= float(cfg.get("high_uncertainty_tissue_recovery_min_atlas_coverage", 0.70))
            and current_dice >= float(cfg.get("high_uncertainty_tissue_recovery_min_current_dice", 0.60))
            and unc_score >= float(cfg.get("high_uncertainty_tissue_recovery_min_uncertainty_score", 0.90))
        ):
            score += float(cfg.get("high_uncertainty_tissue_recovery_bonus", 0.065))
    elif kind == "atlas":
        score += 0.03 if mismatch or high_unc else 0.0
        if "family_consensus_compact" in name:
            if current_dice < float(cfg.get("atlas_switch_min_current_dice", 0.86)):
                score -= float(cfg.get("compact_consensus_large_change_penalty", 0.22))
            if (raw_bad or raw_undercoverage) and prob < 0.90:
                score -= float(cfg.get("family_consensus_raw_bad_penalty", 0.08))
            if prob < float(cfg.get("compact_atlas_min_probability_score", 0.78)):
                score -= float(cfg.get("compact_atlas_low_probability_penalty", 0.16))
            if (
                current_dice < float(cfg.get("compact_atlas_large_change_current_dice", 0.90))
                and not _atlas_rescue_signal(metrics, trigger, cfg)
            ):
                score -= float(cfg.get("compact_atlas_large_change_penalty", 0.08))
    elif kind == "current_prune":
        reduction = 1.0 - (volume_cm3 / max(current_volume, 1e-6))
        if surface_mode or raw_overcoverage:
            score += float(cfg.get("current_prune_bonus", 0.06))
            if 0.015 <= reduction <= 0.16:
                score += float(cfg.get("current_prune_good_reduction_bonus", 0.05))
            current_prec = _to_float(trigger.get("current_atlas_precision"), atlas_precision)
            if np.isfinite(current_prec) and atlas_precision >= current_prec + 0.015:
                score += float(cfg.get("current_prune_precision_gain_bonus", 0.05))
        elif high_unc or raw_bad:
            score += 0.02
        if _current_prune_rescue_signal(metrics, trigger, cfg):
            score += float(cfg.get("current_prune_rescue_bonus", 0.12))
            min_prob = float(cfg.get("current_prune_rescue_min_probability_score", 0.84))
            score += float(cfg.get("current_prune_rescue_probability_bonus", 0.06)) * float(
                np.clip((prob - min_prob) / 0.12, 0.0, 1.0)
            )
            target = float(cfg.get("current_prune_rescue_reduction_target", 0.16))
            score += float(cfg.get("current_prune_rescue_reduction_bonus", 0.04)) * float(
                np.exp(-abs(reduction - target) / 0.06)
            )
    elif kind == "backup":
        score -= 0.015
        text = f"{metrics.get('name', '')} {metrics.get('path', '')}".lower()
        if bool(trigger.get("refinement_regression_risk")) and _refinement_backup_median_rescue_signal(
            metrics,
            trigger,
            cfg,
        ):
            score += float(cfg.get("refinement_regression_backup_bonus", 0.34))
            if "pre-adult-refine" in text or "pre_adult_refine" in text:
                score += float(cfg.get("refinement_regression_pre_refine_bonus", 0.08))
    if bool(trigger.get("refinement_regression_risk")) and kind == "current_prune":
        score -= float(cfg.get("refinement_regression_current_prune_penalty", 0.10))
    if stability_risk != "stable":
        if stability_risk == "inconclusive":
            risk_scale = 0.25
        elif stability_risk == "atlas_disagreement":
            risk_scale = 0.60
        else:
            risk_scale = 1.0
        atlas_median = _to_float(trigger.get("stability_atlas_volume_median_cm3"), 0.0)
        compactness = 0.0
        if atlas_median > 0.0 and volume_cm3 > 0.0:
            compactness = float(np.exp(-abs(np.log(volume_cm3 / atlas_median)) / 0.28))
        if metrics.get("is_current"):
            score -= risk_scale * float(cfg.get("stability_current_penalty", 0.18))
        elif kind == "atlas" or "family_consensus_compact" in name:
            score += risk_scale * float(cfg.get("stability_compact_candidate_bonus", 0.14)) * max(compactness, 0.25)
        elif kind == "current_prune":
            if current_volume > 0.0 and volume_cm3 <= 0.995 * current_volume:
                score += risk_scale * float(cfg.get("stability_prune_candidate_bonus", 0.10)) * max(compactness, 0.25)
        elif kind in {"boundary", "tissue"}:
            score += risk_scale * 0.05 * max(compactness, 0.25)
    if _atlas_rescue_signal(metrics, trigger, cfg):
        score += float(cfg.get("atlas_rescue_bonus", 0.18))
        if kind == "atlas":
            score += 0.04
    if _external_family_rescue_signal(metrics, trigger, cfg):
        score += float(cfg.get("external_family_rescue_bonus", 0.30))
        if kind == "atlas":
            score += 0.04
    if _stable_external_rescue_signal(metrics, trigger, cfg):
        score += float(cfg.get("stable_external_rescue_bonus", 0.18))
    if bool(metrics.get("nomask_pair_consensus_context")):
        score += float(cfg.get("nomask_pair_bonus", 0.52))
    if bool(metrics.get("single_nomask_boundary_rescue_context")):
        score += float(cfg.get("single_nomask_boundary_bonus", 0.42))
    if (
        bool(metrics.get("nomask_pair_global_context"))
        and bool(metrics.get("atlas_external_like"))
        and not bool(metrics.get("nomask_pair_consensus_context"))
        and not bool(metrics.get("single_nomask_boundary_rescue_context"))
    ):
        score -= float(cfg.get("nomask_pair_non_nomask_penalty", 0.38))
    if (
        bool(metrics.get("single_nomask_global_context"))
        and bool(metrics.get("atlas_external_like"))
        and not bool(metrics.get("nomask_pair_consensus_context"))
        and not bool(metrics.get("single_nomask_boundary_rescue_context"))
    ):
        score -= float(cfg.get("single_nomask_non_nomask_penalty", 0.24))
    if bool(metrics.get("masked_nomask_conflict")):
        score -= float(cfg.get("masked_nomask_conflict_penalty", 0.32))
    if (
        bool(metrics.get("external_family_rescue_global_context"))
        and kind == "current_prune"
        and (
            raw_bad
            or raw_mismatch
            or raw_undercoverage
            or stability_risk in {"inconclusive", "atlas_disagreement", "low_trust_surface"}
        )
    ):
        score -= float(cfg.get("external_family_rescue_current_prune_penalty", 0.16))
    if metrics.get("is_current"):
        if surface_mode:
            score -= float(cfg.get("surface_current_penalty", 0.06))
        else:
            score += float(cfg.get("current_stable_bonus", 0.08)) if not triggered else -float(
                cfg.get("current_trigger_penalty", 0.14)
            )
        if high_unc:
            score -= float(cfg.get("high_uncertainty_current_penalty", 0.22))
    if float(metrics.get("extent_ratio", 0.0)) > float(cfg["max_extent_ratio"]):
        score -= 0.08
    if int(metrics.get("components", 99)) > int(cfg["max_components"]):
        score -= 0.10
    return float(score)


def _candidate_metrics(
    name: str,
    path: Path | None,
    mask: np.ndarray,
    current: np.ndarray,
    atlas_vote: np.ndarray | None,
    image: np.ndarray,
    ref_img,
    probability: np.ndarray | None,
    uncertainty: np.ndarray | None,
    cfg: Mapping[str, Any],
    *,
    is_current: bool = False,
) -> dict[str, Any]:
    mask = _clean(mask)
    stats = mask_stats(mask, ref_img)
    overlap_current = mask_overlap_stats(mask, current, ref_img)
    atlas_metrics = mask_overlap_stats(mask, atlas_vote, ref_img) if atlas_vote is not None and atlas_vote.any() else {}
    intensity_score, intensity = _intensity_score(mask, image)
    prob_score = 0.5
    if probability is not None and mask.any():
        prob_score = float(np.clip(np.mean(probability[mask]), 0.0, 1.0))
    unc_mean = None
    unc_score = 1.0
    if uncertainty is not None and mask.any():
        unc_mean = float(np.mean(uncertainty[mask]))
        unc_score = float(np.clip(1.0 - unc_mean, 0.0, 1.0))
    volume_cm3 = float(stats["volume_cm3"])
    return {
        "name": name,
        "path": None if path is None else str(path),
        "kind": "current" if is_current else _candidate_kind(name, path),
        "is_current": bool(is_current),
        "volume_cm3": volume_cm3,
        "components": int(stats["components"]),
        "extent_ratio": float(stats.get("extent_ratio", 0.0)),
        "volume_score": _volume_score(volume_cm3, cfg),
        "shape_score": 1.0
        if int(stats["components"]) <= int(cfg["max_components"]) and float(stats.get("extent_ratio", 0.0)) <= float(cfg["max_extent_ratio"])
        else 0.25,
        "current_dice": float(overlap_current["dice"]),
        "change_ratio": float(np.logical_xor(mask, current).sum() / max(current.sum(), 1)),
        "atlas_dice": None if not atlas_metrics else float(atlas_metrics["dice"]),
        "atlas_precision": None if not atlas_metrics else float(atlas_metrics["precision"]),
        "atlas_coverage": None if not atlas_metrics else float(atlas_metrics["coverage"]),
        "probability_score": prob_score,
        "uncertainty_mean": unc_mean,
        "uncertainty_score": unc_score,
        "intensity_score": intensity_score,
        **{f"intensity_{k}": v for k, v in intensity.items()},
        "mask": mask,
    }


def _atlas_rescue_signal(item: Mapping[str, Any], trigger: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    """Allow a strong external-atlas candidate to bypass current-mask inertia.

    This is still a no-gold decision: the signal comes from volume plausibility,
    intensity, uncertainty/probability maps, and agreement with the external
    atlas vote.  It is intentionally limited to atlas-like candidates and is
    disabled for stable subjects.
    """

    if not bool(cfg.get("atlas_rescue_enabled", True)) or bool(item.get("is_current")):
        return False
    trigger_active = any(
        bool(trigger.get(key))
        for key in (
            "high_uncertainty",
            "raw_bad",
            "soft_volume_bad",
            "raw_mismatch",
            "raw_overcoverage",
            "raw_undercoverage",
            "mild_overcoverage",
            "surface_outlier_risk",
        )
    )
    if not trigger_active:
        return False

    kind = str(item.get("kind", "")).lower()
    text = f"{item.get('name', '')} {item.get('path', '')}".lower()
    atlas_like = kind == "atlas" or (
        kind == "candidate"
        and any(token in text for token in ("atlas", "mni152", "oasis30ants", "sri24", "colin27", "icbm"))
    )
    if not atlas_like or kind in {"current", "backup", "current_prune"}:
        return False

    undercoverage_mode = bool(trigger.get("raw_undercoverage") or trigger.get("raw_bad") or trigger.get("soft_volume_bad"))
    current_dice = _to_float(item.get("current_dice"), 0.0)
    min_current = (
        float(cfg.get("atlas_rescue_undercoverage_min_current_dice", cfg.get("atlas_rescue_min_current_dice", 0.55)))
        if undercoverage_mode
        else float(cfg.get("atlas_rescue_min_current_dice", 0.55))
    )
    if current_dice < min_current:
        return False
    if _to_float(item.get("volume_score"), 0.0) < float(cfg.get("atlas_rescue_min_volume_score", 0.45)):
        return False
    if _to_float(item.get("intensity_score"), 0.0) < float(cfg.get("atlas_rescue_min_intensity_score", 0.45)):
        return False
    min_probability = (
        float(cfg.get("atlas_rescue_undercoverage_min_probability_score", cfg.get("atlas_rescue_min_probability_score", 0.75)))
        if undercoverage_mode
        else float(cfg.get("atlas_rescue_min_probability_score", 0.75))
    )
    if _to_float(item.get("probability_score"), 0.5) < min_probability:
        return False
    min_uncertainty = (
        float(cfg.get("atlas_rescue_undercoverage_min_uncertainty_score", cfg.get("atlas_rescue_min_uncertainty_score", 0.75)))
        if undercoverage_mode
        else float(cfg.get("atlas_rescue_min_uncertainty_score", 0.75))
    )
    if _to_float(item.get("uncertainty_score"), 1.0) < min_uncertainty:
        return False

    atlas_precision = _to_float(item.get("atlas_precision"), 0.0)
    atlas_coverage = _to_float(item.get("atlas_coverage"), 0.0)
    strict_external_vote = (
        atlas_precision
        >= (
            float(cfg.get("atlas_rescue_undercoverage_min_precision", cfg.get("atlas_rescue_min_precision", 0.985)))
            if undercoverage_mode
            else float(cfg.get("atlas_rescue_min_precision", 0.985))
        )
        and atlas_coverage
        >= (
            float(cfg.get("atlas_rescue_undercoverage_min_coverage", cfg.get("atlas_rescue_min_coverage", 0.80)))
            if undercoverage_mode
            else float(cfg.get("atlas_rescue_min_coverage", 0.80))
        )
    )
    named_external_template = any(token in text for token in ("mni152", "oasis30ants", "sri24", "colin27", "icbm"))
    relaxed_external_vote = (
        named_external_template
        and atlas_precision >= float(cfg.get("atlas_rescue_relaxed_min_precision", 0.70))
        and atlas_coverage >= float(cfg.get("atlas_rescue_relaxed_min_coverage", 0.90))
    )
    if not (strict_external_vote or relaxed_external_vote):
        return False

    current_volume = _to_float(trigger.get("current_volume_cm3"), 0.0)
    candidate_volume = _to_float(item.get("volume_cm3"), 0.0)
    if bool(trigger.get("raw_overcoverage")) and current_volume > 0.0 and candidate_volume > 0.0:
        # In over-segmentation failures, a relaxed atlas rescue must actually
        # shrink the current mask.  This blocks large atlas candidates that only
        # agree with the current over-segmentation.
        max_ratio = float(cfg.get("atlas_rescue_overcoverage_max_current_volume_ratio", 0.93))
        if not strict_external_vote and candidate_volume / current_volume > max_ratio:
            return False
    return True


def _external_family_rescue_signal(item: Mapping[str, Any], trigger: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    """Rescue an external candidate when independent atlas families agree.

    This is intentionally weaker than the original nomask-pair rescue: it can
    fire with lower SAM probability on difficult datasets, but only when the
    candidate is external, volume-plausible, and supported by another family.
    """

    if not bool(cfg.get("external_family_rescue_enabled", True)) or bool(item.get("is_current")):
        return False
    if not bool(item.get("atlas_external_like")):
        return False
    if not bool(item.get("external_family_rescue_context")):
        return False
    if str(trigger.get("stability_risk_class", "stable")) == "stable" and not bool(trigger.get("triggered")):
        return False
    return True


def _stable_external_rescue_signal(item: Mapping[str, Any], trigger: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    """Allow a compact external candidate for false-stable boundary failures."""

    if not bool(cfg.get("stable_external_rescue_enabled", False)) or bool(item.get("is_current")):
        return False
    if bool(trigger.get("triggered")) or str(trigger.get("stability_risk_class", "stable")) != "stable":
        return False
    if not bool(item.get("atlas_external_like")):
        return False
    current_volume = _to_float(trigger.get("current_volume_cm3"), 0.0)
    candidate_volume = _to_float(item.get("volume_cm3"), 0.0)
    if current_volume <= 0.0 or candidate_volume <= 0.0:
        return False
    ratio = candidate_volume / current_volume
    if ratio < float(cfg.get("stable_external_rescue_min_volume_ratio_to_current", 0.88)):
        return False
    if ratio > float(cfg.get("stable_external_rescue_max_volume_ratio_to_current", 0.99)):
        return False
    current_atlas_dice = _to_float(trigger.get("current_atlas_dice"), 1.0)
    vote_current_atlas_dice = _to_float(trigger.get("vote_current_atlas_dice"), current_atlas_dice)
    if min(current_atlas_dice, vote_current_atlas_dice) > float(
        cfg.get("stable_external_rescue_max_current_atlas_dice", 0.96)
    ):
        return False
    if _to_float(item.get("current_dice"), 0.0) < float(cfg.get("stable_external_rescue_min_current_dice", 0.88)):
        return False
    if _to_float(item.get("volume_score"), 0.0) < float(cfg.get("stable_external_rescue_min_volume_score", 0.90)):
        return False
    if _to_float(item.get("intensity_score"), 0.0) < float(cfg.get("stable_external_rescue_min_intensity_score", 0.80)):
        return False
    if _to_float(item.get("probability_score"), 0.0) < float(
        cfg.get("stable_external_rescue_min_probability_score", 0.82)
    ):
        return False
    if _to_float(item.get("uncertainty_score"), 0.0) < float(
        cfg.get("stable_external_rescue_min_uncertainty_score", 0.84)
    ):
        return False
    if _to_float(item.get("atlas_precision"), 0.0) < float(
        cfg.get("stable_external_rescue_min_atlas_precision", 0.92)
    ):
        return False
    if _to_float(item.get("atlas_coverage"), 0.0) < float(
        cfg.get("stable_external_rescue_min_atlas_coverage", 0.78)
    ):
        return False
    return True


def _current_prune_rescue_signal(item: Mapping[str, Any], trigger: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    if not bool(cfg.get("current_prune_rescue_enabled", True)):
        return False
    if str(item.get("kind", "")).lower() != "current_prune":
        return False
    if not bool(
        trigger.get("raw_mismatch")
        or trigger.get("raw_overcoverage")
        or trigger.get("raw_undercoverage")
        or trigger.get("mild_overcoverage")
        or trigger.get("surface_outlier_risk")
        or str(trigger.get("stability_risk_class", "stable")) != "stable"
    ):
        return False
    current_dice = _to_float(item.get("current_dice"), 0.0)
    probability = _to_float(item.get("probability_score"), 0.0)
    uncertainty = _to_float(item.get("uncertainty_score"), 0.0)
    current_volume = _to_float(trigger.get("current_volume_cm3"), 0.0)
    candidate_volume = _to_float(item.get("volume_cm3"), 0.0)
    if current_dice < float(cfg.get("current_prune_rescue_min_current_dice", 0.965)):
        return False
    if probability < float(cfg.get("current_prune_rescue_min_probability_score", 0.84)):
        return False
    if uncertainty < float(cfg.get("current_prune_rescue_min_uncertainty_score", 0.84)):
        return False
    if current_volume > 0.0 and candidate_volume > 0.0:
        if candidate_volume / current_volume > float(cfg.get("current_prune_rescue_max_volume_ratio", 0.995)):
            return False
    return True


def _refinement_backup_rescue_signal(
    item: Mapping[str, Any],
    trigger: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> bool:
    """Allow the pre-adult-refine backup when adult refinement itself is risky."""

    if not bool(trigger.get("refinement_regression_risk")):
        return False
    if str(item.get("kind", "")).lower() != "backup":
        return False
    text = f"{item.get('name', '')} {item.get('path', '')}".lower()
    if "pre-adult-refine" not in text and "pre_adult_refine" not in text:
        return False
    if not _refinement_backup_median_rescue_signal(item, trigger, cfg):
        return False
    if _to_float(item.get("current_dice"), 0.0) < float(
        cfg.get("refinement_regression_backup_rescue_min_current_dice", 0.90)
    ):
        return False
    if _to_float(item.get("atlas_coverage"), 0.0) < float(
        cfg.get("refinement_regression_backup_rescue_min_atlas_coverage", 0.94)
    ):
        return False
    if _to_float(item.get("probability_score"), 0.0) < float(
        cfg.get("refinement_regression_backup_rescue_min_probability_score", 0.80)
    ):
        return False
    if _to_float(item.get("uncertainty_score"), 0.0) < float(
        cfg.get("refinement_regression_backup_rescue_min_uncertainty_score", 0.85)
    ):
        return False
    return True


def _refinement_backup_median_rescue_signal(
    item: Mapping[str, Any],
    trigger: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> bool:
    if not bool(trigger.get("refinement_regression_risk")):
        return False
    text = f"{item.get('name', '')} {item.get('path', '')}".lower()
    if "pre-adult-refine" not in text and "pre_adult_refine" not in text:
        return False
    atlas_median = _to_float(trigger.get("stability_atlas_volume_median_cm3"), 0.0)
    current_volume = _to_float(trigger.get("current_volume_cm3"), 0.0)
    candidate_volume = _to_float(item.get("volume_cm3"), 0.0)
    if atlas_median <= 0.0 or current_volume <= 0.0 or candidate_volume <= 0.0:
        return True
    current_delta = abs(float(np.log(current_volume / atlas_median)))
    candidate_delta = abs(float(np.log(candidate_volume / atlas_median)))
    min_gain = float(cfg.get("refinement_regression_backup_min_median_closeness_gain", 0.01))
    return candidate_delta <= current_delta - min_gain


def _allowed_candidate(item: dict[str, Any], trigger: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[bool, str]:
    if item.get("is_current"):
        return True, "current"
    volume = float(item["volume_cm3"])
    if volume < float(cfg["min_volume_cm3"]) or volume > float(cfg["max_volume_cm3"]):
        return False, "volume_out_of_bounds"
    if int(item.get("components", 99)) > int(cfg["max_components"]):
        return False, "too_many_components"
    if float(item.get("extent_ratio", 0.0)) > float(cfg["max_extent_ratio"]):
        return False, "extent_too_large"
    if float(item.get("intensity_score", 0.0)) < float(cfg["min_candidate_intensity_score"]):
        return False, "intensity_too_low"
    current_dice = float(item.get("current_dice", 0.0))
    kind = str(item.get("kind", ""))
    name = str(item.get("name", "")).lower()
    text = f"{name} {item.get('path', '')}".lower()
    atlas_rescue = _atlas_rescue_signal(item, trigger, cfg)
    external_family_rescue = _external_family_rescue_signal(item, trigger, cfg)
    stable_external_rescue = _stable_external_rescue_signal(item, trigger, cfg)
    prune_rescue = _current_prune_rescue_signal(item, trigger, cfg)
    refinement_backup_rescue = _refinement_backup_rescue_signal(item, trigger, cfg)
    nomask_pair_rescue = bool(item.get("nomask_pair_consensus_context"))
    single_nomask_rescue = bool(item.get("single_nomask_boundary_rescue_context"))
    low_current_atlas_switch = (
        bool(item.get("atlas_external_like"))
        or kind == "atlas"
        or "family_consensus_compact" in text
        or "external_atlas" in text
    ) and current_dice < float(cfg.get("atlas_switch_min_current_dice", 0.86))
    if low_current_atlas_switch and not (
        nomask_pair_rescue
        or single_nomask_rescue
        or atlas_rescue
        or external_family_rescue
        or stable_external_rescue
    ):
        return False, "atlas_switch_current_dice_too_low"
    if (
        bool(trigger.get("refinement_regression_risk"))
        and (kind == "atlas" or "family_consensus_compact" in text)
        and not bool(trigger.get("raw_bad"))
        and not bool(trigger.get("raw_undercoverage"))
        and not (atlas_rescue or external_family_rescue or stable_external_rescue)
    ):
        return False, "refinement_regression_external_atlas_blocked"
    if str(trigger.get("stability_risk_class", "stable")) == "inconclusive":
        if kind == "atlas" or any(
            token in text
            for token in (
                "family_consensus_compact",
                "external_atlas_consensus",
                "external_adult_t1_adaptive_atlas",
                "brain_mask_mni152",
                "brain_mask_oasis30ants",
                "brain_mask_sri24",
                "brain_mask_colin27",
                "brain_mask_icbm",
            )
        ):
            if not (
                atlas_rescue
                or external_family_rescue
                or stable_external_rescue
                or nomask_pair_rescue
                or single_nomask_rescue
            ):
                return False, "stability_inconclusive_atlas_blocked"
        if kind in {"boundary", "tissue", "backup"} and current_dice < float(
            cfg.get("min_inconclusive_candidate_current_dice", 0.94)
        ) and not (prune_rescue or refinement_backup_rescue or nomask_pair_rescue or single_nomask_rescue):
            return False, "stability_inconclusive_change_too_large"
    if bool(trigger["high_uncertainty"]):
        threshold = float(cfg["min_high_uncertainty_candidate_current_dice"])
    elif bool(trigger["raw_bad"] or trigger["soft_volume_bad"]):
        threshold = float(cfg["min_raw_bad_candidate_current_dice"])
    elif bool(trigger.get("mild_overcoverage")):
        threshold = float(cfg.get("min_mild_overcoverage_candidate_current_dice", 0.93))
    elif bool(trigger.get("surface_outlier_risk")):
        threshold = float(cfg.get("min_surface_candidate_current_dice", 0.88))
    elif bool(trigger["triggered"]):
        threshold = float(cfg["min_triggered_candidate_current_dice"])
    else:
        threshold = float(cfg["min_stable_candidate_current_dice"])
    if current_dice < threshold and not (
        atlas_rescue
        or external_family_rescue
        or stable_external_rescue
        or prune_rescue
        or refinement_backup_rescue
        or nomask_pair_rescue
        or single_nomask_rescue
    ):
        return False, f"current_dice_below_{threshold:.2f}"
    if (
        str(item.get("kind")) == "current_prune"
        and bool(trigger["triggered"])
        and (
            bool(trigger.get("raw_mismatch"))
            or bool(trigger.get("raw_overcoverage"))
            or bool(trigger.get("raw_undercoverage"))
        )
        and not bool(trigger.get("mild_overcoverage"))
        and not prune_rescue
    ):
        return False, "current_prune_blocked_for_severe_atlas_trigger"
    if (
        str(item.get("kind")) == "current_prune"
        and bool(trigger["triggered"])
        and not bool(trigger.get("mild_overcoverage"))
        and not bool(trigger.get("surface_outlier_risk"))
        and current_dice < float(cfg.get("min_current_prune_triggered_candidate_current_dice", 0.90))
        and not prune_rescue
    ):
        return False, "current_prune_change_too_large"
    return True, "ok"


def _generated_candidate_path(mask_path: Path, name: str) -> Path:
    stem = mask_path.name.replace(".nii.gz", "").replace(".nii", "")
    return mask_path.parent / "adult_candidate_adjudicator_candidates" / f"{stem}_{_safe_name(name)}.nii.gz"


def _stability_override_candidate(
    candidates: list[dict[str, Any]],
    current_item: dict[str, Any],
    trigger: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    risk = str(trigger.get("stability_risk_class", "stable"))
    info: dict[str, Any] = {
        "stability_current_blocked": False,
        "stability_selected_reason": "score_selection",
    }
    current_score = float(current_item.get("score", -1e9))
    current_volume = _to_float(trigger.get("current_volume_cm3"), 0.0)
    atlas_median = _to_float(trigger.get("stability_atlas_volume_median_cm3"), 0.0)
    if risk == "stable":
        stable_external = [
            item
            for item in candidates
            if not bool(item.get("is_current"))
            and bool(item.get("allowed"))
            and _stable_external_rescue_signal(item, trigger, cfg)
        ]
        if stable_external:
            for item in stable_external:
                item["stable_external_rescue_context"] = True
            best = max(
                stable_external,
                key=lambda item: (
                    float(item.get("score", -1e9)),
                    _to_float(item.get("probability_score"), 0.0),
                    _to_float(item.get("uncertainty_score"), 0.0),
                    _to_float(item.get("current_dice"), 0.0),
                ),
            )
            if float(best.get("score", -1e9)) >= current_score - float(
                cfg.get("stable_external_rescue_score_drop_tolerance", 0.12)
            ):
                info["stability_selected_reason"] = "stable_external_rescue"
                return best, info
        info["stability_selected_reason"] = "stable_do_no_harm"
        return None, info

    min_score = float(cfg.get("stability_min_noncurrent_score", 0.80))
    max_drop = float(cfg.get("stability_candidate_score_drop_tolerance", 0.04))
    force_drop = float(cfg.get("stability_force_switch_score_drop_tolerance", 0.08))
    non_current = [
        item
        for item in candidates
        if not bool(item.get("is_current"))
        and bool(item.get("allowed"))
        and float(item.get("score", -1e9)) >= min_score
    ]
    if not non_current:
        info["stability_selected_reason"] = "inconclusive_no_allowed_noncurrent"
        return None, info

    def stability_rank(item: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
        score = float(item.get("score", -1e9))
        volume = _to_float(item.get("volume_cm3"), 0.0)
        compactness = 0.0
        if atlas_median > 0.0 and volume > 0.0:
            compactness = float(np.exp(-abs(np.log(volume / atlas_median)) / 0.28))
        shrink = 0.0
        if current_volume > 0.0 and volume > 0.0:
            shrink = float(np.clip(1.0 - volume / current_volume, -1.0, 1.0))
        return (
            score,
            _to_float(item.get("probability_score"), 0.0),
            _to_float(item.get("current_dice"), 0.0),
            compactness,
            shrink,
        )

    best = max(non_current, key=stability_rank)
    best_score = float(best.get("score", -1e9))
    current_blocked = risk in {"overseg_risk", "low_trust_surface"}
    if best_score >= current_score + float(cfg.get("triggered_selection_margin", 0.015)):
        info["stability_current_blocked"] = bool(current_blocked)
        info["stability_selected_reason"] = f"{risk}_candidate_score_gain"
        return best, info
    if current_blocked and best_score >= current_score - force_drop:
        info["stability_current_blocked"] = True
        info["stability_selected_reason"] = f"{risk}_forced_compact_candidate"
        return best, info
    if best_score >= current_score - max_drop:
        info["stability_selected_reason"] = f"{risk}_near_tie_do_no_harm_candidate"
        return best, info

    info["stability_selected_reason"] = f"{risk}_inconclusive_current_kept"
    return None, info


def apply_adult_candidate_adjudicator(
    mask_path: str | Path,
    brain_path: str | Path,
    input_image_path: str | Path,
    *,
    atlas_paths: Iterable[str | Path | None] = (),
    candidate_paths: Iterable[str | Path | None] = (),
    candidate_search_dirs: Iterable[str | Path | None] = (),
    probability_path: str | Path | None = None,
    uncertainty_path: str | Path | None = None,
    context: Mapping[str, Any] | None = None,
    metrics_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select a final mask from M1-internal adult candidates without gold labels."""

    cfg = _cfg(config)
    result: dict[str, Any] = {
        "candidate_adjudicator_enabled": bool(cfg.get("enabled", True)),
        "candidate_adjudicator_used": False,
        "candidate_adjudicator_trigger": "disabled" if not cfg.get("enabled", True) else "stable",
    }
    if not cfg.get("enabled", True):
        return result

    mask_path = Path(mask_path)
    brain_path = Path(brain_path)
    input_image_path = Path(input_image_path)
    if not mask_path.exists() or not input_image_path.exists():
        result["candidate_adjudicator_trigger"] = "missing_mask_or_input"
        return result

    ref_img = _as_ref_img(input_image_path, order=1)
    image = np.nan_to_num(ref_img.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    current = _clean(_read_mask(mask_path, ref_img))
    if not current.any():
        result["candidate_adjudicator_trigger"] = "empty_current"
        return result
    probability = _read_float(probability_path, ref_img)
    uncertainty = _read_float(uncertainty_path, ref_img)
    atlas_vote, used_atlases, atlas_summary = _atlas_vote(
        atlas_paths,
        ref_img,
        image=image,
        probability=probability,
        uncertainty=uncertainty,
        cfg=cfg,
    )
    context = dict(context or {})
    if probability is not None:
        context["_candidate_adjudicator_probability"] = probability
    trigger = _trigger_state(current, ref_img, atlas_vote, uncertainty, context, atlas_summary, cfg)

    candidates: list[dict[str, Any]] = []
    current_metrics = _candidate_metrics(
        "current",
        mask_path,
        current,
        current,
        atlas_vote,
        image,
        ref_img,
        probability,
        uncertainty,
        cfg,
        is_current=True,
    )
    candidates.append(current_metrics)

    generated_count = 0
    generated_saved_count = 0
    boundary_energy_generated_count = 0
    generated_save_limit = int(cfg.get("generated_candidate_save_limit", 32))

    def maybe_save_generated(name: str, mask: np.ndarray) -> Path | None:
        nonlocal generated_saved_count
        if not bool(cfg.get("save_generated_candidates", True)):
            return None
        if generated_saved_count >= generated_save_limit:
            return None
        out_path = _generated_candidate_path(mask_path, name)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_mask(mask, ref_img, out_path)
        generated_saved_count += 1
        return out_path

    if atlas_vote is not None and atlas_vote.any():
        generated_count += 1
        path = maybe_save_generated("family_consensus_compact", atlas_vote)
        candidates.append(
            _candidate_metrics(
                "family_consensus_compact",
                path,
                atlas_vote,
                current,
                atlas_vote,
                image,
                ref_img,
                probability,
                uncertainty,
                cfg,
            )
        )

    for name, mask in _generated_current_pruning_candidates(current, atlas_vote, image, probability, uncertainty, cfg):
        generated_count += 1
        path = maybe_save_generated(name, mask)
        candidates.append(
            _candidate_metrics(name, path, mask, current, atlas_vote, image, ref_img, probability, uncertainty, cfg)
        )

    for name, mask in _generated_boundary_energy_candidates(current, atlas_vote, image, probability, uncertainty, cfg):
        generated_count += 1
        boundary_energy_generated_count += 1
        path = maybe_save_generated(name, mask)
        candidates.append(
            _candidate_metrics(name, path, mask, current, atlas_vote, image, ref_img, probability, uncertainty, cfg)
        )

    for path in _collect_path_candidates(candidate_paths, candidate_search_dirs, cfg):
        try:
            if path.resolve() == mask_path.resolve():
                continue
        except Exception:
            pass
        try:
            mask = _read_mask(path, ref_img)
        except Exception:
            continue
        if not mask.any():
            continue
        name = _safe_name(path)
        candidates.append(
            _candidate_metrics(name, path, mask, current, atlas_vote, image, ref_img, probability, uncertainty, cfg)
        )

    support_info = _annotate_atlas_support(candidates, trigger, cfg)
    for item in candidates:
        allowed, reason = _allowed_candidate(item, trigger, cfg)
        item["allowed"] = bool(allowed)
        item["reject_reason"] = reason
        item["score"] = _score_candidate(item, trigger, cfg) if allowed else -1e9
    support_info = _annotate_atlas_support(candidates, trigger, cfg)
    for item in candidates:
        allowed, reason = _allowed_candidate(item, trigger, cfg)
        item["allowed"] = bool(allowed)
        item["reject_reason"] = reason
        item["score"] = _score_candidate(item, trigger, cfg) if allowed else -1e9

    candidates.sort(key=lambda value: float(value["score"]), reverse=True)
    current_item = next(item for item in candidates if item.get("is_current"))
    best = next((item for item in candidates if item.get("allowed")), current_item)
    if str(trigger.get("stability_risk_class", "stable")) == "inconclusive":
        margin = float(cfg.get("stability_inconclusive_selection_margin", 0.08))
    else:
        margin = (
            float(cfg["triggered_selection_margin"])
            if bool(trigger["triggered"])
            else float(cfg["stable_selection_margin"])
        )
    if not bool(trigger["triggered"]):
        if float(best["score"]) < float(current_item["score"]) + margin:
            best = current_item
    elif not best.get("is_current") and float(best["score"]) < float(current_item["score"]) + margin:
        best = current_item

    stability_best, stability_info = _stability_override_candidate(candidates, current_item, trigger, cfg)
    if stability_best is not None:
        best = stability_best
    if bool(trigger.get("refinement_regression_risk")):
        pre_refine_backups = [
            item
            for item in candidates
            if bool(item.get("allowed"))
            and str(item.get("kind", "")).lower() == "backup"
            and _refinement_backup_median_rescue_signal(item, trigger, cfg)
        ]
        if pre_refine_backups:
            best = max(pre_refine_backups, key=lambda item: float(item.get("score", -1e9)))
            stability_info["stability_selected_reason"] = "refinement_regression_pre_refine_backup"

    result.update(
        {
            "candidate_adjudicator_trigger": trigger["trigger"],
            "candidate_adjudicator_selected_candidate": str(best["name"]),
            "candidate_adjudicator_selected_path": "" if best.get("path") is None else str(best["path"]),
            "candidate_adjudicator_current_score": float(current_item["score"]),
            "candidate_adjudicator_selected_score": float(best["score"]),
            "candidate_adjudicator_candidate_count": int(len(candidates)),
            "candidate_adjudicator_generated_candidate_count": int(generated_count),
            "boundary_energy_candidate_count": int(boundary_energy_generated_count),
            "candidate_adjudicator_used_atlases": used_atlases,
            "stability_candidate_saved_count": int(generated_saved_count),
            **support_info,
            "nomask_pair_consensus_selected": bool(best.get("nomask_pair_consensus_context")),
            "external_family_rescue_selected": bool(best.get("external_family_rescue_context")),
            "single_nomask_boundary_rescue_selected": bool(best.get("single_nomask_boundary_rescue_context")),
            "stable_external_rescue_selected": bool(best.get("stable_external_rescue_context")),
            "stability_risk_class": trigger.get("stability_risk_class", "stable"),
            "stability_current_volume_ratio_to_median": trigger.get("stability_current_volume_ratio_to_median"),
            "stability_atlas_disagreement": trigger.get("stability_atlas_disagreement"),
            "stability_atlas_disagreement_risk": trigger.get("stability_atlas_disagreement_risk"),
            "stability_overseg_risk": trigger.get("stability_overseg_risk"),
            "stability_low_trust_surface": trigger.get("stability_low_trust_surface"),
            "stability_ratio_close_to_atlas_median": trigger.get("stability_ratio_close_to_atlas_median"),
            **{k: v for k, v in atlas_summary.items() if k != "stability_atlas_vote_items"},
            **stability_info,
            **{f"candidate_adjudicator_{k}": v for k, v in trigger.items() if k != "labels"},
        }
    )
    result["stability_atlas_vote_items"] = atlas_summary.get("stability_atlas_vote_items", [])

    slim = []
    for item in candidates[: int(cfg.get("save_top_candidates", 24))]:
        slim.append({k: v for k, v in item.items() if k != "mask"})
    result["candidate_adjudicator_candidates"] = slim
    qc_path = mask_path.with_name(mask_path.name.replace(".nii.gz", "_candidate-adjudicator_qc.json"))
    result["candidate_adjudicator_qc_path"] = str(qc_path)
    qc_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    if best.get("is_current"):
        result["candidate_adjudicator_reason"] = "current_selected"
        if metrics_path:
            update_metrics_json(metrics_path, result)
        qc_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    selected = np.asarray(best["mask"], dtype=bool)
    backup = mask_path.with_name(mask_path.name.replace(".nii.gz", f"{cfg['backup_suffix']}.nii.gz"))
    if not backup.exists():
        shutil.copyfile(mask_path, backup)
    save_mask(selected, ref_img, mask_path)
    save_float(image * selected.astype(np.float32), ref_img, brain_path)
    result.update(
        {
            "candidate_adjudicator_used": True,
            "candidate_adjudicator_reason": "triggered_candidate_selected"
            if bool(trigger["triggered"])
            else "stable_candidate_selected",
            "candidate_adjudicator_backup_mask": str(backup),
            "candidate_adjudicator_volume_after_cm3": float(mask_stats(selected, ref_img)["volume_cm3"]),
        }
    )
    if metrics_path:
        update_metrics_json(metrics_path, result)
    qc_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
