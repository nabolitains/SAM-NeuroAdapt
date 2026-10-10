"""iSeg2017 adaptive guidance atlas without leave-one-out labels.

The helper keeps the notebook pipeline shape unchanged: external atlas
registration still happens first, then this module builds a better guidance
mask from the registered external variants plus the subject T1/T2 signal.  It
does not read the subject gold mask and does not use iSeg LOO atlas entries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi

try:
    from .neuro_utils import largest_component, mask_volume_cm3, save_mask
except ImportError:  # pragma: no cover
    from neuro_utils import largest_component, mask_volume_cm3, save_mask


def _ball(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    grid = np.ogrid[tuple(slice(-radius, radius + 1) for _ in range(3))]
    return sum(g.astype(float) ** 2 for g in grid) <= radius**2


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    iterations = max(0, int(iterations))
    if iterations == 0:
        return np.asarray(mask, dtype=bool)
    structure = ndi.generate_binary_structure(3, 1)
    return ndi.binary_dilation(mask, structure=structure, iterations=iterations)


def _as_3d_img(path: str | Path, ref_img=None, order: int = 0):
    img = nib.load(str(path))
    if img.ndim > 3:
        data = np.asarray(img.dataobj)[..., 0]
        img = nib.Nifti1Image(data, img.affine, img.header)
    if ref_img is not None and (
        img.shape[:3] != ref_img.shape[:3] or not np.allclose(img.affine, ref_img.affine, atol=1e-3)
    ):
        img = resample_from_to(img, ref_img, order=order, cval=0)
    return img


def _mask_stats(mask: np.ndarray, ref_img) -> dict[str, Any]:
    lab, ncomp = ndi.label(mask)
    stats = {
        "volume_cm3": mask_volume_cm3(mask, ref_img),
        "components": int(ncomp),
        "voxels": int(mask.sum()),
    }
    if mask.any():
        coords = np.where(mask)
        stats["extent_ratio"] = float(max((c.max() - c.min() + 1) / s for c, s in zip(coords, mask.shape)))
    else:
        stats["extent_ratio"] = 0.0
    return stats


def _norm_for_score(data: np.ndarray, region: np.ndarray) -> np.ndarray:
    values = data[region & np.isfinite(data)]
    if values.size < 16:
        return np.zeros(data.shape, dtype=np.float32)
    lo, hi = np.percentile(values, [2, 98])
    if hi <= lo:
        return np.zeros(data.shape, dtype=np.float32)
    return np.clip((data - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _registered_variant_paths(anat_dir: str | Path, subject_id: str) -> dict[str, Path]:
    adv_dir = Path(anat_dir) / "atlases_advanced"
    prefix = f"sub-{subject_id}_brain_mask"
    variants = {
        "aff": adv_dir / f"{prefix}_aff.nii.gz",
        "aff_nomask": adv_dir / f"{prefix}_aff_nomask.nii.gz",
        "syn": adv_dir / f"{prefix}_syn.nii.gz",
    }
    return {name: path for name, path in variants.items() if path.exists()}


def build_iseg_adaptive_guidance_atlas(
    anat_dir: str | Path,
    subject_id: str,
    input_image_path: str | Path,
    t2_path: str | Path | None = None,
    output_path: str | Path | None = None,
    force: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a subject-adaptive atlas from registered external atlas variants.

    No gold mask and no iSeg LOO mask are read.  The output is intended to be
    passed as the usual ``brain_mask`` atlas to the existing SAM2 extractor.
    """

    cfg = {
        "min_volume_cm3": 550.0,
        "max_volume_cm3": 1100.0,
        "seed_prefer": "aff",
        "shell_width": 24,
        "envelope_dilation": 3,
        "t1_percentile": 3.0,
        "t2_percentile": 3.0,
        "target_from_envelope_ratio": 0.90,
        "target_seed_gain_ratio": 1.12,
        "target_max_cm3": 980.0,
        "max_added_cm3": 260.0,
        "min_added_cm3": 5.0,
    }
    if config:
        cfg.update(config)

    anat_dir = Path(anat_dir)
    ref_img = _as_3d_img(input_image_path, order=1)
    out_path = Path(output_path) if output_path else anat_dir / "atlases_advanced" / f"sub-{subject_id}_brain_mask_adaptive.nii.gz"
    qc_path = out_path.with_name(out_path.name.replace(".nii.gz", "_qc.json"))
    if out_path.exists() and qc_path.exists() and not force:
        try:
            cached = json.loads(qc_path.read_text())
            cached["output_path"] = str(out_path)
            cached["used_cache"] = True
            return cached
        except Exception:
            pass

    variants: list[dict[str, Any]] = []
    for name, path in _registered_variant_paths(anat_dir, subject_id).items():
        img = _as_3d_img(path, ref_img, order=0)
        mask = img.get_fdata(dtype=np.float32) > 0.5
        stats = _mask_stats(mask, ref_img)
        if (
            stats["volume_cm3"] >= float(cfg["min_volume_cm3"])
            and stats["volume_cm3"] <= float(cfg["max_volume_cm3"])
            and stats["components"] <= 2
        ):
            variants.append({"name": name, "path": str(path), "mask": mask, **stats})

    result: dict[str, Any] = {
        "adaptive_atlas_used": False,
        "subject_id": subject_id,
        "output_path": str(out_path),
        "variant_count": len(variants),
        "variants": [
            {k: v for k, v in item.items() if k != "mask"}
            for item in variants
        ],
    }
    if not variants:
        result["adaptive_atlas_reason"] = "no_plausible_registered_variants"
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    prefer = str(cfg.get("seed_prefer", "aff"))
    seed_item = next((v for v in variants if v["name"] == prefer), None)
    if seed_item is None:
        seed_item = min(variants, key=lambda v: v["volume_cm3"])
    seed = np.asarray(seed_item["mask"], dtype=bool)
    envelope = np.logical_or.reduce([np.asarray(v["mask"], dtype=bool) for v in variants])
    envelope = _dilate(envelope, int(cfg["envelope_dilation"]))
    envelope = ndi.binary_fill_holes(envelope).astype(bool)

    t1 = ref_img.get_fdata(dtype=np.float32)
    t2 = None
    if t2_path and Path(t2_path).exists():
        try:
            t2 = _as_3d_img(t2_path, ref_img, order=1).get_fdata(dtype=np.float32)
        except Exception:
            t2 = None

    seed_vol = mask_volume_cm3(seed, ref_img)
    envelope_vol = mask_volume_cm3(envelope, ref_img)
    target_vol = max(
        seed_vol * float(cfg["target_seed_gain_ratio"]),
        envelope_vol * float(cfg["target_from_envelope_ratio"]),
    )
    target_vol = min(float(cfg["target_max_cm3"]), float(cfg["max_volume_cm3"]), target_vol)
    target_vol = max(seed_vol, target_vol)
    max_added_cm3 = min(float(cfg["max_added_cm3"]), max(0.0, target_vol - seed_vol))

    shell = _dilate(seed, int(cfg["shell_width"])) & (~seed) & envelope
    additions = np.zeros(seed.shape, dtype=bool)
    if shell.any() and seed.any() and max_added_cm3 > 0:
        t1_thr = float(np.percentile(t1[seed], float(cfg["t1_percentile"])))
        additions = shell & (t1 >= t1_thr)
        t2_thr = None
        if t2 is not None:
            t2_thr = float(np.percentile(t2[seed], float(cfg["t2_percentile"])))
            additions &= t2 >= t2_thr

        max_add_vox = int(max_added_cm3 / max(float(np.prod(ref_img.header.get_zooms()[:3]) / 1000.0), 1e-9))
        if additions.sum() > max_add_vox > 0:
            score_region = envelope | seed | additions
            score = _norm_for_score(t1, score_region)
            if t2 is not None:
                score = score + _norm_for_score(t2, score_region)
            distance = ndi.distance_transform_edt(~seed)
            score = score - 0.015 * distance.astype(np.float32)
            values = score[additions]
            cutoff = np.partition(values, max(0, values.size - max_add_vox))[max(0, values.size - max_add_vox)]
            additions &= score >= cutoff

        result["adaptive_atlas_t1_threshold"] = t1_thr
        if t2 is not None:
            result["adaptive_atlas_t2_threshold"] = t2_thr

    adaptive = seed | additions
    adaptive = ndi.binary_fill_holes(adaptive).astype(bool)
    adaptive = largest_component(adaptive)
    adaptive_stats = _mask_stats(adaptive, ref_img)

    if adaptive_stats["volume_cm3"] > float(cfg["max_volume_cm3"]):
        adaptive = seed
        adaptive_stats = _mask_stats(adaptive, ref_img)
        result["adaptive_atlas_reason"] = "proposal_above_max_volume_reverted_to_seed"
    elif adaptive_stats["volume_cm3"] < seed_vol + float(cfg["min_added_cm3"]):
        result["adaptive_atlas_reason"] = "minimal_tissue_recovery"
    else:
        result["adaptive_atlas_reason"] = "t1_t2_envelope_recovery"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_mask(adaptive, ref_img, out_path)
    result.update(
        {
            "adaptive_atlas_used": True,
            "seed_variant": seed_item["name"],
            "seed_volume_cm3": float(seed_vol),
            "envelope_volume_cm3": float(envelope_vol),
            "target_volume_cm3": float(target_vol),
            "adaptive_added_cm3": float(max(0.0, adaptive_stats["volume_cm3"] - seed_vol)),
            "adaptive_volume_cm3": float(adaptive_stats["volume_cm3"]),
            "adaptive_components": int(adaptive_stats["components"]),
            "output_path": str(out_path),
        }
    )
    qc_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
