"""Small NIfTI/mask helpers shared by notebook support scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

try:
    import nibabel as nib
except Exception:  # pragma: no cover
    nib = None


def load_img(path: str | Path):
    if nib is None:
        raise RuntimeError("nibabel is required for NIfTI operations")
    return nib.load(str(path))


def load_data(path: str | Path, dtype=np.float32) -> np.ndarray:
    return load_img(path).get_fdata(dtype=dtype)


def load_mask(path: str | Path, threshold: float = 0.5) -> np.ndarray:
    return load_data(path, np.float32) > threshold


def save_mask(mask: np.ndarray, ref_img, path: str | Path, dtype=np.uint8) -> Path:
    if nib is None:
        raise RuntimeError("nibabel is required for NIfTI operations")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(mask.astype(dtype), ref_img.affine, ref_img.header)
    img.set_data_dtype(dtype)
    nib.save(img, str(p))
    return p


def save_float(data: np.ndarray, ref_img, path: str | Path, dtype=np.float32) -> Path:
    if nib is None:
        raise RuntimeError("nibabel is required for NIfTI operations")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(dtype), ref_img.affine, ref_img.header)
    img.set_data_dtype(dtype)
    nib.save(img, str(p))
    return p


def voxel_volume_cm3(img_or_path) -> float:
    img = load_img(img_or_path) if isinstance(img_or_path, (str, Path)) else img_or_path
    return float(np.prod(img.header.get_zooms()[:3]) / 1000.0)


def mask_volume_cm3(mask: np.ndarray, ref_img) -> float:
    return float(mask.sum()) * voxel_volume_cm3(ref_img)


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def mask_stats(mask: np.ndarray, ref_img) -> dict[str, Any]:
    lab, n = ndi.label(mask)
    vol = mask_volume_cm3(mask, ref_img)
    extent = 0.0
    if mask.any():
        coords = np.where(mask)
        spans = [int(c.max() - c.min() + 1) / max(1, s) for c, s in zip(coords, mask.shape)]
        extent = float(max(spans))
    return {"volume_cm3": vol, "components": int(n), "extent_ratio": extent, "voxels": int(mask.sum())}


def mask_overlap_stats(pred: np.ndarray, ref: np.ndarray, ref_img) -> dict[str, float]:
    pred = pred.astype(bool)
    ref = ref.astype(bool)
    inter = np.logical_and(pred, ref).sum()
    pred_sum = pred.sum()
    ref_sum = ref.sum()
    dice = 2.0 * inter / (pred_sum + ref_sum + 1e-9)
    precision = inter / (pred_sum + 1e-9)
    coverage = inter / (ref_sum + 1e-9)
    vox = voxel_volume_cm3(ref_img)
    return {
        "dice": float(dice),
        "precision": float(precision),
        "coverage": float(coverage),
        "fp_cm3": float(np.logical_and(pred, ~ref).sum() * vox),
        "fn_cm3": float(np.logical_and(~pred, ref).sum() * vox),
    }


def update_metrics_json(path: str | Path, updates: dict[str, Any]) -> None:
    p = Path(path)
    data: dict[str, Any] = {}
    if p.exists() and p.stat().st_size > 0:
        try:
            data = json.loads(p.read_text())
        except Exception:
            data = {}
    data.update({k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v) for k, v in updates.items()})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def write_brain_from_mask(image_path: str | Path, mask_path: str | Path, out_path: str | Path) -> Path:
    img = load_img(image_path)
    data = img.get_fdata(dtype=np.float32)
    mask = load_mask(mask_path)
    return save_float(data * mask, img, out_path, np.float32)


def resample_like_nearest(source_path: str | Path, reference_path: str | Path, out_path: str | Path) -> Path:
    if nib is None:
        raise RuntimeError("nibabel is required for NIfTI operations")
    from nibabel.processing import resample_from_to

    src = load_img(source_path)
    ref = load_img(reference_path)
    res = resample_from_to(src, ref, order=0)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    res.set_data_dtype(np.uint8)
    nib.save(res, str(p))
    return p

