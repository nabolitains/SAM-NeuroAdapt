#!/usr/bin/env python3
"""Standardize local downloaded neuro datasets for SAM-NeuroAdapt M1 runs.

The script treats ``data/`` as immutable/raw input and writes real file copies
under ``datasets/``. It also builds CSV manifests and LOO atlas libraries using
the conventions consumed by the notebook support code.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import nibabel as nib
    from nibabel.processing import resample_from_to
except Exception:  # pragma: no cover
    nib = None
    resample_from_to = None


STANDARD_FIELDS = [
    "dataset",
    "subject",
    "session",
    "split",
    "has_gold",
    "qc_only",
    "atlas_eligible",
    "modality",
    "profile",
    "t1_path",
    "t2_path",
    "input_path",
    "brain_mask_path",
    "label_path",
    "source_path",
    "standardized_dir",
    "shape",
    "zooms",
    "mask_voxels",
    "mask_volume_cm3",
    "notes",
]

IBSR18_SEG_OFFSETS = {
    "1_24": -4,
    "2_4": 6,
    "5_8": 1,
    "4_8": 8,
    "6_10": 0,
    "7_8": 0,
    "8_4": 2,
    "11_3": 1,
    "12_3": 0,
    "13_3": 0,
    "15_3": 0,
    "16_3": 0,
    "17_3": 3,
    "100_23": 0,
    "110_3": 1,
    "111_2": 2,
    "112_2": 1,
    "191_3": 3,
    "202_3": 3,
    "205_3": 2,
}

DATASET_ALIASES = {
    "BraTS-PEDs": "BraTSPEDs",
    "BraTS_PEDs": "BraTSPEDs",
    "BraTS_PED": "BraTSPEDs",
    "BRATS-PEDS": "BraTSPEDs",
    "BRATSPEDS": "BraTSPEDs",
    "CANDI": "CANDIShare",
    "LowFieldInfant": "LowFieldInfantMRI",
    "LowFieldInfantMR": "LowFieldInfantMRI",
    "LowFieldMRI": "LowFieldInfantMRI",
    "ScienceDBLowField": "LowFieldInfantMRI",
    "LISA": "LISA2025",
    "LISA25": "LISA2025",
    "M-CRIB": "MCRIB",
    "M_CRIB": "MCRIB",
    "MCRIB2": "MCRIB_WM",
    "M-CRIB2": "MCRIB_WM",
    "MCRIB-WM": "MCRIB_WM",
    "M-CRIB-WM": "MCRIB_WM",
    "MCRIBWM": "MCRIB_WM",
    "dHCP": "dHCPNeonatal",
    "DHCP": "dHCPNeonatal",
    "NIH-PD": "NIHPD",
    "NIH_Pediatric_MRI": "NIHPD",
    "ADHD-200": "ADHD200",
    "ADHD_200": "ADHD200",
}

MANIFEST_ALIASES = {
    "MCRIB_WM": ["MCRIB2"],
}

MANUAL_ACCESS_DATASETS = {
    "BraTSPEDs": {
        "status": "manual_required",
        "reason": "TCIA/NBIA collection must be downloaded manually or with NBIA tooling, then placed under data/BraTSPEDs.",
        "url": "https://www.cancerimagingarchive.net/collection/brats-peds/",
        "expected_target": "data/BraTSPEDs",
        "benchmark_group": "pathology_manual_tumor_or_hydro",
        "estimated_subjects": 457,
    },
    "HyKid": {
        "status": "manual_required",
        "reason": "Synapse download requires account/terms unless already present under data/HyKid.",
        "url": "https://www.synapse.org/Synapse:syn68544889",
        "expected_target": "data/HyKid",
        "benchmark_group": "pathology_manual_tumor_or_hydro",
        "estimated_subjects": 48,
    },
    "iSeg2019": {
        "status": "manual_required",
        "reason": "UNC iSeg-2019 requires team registration, institutional email, and signed agreement before download.",
        "url": "https://iseg2019.web.unc.edu/data/",
        "expected_target": "data/iSeg2019",
    },
    "CANDIShare": {
        "status": "manual_required",
        "reason": "NITRC login is required for CANDIShare image+segmentation archives.",
        "url": "https://www.nitrc.org/frs/?group_id=377",
        "expected_target": "data/CANDIShare",
        "benchmark_group": "clinical_manual_non_tumor",
        "estimated_subjects": 103,
    },
    "ALBERTs": {
        "status": "manual_required",
        "reason": "Brain Development ALBERTs requires accepting the academic licence and filling the download form.",
        "url": "https://brain-development.org/brain-atlases/neonatal-brain-atlases/neonatal-brain-atlas-gousias/",
        "expected_target": "data/ALBERTs",
    },
    "dHCPNeonatal": {
        "status": "excluded_non_strict",
        "reason": "dHCP requires institutional registration and data sharing agreement; labels are pipeline derivatives, not strict manual GT.",
        "url": "https://www.developingconnectome.org/data-release/second-data-release/information-registration-and-download/",
        "expected_target": "data/dHCP",
        "estimated_subjects": 783,
    },
    "NeoBrainS12": {
        "status": "manual_required",
        "reason": "Official NeoBrainS12 host currently does not resolve; use only an official mirror or organizer-provided access.",
        "url": "http://neobrains12.isi.uu.nl/",
        "expected_target": "data/NeoBrainS12",
    },
    "LowFieldInfantMRI": {
        "status": "manual_required",
        "reason": "Science Data Bank access requires account registration and DUA review before download.",
        "url": "https://doi.org/10.57760/sciencedb.10690",
        "expected_target": "data/LowFieldInfantMRI",
        "benchmark_group": "clinical_manual_non_tumor",
        "estimated_subjects": 100,
    },
    "LISA2025": {
        "status": "manual_required",
        "reason": "Use official LISA challenge data only after verifying that images and labels are distributable outside the challenge.",
        "url": "https://zenodo.org/records/15081583",
        "expected_target": "data/LISA2025",
        "benchmark_group": "clinical_manual_non_tumor",
        "estimated_subjects": 79,
    },
    "NIHPD": {
        "status": "excluded_non_strict",
        "reason": "Large controlled pediatric release, but label provenance is not strict manual GT for this benchmark.",
        "url": "https://nda.nih.gov/ndarpublicweb/Documents/Pediatric_MRI_Data.pdf",
        "expected_target": "data/NIHPD",
        "estimated_subjects": 550,
    },
    "ABCD": {
        "status": "excluded_non_strict",
        "reason": "FreeSurfer derivatives are automated labels, not strict manual GT.",
        "url": "https://docs.abcdstudy.org/v/6_0_0/documentation/imaging/data_derivatives.html",
        "expected_target": "data/ABCD",
        "estimated_subjects": 10000,
    },
    "HBN": {
        "status": "excluded_non_strict",
        "reason": "Large pediatric MRI resource but no strict manual segmentation GT for this benchmark.",
        "url": "https://childmind.org/science/global-open-science/healthy-brain-network/",
        "expected_target": "data/HBN",
        "estimated_subjects": 2500,
    },
    "ABIDE": {
        "status": "excluded_non_strict",
        "reason": "Preprocessed derivatives are automated and the cohort is not pediatric-only.",
        "url": "https://preprocessed-connectomes-project.org/abide/index.html",
        "expected_target": "data/ABIDE",
        "estimated_subjects": 1112,
    },
    "ADHD200": {
        "status": "excluded_non_strict",
        "reason": "Preprocessed derivatives are automated pipeline labels, not strict manual GT.",
        "url": "https://preprocessed-connectomes-project.org/adhd200",
        "expected_target": "data/ADHD200",
        "estimated_subjects": 947,
    },
}

DEFAULT_DATASET_ORDER = [
    "IBSR18",
    "NFBS",
    "iSeg2017",
    "MRBrainS13",
    "SynthStrip",
    "SynthStripInfant",
    "BOBs",
    "BrainMetShare",
    "LPBA40",
    "MCRIB",
    "MCRIB_WM",
    "ALBERTs",
    "FetalFMRI",
    "BraTSPEDs",
    "HyKid",
    "CANDIShare",
    "LowFieldInfantMRI",
    "LISA2025",
]

STRICT_PEDIATRIC_SOURCE_METADATA = {
    "BOBs": {
        "gt_provenance": "manual_corrected_segmentation",
        "benchmark_group": "core_manual_developmental",
        "license": "OpenNeuro dataset terms",
        "access": "open",
        "count_policy": "subject_session",
        "skull_stripping_eligible": "true",
    },
    "iSeg2017": {
        "gt_provenance": "manual_tissue_labels",
        "benchmark_group": "segmentation_only_prestripped_input",
        "license": "iSeg challenge terms",
        "access": "local_archive",
        "count_policy": "subject_session",
        "skull_stripping_eligible": "false",
        "input_state": "prestripped_brain_only",
    },
    "MCRIB": {
        "gt_provenance": "manual_parcellation",
        "benchmark_group": "segmentation_only_preprocessed_input",
        "license": "OSF project terms",
        "access": "open",
        "count_policy": "subject",
        "skull_stripping_eligible": "false",
        "input_state": "preprocessed_brain_focused",
    },
    "MCRIB_WM": {
        "gt_provenance": "manual_parcellation",
        "benchmark_group": "segmentation_only_preprocessed_input",
        "license": "OSF project terms",
        "access": "open",
        "count_policy": "annotation_variant_no_new_subjects",
        "annotation_variant_of": "MCRIB",
        "skull_stripping_eligible": "false",
        "input_state": "preprocessed_brain_focused",
    },
    "SynthStripInfant": {
        "gt_provenance": "manually_refined_binary_brain_mask",
        "benchmark_group": "core_manual_developmental",
        "license": "SynthStrip dataset source terms",
        "access": "open",
        "count_policy": "subject",
        "target_mask": "brain_extraction",
        "skull_stripping_eligible": "true",
    },
    "ALBERTs": {
        "gt_provenance": "manual_50_region_neonatal_parcellation",
        "benchmark_group": "conditional_proxy_not_dedicated_brain_mask",
        "license": "ALBERTs free academic end user licence",
        "access": "academic_form",
        "count_policy": "subject",
        "target_mask": "brain_extraction_from_union_of_manual_labels",
        "skull_stripping_eligible": "false",
        "conditional_proxy": "true",
    },
    "FetalFMRI": {
        "gt_provenance": "manually_drawn_fetal_brain_masks",
        "benchmark_group": "stress_test_manual_fetal_functional",
        "license": "CC0",
        "access": "open",
        "count_policy": "unique_subject_not_frame",
        "target_mask": "brain_extraction",
        "skull_stripping_eligible": "true",
    },
    "BraTSPEDs": {
        "gt_provenance": "expert_refined_tumor_annotations",
        "benchmark_group": "pathology_manual_tumor_or_hydro",
        "license": "TCIA data usage policy",
        "access": "open_tcia_large_manual_or_nbia",
        "count_policy": "subject",
        "target_mask": "tumor_regions",
        "skull_stripping_eligible": "false",
    },
    "HyKid": {
        "gt_provenance": "expert_corrected_multistructure_labels",
        "benchmark_group": "pathology_manual_tumor_or_hydro",
        "license": "Synapse dataset terms",
        "access": "synapse_account_or_terms",
        "count_policy": "subject",
        "target_mask": "hydrocephalus_structures",
        "skull_stripping_eligible": "false",
        "conditional_proxy": "true",
    },
    "CANDIShare": {
        "gt_provenance": "CMA_general_segmentation",
        "benchmark_group": "clinical_manual_non_tumor",
        "license": "Creative Commons Attribution plus NITRC click-through terms",
        "access": "nitrc_login",
        "count_policy": "subject",
        "skull_stripping_eligible": "false",
        "conditional_proxy": "true",
    },
    "LowFieldInfantMRI": {
        "gt_provenance": "manual_brain_mask",
        "benchmark_group": "clinical_manual_non_tumor",
        "license": "Science Data Bank DUA",
        "access": "account_dua_required",
        "count_policy": "subject",
        "target_mask": "brain_extraction",
        "skull_stripping_eligible": "true",
    },
    "LISA2025": {
        "gt_provenance": "challenge_expert_structure_labels",
        "benchmark_group": "clinical_manual_non_tumor",
        "license": "LISA challenge terms",
        "access": "challenge_access_verify_release",
        "count_policy": "subject",
        "target_mask": "subcortical_structures",
        "skull_stripping_eligible": "false",
    },
}


def _natural_key(text: str | Path) -> list[Any]:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(text))]


def _canonical_dataset_name(name: str) -> str:
    value = str(name).strip()
    return DATASET_ALIASES.get(value, value)


def _parse_dataset_list(value: str | None) -> set[str] | None:
    if not value:
        return None
    items: set[str] = set()
    for chunk in re.split(r"[,;\s]+", value):
        chunk = chunk.strip()
        if chunk:
            items.add(_canonical_dataset_name(chunk))
    return items or None


def _bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def _is_junk_member(name: str) -> bool:
    p = Path(name)
    return (
        name.startswith("__MACOSX/")
        or any(part == "__MACOSX" for part in p.parts)
        or p.name.startswith("._")
        or p.name.endswith(":Zone.Identifier")
        or not p.name
    )


def _safe_target(base: Path, member_name: str) -> Path:
    target = (base / member_name).resolve()
    root = base.resolve()
    if not str(target).startswith(str(root) + os.sep) and target != root:
        raise RuntimeError(f"unsafe archive member: {member_name}")
    return target


def _copy_file(src: Path, dst: Path, force: bool = False) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if force or not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return dst


def _first_existing_dir(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_dir():
            return path
    return None


def _first_match(root: Path, patterns: Iterable[str], *, reject: Iterable[str] = ()) -> Path | None:
    reject_l = tuple(r.lower() for r in reject)
    for pattern in patterns:
        matches = sorted(root.rglob(pattern), key=_natural_key)
        for path in matches:
            text = str(path).lower()
            if path.is_file() and not any(r in text for r in reject_l):
                return path
    return None


def _best_file_match(
    root: Path,
    patterns: Iterable[str],
    *,
    prefer: Iterable[str] = (),
    reject: Iterable[str] = (),
) -> Path | None:
    candidates: dict[Path, tuple[int, int, str]] = {}
    prefer_l = tuple(p.lower() for p in prefer)
    reject_l = tuple(r.lower() for r in reject)
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            text = str(path).lower()
            prefer_score = sum(1 for token in prefer_l if token and token in text)
            reject_score = sum(1 for token in reject_l if token and token in text)
            candidates[path] = (-prefer_score, reject_score, str(path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (candidates[p][0], candidates[p][1], _natural_key(candidates[p][2])))[0]


def _copy_tree_files(src_dir: Path, dst_dir: Path, force: bool = False) -> int:
    count = 0
    for src in sorted(src_dir.rglob("*"), key=_natural_key):
        if not src.is_file() or _is_junk_member(str(src.relative_to(src_dir))):
            continue
        rel = src.relative_to(src_dir)
        _copy_file(src, dst_dir / rel, force)
        count += 1
    return count


def _safe_extract_zip(zip_path: Path, out_dir: Path, force: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or _is_junk_member(info.filename):
                continue
            target = _safe_target(out_dir, info.filename)
            if target.exists() and not force:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def _safe_extract_tar(tar_path: Path, out_dir: Path, force: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            if not member.isfile() or _is_junk_member(member.name):
                continue
            target = _safe_target(out_dir, member.name)
            if target.exists() and not force:
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def _read_nifti(path: Path):
    if nib is None:
        raise RuntimeError("nibabel is required")
    return nib.load(str(path))


def _save_nifti(data: np.ndarray, affine: np.ndarray, header: Any, out_path: Path, dtype=np.float32) -> Path:
    if nib is None:
        raise RuntimeError("nibabel is required")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(dtype), affine, header)
    img.set_data_dtype(dtype)
    nib.save(img, str(out_path))
    return out_path


def _same_spatial_grid(img: Any, ref_img: Any, atol: float = 1e-3) -> bool:
    return img.shape[:3] == ref_img.shape[:3] and np.allclose(img.affine, ref_img.affine, atol=atol)


def _needs_spatial_regeneration(path: Path, ref_img: Any, force: bool = False) -> bool:
    if force or not path.exists():
        return True
    try:
        return not _same_spatial_grid(_read_nifti(path), ref_img)
    except Exception:
        return True


def _resample_label_to_reference(label_img: Any, ref_img: Any) -> np.ndarray:
    if _same_spatial_grid(label_img, ref_img):
        return np.asanyarray(label_img.dataobj)
    if resample_from_to is None:
        raise RuntimeError("nibabel.processing.resample_from_to is required for spatial label resampling")
    label_resampled = resample_from_to(label_img, ref_img, order=0, cval=0)
    return np.asanyarray(label_resampled.dataobj)


def _convert_nifti(src: Path, dst: Path, force: bool = False, dtype=None) -> Path:
    if not force and dst.exists():
        return dst
    img = _read_nifti(src)
    data = np.asanyarray(img.dataobj)
    if dtype is None:
        dtype = data.dtype
    return _save_nifti(data, img.affine, img.header, dst, dtype=dtype)


def _copy_or_convert_nifti(src: Path, dst: Path, force: bool = False, dtype=None) -> Path:
    if src.name.lower().endswith(".nii.gz") and (dtype is None or dtype == np.float32):
        return _copy_file(src, dst, force)
    return _convert_nifti(src, dst, force=force, dtype=dtype)


def _squeeze_singleton_volume(data: np.ndarray) -> np.ndarray:
    if data.ndim == 4 and data.shape[-1] == 1:
        return np.ascontiguousarray(data[..., 0])
    return data


def _convert_lpba_analyze_to_nifti(
    src: Path,
    dst: Path,
    force: bool = False,
    *,
    dtype=np.float32,
    binary_mask: bool = False,
) -> Path:
    """Convert LPBA Analyze pairs to 3D NIfTI for runtime tools.

    LPBA40 is distributed as ``.hdr`` + compressed ``.img.gz`` pairs. nibabel
    can read that directly, but other tools in the M1 chain may request the
    classic uncompressed ``.img`` sidecar. Converting once to NIfTI avoids that
    brittle mixed Analyze behavior.
    """

    if not force and dst.exists():
        return dst
    read_src = src
    tmp_ctx = None
    if not src.with_suffix(".img").exists() and src.with_suffix(".img.gz").exists():
        tmp_ctx = tempfile.TemporaryDirectory()
        tmp_dir = Path(tmp_ctx.name)
        read_src = tmp_dir / src.name
        shutil.copy2(src, read_src)
        with gzip.open(src.with_suffix(".img.gz"), "rb") as gz_src, read_src.with_suffix(".img").open("wb") as raw_dst:
            shutil.copyfileobj(gz_src, raw_dst)
    try:
        img = _read_nifti(read_src)
        data = _squeeze_singleton_volume(np.asanyarray(img.dataobj).copy())
        affine = np.array(img.affine, copy=True)
        header = img.header.copy()
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
    if binary_mask:
        data = data > 0
        dtype = np.uint8
    return _save_nifti(data, affine, header, dst, dtype=dtype)


def _save_mask(mask: np.ndarray, ref_img: Any, dst: Path, force: bool = False) -> Path:
    if not force and dst.exists():
        return dst
    return _save_nifti(mask.astype(np.uint8), ref_img.affine, ref_img.header, dst, dtype=np.uint8)


def _save_filled_mask(mask: np.ndarray, ref_img: Any, dst: Path) -> Path:
    fslmaths = shutil.which("fslmaths")
    if not fslmaths:
        raise RuntimeError("fslmaths is required to fill holes in the ALBERTs brain envelope")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent) as tmp:
        raw_mask = Path(tmp) / "unfilled_mask.nii.gz"
        _save_mask(mask, ref_img, raw_mask, force=True)
        proc = subprocess.run(
            [fslmaths, str(raw_mask), "-bin", "-fillh26", str(dst)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(f"fslmaths -fillh26 failed for {dst}: {proc.stdout[-1000:]}")
    return dst


def _mask_stats(path: Path) -> dict[str, Any]:
    try:
        img = _read_nifti(path)
        mask = np.asanyarray(img.dataobj) > 0
        vox = int(mask.sum())
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        return {
            "mask_voxels": vox,
            "mask_volume_cm3": float(vox * np.prod(zooms) / 1000.0),
            "shape": "x".join(str(int(v)) for v in img.shape[:3]),
            "zooms": "x".join(f"{float(v):g}" for v in zooms),
        }
    except Exception as exc:
        return {"mask_voxels": "", "mask_volume_cm3": "", "shape": "", "zooms": "", "notes": f"mask_stats_error={exc}"}


def _image_shape_zooms(path: Path) -> dict[str, str]:
    try:
        img = _read_nifti(path)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        return {
            "shape": "x".join(str(int(v)) for v in img.shape[:3]),
            "zooms": "x".join(f"{float(v):g}" for v in zooms),
        }
    except Exception:
        return {"shape": "", "zooms": ""}


def _row(**kwargs: Any) -> dict[str, str]:
    data = {k: "" for k in STANDARD_FIELDS}
    for key, value in kwargs.items():
        if key not in data:
            continue
        if isinstance(value, bool):
            data[key] = _bool(value)
        elif isinstance(value, Path):
            data[key] = str(value)
        elif value is None:
            data[key] = ""
        elif isinstance(value, float):
            data[key] = f"{value:.6g}"
        else:
            data[key] = str(value)
    return data


def _note_join(*parts: Any) -> str:
    tokens: list[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip().strip(";")
        if text:
            tokens.append(text)
    return ";".join(tokens)


def _strict_source_notes(dataset: str, *extra: Any) -> str:
    meta = STRICT_PEDIATRIC_SOURCE_METADATA.get(dataset, {})
    tokens = [f"{key}={value}" for key, value in meta.items() if value not in {"", None}]
    return _note_join(*tokens, *extra)


def _safe_subject_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return cleaned or "unknown"


def _is_nifti_like(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".nii", ".nii.gz", ".hdr"))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STANDARD_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _truthy(row: dict[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in {"1", "true", "yes", "y"}


def _atlas_library_name(dataset: str) -> str:
    mapping = {
        "IBSR18": "ibsr18_loo_full",
        "LPBA40": "lpba40_loo",
        "iSeg2017": "iseg2017_loo",
        "NFBS": "nfbs_loo",
        "SynthStripT1": "synthstript1_loo",
        "BOBs": "bobs_loo",
        "MRBrainS13": "mrbrains13_loo",
    }
    return mapping.get(dataset, f"{dataset.lower()}_loo")


class DatasetStandardizer:
    def __init__(
        self,
        data_root: Path,
        output_root: Path,
        force: bool = False,
        smoke: int | None = None,
        datasets: set[str] | None = None,
    ):
        self.data_root = data_root
        self.output_root = output_root
        self.force = force
        self.smoke = smoke
        self.requested_datasets = datasets
        self.manifest_dir = output_root / "manifests"
        self.rows_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.summary: dict[str, Any] = {
            "data_root": str(data_root),
            "output_root": str(output_root),
            "force": force,
            "requested_datasets": sorted(datasets) if datasets else "all",
            "datasets": {},
            "warnings": [],
        }

    def run(self) -> dict[str, Any]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self._preflight()
        if self._selected("IBSR18"):
            self._standardize_ibsr18()
        if self._selected("NFBS"):
            self._standardize_nfbs()
        if self._selected("iSeg2017"):
            self._standardize_iseg2017()
        if self._selected("MRBrainS13"):
            self._standardize_mrbrains13()
        if self._selected("SynthStrip", "SynthStripT1", "SynthStripInfant"):
            self._standardize_synthstrip()
        if self._selected("BOBs"):
            self._standardize_bobs()
        if self._selected("BrainMetShare"):
            self._standardize_brainmetshare()
        if self._selected("LPBA40"):
            self._standardize_lpba40_if_present()
        if self._selected("MCRIB"):
            self._standardize_mcrib()
        if self._selected("MCRIB_WM"):
            self._standardize_mcrib_wm()
        if self._selected("ALBERTs"):
            self._standardize_alberts()
        if self._selected("FetalFMRI"):
            self._standardize_fetal_fmri()
        if self._selected("BraTSPEDs"):
            self._standardize_brats_peds()
        if self._selected("HyKid"):
            self._standardize_hykid()
        if self._selected("CANDIShare"):
            self._standardize_candishare()
        if self._selected("LowFieldInfantMRI"):
            self._standardize_lowfield_infant_mri()
        if self._selected("LISA2025"):
            self._standardize_lisa2025()
        self._record_manual_access_statuses()
        self._write_manifests()
        self._build_loo_libraries()
        _write_json(self.manifest_dir / "standardization_summary.json", self.summary)
        return self.summary

    def _selected(self, *datasets: str) -> bool:
        if self.requested_datasets is None:
            return True
        return any(_canonical_dataset_name(ds) in self.requested_datasets for ds in datasets)

    def _record_manual_access_statuses(self) -> None:
        if self.requested_datasets is None:
            return
        for dataset in sorted(self.requested_datasets):
            if dataset not in MANUAL_ACCESS_DATASETS:
                continue
            status = dict(MANUAL_ACCESS_DATASETS[dataset])
            status.setdefault("rows", 0)
            self.summary["datasets"].setdefault(dataset, status)

    def _preflight(self) -> None:
        usage = shutil.disk_usage(self.output_root if self.output_root.exists() else ".")
        source_size = 0
        for p in self.data_root.rglob("*"):
            if p.is_file() and not p.name.endswith(":Zone.Identifier"):
                source_size += p.stat().st_size
        self.summary["preflight"] = {
            "source_size_gb": round(source_size / (1024**3), 3),
            "free_space_gb": round(usage.free / (1024**3), 3),
        }
        if usage.free < source_size * 1.15:
            self.summary["warnings"].append("free space may be tight for full real-copy standardization")

    def _add_rows(self, dataset: str, rows: Iterable[dict[str, str]]) -> None:
        incoming = list(rows)
        self.rows_by_dataset[dataset].extend(incoming)
        self.summary["datasets"].setdefault(dataset, {})
        self.summary["datasets"][dataset]["rows"] = len(self.rows_by_dataset[dataset])

    def _standardize_ibsr18(self) -> None:
        src = self.data_root / "IBSR"
        if not src.exists():
            self.summary["warnings"].append("IBSR source missing")
            return
        raw_root = self.output_root / "IBSR18" / "raw"
        top_root = self.output_root / "IBSR18"
        t1_tgz = src / "20Normals_T1.tgz"
        brain_tgz = src / "20Normals_T1_brain.tgz"
        if not t1_tgz.exists() or not brain_tgz.exists():
            self.summary["warnings"].append("IBSR archives incomplete")
            return
        _safe_extract_tar(t1_tgz, raw_root, self.force)
        _safe_extract_tar(brain_tgz, raw_root, self.force)
        # Legacy and notebook-compatible aliases as real copies.
        if (raw_root / "20Normals_T1").exists():
            _copy_tree_files(raw_root / "20Normals_T1", top_root / "20Normals_T1", self.force)
        seg_src = raw_root / "20Normals_T1_brain"
        if seg_src.exists():
            _copy_tree_files(seg_src, top_root / "20Normals_T1_brain", self.force)
            _copy_tree_files(seg_src, raw_root / "20Normals_T1_seg", self.force)

        nifti_root = top_root / "nifti"
        rows = []
        for subj_dir in sorted((raw_root / "20Normals_T1").iterdir(), key=lambda p: _natural_key(p.name)):
            if not subj_dir.is_dir():
                continue
            raw = subj_dir.name
            if raw not in IBSR18_SEG_OFFSETS:
                continue
            seg_hdr = raw_root / "20Normals_T1_seg" / f"{raw}.hdr"
            seg_buchar = raw_root / "20Normals_T1_seg" / f"{raw}.buchar"
            if not seg_hdr.exists() or not seg_buchar.exists():
                continue
            t1_data, mask_data, meta = self._load_ibsr_pair(raw, subj_dir, seg_hdr, seg_buchar)
            affine = np.diag([1.0, 3.0, 1.0, 1.0])
            t1_out = nifti_root / f"{raw}_T1w.nii.gz"
            mask_out = nifti_root / f"{raw}_brainmask.nii.gz"
            if self.force or not t1_out.exists():
                _save_nifti(t1_data, affine, None, t1_out, dtype=np.float32)
            if self.force or not mask_out.exists():
                _save_nifti(mask_data > 0, affine, None, mask_out, dtype=np.uint8)
            stats = _mask_stats(mask_out)
            rows.append(
                _row(
                    dataset="IBSR18",
                    subject=f"IBSR_{raw}",
                    split="gold",
                    has_gold=True,
                    qc_only=False,
                    atlas_eligible=True,
                    modality="T1w",
                    profile="adult_t1",
                    t1_path=t1_out,
                    input_path=t1_out,
                    brain_mask_path=mask_out,
                    label_path=seg_buchar,
                    source_path=subj_dir,
                    standardized_dir=nifti_root,
                    shape=stats.get("shape"),
                    zooms=stats.get("zooms"),
                    mask_voxels=stats.get("mask_voxels"),
                    mask_volume_cm3=stats.get("mask_volume_cm3"),
                    notes=f"seg_offset={meta['seg_offset']}",
                )
            )
        self._add_rows("IBSR18", rows)

    def _load_ibsr_pair(self, raw: str, t1_dir: Path, seg_hdr: Path, seg_buchar: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        pairs = []
        for p in t1_dir.glob(f"{raw}_*.img"):
            if p.name.startswith("._"):
                continue
            match = re.search(r"_(-?\d+)\.img$", p.name)
            if match:
                pairs.append((int(match.group(1)), p))
        pairs.sort(key=lambda item: item[0])
        if not pairs:
            raise RuntimeError(f"No IBSR slices for {raw}")
        t1_slices = []
        for _, p in pairs:
            arr = np.fromfile(p, dtype=">u2")
            if arr.size != 256 * 256:
                raise RuntimeError(f"Invalid IBSR slice {p}: {arr.size}")
            t1_slices.append(arr.reshape(256, 256).T)
        t1_native = np.stack(t1_slices, axis=-1).astype(np.float32)
        cols, rows, n_seg, _byteswap = [int(v) for v in seg_hdr.read_text().split()[:4]]
        seg = np.fromfile(seg_buchar, dtype=np.uint8)
        if seg.size != cols * rows * n_seg:
            plane = cols * rows
            if plane > 0 and seg.size % plane == 0:
                n_seg = int(seg.size // plane)
            else:
                raise RuntimeError(f"Invalid IBSR segmentation {seg_buchar}: {seg.size}")
        seg_xyz = seg.reshape((n_seg, rows, cols)).transpose(2, 1, 0)
        offset = IBSR18_SEG_OFFSETS[raw]
        mask_native = np.zeros(t1_native.shape, dtype=np.uint8)
        slice_numbers = [n for n, _ in pairs]
        for z, t1_slice_number in enumerate(slice_numbers):
            seg_index = t1_slice_number - offset
            if 1 <= seg_index <= n_seg:
                mask_native[:, :, z] = (seg_xyz[:, :, seg_index - 1] > 0).astype(np.uint8)

        def to_ras(data: np.ndarray) -> np.ndarray:
            ras = np.transpose(data, (0, 2, 1))
            ras = ras[:, :, ::-1]
            return np.ascontiguousarray(ras)

        return to_ras(t1_native), to_ras(mask_native), {"seg_offset": offset}

    def _standardize_nfbs(self) -> None:
        src_base = self.data_root / "NFBS" / "NFBS_Dataset" / "NFBS_Dataset"
        if not src_base.exists():
            legacy = self.data_root / "NBSR" / "NFBS_Dataset" / "NFBS_Dataset"
            src_base = legacy if legacy.exists() else src_base
        if not src_base.exists():
            self.summary["warnings"].append("NFBS source missing")
            return
        legacy_dst = self.output_root / "NFBS" / "NFBS_Dataset" / "NFBS_Dataset"
        rows = []
        for subj_dir in sorted([p for p in src_base.iterdir() if p.is_dir()], key=lambda p: _natural_key(p.name)):
            sid = subj_dir.name
            t1 = next(iter(sorted(subj_dir.glob("*_T1w.nii.gz"))), None)
            mask = next(iter(sorted(subj_dir.glob("*_brainmask.nii.gz"))), None)
            brain = next(iter(sorted(subj_dir.glob("*_brain.nii.gz"))), None)
            if not t1 or not mask:
                continue
            compat_dir = legacy_dst / sid
            compat_t1 = _copy_file(t1, compat_dir / t1.name, self.force)
            compat_mask = _copy_file(mask, compat_dir / mask.name, self.force)
            if brain:
                _copy_file(brain, compat_dir / brain.name, self.force)
            anat = self.output_root / "standardized" / "NFBS" / sid / "anat"
            std_t1 = _copy_file(t1, anat / f"sub-{sid}_T1w.nii.gz", self.force)
            std_mask = _copy_file(mask, anat / f"sub-{sid}_desc-brain_mask.nii.gz", self.force)
            stats = _mask_stats(std_mask)
            rows.append(
                _row(
                    dataset="NFBS",
                    subject=sid,
                    split="gold",
                    has_gold=True,
                    qc_only=False,
                    atlas_eligible=True,
                    modality="T1w",
                    profile="adult_t1",
                    t1_path=std_t1,
                    input_path=std_t1,
                    brain_mask_path=std_mask,
                    source_path=compat_t1,
                    standardized_dir=anat,
                    shape=stats.get("shape"),
                    zooms=stats.get("zooms"),
                    mask_voxels=stats.get("mask_voxels"),
                    mask_volume_cm3=stats.get("mask_volume_cm3"),
                )
            )
        self._add_rows("NFBS", rows)

    def _standardize_iseg2017(self) -> None:
        zip_path = self.data_root / "iSeg-2017" / "iSeg-2017.zip"
        if not zip_path.exists():
            self.summary["warnings"].append("iSeg-2017 archive missing")
            return
        raw_root = self.output_root / "iSeg2017" / "raw"
        _safe_extract_zip(zip_path, raw_root, self.force)
        rows = []
        for split_dir, split, has_gold in [
            (raw_root / "iSeg-2017-Training", "train_labeled", True),
            (raw_root / "iSeg-2017-Testing", "test_unlabeled", False),
        ]:
            if not split_dir.exists():
                continue
            t1_files = sorted(split_dir.glob("subject-*-T1.hdr"), key=lambda p: _natural_key(p.name))
            for t1_hdr in t1_files:
                match = re.search(r"subject-(\d+)-T1\.hdr", t1_hdr.name)
                if not match:
                    continue
                num = int(match.group(1))
                subject = f"sub-{num:03d}"
                t2_hdr = split_dir / f"subject-{num}-T2.hdr"
                label_hdr = split_dir / f"subject-{num}-label.hdr"
                anat = self.output_root / "standardized" / "iSeg2017" / subject / "anat"
                t1_out = _convert_nifti(t1_hdr, anat / f"{subject}_T1w.nii.gz", self.force, dtype=np.float32)
                t2_out = _convert_nifti(t2_hdr, anat / f"{subject}_T2w.nii.gz", self.force, dtype=np.float32) if t2_hdr.exists() else Path("")
                label_out = Path("")
                mask_out = Path("")
                stats: dict[str, Any] = _image_shape_zooms(t1_out)
                if has_gold and label_hdr.exists():
                    label_img = _read_nifti(label_hdr)
                    label = np.asanyarray(label_img.dataobj)
                    label_out = _convert_nifti(label_hdr, anat / f"{subject}_desc-label_dseg.nii.gz", self.force, dtype=np.uint8)
                    mask_out = _save_mask(label > 0, label_img, anat / f"{subject}_desc-brain_mask.nii.gz", self.force)
                    stats = _mask_stats(mask_out)
                row = _row(
                    dataset="iSeg2017",
                    subject=subject,
                    split=split,
                    has_gold=has_gold and bool(mask_out),
                    qc_only=not has_gold,
                    atlas_eligible=has_gold and bool(mask_out),
                    modality="T1w+T2w",
                    profile="infant_t1t2",
                    t1_path=t1_out,
                    t2_path=t2_out,
                    input_path=t1_out,
                    brain_mask_path=mask_out,
                    label_path=label_out,
                    source_path=t1_hdr,
                    standardized_dir=anat,
                    shape=stats.get("shape"),
                    zooms=stats.get("zooms"),
                    mask_voxels=stats.get("mask_voxels", ""),
                    mask_volume_cm3=stats.get("mask_volume_cm3", ""),
                    notes=_strict_source_notes("iSeg2017") if has_gold and bool(mask_out) else "unlabeled_test_split",
                )
                rows.append(row)
        self._add_rows("iSeg2017", rows)
        # Exact notebook manifest name/columns.
        train_rows = [r for r in rows if r["split"] == "train_labeled" and _truthy(r, "has_gold")]
        manifest = self.output_root / "iSeg2017" / "training_manifest.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="") as f:
            fields = ["subject", "split", "has_gold", "t1_path", "t2_path", "brain_mask_path", "label_path"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in train_rows:
                writer.writerow({k: r[k] for k in fields})

    def _standardize_mrbrains13(self) -> None:
        archive = self.data_root / "MRBrainS13" / "MRBrainS13-doi-10.34894-645zin.zip"
        if not archive.exists():
            self.summary["warnings"].append("MRBrainS13 archive missing")
            return
        raw_root = self.output_root / "MRBrainS13" / "raw"
        raw_root.mkdir(parents=True, exist_ok=True)
        _safe_extract_zip(archive, raw_root, self.force)
        for inner in ["TrainingData.zip", "TestData.zip"]:
            p = raw_root / inner
            if p.exists():
                _safe_extract_zip(p, raw_root, self.force)
        rows = []
        for split_name, split in [("TrainingData", "train"), ("TestData", "test")]:
            base = raw_root / split_name
            if not base.exists():
                continue
            for subj_dir in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: _natural_key(p.name)):
                sid = f"{split_name.lower()}-{int(subj_dir.name):02d}" if subj_dir.name.isdigit() else f"{split_name.lower()}-{subj_dir.name}"
                label = subj_dir / "LabelsForTesting.nii"
                if not label.exists():
                    label = subj_dir / "LabelsForTraining.nii"
                if not label.exists():
                    continue
                t1_candidates = [subj_dir / "T1_1mm.nii", subj_dir / "T1.nii"]
                t1 = next((p for p in t1_candidates if p.exists()), None)
                if t1 is None:
                    continue
                anat = self.output_root / "standardized" / "MRBrainS13" / sid / "anat"
                t1_out = _convert_nifti(t1, anat / f"sub-{sid}_T1w.nii.gz", self.force, dtype=np.float32)
                t1_img = _read_nifti(t1_out)
                label_img = _read_nifti(label)
                label_data = _resample_label_to_reference(label_img, t1_img)
                label_out = anat / f"sub-{sid}_desc-label_dseg.nii.gz"
                if _needs_spatial_regeneration(label_out, t1_img, self.force):
                    _save_nifti(label_data, t1_img.affine, t1_img.header, label_out, dtype=np.uint8)
                mask_out = anat / f"sub-{sid}_desc-brain_mask.nii.gz"
                if _needs_spatial_regeneration(mask_out, t1_img, self.force):
                    _save_mask(label_data > 0, t1_img, mask_out, force=True)
                stats = _mask_stats(mask_out)
                rows.append(
                    _row(
                        dataset="MRBrainS13",
                        subject=sid,
                        split=split,
                        has_gold=True,
                        qc_only=False,
                        atlas_eligible=True,
                        modality="T1w",
                        profile="adult_t1",
                        t1_path=t1_out,
                        input_path=t1_out,
                        brain_mask_path=mask_out,
                        label_path=label_out,
                        source_path=t1,
                        standardized_dir=anat,
                        shape=stats.get("shape"),
                        zooms=stats.get("zooms"),
                        mask_voxels=stats.get("mask_voxels"),
                        mask_volume_cm3=stats.get("mask_volume_cm3"),
                        notes="mrbrains13_labels_resampled_to_t1_grid",
                    )
                )
        self._add_rows("MRBrainS13", rows)

    def _standardize_synthstrip(self) -> None:
        src = self.data_root / "SynthStrip" / "synthstrip_data_v1.5"
        if not src.exists():
            self.summary["warnings"].append("SynthStrip source missing")
            return
        rows_all = []
        rows_t1 = []
        rows_infant = []
        for subj_dir in sorted([p for p in src.iterdir() if p.is_dir()], key=lambda p: _natural_key(p.name)):
            image = subj_dir / "image.nii.gz"
            mask = subj_dir / "mask.nii.gz"
            if not image.exists() or not mask.exists():
                continue
            sid = subj_dir.name
            prefix = sid.split("_", 1)[0]
            modality = sid.split("_", 2)[1] if "_" in sid else "unknown"
            is_t1 = bool(re.search(r"(^|_)t1(_|$)", sid)) or sid.startswith(("fsm_t1_", "asl_t1_", "infant_t1_"))
            anat = self.output_root / "standardized" / "SynthStrip" / sid / "anat"
            std_img = _copy_file(image, anat / f"sub-{sid}_T1w.nii.gz", self.force)
            std_mask = _copy_file(mask, anat / f"sub-{sid}_desc-brain_mask.nii.gz", self.force)
            labels = subj_dir / "labels.nii.gz"
            std_label: Path | str = (
                _copy_file(labels, anat / f"sub-{sid}_desc-label_dseg.nii.gz", self.force) if labels.exists() else ""
            )
            stats = _mask_stats(std_mask)
            row = _row(
                dataset="SynthStrip",
                subject=sid,
                split="gold",
                has_gold=True,
                qc_only=not is_t1,
                atlas_eligible=is_t1,
                modality=modality,
                profile="infant_t1" if sid.startswith("infant_") else "adult_t1" if is_t1 else "non_t1_qc",
                t1_path=std_img if is_t1 else "",
                input_path=std_img,
                brain_mask_path=std_mask,
                label_path=std_label,
                source_path=image,
                standardized_dir=anat,
                shape=stats.get("shape"),
                zooms=stats.get("zooms"),
                mask_voxels=stats.get("mask_voxels"),
                mask_volume_cm3=stats.get("mask_volume_cm3"),
                notes=f"source_subset={prefix};t1_like={_bool(is_t1)}",
            )
            rows_all.append(row)
            if is_t1:
                t1_row = dict(row)
                t1_row["dataset"] = "SynthStripT1"
                rows_t1.append(t1_row)
            if sid.startswith("infant_t1_"):
                infant_row = dict(row)
                infant_row.update(
                    {
                        "dataset": "SynthStripInfant",
                        "qc_only": "false",
                        "atlas_eligible": "false",
                        "modality": "T1w",
                        "profile": "infant_t1",
                        "notes": _strict_source_notes(
                            "SynthStripInfant",
                            "source_subset=infant",
                            "full_head_image=true",
                            "evaluation_only=true",
                        ),
                    }
                )
                rows_infant.append(infant_row)
        self._add_rows("SynthStrip", rows_all)
        self._add_rows("SynthStripT1", rows_t1)
        self._add_rows("SynthStripInfant", rows_infant)

    def _standardize_bobs(self) -> None:
        src = self.data_root / "BOBs" / "ds005450"
        if not src.exists():
            self.summary["warnings"].append("BOBs source missing")
            return
        rows = []
        for anat_src in sorted(src.glob("sub-*/ses-*/anat"), key=lambda p: _natural_key(str(p))):
            subject_id = anat_src.parts[-3]
            session = anat_src.parts[-2]
            sid = f"{subject_id}_{session}"
            t1 = next(iter(sorted(anat_src.glob("*_T1w.nii.gz"))), None)
            t2 = next(iter(sorted(anat_src.glob("*_T2w.nii.gz"))), None)
            mask = next(iter(sorted(anat_src.glob("*desc-aseg_brainmask.nii.gz"))), None)
            dseg = next(iter(sorted(anat_src.glob("*desc-aseg_dseg.nii.gz"))), None)
            if not t1 or not mask:
                continue
            anat = self.output_root / "standardized" / "BOBs" / sid / "anat"
            std_t1 = _copy_file(t1, anat / f"{subject_id}_{session}_T1w.nii.gz", self.force)
            std_t2 = _copy_file(t2, anat / f"{subject_id}_{session}_T2w.nii.gz", self.force) if t2 else Path("")
            std_mask = _copy_file(mask, anat / f"{subject_id}_{session}_desc-brain_mask.nii.gz", self.force)
            std_dseg = _copy_file(dseg, anat / f"{subject_id}_{session}_desc-label_dseg.nii.gz", self.force) if dseg else Path("")
            stats = _mask_stats(std_mask)
            rows.append(
                _row(
                    dataset="BOBs",
                    subject=sid,
                    session=session,
                    split="gold",
                    has_gold=True,
                    qc_only=False,
                    atlas_eligible=True,
                    modality="T1w+T2w",
                    profile="infant_t1t2",
                    t1_path=std_t1,
                    t2_path=std_t2,
                    input_path=std_t1,
                    brain_mask_path=std_mask,
                    label_path=std_dseg,
                    source_path=t1,
                    standardized_dir=anat,
                    shape=stats.get("shape"),
                    zooms=stats.get("zooms"),
                    mask_voxels=stats.get("mask_voxels"),
                    mask_volume_cm3=stats.get("mask_volume_cm3"),
                    notes=_strict_source_notes("BOBs"),
                )
            )
        self._add_rows("BOBs", rows)

    def _mcrib_subject_ids(self, src: Path, label_patterns: Iterable[str]) -> list[str]:
        ids: set[str] = set()
        for pattern in label_patterns:
            for path in src.rglob(pattern):
                if not path.is_file():
                    continue
                match = re.search(r"_P(\d+)_|_P(\d+)\b|P(\d+)_", path.name)
                if match:
                    ids.add(next(g for g in match.groups() if g))
        return sorted(ids, key=_natural_key)

    def _standardize_mcrib_like(
        self,
        *,
        dataset: str,
        source_dirs: Iterable[Path],
        subject_prefix: str,
        label_patterns: Iterable[str],
        label_prefer: Iterable[str],
        notes: str,
    ) -> None:
        src = _first_existing_dir(source_dirs)
        if src is None:
            self.summary["warnings"].append(f"{dataset} source missing")
            return
        rows = []
        subject_ids = self._mcrib_subject_ids(src, label_patterns)
        for raw_id in subject_ids:
            nn = int(raw_id)
            label = _best_file_match(
                src,
                [pattern.replace("P??", f"P{nn:02d}") for pattern in label_patterns],
                prefer=label_prefer,
                reject=("readme", "license"),
            )
            t1 = _best_file_match(
                src,
                [
                    f"*P{nn:02d}*T1*registered*to*T2*.nii*",
                    f"*P{nn:02d}*T1*registered*.nii*",
                    f"*P{nn:02d}*T1*.nii*",
                ],
                prefer=("preprocessed", "registered", "to_t2"),
                reject=("raw", "diffusion", "parc", "label", "mask"),
            )
            t2 = _best_file_match(
                src,
                [
                    f"*P{nn:02d}*T2*.nii*",
                ],
                prefer=("preprocessed",),
                reject=("raw", "parc", "label", "mask"),
            )
            if label is None or t1 is None:
                self.summary["warnings"].append(f"{dataset} P{nn:02d} skipped: missing T1 or parcellation")
                continue
            subject = f"sub-{subject_prefix}-P{nn:02d}"
            anat = self.output_root / "standardized" / dataset / subject / "anat"
            std_t1 = _copy_file(t1, anat / f"{subject}_T1w.nii.gz", self.force)
            std_t2 = _copy_file(t2, anat / f"{subject}_T2w.nii.gz", self.force) if t2 else Path("")
            t1_img = _read_nifti(std_t1)
            label_img = _read_nifti(label)
            label_data = _resample_label_to_reference(label_img, t1_img)
            label_out = anat / f"{subject}_desc-label_dseg.nii.gz"
            if _needs_spatial_regeneration(label_out, t1_img, self.force):
                _save_nifti(label_data, t1_img.affine, t1_img.header, label_out, dtype=np.uint16)
            mask_out = anat / f"{subject}_desc-brain_mask.nii.gz"
            if _needs_spatial_regeneration(mask_out, t1_img, self.force):
                _save_mask(label_data > 0, t1_img, mask_out, force=True)
            stats = _mask_stats(mask_out)
            rows.append(
                _row(
                    dataset=dataset,
                    subject=subject,
                    session="ses-0mo",
                    split="gold",
                    has_gold=True,
                    qc_only=False,
                    atlas_eligible=True,
                    modality="T1w+T2w" if t2 else "T1w",
                    profile="infant_t1t2" if t2 else "infant_t1",
                    t1_path=std_t1,
                    t2_path=std_t2,
                    input_path=std_t1,
                    brain_mask_path=mask_out,
                    label_path=label_out,
                    source_path=t1,
                    standardized_dir=anat,
                    shape=stats.get("shape"),
                    zooms=stats.get("zooms"),
                    mask_voxels=stats.get("mask_voxels"),
                    mask_volume_cm3=stats.get("mask_volume_cm3"),
                    notes=_strict_source_notes(
                        dataset,
                        "GT derived from labels",
                        "label_gt0",
                        "segmentation_only_preprocessed_input=true",
                        f"source_cohort_subject=P{nn:02d}",
                        notes,
                    ),
                )
            )
        self._add_rows(dataset, rows)

    def _standardize_mcrib(self) -> None:
        self._standardize_mcrib_like(
            dataset="MCRIB",
            source_dirs=(self.data_root / "MCRIB", self.data_root / "M-CRIB"),
            subject_prefix="MCRIB",
            label_patterns=(
                "*M-CRIB_orig_P??_parc*.nii*",
                "*P??*parc*.nii*",
            ),
            label_prefer=("original m-crib parcellations", "orig", "parc"),
            notes="mcrib_original_manual_parcellation",
        )

    def _standardize_mcrib_wm(self) -> None:
        self._standardize_mcrib_like(
            dataset="MCRIB_WM",
            source_dirs=(self.data_root / "MCRIB_WM", self.data_root / "MCRIB2", self.data_root / "M-CRIB-WM"),
            subject_prefix="MCRIBWM",
            label_patterns=(
                "*M-CRIB-WM_and_M-CRIB_2-0_P??_parc_smallrange.nii*",
                "*M-CRIB-WM_and_M-CRIB_2-0_P??_parc*.nii*",
                "*P??*parc*smallrange*.nii*",
                "*P??*parc*.nii*",
            ),
            label_prefer=("combined whole brain", "smallrange", "m-crib 2-0", "m-crib-wm"),
            notes="mcrib_wm_mcrib2_combined_manual_parcellation",
        )

    def _standardize_alberts(self) -> None:
        src = self.data_root / "external_atlases" / "alberts_gousias"
        image_root = src / "images"
        label_root = src / "segmentations"
        if not image_root.exists() or not label_root.exists():
            self.summary["warnings"].append("ALBERTs source missing")
            return
        if nib is None:
            raise RuntimeError("nibabel is required")

        def canonical_3d(image: Any) -> Any:
            data = _squeeze_singleton_volume(np.asanyarray(image.dataobj))
            return nib.as_closest_canonical(nib.Nifti1Image(data, image.affine, image.header))

        rows = []
        for number in range(1, 21):
            token = f"{number:02d}"
            t1 = image_root / f"{token}_T1.nii"
            t2 = image_root / f"{token}_T2.nii"
            label = label_root / f"ALBERT_{token}_ISG_seg50.nii"
            if not t1.exists() or not label.exists():
                self.summary["warnings"].append(f"ALBERTs {token} skipped: missing T1 or manual parcellation")
                continue

            subject = f"sub-ALBERT-{token}"
            anat = self.output_root / "standardized" / "ALBERTs" / subject / "anat"
            t1_out = anat / f"{subject}_T1w.nii.gz"
            t1_src_img = canonical_3d(_read_nifti(t1))
            if _needs_spatial_regeneration(t1_out, t1_src_img, self.force):
                _save_nifti(
                    np.asanyarray(t1_src_img.dataobj),
                    t1_src_img.affine,
                    t1_src_img.header,
                    t1_out,
                    dtype=np.float32,
                )
            t1_img = _read_nifti(t1_out)

            t2_out = Path("")
            if t2.exists():
                t2_out = anat / f"{subject}_T2w.nii.gz"
                t2_src_img = canonical_3d(_read_nifti(t2))
                if _needs_spatial_regeneration(t2_out, t2_src_img, self.force):
                    _save_nifti(
                        np.asanyarray(t2_src_img.dataobj),
                        t2_src_img.affine,
                        t2_src_img.header,
                        t2_out,
                        dtype=np.float32,
                    )

            label_src_img = canonical_3d(_read_nifti(label))
            label_data = _resample_label_to_reference(label_src_img, t1_img)

            label_out = anat / f"{subject}_desc-label_dseg.nii.gz"
            if _needs_spatial_regeneration(label_out, t1_img, self.force):
                _save_nifti(label_data, t1_img.affine, t1_img.header, label_out, dtype=np.uint16)
            mask_out = anat / f"{subject}_desc-brain_mask.nii.gz"
            _save_filled_mask(label_data > 0, t1_img, mask_out)

            stats = _mask_stats(mask_out)
            rows.append(
                _row(
                    dataset="ALBERTs",
                    subject=subject,
                    session="ses-term",
                    split="gold",
                    has_gold=True,
                    qc_only=False,
                    atlas_eligible=False,
                    modality="T1w+T2w" if t2_out else "T1w",
                    profile="infant_t1t2" if t2_out else "infant_t1",
                    t1_path=t1_out,
                    t2_path=t2_out,
                    input_path=t1_out,
                    brain_mask_path=mask_out,
                    label_path=label_out,
                    source_path=t1,
                    standardized_dir=anat,
                    shape=stats.get("shape"),
                    zooms=stats.get("zooms"),
                    mask_voxels=stats.get("mask_voxels"),
                    mask_volume_cm3=stats.get("mask_volume_cm3"),
                    notes=_strict_source_notes(
                        "ALBERTs",
                        "GT derived from union of manual labels",
                        "label_gt0",
                        "brain_envelope_holes_filled=fsl_fillh26",
                        "full_head_nonbrain_signal_verified=true",
                        "evaluation_only=true",
                    ),
                )
            )
        self._add_rows("ALBERTs", rows)

    def _standardize_fetal_fmri(self) -> None:
        src = self.data_root / "FetalFMRI" / "ds003090"
        mask_root = src / "derivatives" / "manual-masks"
        if not src.exists() or not mask_root.exists():
            self.summary["warnings"].append("FetalFMRI OpenNeuro ds003090 source missing")
            return
        if nib is None:
            raise RuntimeError("nibabel is required")

        rows = []
        duplicate_frames_skipped = 0
        for image in sorted(src.glob("sub-*/ses-*/func/*_bold.nii.gz"), key=lambda p: _natural_key(str(p))):
            relative = image.relative_to(src)
            mask = mask_root / relative
            if not mask.exists():
                self.summary["warnings"].append(f"FetalFMRI skipped without paired mask: {relative}")
                continue

            image_4d = nib.as_closest_canonical(_read_nifti(image))
            mask_4d = nib.as_closest_canonical(_read_nifti(mask))
            if image_4d.shape != mask_4d.shape or len(image_4d.shape) != 4:
                self.summary["warnings"].append(
                    f"FetalFMRI skipped incompatible image/mask shapes: {relative} "
                    f"{image_4d.shape} vs {mask_4d.shape}"
                )
                continue

            subject = relative.parts[0]
            source_session = relative.parts[1]
            image_data = np.asanyarray(image_4d.dataobj)
            mask_data = np.asanyarray(mask_4d.dataobj)
            seen_frame_signatures: set[str] = set()
            for frame in range(image_4d.shape[3]):
                image_frame = np.ascontiguousarray(image_data[..., frame])
                mask_frame = np.ascontiguousarray(mask_data[..., frame] > 0)
                digest = hashlib.sha256()
                digest.update(image_frame.view(np.uint8))
                digest.update(mask_frame.view(np.uint8))
                signature = digest.hexdigest()
                if signature in seen_frame_signatures:
                    duplicate_frames_skipped += 1
                    continue
                seen_frame_signatures.add(signature)
                frame_token = f"frame-{frame:04d}"
                session = f"{source_session}_{frame_token}"
                anat = self.output_root / "standardized" / "FetalFMRI" / subject / source_session / frame_token / "anat"
                image_out = anat / f"{subject}_{source_session}_{frame_token}_bold.nii.gz"
                mask_out = anat / f"{subject}_{source_session}_{frame_token}_desc-brain_mask.nii.gz"
                if self.force or not image_out.exists():
                    _save_nifti(
                        image_data[..., frame],
                        image_4d.affine,
                        image_4d.header,
                        image_out,
                        dtype=np.float32,
                    )
                image_ref = _read_nifti(image_out)
                if _needs_spatial_regeneration(mask_out, image_ref, self.force):
                    _save_mask(mask_frame, image_ref, mask_out, force=True)
                stats = _mask_stats(mask_out)
                rows.append(
                    _row(
                        dataset="FetalFMRI",
                        subject=subject,
                        session=session,
                        split="gold_stress_test",
                        has_gold=True,
                        qc_only=False,
                        atlas_eligible=False,
                        modality="BOLD",
                        profile="fetal_fmri",
                        input_path=image_out,
                        brain_mask_path=mask_out,
                        source_path=image,
                        standardized_dir=anat,
                        shape=stats.get("shape"),
                        zooms=stats.get("zooms"),
                        mask_voxels=stats.get("mask_voxels"),
                        mask_volume_cm3=stats.get("mask_volume_cm3"),
                        notes=_strict_source_notes(
                            "FetalFMRI",
                            f"source_cohort_subject={subject}",
                            f"source_session={source_session}",
                            f"frame_index={frame}",
                            f"source_mask={mask}",
                            "full_field_fetal_functional_image=true",
                            "count_as_unique_subject_not_frame=true",
                        ),
                    )
                )
        self._add_rows("FetalFMRI", rows)
        self.summary["datasets"]["FetalFMRI"]["duplicate_frames_skipped"] = duplicate_frames_skipped

    def _manual_search_roots(self, dataset: str, source_names: Iterable[str]) -> list[Path]:
        roots: list[Path] = []
        for name in source_names:
            root = self.data_root / name
            if root.exists() and root.is_dir():
                roots.append(root)
                raw_root = self.output_root / dataset / "raw"
                extracted = 0
                for archive in sorted(root.glob("*"), key=_natural_key):
                    lower = archive.name.lower()
                    try:
                        if lower.endswith(".zip"):
                            extracted += _safe_extract_zip(archive, raw_root / archive.stem, self.force)
                        elif lower.endswith((".tar.gz", ".tgz", ".tar")):
                            extracted += _safe_extract_tar(archive, raw_root / re.sub(r"\.(tar\.gz|tgz|tar)$", "", archive.name, flags=re.I), self.force)
                    except Exception as exc:
                        self.summary["warnings"].append(f"{dataset} archive extraction skipped for {archive}: {exc}")
                if raw_root.exists() and (extracted or any(raw_root.rglob("*"))):
                    roots.insert(0, raw_root)
        # Preserve order but avoid scanning the same tree twice.
        unique: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            resolved = root.resolve()
            if resolved not in seen:
                unique.append(root)
                seen.add(resolved)
        return unique

    def _subject_session_from_path(
        self,
        path: Path,
        root: Path,
        *,
        regex: str | None = None,
    ) -> tuple[str, str]:
        text = str(path)
        if regex:
            match = re.search(regex, text)
            if match:
                subject = match.group(1) if match.groups() else match.group(0)
                return _safe_subject_token(subject), ""
        try:
            parts = path.relative_to(root).parts
        except Exception:
            parts = path.parts
        subject = next((p for p in parts if p.lower().startswith("sub-")), "")
        session = next((p for p in parts if p.lower().startswith("ses-")), "")
        if not subject:
            for parent in [path.parent, *path.parents]:
                name = parent.name
                if name.lower() in {"anat", "mri", "seg", "segs", "label", "labels", "derivatives", "manual", "masks"}:
                    continue
                if name and name != root.name:
                    subject = name
                    break
        return _safe_subject_token(subject), _safe_subject_token(session) if session else ""

    def _find_image_for_label(
        self,
        root: Path,
        label: Path,
        subject: str,
        session: str,
        image_patterns: Iterable[str],
        image_prefer: Iterable[str],
    ) -> Path | None:
        reject = (
            "seg",
            "dseg",
            "label",
            "labels",
            "mask",
            "manual",
            "annotation",
            "parc",
            label.name.lower(),
        )
        search_dirs: list[Path] = []
        current = label.parent
        while True:
            search_dirs.append(current)
            if current == root or current.parent == current:
                break
            current = current.parent
            if len(search_dirs) >= 5:
                break
        for directory in search_dirs:
            image = _best_file_match(directory, image_patterns, prefer=image_prefer, reject=reject)
            if image is not None and image != label:
                return image
        subject_tokens = [subject]
        if subject.startswith("sub-"):
            subject_tokens.append(subject[4:])
        candidates: list[Path] = []
        for pattern in image_patterns:
            for image in root.rglob(pattern):
                if not image.is_file() or image == label or not _is_nifti_like(image):
                    continue
                text = str(image).lower()
                if any(r in text for r in reject):
                    continue
                if all(token.lower() not in text for token in subject_tokens if token):
                    continue
                if session and session.lower() not in text:
                    continue
                candidates.append(image)
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda p: (
                -sum(1 for token in image_prefer if token.lower() in str(p).lower()),
                _natural_key(str(p)),
            ),
        )[0]

    def _standardize_generic_label_dataset(
        self,
        *,
        dataset: str,
        source_names: Iterable[str],
        label_patterns: Iterable[str],
        label_prefer: Iterable[str],
        image_patterns: Iterable[str],
        image_prefer: Iterable[str],
        modality: str,
        profile: str,
        subject_regex: str | None = None,
        atlas_eligible: bool = False,
        extra_note: str = "",
    ) -> None:
        roots = self._manual_search_roots(dataset, source_names)
        if not roots:
            self.summary["warnings"].append(f"{dataset} source missing")
            return
        grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
        for root in roots:
            for pattern in label_patterns:
                for label in sorted(root.rglob(pattern), key=_natural_key):
                    if not label.is_file() or not _is_nifti_like(label):
                        continue
                    if _is_junk_member(str(label)):
                        continue
                    subject, session = self._subject_session_from_path(label, root, regex=subject_regex)
                    if subject:
                        grouped[(subject, session)].append(label)
        rows = []
        seen: set[tuple[str, str]] = set()
        for (subject, session), labels in sorted(grouped.items(), key=lambda kv: _natural_key("_".join(kv[0]))):
            if (subject, session) in seen:
                continue
            label = sorted(
                set(labels),
                key=lambda p: (
                    -sum(1 for token in label_prefer if token.lower() in str(p).lower()),
                    _natural_key(str(p)),
                ),
            )[0]
            root = next((r for r in roots if str(label.resolve()).startswith(str(r.resolve()))), roots[0])
            image = self._find_image_for_label(root, label, subject, session, image_patterns, image_prefer)
            if image is None:
                self.summary["warnings"].append(f"{dataset} {subject} skipped: no MRI image matched label {label}")
                continue
            stem = subject if not session else f"{subject}_{session}"
            anat = self.output_root / "standardized" / dataset / stem / "anat"
            suffix = "T2w" if modality.lower().startswith("t2") else "T1w" if modality.lower().startswith("t1") else "MRI"
            image_out = _copy_or_convert_nifti(image, anat / f"{stem}_{suffix}.nii.gz", self.force, dtype=np.float32)
            ref_img = _read_nifti(image_out)
            label_img = _read_nifti(label)
            label_data = _resample_label_to_reference(label_img, ref_img)
            label_out = anat / f"{stem}_desc-label_dseg.nii.gz"
            if _needs_spatial_regeneration(label_out, ref_img, self.force):
                _save_nifti(label_data, ref_img.affine, ref_img.header, label_out, dtype=np.uint16)
            mask_out = anat / f"{stem}_desc-eval_mask.nii.gz"
            if _needs_spatial_regeneration(mask_out, ref_img, self.force):
                _save_mask(label_data > 0, ref_img, mask_out, force=True)
            stats = _mask_stats(mask_out)
            rows.append(
                _row(
                    dataset=dataset,
                    subject=subject,
                    session=session,
                    split="gold",
                    has_gold=True,
                    qc_only=False,
                    atlas_eligible=atlas_eligible,
                    modality=modality,
                    profile=profile,
                    t1_path=image_out if suffix == "T1w" else "",
                    t2_path=image_out if suffix == "T2w" else "",
                    input_path=image_out,
                    brain_mask_path=mask_out,
                    label_path=label_out,
                    source_path=image,
                    standardized_dir=anat,
                    shape=stats.get("shape"),
                    zooms=stats.get("zooms"),
                    mask_voxels=stats.get("mask_voxels"),
                    mask_volume_cm3=stats.get("mask_volume_cm3"),
                    notes=_strict_source_notes(dataset, extra_note, f"source_label={label}"),
                )
            )
            seen.add((subject, session))
        self._add_rows(dataset, rows)

    def _standardize_brats_peds(self) -> None:
        roots = self._manual_search_roots("BraTSPEDs", ("BraTSPEDs", "BraTS-PEDs", "BraTS_PEDs", "BRATS-PEDS"))
        if not roots:
            self.summary["warnings"].append("BraTSPEDs source missing")
            return
        segs: dict[str, Path] = {}
        for root in roots:
            for seg in sorted(root.rglob("*seg.nii*"), key=_natural_key):
                if not seg.is_file():
                    continue
                match = re.search(r"(BraTS-PED-\d+-\d+)", str(seg))
                subject = match.group(1) if match else seg.parent.name
                segs.setdefault(_safe_subject_token(subject), seg)
        rows = []
        for subject, seg in sorted(segs.items(), key=lambda kv: _natural_key(kv[0])):
            subj_dir = seg.parent
            t1n = _best_file_match(subj_dir, (f"*{subject}*t1n.nii*", "*t1n.nii*"), prefer=("t1n",), reject=("seg",))
            t1c = _best_file_match(subj_dir, (f"*{subject}*t1c.nii*", "*t1c.nii*"), prefer=("t1c",), reject=("seg",))
            t2w = _best_file_match(subj_dir, (f"*{subject}*t2w.nii*", "*t2w.nii*"), prefer=("t2w",), reject=("seg",))
            t2f = _best_file_match(subj_dir, (f"*{subject}*t2f.nii*", "*t2f.nii*", "*flair.nii*"), prefer=("t2f", "flair"), reject=("seg",))
            primary = t1n or t1c or t2w or t2f
            if primary is None:
                self.summary["warnings"].append(f"BraTSPEDs {subject} skipped: missing MRI modalities")
                continue
            anat = self.output_root / "standardized" / "BraTSPEDs" / subject / "anat"
            std_t1 = _copy_or_convert_nifti(primary, anat / f"{subject}_T1w.nii.gz", self.force, dtype=np.float32)
            std_t2 = _copy_or_convert_nifti(t2w, anat / f"{subject}_T2w.nii.gz", self.force, dtype=np.float32) if t2w else Path("")
            ref_img = _read_nifti(std_t1)
            label_img = _read_nifti(seg)
            label_data = _resample_label_to_reference(label_img, ref_img)
            label_out = anat / f"{subject}_desc-label_dseg.nii.gz"
            if _needs_spatial_regeneration(label_out, ref_img, self.force):
                _save_nifti(label_data, ref_img.affine, ref_img.header, label_out, dtype=np.uint8)
            mask_out = anat / f"{subject}_desc-tumor_eval_mask.nii.gz"
            if _needs_spatial_regeneration(mask_out, ref_img, self.force):
                _save_mask(label_data > 0, ref_img, mask_out, force=True)
            stats = _mask_stats(mask_out)
            rows.append(
                _row(
                    dataset="BraTSPEDs",
                    subject=subject,
                    split="gold",
                    has_gold=True,
                    qc_only=False,
                    atlas_eligible=False,
                    modality="T1n+T1c+T2w+T2f",
                    profile="pediatric_tumor",
                    t1_path=std_t1,
                    t2_path=std_t2,
                    input_path=std_t1,
                    brain_mask_path=mask_out,
                    label_path=label_out,
                    source_path=primary,
                    standardized_dir=anat,
                    shape=stats.get("shape"),
                    zooms=stats.get("zooms"),
                    mask_voxels=stats.get("mask_voxels"),
                    mask_volume_cm3=stats.get("mask_volume_cm3"),
                    notes=_strict_source_notes(
                        "BraTSPEDs",
                        "not_brain_mask",
                        "separate_pathology_benchmark",
                        f"source_label={seg}",
                    ),
                )
            )
        self._add_rows("BraTSPEDs", rows)

    def _standardize_hykid(self) -> None:
        self._standardize_generic_label_dataset(
            dataset="HyKid",
            source_names=("HyKid", "hykid", "Synapse_HyKid"),
            label_patterns=("*label*.nii*", "*seg*.nii*", "*mask*.nii*", "*CP*.nii*", "*cp*.nii*"),
            label_prefer=("label", "seg", "mask", "cp"),
            image_patterns=("*T2*.nii*", "*T1*.nii*", "*image*.nii*", "*MRI*.nii*", "*.nii*"),
            image_prefer=("T2", "image", "MRI", "T1"),
            modality="MRI",
            profile="pediatric_hydrocephalus",
            atlas_eligible=False,
            extra_note="separate_pathology_benchmark",
        )

    def _standardize_candishare(self) -> None:
        self._standardize_generic_label_dataset(
            dataset="CANDIShare",
            source_names=("CANDIShare", "CANDI", "candi_share"),
            label_patterns=("*seg*.nii*", "*seg*.hdr", "*label*.nii*", "*label*.hdr", "*parc*.nii*", "*parc*.hdr"),
            label_prefer=("seg", "label", "parc"),
            image_patterns=("*mprage*.nii*", "*mprage*.hdr", "*mri*.nii*", "*mri*.hdr", "*T1*.nii*", "*T1*.hdr", "*.nii*", "*.hdr"),
            image_prefer=("mprage", "mri", "T1"),
            modality="T1w",
            profile="pediatric_t1",
            atlas_eligible=True,
        )

    def _standardize_lowfield_infant_mri(self) -> None:
        self._standardize_generic_label_dataset(
            dataset="LowFieldInfantMRI",
            source_names=("LowFieldInfantMRI", "LowFieldInfant", "ScienceDBLowField", "sciencedb_10690"),
            label_patterns=("*brain*mask*.nii*", "*mask*.nii*", "*manual*.nii*", "*seg*.nii*"),
            label_prefer=("brain", "manual", "mask", "seg"),
            image_patterns=("*T2*.nii*", "*T2w*.nii*", "*image*.nii*", "*anat*.nii*", "*.nii*"),
            image_prefer=("T2", "T2w", "image", "anat"),
            modality="T2w",
            profile="infant_lowfield_t2",
            atlas_eligible=True,
        )

    def _standardize_lisa2025(self) -> None:
        self._standardize_generic_label_dataset(
            dataset="LISA2025",
            source_names=("LISA2025", "LISA", "LowFieldPediatricLISA"),
            label_patterns=("*label*.nii*", "*seg*.nii*", "*mask*.nii*"),
            label_prefer=("label", "seg", "mask"),
            image_patterns=("*T2*.nii*", "*T2w*.nii*", "*image*.nii*", "*.nii*"),
            image_prefer=("T2", "T2w", "image"),
            modality="T2w",
            profile="pediatric_lowfield_t2",
            atlas_eligible=False,
            extra_note="candidate_requires_release_verification",
        )

    def _standardize_brainmetshare(self) -> None:
        src = self.data_root / "BrainMetShare"
        if not src.exists():
            self.summary["warnings"].append("BrainMetShare source missing")
            return
        raw_root = self.output_root / "BrainMetShare" / "raw"
        for archive_name in ["train_NifTi_files.zip", "test_NifTi_files.zip"]:
            archive = src / archive_name
            if archive.exists():
                _safe_extract_zip(archive, raw_root / archive_name.replace("_NifTi_files.zip", ""), self.force)
        rows = []
        for split in ["train", "test"]:
            base = raw_root / split
            if not base.exists():
                continue
            for subj_dir in sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: _natural_key(p.name)):
                sid = subj_dir.name
                preferred = [subj_dir / "bravo.nii.gz", subj_dir / "t1_pre.nii.gz", subj_dir / "t1_gd.nii.gz"]
                t1 = next((p for p in preferred if p.exists()), None)
                if t1 is None:
                    continue
                seg = subj_dir / "seg.nii.gz"
                anat = self.output_root / "standardized" / "BrainMetShare" / sid / "anat"
                std_t1 = _copy_file(t1, anat / f"sub-{sid}_T1w.nii.gz", self.force)
                std_seg = _copy_file(seg, anat / f"sub-{sid}_desc-metastasis_mask.nii.gz", self.force) if seg.exists() else Path("")
                info = _image_shape_zooms(std_t1)
                rows.append(
                    _row(
                        dataset="BrainMetShare",
                        subject=sid,
                        split=split,
                        has_gold=False,
                        qc_only=True,
                        atlas_eligible=False,
                        modality="T1w",
                        profile="pathology_qc_only",
                        t1_path=std_t1,
                        input_path=std_t1,
                        brain_mask_path="",
                        label_path=std_seg,
                        source_path=t1,
                        standardized_dir=anat,
                        shape=info.get("shape"),
                        zooms=info.get("zooms"),
                        notes="segmentation_is_metastasis_not_brain_mask",
                    )
                )
        self._add_rows("BrainMetShare", rows)

    def _standardize_lpba40_if_present(self) -> None:
        candidates = []
        for root in [self.data_root / "LPBA40", self.data_root]:
            if not root.exists():
                continue
            for p in root.rglob("*native.mri.hdr"):
                if "lpba" in str(p).lower() or p.parent.name.startswith("S"):
                    candidates.append(p.parent)
        subj_dirs = sorted(set(candidates), key=lambda p: _natural_key(str(p)))
        if not subj_dirs:
            self.summary["warnings"].append("LPBA40 not detected under data/")
            return
        dst_base = self.output_root / "LPBA40" / "LPBA40" / "native_space"
        rows = []
        for subj_dir in subj_dirs:
            sid = subj_dir.name
            t1 = subj_dir / f"{sid}.native.mri.hdr"
            mask = subj_dir / f"{sid}.native.brain.mask.hdr"
            if not t1.exists() or not mask.exists():
                continue
            raw_dst = dst_base / sid
            _copy_tree_files(subj_dir, raw_dst, self.force)
            anat = self.output_root / "standardized" / "LPBA40" / sid / "anat"
            std_t1 = _convert_lpba_analyze_to_nifti(
                t1,
                anat / f"sub-{sid}_T1w.nii.gz",
                self.force,
                dtype=np.float32,
            )
            std_mask = _convert_lpba_analyze_to_nifti(
                mask,
                anat / f"sub-{sid}_desc-brain_mask.nii.gz",
                self.force,
                dtype=np.uint8,
                binary_mask=True,
            )
            stats = _mask_stats(std_mask)
            rows.append(
                _row(
                    dataset="LPBA40",
                    subject=sid,
                    split="gold",
                    has_gold=True,
                    qc_only=False,
                    atlas_eligible=True,
                    modality="T1w",
                    profile="adult_t1",
                    t1_path=std_t1,
                    input_path=std_t1,
                    brain_mask_path=std_mask,
                    source_path=t1,
                    standardized_dir=anat,
                    shape=stats.get("shape"),
                    zooms=stats.get("zooms"),
                    mask_voxels=stats.get("mask_voxels"),
                    mask_volume_cm3=stats.get("mask_volume_cm3"),
                    notes="runtime_nifti_from_analyze_hdr_imggz",
                )
            )
        self._add_rows("LPBA40", rows)

    def _write_manifests(self) -> None:
        all_rows = []
        for dataset, rows in sorted(self.rows_by_dataset.items()):
            _write_csv(self.manifest_dir / f"{dataset}.csv", rows)
            self.summary["datasets"].setdefault(dataset, {})
            self.summary["datasets"][dataset].update(
                {
                    "manifest": str(self.manifest_dir / f"{dataset}.csv"),
                    "gold_rows": sum(1 for r in rows if _truthy(r, "has_gold") and not _truthy(r, "qc_only")),
                    "atlas_eligible_rows": sum(1 for r in rows if _truthy(r, "atlas_eligible")),
                }
            )
            for alias in MANIFEST_ALIASES.get(dataset, []):
                alias_rows = [dict(r, dataset=alias) for r in rows]
                _write_csv(self.manifest_dir / f"{alias}.csv", alias_rows)
                self.summary["datasets"][dataset].setdefault("manifest_aliases", []).append(
                    str(self.manifest_dir / f"{alias}.csv")
                )
            all_rows.extend(rows)
        _write_csv(self.manifest_dir / "all_datasets.csv", all_rows)

    def _build_loo_libraries(self) -> None:
        for dataset, rows in sorted(self.rows_by_dataset.items()):
            eligible = [
                r
                for r in rows
                if _truthy(r, "has_gold")
                and _truthy(r, "atlas_eligible")
                and r.get("t1_path")
                and r.get("brain_mask_path")
                and Path(r["t1_path"]).exists()
                and Path(r["brain_mask_path"]).exists()
            ]
            if len(eligible) < 2:
                continue
            lib_name = _atlas_library_name(dataset)
            lib_dir = self.output_root / "atlas_library" / lib_name
            subjects = []
            for r in eligible:
                sid = re.sub(r"[^A-Za-z0-9_.-]+", "_", r["subject"])
                out = lib_dir / sid
                t1_dst = _copy_file(Path(r["t1_path"]), out / "MNI152_T1_2mm.nii.gz", self.force)
                mask_dst = _copy_file(Path(r["brain_mask_path"]), out / "brain_mask.nii.gz", self.force)
                subjects.append({"subject": sid, "t1": str(t1_dst), "mask": str(mask_dst)})
            manifest = {
                "dataset": dataset,
                "library": lib_name,
                "n_subjects": len(subjects),
                "subjects": subjects,
                "anti_leakage": "callers must exclude the held-out subject id before topK selection",
            }
            _write_json(lib_dir / "manifest.json", manifest)
            self.summary["datasets"].setdefault(dataset, {})
            self.summary["datasets"][dataset]["loo_library"] = str(lib_dir)
            self.summary["datasets"][dataset]["loo_subjects"] = len(subjects)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="datasets")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma/space separated dataset names to standardize, e.g. iSeg2017,BOBs,MCRIB,MCRIB_WM.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", type=int, default=None, help="Reserved for future partial standardization.")
    args = parser.parse_args(argv)
    standardizer = DatasetStandardizer(
        Path(args.data_root),
        Path(args.output_root),
        force=args.force,
        smoke=args.smoke,
        datasets=_parse_dataset_list(args.datasets),
    )
    summary = standardizer.run()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
