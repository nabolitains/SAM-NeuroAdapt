#!/usr/bin/env python3
"""Prepare external atlases for lightweight M1 atlas guidance.

Raw downloads are kept under ``data/``. Runtime-ready files are written under
``dataset/`` so notebook cells can use stable, small paths without touching the
original archives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output
from scipy import ndimage as ndi


TEMPLATEFLOW_S3 = "https://templateflow.s3.amazonaws.com"
UNC_BCP_4D_V2_URL = "https://www.nitrc.org/frs/download.php/17111/UNC-BCP-4D-Infant-Brain-Volumetric-Atlas-Ver2.zip"
# dHCP neonatal volumetric atlas (Schuh 2018) sur G-Node : fichiers par semaine
# sous mean/ga_XX/ (template_t1/t2.nii.gz + mask.nii.gz), servis en NIfTI réel via
# /raw/ (pas de pointeur git-annex). On ne tire que les semaines utiles (≪ 1 Go).
DHCP_NEONATAL_RAW = "https://gin.g-node.org/BioMedIA/dhcp-volumetric-atlas-groupwise/raw/master/mean"
DHCP_NEONATAL_GA_WEEKS = (36, 37, 38, 39, 40, 41, 42, 43, 44)
# Cohortes MNIInfant complètes (1-11 = 0-60 mois). Les cohortes 6-11 (14-60 mo)
# comblent le gap pour les datasets à âge inconnu (ex. SynthStripInfant), dont le
# routage référence c07/c09. Datasets gagnants (BOBs 1-9mo → c1-4, iSeg 6mo → c3)
# n'utilisent jamais c06+ → ajout sans risque de régression. TemplateFlow ungated.
DEFAULT_MNIINFANT_COHORTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
DEFAULT_ADULT_TEMPLATE_IDS = (
    "oasis30ants",
    "mni152_nlin_2009c",
    "sri24",
    "colin27_2008",
    "icbm452",
    "lpba40_average",
    "miitra_older_adult",
)
DEFAULT_PEDIATRIC_TEMPLATE_IDS = (
    "fbn125_neonatal",
    "alberts_gousias",
)
SRI24_URLS = {
    "anatomy": "https://www.nitrc.org/frs/download.php/4499/sri24_anatomy_nifti.zip",
    "unstripped": "https://www.nitrc.org/frs/download.php/4502/sri24_anatomy_unstripped_nifti.zip",
}
COLIN27_2008_URL = "https://packages.bic.mni.mcgill.ca/mni-models/colin27/mni_colin27_2008_nifti.zip"
ADULT_TEMPLATE_MANUAL_ROOT = Path("dataset/raw_external_atlases")


def _download(url: str, dst: Path, force: bool = False) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return dst
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(dst)
    return dst


def _copy(src: Path, dst: Path, force: bool = False) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return dst
    shutil.copy2(src, dst)
    return dst


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_nifti(path: Path, *, is_mask: bool = False) -> dict:
    img = nib.load(str(path))
    shape = tuple(int(v) for v in img.shape[:3])
    zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
    payload: dict[str, Any] = {
        "path": str(path),
        "shape": shape,
        "zooms": zooms,
        "axcodes": "".join(nib.orientations.aff2axcodes(img.affine)),
        "qform_code": int(img.header["qform_code"]),
        "sform_code": int(img.header["sform_code"]),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if is_mask:
        data = img.get_fdata(dtype=np.float32) > 0.5
        lab, components = ndi.label(data)
        volume_cm3 = float(data.sum() * np.prod(zooms) / 1000.0)
        extent = 0.0
        if data.any():
            coords = np.where(data)
            spans = [int(c.max() - c.min() + 1) / max(1, s) for c, s in zip(coords, shape)]
            extent = float(max(spans))
        payload.update(
            {
                "voxels": int(data.sum()),
                "volume_cm3": volume_cm3,
                "components": int(components),
                "extent_ratio": extent,
                "adult_volume_plausible": bool(650.0 <= volume_cm3 <= 2100.0),
            }
        )
    return payload


def _safe_extract_zip(zip_path: Path, extract_root: Path, force: bool = False) -> Path:
    marker = extract_root / ".extract_complete"
    if marker.exists() and not force:
        return extract_root
    if extract_root.exists() and force:
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    root_resolved = extract_root.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (extract_root / info.filename).resolve()
            if not str(target).startswith(str(root_resolved)):
                raise RuntimeError(f"unsafe archive member: {info.filename}")
            zf.extract(info, extract_root)
    marker.write_text(str(zip_path))
    return extract_root


def _safe_extract_tar(tar_path: Path, extract_root: Path, force: bool = False) -> Path:
    marker = extract_root / ".extract_complete"
    if marker.exists() and not force:
        return extract_root
    if extract_root.exists() and force:
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    root_resolved = extract_root.resolve()
    mode = "r:gz" if tar_path.name.endswith((".tar.gz", ".tgz")) else "r:"
    with tarfile.open(tar_path, mode) as tf:
        for member in tf.getmembers():
            target = (extract_root / member.name).resolve()
            if not str(target).startswith(str(root_resolved)):
                raise RuntimeError(f"unsafe archive member: {member.name}")
        tf.extractall(extract_root)
    marker.write_text(str(tar_path))
    return extract_root


def _safe_extract_nested_zip(zip_path: Path, extract_root: Path, force: bool = False) -> Path:
    """Extract a LONI archive that may contain a single nested zip file."""

    marker = extract_root / ".extract_complete"
    if marker.exists() and not force:
        return extract_root
    if extract_root.exists() and force:
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    root_resolved = extract_root.resolve()
    with zipfile.ZipFile(zip_path) as outer:
        inner_zips = [info for info in outer.infolist() if info.filename.lower().endswith(".zip")]
        if len(inner_zips) == 1:
            inner_tmp = extract_root / "_nested_archive.zip"
            with outer.open(inner_zips[0]) as src, inner_tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            with zipfile.ZipFile(inner_tmp) as inner:
                for info in inner.infolist():
                    target = (extract_root / info.filename).resolve()
                    if not str(target).startswith(str(root_resolved)):
                        raise RuntimeError(f"unsafe nested archive member: {info.filename}")
                    inner.extract(info, extract_root)
            inner_tmp.unlink(missing_ok=True)
        else:
            for info in outer.infolist():
                target = (extract_root / info.filename).resolve()
                if not str(target).startswith(str(root_resolved)):
                    raise RuntimeError(f"unsafe archive member: {info.filename}")
                outer.extract(info, extract_root)
    marker.write_text(str(zip_path))
    return extract_root


def _find_one(root: Path, patterns: Sequence[str]) -> Path | None:
    for pattern in patterns:
        found = sorted(root.rglob(pattern))
        found = [
            p
            for p in found
            if p.is_file()
            and not p.name.endswith(".tmp")
            and not p.name.endswith(":Zone.Identifier")
            and p.name != ".DS_Store"
            and not p.name.startswith("._")
            and "__MACOSX" not in p.parts
        ]
        if found:
            return found[0]
    return None


def _find_any(roots: Sequence[Path], patterns: Sequence[str]) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        found = _find_one(root, patterns)
        if found is not None:
            return found
    return None


def _manual_dirs(manual_root: Path, atlas_id: str) -> list[Path]:
    return [
        manual_root / atlas_id,
        manual_root,
        manual_root / "Looni-atlas",
        manual_root / "MIITRA",
        Path("data/external_atlases") / atlas_id,
        Path("data") / atlas_id,
    ]


def _manual_file(manual_root: Path, atlas_id: str, names: Sequence[str]) -> Path | None:
    for directory in _manual_dirs(manual_root, atlas_id):
        if not directory.exists():
            continue
        for name in names:
            direct = directory / name
            if direct.exists() and direct.stat().st_size > 0:
                return direct
        found = _find_one(directory, names)
        if found is not None:
            return found
    return None


def _download_or_manual(
    url: str,
    dst: Path,
    *,
    atlas_id: str,
    manual_root: Path,
    names: Sequence[str],
    force: bool = False,
) -> Path:
    manual = _manual_file(manual_root, atlas_id, names)
    if manual is not None and not force:
        return manual
    try:
        return _download(url, dst, force)
    except Exception:
        if manual is not None:
            return manual
        raise


def _runtime_image(
    src: Path,
    dst: Path,
    *,
    force: bool = False,
    order: int = 1,
    voxel_size_mm: float | None = None,
    is_mask: bool = False,
) -> Path:
    if dst.exists() and not force:
        return dst
    img = nib.load(str(src))
    if len(img.shape) > 3:
        data = np.asarray(img.dataobj)[..., 0]
        img = nib.Nifti1Image(data, img.affine, img.header)
    if voxel_size_mm:
        img = resample_to_output(img, voxel_sizes=(voxel_size_mm, voxel_size_mm, voxel_size_mm), order=order)
    if is_mask:
        data = img.get_fdata(dtype=np.float32) > 0.5
        img = nib.Nifti1Image(data.astype(np.uint8), img.affine, img.header)
        img.set_data_dtype(np.uint8)
    else:
        data = np.nan_to_num(img.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        img = nib.Nifti1Image(data.astype(np.float32), img.affine, img.header)
        img.set_data_dtype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(dst))
    return dst


def _save_mask_from_array(mask: np.ndarray, ref_img, dst: Path, force: bool = False) -> Path:
    if dst.exists() and not force:
        return dst
    out = nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine, ref_img.header)
    out.set_data_dtype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))
    return dst


def _first_volume_image(path: Path) -> nib.Nifti1Image:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Image non-3D non supportee: {path} shape={img.shape}")
    out = nib.Nifti1Image(np.asarray(data), img.affine, img.header)
    return out


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(np.asarray(mask, dtype=bool))
    if n <= 1:
        return np.asarray(mask, dtype=bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(np.argmax(sizes))


def _unavailable_manifest(atlas_id: str, reason: str, manual_root: Path, expected: Sequence[str], sources: Sequence[str]) -> dict:
    target = manual_root / atlas_id
    return {
        "atlas": atlas_id,
        "available": False,
        "reason": reason,
        "manual_install": {
            "drop_archives_or_files_under": str(target),
            "expected": list(expected),
            "rerun": "python scripts/prepare_external_atlases.py --adult-templates",
        },
        "sources": list(sources),
    }


def _atlas_manifest(out: Path, payload: dict) -> dict:
    _write_manifest(out / "atlas_manifest.json", payload)
    return payload


def _save_mask_from_labels(label_path: Path, dst: Path, force: bool = False) -> Path:
    if dst.exists() and not force:
        return dst
    img = nib.load(str(label_path))
    mask = img.get_fdata(dtype=np.float32) > 0
    out = nib.Nifti1Image(mask.astype(np.uint8), img.affine, img.header)
    out.set_data_dtype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))
    return dst


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(text)
    if path.name == "atlas_manifest.json":
        (path.parent / "manifest.json").write_text(text)


def prepare_runtime_layout(raw_root: Path, runtime_root: Path, force: bool = False) -> dict:
    """Create ``dataset/standardized`` and rewritten manifests when possible."""

    report: dict = {"standardized": None, "manifests": []}
    src_standardized = Path("datasets/standardized")
    dst_standardized = runtime_root / "standardized"
    runtime_root.mkdir(parents=True, exist_ok=True)
    if src_standardized.exists() and not dst_standardized.exists():
        dst_standardized.symlink_to(Path("..") / "datasets" / "standardized")
    if dst_standardized.exists():
        report["standardized"] = str(dst_standardized)

    src_manifest_dir = Path("datasets/manifests")
    dst_manifest_dir = runtime_root / "manifests"
    if src_manifest_dir.exists():
        dst_manifest_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_manifest_dir.glob("*.csv")):
            dst = dst_manifest_dir / src.name
            if dst.exists() and not force:
                report["manifests"].append(str(dst))
                continue
            text = src.read_text()
            text = text.replace("datasets/standardized/", "dataset/standardized/")
            dst.write_text(text)
            report["manifests"].append(str(dst))
    return report


def prepare_oasis30ants(raw_root: Path, runtime_root: Path, force: bool = False) -> dict:
    raw = raw_root / "templateflow" / "tpl-OASIS30ANTs"
    out = runtime_root / "atlas_library" / "oasis30ants"
    files = {
        "t1": "tpl-OASIS30ANTs_res-01_T1w.nii.gz",
        "mask": "tpl-OASIS30ANTs_res-01_desc-brain_mask.nii.gz",
    }
    downloaded = {}
    for key, name in files.items():
        downloaded[key] = _download(f"{TEMPLATEFLOW_S3}/tpl-OASIS30ANTs/{name}", raw / name, force)

    _copy(downloaded["t1"], out / files["t1"], force)
    _copy(downloaded["mask"], out / files["mask"], force)
    _copy(downloaded["t1"], out / "MNI152_T1_2mm.nii.gz", force)
    _copy(downloaded["mask"], out / "brain_mask.nii.gz", force)
    manifest = {
        "atlas": "OASIS30ANTs",
        "source": "TemplateFlow tpl-OASIS30ANTs",
        "raw_files": {k: str(v) for k, v in downloaded.items()},
        "runtime_files": {
            "t1": str(out / "MNI152_T1_2mm.nii.gz"),
            "brain_mask": str(out / "brain_mask.nii.gz"),
        },
        "validation": {
            "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
            "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
        },
    }
    _write_manifest(out / "atlas_manifest.json", manifest)
    return manifest


def prepare_mni152(runtime_root: Path, force: bool = False) -> dict:
    src_base = Path("sam_neuroadapt_v2_real_atlases/atlases")
    out = runtime_root / "atlas_library" / "mni152_nlin_2009c"
    t1_src = src_base / "tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz"
    mask_src = src_base / "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
    _copy(t1_src, out / "tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz", force)
    _copy(mask_src, out / "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz", force)
    _copy(t1_src, out / "MNI152_T1_2mm.nii.gz", force)
    _copy(mask_src, out / "brain_mask.nii.gz", force)
    manifest = {
        "atlas": "MNI152NLin2009cAsym",
        "source": str(src_base),
        "runtime_files": {
            "t1": str(out / "MNI152_T1_2mm.nii.gz"),
            "brain_mask": str(out / "brain_mask.nii.gz"),
        },
        "validation": {
            "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
            "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
        },
    }
    _write_manifest(out / "atlas_manifest.json", manifest)
    return manifest


def prepare_mniinfant(raw_root: Path, runtime_root: Path, cohorts: Iterable[int], force: bool = False) -> list[dict]:
    reports = []
    for cohort in cohorts:
        raw = raw_root / "templateflow" / "tpl-MNIInfant" / f"cohort-{cohort}"
        out = runtime_root / "atlas_library" / f"mniinfant_cohort_{cohort:02d}"
        names = {
            "t1": f"tpl-MNIInfant_cohort-{cohort}_res-2_T1w.nii.gz",
            "t2": f"tpl-MNIInfant_cohort-{cohort}_res-2_T2w.nii.gz",
            "mask": f"tpl-MNIInfant_cohort-{cohort}_res-2_desc-brain_mask.nii.gz",
        }
        downloaded = {}
        for key, name in names.items():
            downloaded[key] = _download(
                f"{TEMPLATEFLOW_S3}/tpl-MNIInfant/cohort-{cohort}/{name}",
                raw / name,
                force,
            )
            _copy(downloaded[key], out / name, force)
        _copy(downloaded["t1"], out / "MNI152_T1_2mm.nii.gz", force)
        _copy(downloaded["mask"], out / "brain_mask.nii.gz", force)
        manifest = {
            "atlas": "MNIInfant",
            "cohort": cohort,
            "source": "TemplateFlow tpl-MNIInfant",
            "raw_files": {k: str(v) for k, v in downloaded.items()},
            "runtime_files": {
                "t1": str(out / "MNI152_T1_2mm.nii.gz"),
                "t2": str(out / names["t2"]),
                "brain_mask": str(out / "brain_mask.nii.gz"),
            },
            "validation": {
                "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
            },
        }
        _write_manifest(out / "atlas_manifest.json", manifest)
        reports.append(manifest)
    return reports


def _extract_unc_zip(raw_unc_root: Path, force: bool = False) -> Path | None:
    candidates = sorted(raw_unc_root.glob("UNCInfant012Atlases*.zip"))
    if not candidates:
        return None
    zip_path = candidates[0]
    extract_root = raw_unc_root / "extracted"
    marker = extract_root / ".extract_complete"
    if marker.exists() and not force:
        return extract_root
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_root)
    marker.write_text(str(zip_path))
    return extract_root


def prepare_unc_infant(raw_root: Path, runtime_root: Path, force: bool = False) -> list[dict]:
    raw_unc_root = Path("data/UNCInfant012Atlases")
    extract_root = _extract_unc_zip(raw_unc_root, force)
    if extract_root is None:
        return []
    atlas_root = next((p for p in extract_root.iterdir() if p.is_dir() and p.name.startswith("UNCInfant012Atlases")), extract_root)
    specs = {
        "neo": {"t1": "infant-neo-withSkull.nii.gz", "labels": "infant-neo-seg.nii.gz"},
        "1yr": {"t1": "infant-1yr-withSkull.nii.gz", "labels": "infant-1yr-seg.nii.gz"},
        "2yr": {"t1": "infant-2yr-withSkull.nii.gz", "labels": "infant-2yr-seg.nii.gz"},
    }
    reports = []
    for age, spec in specs.items():
        out = runtime_root / "atlas_library" / f"unc_infant_{age}"
        t1_src = atlas_root / spec["t1"]
        label_src = atlas_root / spec["labels"]
        _copy(t1_src, out / spec["t1"], force)
        _copy(label_src, out / spec["labels"], force)
        _copy(t1_src, out / "MNI152_T1_2mm.nii.gz", force)
        _save_mask_from_labels(label_src, out / "brain_mask.nii.gz", force)
        manifest = {
            "atlas": "UNCInfant012Atlases",
            "age": age,
            "source_archive": str(raw_unc_root),
            "runtime_files": {
                "t1": str(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": str(out / "brain_mask.nii.gz"),
                "labels": str(out / spec["labels"]),
            },
            "validation": {
                "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
            },
        }
        _write_manifest(out / "atlas_manifest.json", manifest)
        reports.append(manifest)
    return reports


def _extract_unc_bcp_06m(zip_path: Path, raw_dir: Path, force: bool = False) -> Path:
    extract_root = raw_dir / "extracted"
    month_root = extract_root / "BCP-atlas-for_release-Ver2.0.0" / "06Month"
    required = [
        month_root / "BCP-06M-T1.nii.gz",
        month_root / "BCP-06M-T2.nii.gz",
        month_root / "BCP-06M-Mask.nii.gz",
        month_root / "BCP-06M-Seg.nii.gz",
    ]
    if all(p.exists() for p in required) and not force:
        return month_root
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if "/06Month/" not in info.filename:
                continue
            target = (extract_root / info.filename).resolve()
            if not str(target).startswith(str(extract_root.resolve())):
                raise RuntimeError(f"unsafe archive member: {info.filename}")
            zf.extract(info, extract_root)
    return month_root


def prepare_unc_bcp_06m(raw_root: Path, runtime_root: Path, force: bool = False) -> dict | None:
    raw = raw_root / "unc_bcp_4d"
    zip_path = raw / "UNC-BCP-4D-Infant-Brain-Volumetric-Atlas-Ver2.zip"
    try:
        _download(UNC_BCP_4D_V2_URL, zip_path, force)
    except Exception:
        if not zip_path.exists():
            return None
    month_root = _extract_unc_bcp_06m(zip_path, raw, force)
    t1_src = month_root / "BCP-06M-T1.nii.gz"
    t2_src = month_root / "BCP-06M-T2.nii.gz"
    mask_src = month_root / "BCP-06M-Mask.nii.gz"
    seg_src = month_root / "BCP-06M-Seg.nii.gz"
    out = runtime_root / "atlas_library" / "unc_bcp_06m"
    _copy(t1_src, out / "BCP-06M-T1.nii.gz", force)
    _copy(t2_src, out / "BCP-06M-T2.nii.gz", force)
    _copy(mask_src, out / "BCP-06M-Mask.nii.gz", force)
    _copy(seg_src, out / "BCP-06M-Seg.nii.gz", force)
    _copy(t1_src, out / "MNI152_T1_2mm.nii.gz", force)
    _copy(t1_src, out / "tpl-UNCBCP06M_T1w.nii.gz", force)
    _save_mask_from_labels(mask_src, out / "brain_mask.nii.gz", force)
    _save_mask_from_labels(mask_src, out / "tpl-UNCBCP06M_desc-brain_mask.nii.gz", force)
    manifest = {
        "atlas": "UNC-BCP 4D Infant Brain Volumetric Atlas",
        "age": "06M",
        "source": "NITRC UNC-BCP-4D-Infant-Brain-Volumetric-Atlas-Ver2",
        "source_archive": str(zip_path),
        "runtime_files": {
            "t1": str(out / "MNI152_T1_2mm.nii.gz"),
            "t2": str(out / "BCP-06M-T2.nii.gz"),
            "brain_mask": str(out / "brain_mask.nii.gz"),
            "seg": str(out / "BCP-06M-Seg.nii.gz"),
        },
        "validation": {
            "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
            "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
        },
    }
    _write_manifest(out / "atlas_manifest.json", manifest)
    return manifest


# Bins mensuels disponibles dans l'archive UNC-BCP 4D Ver2.
UNC_BCP_MONTHS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 21, 24, 36, 48, 60)


def _extract_unc_bcp_month(zip_path: Path, raw_dir: Path, month: int, force: bool = False) -> Path:
    """Extrait un bin mensuel arbitraire de l'archive UNC-BCP 4D (généralise 06m)."""
    extract_root = raw_dir / "extracted"
    folder = f"{month:02d}Month"
    month_root = extract_root / "BCP-atlas-for_release-Ver2.0.0" / folder
    required = [month_root / f"BCP-{month:02d}M-T1.nii.gz", month_root / f"BCP-{month:02d}M-Mask.nii.gz"]
    if all(p.exists() for p in required) and not force:
        return month_root
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if f"/{folder}/" not in info.filename:
                continue
            target = (extract_root / info.filename).resolve()
            if not str(target).startswith(str(extract_root.resolve())):
                raise RuntimeError(f"unsafe archive member: {info.filename}")
            zf.extract(info, extract_root)
    return month_root


def prepare_unc_bcp_month(raw_root: Path, runtime_root: Path, month: int, force: bool = False) -> dict | None:
    """Prépare un bin mensuel UNC-BCP 4D en atlas runtime ``unc_bcp_<XX>m``.

    Donne des templates T1+T2 par mois (1-9 = nourrisson), bien plus appariés
    qu'un cohort MNIInfant unique pour les datasets infant à âge inconnu
    (SynthStripInfant) ou par mois (BOBs). Contrat de sortie identique à
    ``prepare_unc_bcp_06m`` : ``brain_mask.nii.gz`` + alias ``MNI152_T1_2mm.nii.gz``.
    """
    raw = raw_root / "unc_bcp_4d"
    zip_path = raw / "UNC-BCP-4D-Infant-Brain-Volumetric-Atlas-Ver2.zip"
    try:
        _download(UNC_BCP_4D_V2_URL, zip_path, force)
    except Exception:
        if not zip_path.exists():
            return None
    month_root = _extract_unc_bcp_month(zip_path, raw, month, force)
    tag = f"{month:02d}M"
    t1_src = month_root / f"BCP-{tag}-T1.nii.gz"
    t2_src = month_root / f"BCP-{tag}-T2.nii.gz"
    mask_src = month_root / f"BCP-{tag}-Mask.nii.gz"
    if not (t1_src.exists() and mask_src.exists()):
        return None
    out = runtime_root / "atlas_library" / f"unc_bcp_{month:02d}m"
    _copy(t1_src, out / f"BCP-{tag}-T1.nii.gz", force)
    if t2_src.exists():
        _copy(t2_src, out / f"BCP-{tag}-T2.nii.gz", force)
    _copy(mask_src, out / f"BCP-{tag}-Mask.nii.gz", force)
    _copy(t1_src, out / "MNI152_T1_2mm.nii.gz", force)
    _copy(t1_src, out / f"tpl-UNCBCP{tag}_T1w.nii.gz", force)
    _save_mask_from_labels(mask_src, out / "brain_mask.nii.gz", force)
    _save_mask_from_labels(mask_src, out / f"tpl-UNCBCP{tag}_desc-brain_mask.nii.gz", force)
    manifest = {
        "atlas": "UNC-BCP 4D Infant Brain Volumetric Atlas",
        "age": tag,
        "source": "NITRC UNC-BCP-4D-Infant-Brain-Volumetric-Atlas-Ver2",
        "source_archive": str(zip_path),
        "runtime_files": {
            "t1": str(out / "MNI152_T1_2mm.nii.gz"),
            "t2": str(out / f"BCP-{tag}-T2.nii.gz") if t2_src.exists() else None,
            "brain_mask": str(out / "brain_mask.nii.gz"),
        },
        "validation": {
            "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
            "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
        },
    }
    _write_manifest(out / "atlas_manifest.json", manifest)
    return manifest


def prepare_unc_bcp_4d_months(
    raw_root: Path, runtime_root: Path, months: Iterable[int], force: bool = False
) -> list[dict]:
    """Prépare plusieurs bins mensuels UNC-BCP 4D (idempotent)."""
    reports = []
    for m in months:
        rep = prepare_unc_bcp_month(raw_root, runtime_root, int(m), force)
        if rep is not None:
            reports.append(rep)
    return reports


import re as _re_fetal


def _parse_ga_weeks(name: str) -> int | None:
    """Extrait l'âge gestationnel (semaines, 18-42) d'un nom de fichier."""
    for pat in (r"ga[-_ ]?(\d{2})", r"sta[-_ ]?(\d{2})", r"\b(\d{2})\s*w(?:k|eek)?s?\b", r"_(\d{2})w"):
        m = _re_fetal.search(pat, name, _re_fetal.IGNORECASE)
        if m:
            wk = int(m.group(1))
            if 18 <= wk <= 42:
                return wk
    return None


def _is_mask_name(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in ("mask", "brain_mask", "brainmask", "_seg", "tissue"))


def prepare_fetal_manual_bank(
    manual_root: Path,
    runtime_root: Path,
    source: str = "serag",
    force: bool = False,
) -> list[dict]:
    """Construit une banque d'atlas FŒTAUX GA-binnée depuis un dépôt MANUEL.

    Les atlas fœtaux (Serag, CRL-2025, dHCP-fetal) sont gated (formulaire/licence)
    et sont déposés à la main par l'utilisateur — voir
    ``dataset/raw_external_atlases/fetal/DEPOT_README.md``. Cette fonction scanne le
    dépôt, apparie un template (T2w) avec son masque cerveau par âge gestationnel
    (parsé du nom de fichier), et écrit des atlas runtime
    ``fetal_<source>_ga<WW>/`` au format attendu par le runner M1
    (``brain_mask.nii.gz`` + alias ``MNI152_T1_2mm.nii.gz``).

    Idempotent et tolérant : signale les bins incomplets, n'écrase rien sauf force.
    """
    deposit = manual_root / "fetal" / source
    if not deposit.exists():
        deposit = manual_root / "fetal"
    reports: list[dict] = []
    if not deposit.exists():
        print(f"[fetal] dépôt absent: {manual_root/'fetal'} — voir DEPOT_README.md")
        return reports

    niftis = [p for p in deposit.rglob("*.nii*") if p.is_file()]
    by_ga: dict[int, dict[str, Path]] = {}
    for p in niftis:
        ga = _parse_ga_weeks(p.name)
        if ga is None:
            continue
        slot = by_ga.setdefault(ga, {})
        if _is_mask_name(p.name):
            slot.setdefault("mask", p)
        else:
            slot.setdefault("template", p)

    for ga in sorted(by_ga):
        slot = by_ga[ga]
        if "template" not in slot or "mask" not in slot:
            missing = ", ".join(part for part in ("template", "mask") if part not in slot)
            print(f"[fetal] GA {ga}w incomplet (manque: {missing}); ignoré")
            continue
        out = runtime_root / "atlas_library" / f"fetal_{source}_ga{ga:02d}"
        _copy(slot["template"], out / "MNI152_T1_2mm.nii.gz", force)
        _copy(slot["template"], out / f"tpl-fetal{source}GA{ga:02d}_T2w.nii.gz", force)
        _save_mask_from_labels(slot["mask"], out / "brain_mask.nii.gz", force)
        manifest = {
            "atlas": f"Fetal manual ({source})",
            "ga_weeks": ga,
            "source": f"manual deposit: {slot['template'].parent}",
            "runtime_files": {
                "t1": str(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": str(out / "brain_mask.nii.gz"),
            },
            "validation": {
                "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
            },
        }
        _write_manifest(out / "atlas_manifest.json", manifest)
        reports.append(manifest)
        print(f"[fetal] GA {ga}w → {out.name} OK")
    if not reports:
        print(f"[fetal] aucun bin GA complet trouvé sous {deposit}")
    return reports


def prepare_dhcp_neonatal(
    raw_root: Path,
    runtime_root: Path,
    ga_weeks: Iterable[int] = DHCP_NEONATAL_GA_WEEKS,
    force: bool = False,
) -> list[dict]:
    """Prépare l'atlas dHCP neonatal volumetric (Schuh 2018) par semaine (GA).

    Pour chaque semaine, télécharge template_t1/t2 + mask depuis G-Node (/raw,
    NIfTI réel) et écrit ``dhcp_neonatal_ga<WW>/`` au format runtime M1
    (``brain_mask.nii.gz`` + alias ``MNI152_T1_2mm.nii.gz``). Cible MCRIB/MCRIB2
    (term-néonatal) : templates 0.5 mm, grande cohorte non-biaisée.
    """
    reports: list[dict] = []
    for ga in ga_weeks:
        raw = raw_root / "dhcp_neonatal" / f"ga_{ga}"
        try:
            t1 = _download(f"{DHCP_NEONATAL_RAW}/ga_{ga}/template_t1.nii.gz", raw / "template_t1.nii.gz", force)
            t2 = _download(f"{DHCP_NEONATAL_RAW}/ga_{ga}/template_t2.nii.gz", raw / "template_t2.nii.gz", force)
            mask = _download(f"{DHCP_NEONATAL_RAW}/ga_{ga}/mask.nii.gz", raw / "mask.nii.gz", force)
        except Exception as exc:
            print(f"[dhcp_neonatal] ga_{ga} échec téléchargement: {str(exc)[:100]}")
            continue
        out = runtime_root / "atlas_library" / f"dhcp_neonatal_ga{ga:02d}"
        _copy(t1, out / "template_t1.nii.gz", force)
        _copy(t2, out / "template_t2.nii.gz", force)
        _copy(t1, out / "MNI152_T1_2mm.nii.gz", force)
        _copy(t1, out / f"tpl-dHCPneo{ga:02d}_T1w.nii.gz", force)
        _save_mask_from_labels(mask, out / "brain_mask.nii.gz", force)
        manifest = {
            "atlas": "dHCP neonatal volumetric (groupwise, Schuh 2018)",
            "ga_weeks": ga,
            "source": "G-Node BioMedIA/dhcp-volumetric-atlas-groupwise (CC-BY 4.0)",
            "runtime_files": {
                "t1": str(out / "MNI152_T1_2mm.nii.gz"),
                "t2": str(out / "template_t2.nii.gz"),
                "brain_mask": str(out / "brain_mask.nii.gz"),
            },
            "validation": {
                "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
            },
        }
        _write_manifest(out / "atlas_manifest.json", manifest)
        reports.append(manifest)
    return reports


def prepare_sri24(raw_root: Path, runtime_root: Path, manual_root: Path, force: bool = False) -> dict:
    atlas_id = "sri24"
    raw = raw_root / atlas_id
    out = runtime_root / "atlas_library" / atlas_id
    expected = ("sri24_anatomy_nifti.zip", "sri24_anatomy_unstripped_nifti.zip")
    try:
        anatomy_zip = _download_or_manual(
            SRI24_URLS["anatomy"],
            raw / "sri24_anatomy_nifti.zip",
            atlas_id=atlas_id,
            manual_root=manual_root,
            names=("sri24_anatomy_nifti.zip",),
            force=force,
        )
        unstripped_zip = _download_or_manual(
            SRI24_URLS["unstripped"],
            raw / "sri24_anatomy_unstripped_nifti.zip",
            atlas_id=atlas_id,
            manual_root=manual_root,
            names=("sri24_anatomy_unstripped_nifti.zip",),
            force=force,
        )
        anatomy_root = _safe_extract_zip(anatomy_zip, raw / "anatomy", force)
        unstripped_root = _safe_extract_zip(unstripped_zip, raw / "unstripped", force)
        stripped_t1 = _find_one(anatomy_root, ("spgr.nii", "spgr.nii.gz", "*spgr*.nii*"))
        moving_t1 = _find_one(unstripped_root, ("spgr_unstrip.nii", "spgr_unstrip.nii.gz", "*unstrip*.nii*"))
        if stripped_t1 is None:
            raise FileNotFoundError("sri24 stripped SPGR introuvable")
        if moving_t1 is None:
            moving_t1 = stripped_t1
        t1_out = _runtime_image(moving_t1, out / "sri24_spgr_unstrip_T1w.nii.gz", force=force, order=1)
        _copy(t1_out, out / "MNI152_T1_2mm.nii.gz", force)
        stripped_img = _first_volume_image(stripped_t1)
        moving_img = _first_volume_image(moving_t1)
        mask = stripped_img.get_fdata(dtype=np.float32) > 0
        mask = ndi.binary_fill_holes(mask)
        mask = _largest_component(ndi.binary_opening(mask, iterations=1))
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), stripped_img.affine, stripped_img.header)
        mask_img.set_data_dtype(np.uint8)
        if mask_img.shape[:3] != moving_img.shape[:3] or not np.allclose(mask_img.affine, moving_img.affine, atol=1e-4):
            mask_img = resample_from_to(mask_img, moving_img, order=0, cval=0)
        _save_mask_from_array(mask_img.get_fdata(dtype=np.float32) > 0.5, moving_img, out / "brain_mask.nii.gz", force)
        manifest = {
            "atlas": "SRI24",
            "available": True,
            "source": "NITRC SRI24 Atlas: Normal Adult Brain Anatomy",
            "source_urls": SRI24_URLS,
            "license": "CC BY-SA 3.0; see archive LICENSE",
            "raw_files": {"anatomy": str(anatomy_zip), "unstripped": str(unstripped_zip)},
            "runtime_files": {
                "t1": str(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": str(out / "brain_mask.nii.gz"),
                "native_t1": str(t1_out),
            },
            "validation": {
                "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
            },
        }
        return _atlas_manifest(out, manifest)
    except Exception as exc:
        manifest = _unavailable_manifest(
            atlas_id,
            str(exc),
            manual_root,
            expected,
            ["https://www.nitrc.org/frs/?group_id=214", *SRI24_URLS.values()],
        )
        out.mkdir(parents=True, exist_ok=True)
        return _atlas_manifest(out, manifest)


def prepare_colin27_2008(
    raw_root: Path,
    runtime_root: Path,
    manual_root: Path,
    *,
    force: bool = False,
    spacing_mm: float = 1.0,
) -> dict:
    atlas_id = "colin27_2008"
    raw = raw_root / atlas_id
    out = runtime_root / "atlas_library" / atlas_id
    expected = ("mni_colin27_2008_nifti.zip",)
    try:
        archive = _download_or_manual(
            COLIN27_2008_URL,
            raw / "mni_colin27_2008_nifti.zip",
            atlas_id=atlas_id,
            manual_root=manual_root,
            names=expected,
            force=force,
        )
        extract_root = _safe_extract_zip(archive, raw / "extracted", force)
        t1_src = _find_one(extract_root, ("colin27_t1_tal_hires.nii", "colin27_t1_tal_hires.nii.gz", "*t1*.nii*"))
        cls_src = _find_one(extract_root, ("colin27_cls_tal_hires.nii", "colin27_cls_tal_hires.nii.gz", "*cls*.nii*"))
        if t1_src is None or cls_src is None:
            raise FileNotFoundError("Colin27 T1 ou classification discrete introuvable")
        t1_out = _runtime_image(
            t1_src,
            out / "colin27_2008_t1_tal_res-1mm_T1w.nii.gz",
            force=force,
            order=1,
            voxel_size_mm=spacing_mm,
        )
        cls_img = nib.load(str(cls_src))
        cls_resampled_path = _runtime_image(
            cls_src,
            out / "colin27_2008_cls_tal_res-1mm.nii.gz",
            force=force,
            order=0,
            voxel_size_mm=spacing_mm,
        )
        cls_img = nib.load(str(cls_resampled_path))
        cls = np.rint(cls_img.get_fdata(dtype=np.float32)).astype(np.int16)
        brain_mask = np.isin(cls, [1, 2, 3])
        brain_mask = _largest_component(ndi.binary_fill_holes(brain_mask))
        _save_mask_from_array(brain_mask, cls_img, out / "brain_mask.nii.gz", force)
        _copy(t1_out, out / "MNI152_T1_2mm.nii.gz", force)
        manifest = {
            "atlas": "Colin27 2008",
            "available": True,
            "source": "McGill BIC Colin 27 Average Brain high-resolution version 2008",
            "source_urls": {
                "page": "https://www.mcgill.ca/bic/software/tools-data-analysis/anatomical-mri/atlases/colin-27-2008",
                "archive": COLIN27_2008_URL,
            },
            "license": "McGill BIC permissive license; see source page/README",
            "raw_files": {"archive": str(archive), "t1": str(t1_src), "labels": str(cls_src)},
            "runtime_files": {
                "t1": str(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": str(out / "brain_mask.nii.gz"),
                "labels": str(cls_resampled_path),
            },
            "mask_derivation": "labels 1,2,3 = CSF, GM, WM",
            "validation": {
                "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
            },
        }
        return _atlas_manifest(out, manifest)
    except Exception as exc:
        manifest = _unavailable_manifest(
            atlas_id,
            str(exc),
            manual_root,
            expected,
            [
                "https://www.mcgill.ca/bic/software/tools-data-analysis/anatomical-mri/atlases/colin-27-2008",
                COLIN27_2008_URL,
            ],
        )
        out.mkdir(parents=True, exist_ok=True)
        return _atlas_manifest(out, manifest)


def _prepare_manual_average_template(
    atlas_id: str,
    raw_root: Path,
    runtime_root: Path,
    manual_root: Path,
    *,
    force: bool = False,
    spacing_mm: float = 1.0,
    source_page: str,
    expected: Sequence[str],
    t1_patterns: Sequence[str],
    mask_patterns: Sequence[str],
    allow_nonzero_t1_mask: bool = False,
) -> dict:
    out = runtime_root / "atlas_library" / atlas_id
    manual_dirs = _manual_dirs(manual_root, atlas_id)
    try:
        search_roots = [p for p in manual_dirs if p.exists()]
        if not search_roots:
            raise FileNotFoundError(f"aucun dossier manuel trouve parmi {[str(p) for p in manual_dirs]}")
        t1_src = None
        mask_src = None
        for root in search_roots:
            t1_src = t1_src or _find_one(root, t1_patterns)
            mask_src = mask_src or _find_one(root, mask_patterns)
        if t1_src is None:
            raise FileNotFoundError("T1 template introuvable")
        t1_out = _runtime_image(
            t1_src,
            out / f"{atlas_id}_T1w.nii.gz",
            force=force,
            order=1,
            voxel_size_mm=spacing_mm,
        )
        if mask_src is not None:
            _runtime_image(
                mask_src,
                out / "brain_mask.nii.gz",
                force=force,
                order=0,
                voxel_size_mm=spacing_mm,
                is_mask=True,
            )
            mask_derivation = f"manual mask file: {mask_src}"
        elif allow_nonzero_t1_mask:
            t1_img = nib.load(str(t1_out))
            mask = t1_img.get_fdata(dtype=np.float32) > 0
            mask = _largest_component(ndi.binary_fill_holes(mask))
            _save_mask_from_array(mask, t1_img, out / "brain_mask.nii.gz", force)
            mask_derivation = "nonzero voxels from skull-stripped average template"
        else:
            raise FileNotFoundError("brain mask introuvable")
        _copy(t1_out, out / "MNI152_T1_2mm.nii.gz", force)
        manifest = {
            "atlas": atlas_id,
            "available": True,
            "source": source_page,
            "source_urls": {"page": source_page},
            "license": "manual verification required before publication",
            "raw_files": {"t1": str(t1_src), "brain_mask": None if mask_src is None else str(mask_src)},
            "runtime_files": {
                "t1": str(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": str(out / "brain_mask.nii.gz"),
                "native_t1": str(t1_out),
            },
            "mask_derivation": mask_derivation,
            "validation": {
                "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
            },
        }
        return _atlas_manifest(out, manifest)
    except Exception as exc:
        manifest = _unavailable_manifest(atlas_id, str(exc), manual_root, expected, [source_page])
        out.mkdir(parents=True, exist_ok=True)
        return _atlas_manifest(out, manifest)


def _loni_zip(manual_root: Path, names: Sequence[str]) -> Path | None:
    roots = [manual_root, manual_root / "Looni-atlas", Path("atlases"), Path("atlases") / "Looni-atlas"]
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            direct = root / name
            if direct.exists() and direct.stat().st_size > 0:
                return direct
        found = _find_one(root, names)
        if found is not None:
            return found
    return None


def _copy_runtime_template(t1_out: Path, out: Path, force: bool) -> None:
    _copy(t1_out, out / "MNI152_T1_2mm.nii.gz", force)


def prepare_icbm452(raw_root: Path, runtime_root: Path, manual_root: Path, force: bool = False, spacing_mm: float = 1.0) -> dict:
    atlas_id = "icbm452"
    out = runtime_root / "atlas_library" / atlas_id
    expected = ("ICBM_452_T1_Atlas_Affine_Transformations.zip", "ICBM452 NIfTI/Analyze T1 template")
    source_page = "https://resource.loni.usc.edu/resources/atlases/"
    try:
        archive = _loni_zip(manual_root, ("ICBM_452_T1_Atlas_Affine_Transformations.zip", "*ICBM*452*T1*.zip"))
        if archive is not None:
            extract_root = _safe_extract_nested_zip(archive, raw_root / atlas_id / "loni_icbm452", force)
            t1_src = _find_one(extract_root, ("*icbm452*atlas*air12*sinc.hdr", "*icbm452*.hdr", "*452*.nii*"))
        else:
            t1_src = _find_any(
                _manual_dirs(manual_root, atlas_id),
                ("*icbm*452*T1*.nii*", "*ICBM*452*T1*.nii*", "*452*T1*.nii*", "*icbm452*.hdr"),
            )
        if t1_src is None:
            raise FileNotFoundError("ICBM452 T1 template introuvable")
        t1_out = _runtime_image(
            t1_src,
            out / "icbm452_T1w.nii.gz",
            force=force,
            order=1,
            voxel_size_mm=spacing_mm,
        )
        t1_img = nib.load(str(t1_out))
        t1 = np.nan_to_num(t1_img.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mask = _largest_component(ndi.binary_fill_holes(t1 > 0))
        _save_mask_from_array(mask, t1_img, out / "brain_mask.nii.gz", force)
        _copy_runtime_template(t1_out, out, force)
        manifest = {
            "atlas": "ICBM 452 T1 Atlas",
            "available": True,
            "source": "LONI ICBM 452 T1 Atlas",
            "source_urls": {"page": source_page},
            "license": "LONI Atlas Distribution Agreement; manual verification required before publication",
            "raw_files": {"archive": None if archive is None else str(archive), "t1": str(t1_src)},
            "runtime_files": {
                "t1": str(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": str(out / "brain_mask.nii.gz"),
                "native_t1": str(t1_out),
            },
            "mask_derivation": "nonzero voxels from ICBM452 template, binary_fill_holes, largest component; no gold labels",
            "validation": {
                "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
            },
        }
        return _atlas_manifest(out, manifest)
    except Exception as exc:
        manifest = _unavailable_manifest(atlas_id, str(exc), manual_root, expected, [source_page])
        out.mkdir(parents=True, exist_ok=True)
        return _atlas_manifest(out, manifest)


def prepare_lpba40_average(raw_root: Path, runtime_root: Path, manual_root: Path, force: bool = False, spacing_mm: float = 1.0) -> dict:
    atlas_id = "lpba40_average"
    out = runtime_root / "atlas_library" / atlas_id
    expected = ("LPBA40_AIR_Atlas_NIFTI_format.zip", "LBPA40_Atlas_MBAT_v3.zip")
    source_page = "https://resource.loni.usc.edu/resources/atlases/"
    try:
        archive = _loni_zip(manual_root, ("LPBA40_AIR_Atlas_NIFTI_format.zip", "*LPBA40*AIR*NIFTI*.zip"))
        mask_derivation = "LPBA40 AIR maxprob label > 0"
        if archive is not None:
            extract_root = _safe_extract_nested_zip(archive, raw_root / atlas_id / "loni_lpba40_air", force)
            t1_src = _find_one(extract_root, ("*lpba40.air*brain.avg.nii*", "*brain.avg.nii*"))
            label_src = _find_one(
                extract_root,
                (
                    "lpba40.air.icbm452w5.label.nii.gz",
                    "lpba40.air.icbm452w5.label.nii",
                    "*maxprob/lpba40.air.icbm452w5.label.nii*",
                    "*lpba40.air*label.nii*",
                    "*label.nii*",
                ),
            )
        else:
            archive = _loni_zip(manual_root, ("LBPA40_Atlas_MBAT_v3.zip", "*LPBA40*MBAT*.zip", "*LBPA40*MBAT*.zip"))
            if archive is not None:
                extract_root = _safe_extract_nested_zip(archive, raw_root / atlas_id / "loni_lpba40_mbat", force)
                t1_src = _find_one(extract_root, ("lpba40.hdr", "*lpba40*.nii*"))
                label_src = _find_one(extract_root, ("lpba40.label.hdr", "*label*.nii*"))
                mask_derivation = "LPBA40 MBAT label > 0 fallback"
            else:
                search_roots = _manual_dirs(manual_root, atlas_id)
                t1_src = _find_any(search_roots, ("*lpba40*avg*brain*.nii*", "*LPBA40*avg*brain*.nii*", "*avg*brain*.nii*"))
                label_src = _find_any(search_roots, ("*lpba40*label*.nii*", "*label*.nii*", "*brain*mask*.nii*", "*mask*.nii*"))
        if t1_src is None:
            raise FileNotFoundError("LPBA40 average T1 template introuvable")
        if label_src is None:
            raise FileNotFoundError("LPBA40 label/mask introuvable")
        t1_out = _runtime_image(
            t1_src,
            out / "lpba40_average_T1w.nii.gz",
            force=force,
            order=1,
            voxel_size_mm=spacing_mm,
        )
        mask_out = _runtime_image(
            label_src,
            out / "brain_mask.nii.gz",
            force=force,
            order=0,
            voxel_size_mm=spacing_mm,
            is_mask=True,
        )
        t1_img = nib.load(str(t1_out))
        mask_img = nib.load(str(mask_out))
        if mask_img.shape[:3] != t1_img.shape[:3] or not np.allclose(mask_img.affine, t1_img.affine, atol=1e-4):
            mask_img = resample_from_to(mask_img, t1_img, order=0, cval=0)
            _save_mask_from_array(mask_img.get_fdata(dtype=np.float32) > 0.5, t1_img, out / "brain_mask.nii.gz", force=True)
        _copy_runtime_template(t1_out, out, force)
        manifest = {
            "atlas": "LPBA40 average",
            "available": True,
            "source": "LONI Probabilistic Brain Atlas LPBA40",
            "source_urls": {"page": source_page},
            "license": "LONI Atlas Distribution Agreement; dataset-derived atlas, report separately on LPBA40",
            "raw_files": {
                "archive": None if archive is None else str(archive),
                "t1": str(t1_src),
                "label_or_mask": str(label_src),
            },
            "runtime_files": {
                "t1": str(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": str(out / "brain_mask.nii.gz"),
                "native_t1": str(t1_out),
            },
            "mask_derivation": mask_derivation,
            "dataset_derived_warning": "LPBA40-derived public atlas; exclude from strict LPBA40 M1 unless explicitly enabled",
            "validation": {
                "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
            },
        }
        return _atlas_manifest(out, manifest)
    except Exception as exc:
        manifest = _unavailable_manifest(atlas_id, str(exc), manual_root, expected, [source_page])
        out.mkdir(parents=True, exist_ok=True)
        return _atlas_manifest(out, manifest)


def prepare_miitra_older_adult(
    raw_root: Path,
    runtime_root: Path,
    manual_root: Path,
    force: bool = False,
    spacing_mm: float = 1.0,
) -> dict:
    atlas_id = "miitra_older_adult"
    out = runtime_root / "atlas_library" / atlas_id
    source_page = "https://www.nitrc.org/projects/miitra/"
    expected = ("MIITRA-T1w-1mm.nii.gz", "MIITRA-BrainMask-05mm.nii.gz")
    try:
        search_roots = [
            manual_root / "MIITRA",
            manual_root / atlas_id,
            manual_root,
            Path("atlases") / "MIITRA",
            Path("data/external_atlases") / "MIITRA",
        ]
        t1_src = _find_any(search_roots, ("MIITRA-T1w-1mm.nii.gz", "*MIITRA*T1w*1mm*.nii*"))
        mask_src = _find_any(search_roots, ("MIITRA-BrainMask-05mm.nii.gz", "*MIITRA*BrainMask*.nii*"))
        if t1_src is None:
            raise FileNotFoundError("MIITRA T1w 1mm introuvable")
        if mask_src is None:
            raise FileNotFoundError("MIITRA BrainMask introuvable")
        t1_out = _runtime_image(
            t1_src,
            out / "miitra_older_adult_T1w.nii.gz",
            force=force,
            order=1,
            voxel_size_mm=spacing_mm,
        )
        t1_img = nib.load(str(t1_out))
        mask_img = nib.load(str(mask_src))
        if len(mask_img.shape) > 3:
            mask_img = _first_volume_image(mask_src)
        if mask_img.shape[:3] != t1_img.shape[:3] or not np.allclose(mask_img.affine, t1_img.affine, atol=1e-4):
            mask_img = resample_from_to(mask_img, t1_img, order=0, cval=0)
        mask = mask_img.get_fdata(dtype=np.float32) > 0.5
        mask = _largest_component(ndi.binary_fill_holes(mask))
        _save_mask_from_array(mask, t1_img, out / "brain_mask.nii.gz", force)
        _copy_runtime_template(t1_out, out, force)
        manifest = {
            "atlas": "MIITRA older-adult atlas",
            "available": True,
            "source": "NITRC MIITRA atlas",
            "source_urls": {"page": source_page},
            "license": "MIITRA/NITRC license; manual verification required before publication",
            "population": "non-demented older adults",
            "raw_files": {"t1": str(t1_src), "brain_mask": str(mask_src)},
            "runtime_files": {
                "t1": str(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": str(out / "brain_mask.nii.gz"),
                "native_t1": str(t1_out),
            },
            "mask_derivation": "MIITRA-BrainMask resampled nearest-neighbor to MIITRA-T1w grid, fill holes, largest component",
            "validation": {
                "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
            },
        }
        return _atlas_manifest(out, manifest)
    except Exception as exc:
        manifest = _unavailable_manifest(atlas_id, str(exc), manual_root, expected, [source_page])
        out.mkdir(parents=True, exist_ok=True)
        return _atlas_manifest(out, manifest)


def _manual_archive(
    manual_root: Path,
    atlas_id: str,
    names: Sequence[str],
    *,
    default_roots: Sequence[Path] = (Path("atlases"), Path("atlas")),
) -> Path | None:
    roots = [manual_root / atlas_id, manual_root, *default_roots]
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            direct = root / name
            if direct.exists() and direct.stat().st_size > 0:
                return direct
        found = _find_one(root, names)
        if found is not None:
            return found
    return None


def prepare_fbn125_neonatal(raw_root: Path, runtime_root: Path, manual_root: Path, force: bool = False) -> dict:
    atlas_id = "fbn125_neonatal"
    out = runtime_root / "atlas_library" / atlas_id
    source_page = "https://www.nitrc.org/projects/finnbrain/"
    expected = ("FBN-125-template-et-atlases.zip",)
    try:
        archive = _manual_archive(manual_root, atlas_id, expected)
        if archive is None:
            raise FileNotFoundError("FBN-125-template-et-atlases.zip introuvable")
        extract_root = _safe_extract_zip(archive, raw_root / atlas_id / "extracted", force)
        t1_src = _find_one(extract_root, ("avg_t1.nii", "avg_t1.nii.gz", "*avg*t1*.nii*"))
        t2_src = _find_one(extract_root, ("avg_t2.nii", "avg_t2.nii.gz", "*avg*t2*.nii*"))
        label_src = _find_one(
            extract_root,
            (
                "labels_symmetric.nii",
                "labels_symmetric.nii.gz",
                "labels_FreeSurfer_LUT.nii",
                "labels_FreeSurfer_LUT.nii.gz",
                "labels_SPM_style.nii",
                "labels_SPM_style.nii.gz",
                "*label*.nii*",
            ),
        )
        if t1_src is None:
            raise FileNotFoundError("FBN-125 avg_t1 introuvable")
        if label_src is None:
            raise FileNotFoundError("FBN-125 labels introuvables")
        t1_out = _runtime_image(t1_src, out / "fbn125_neonatal_T1w.nii.gz", force=force, order=1)
        if t2_src is not None:
            t2_out = _runtime_image(t2_src, out / "fbn125_neonatal_T2w.nii.gz", force=force, order=1)
        else:
            t2_out = None
        label_out = _runtime_image(
            label_src,
            out / "fbn125_neonatal_label_dseg.nii.gz",
            force=force,
            order=0,
            is_mask=False,
        )
        t1_img = nib.load(str(t1_out))
        label_img = nib.load(str(label_out))
        if label_img.shape[:3] != t1_img.shape[:3] or not np.allclose(label_img.affine, t1_img.affine, atol=1e-4):
            label_img = resample_from_to(label_img, t1_img, order=0, cval=0)
            nib.save(label_img, str(label_out))
        labels = np.rint(label_img.get_fdata(dtype=np.float32)).astype(np.int16)
        brain_mask = _largest_component(ndi.binary_fill_holes(labels > 0))
        _save_mask_from_array(brain_mask, t1_img, out / "brain_mask.nii.gz", force)
        # Diagnostic-only variant: useful when a dataset excludes CSF from the target mask.
        no_csf = labels > 0
        no_csf &= ~np.isin(labels, [3, 24])
        if np.count_nonzero(no_csf) > 0:
            no_csf = _largest_component(ndi.binary_fill_holes(no_csf))
            _save_mask_from_array(no_csf, t1_img, out / "brain_mask_no_csf.nii.gz", force)
        _copy_runtime_template(t1_out, out, force)
        manifest = {
            "atlas": "FinnBrain FBN-125 neonatal template and atlases",
            "atlas_id": atlas_id,
            "available": True,
            "source": "FBN-125-template-et-atlases.zip",
            "source_urls": {"page": source_page},
            "license": "FinnBrain/NITRC license; manual verification required before publication",
            "population": "term-born neonates; multi-contrast T1/T2/DTI template",
            "raw_files": {
                "archive": str(archive),
                "t1": str(t1_src),
                "t2": None if t2_src is None else str(t2_src),
                "labels": str(label_src),
            },
            "runtime_files": {
                "t1": str(out / "MNI152_T1_2mm.nii.gz"),
                "t2": None if t2_out is None else str(t2_out),
                "brain_mask": str(out / "brain_mask.nii.gz"),
                "labels": str(label_out),
                "brain_mask_no_csf": str(out / "brain_mask_no_csf.nii.gz"),
            },
            "mask_derivation": "FBN labels > 0, binary_fill_holes, largest component; no evaluated GT used",
            "recommended_for": ["MCRIB", "MCRIB_WM", "MCRIB2", "neonatal cohorts"],
            "validation": {
                "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
                "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
            },
        }
        return _atlas_manifest(out, manifest)
    except Exception as exc:
        manifest = _unavailable_manifest(atlas_id, str(exc), manual_root, expected, [source_page])
        out.mkdir(parents=True, exist_ok=True)
        return _atlas_manifest(out, manifest)


def prepare_alberts_gousias(raw_root: Path, runtime_root: Path, manual_root: Path, force: bool = False) -> list[dict]:
    atlas_id = "alberts_gousias"
    source_page = "https://brain-development.org/brain-atlases/neonatal-brain-atlases/neonatal-brain-atlas-albert/"
    expected = (
        "ALBERTs-Gousias-neonatal-atlases-images.tar.gz",
        "ALBERTs-Gousias-neonatal-atlases-segmentations.zip",
    )
    reports: list[dict] = []
    try:
        images_archive = _manual_archive(manual_root, atlas_id, ("ALBERTs-Gousias-neonatal-atlases-images.tar.gz",))
        seg_archive = _manual_archive(manual_root, atlas_id, ("ALBERTs-Gousias-neonatal-atlases-segmentations.zip",))
        if images_archive is None:
            raise FileNotFoundError("ALBERTs images tar.gz introuvable")
        if seg_archive is None:
            raise FileNotFoundError("ALBERTs segmentations zip introuvable")
        image_root = _safe_extract_tar(images_archive, raw_root / atlas_id / "images", force)
        seg_root = _safe_extract_zip(seg_archive, raw_root / atlas_id / "segmentations", force)
        for idx in range(1, 21):
            sid = f"{idx:02d}"
            t1_src = _find_one(image_root, (f"{sid}_T1.nii", f"{sid}_T1.nii.gz"))
            t2_src = _find_one(image_root, (f"{sid}_T2.nii", f"{sid}_T2.nii.gz"))
            label_src = _find_one(seg_root, (f"ALBERT_{sid}_ISG_seg50.nii", f"ALBERT_{sid}_ISG_seg50.nii.gz"))
            if t1_src is None or label_src is None:
                reports.append(
                    {
                        "atlas": f"ALBERTs-Gousias neonatal atlas {sid}",
                        "atlas_id": f"alberts_gousias_{sid}",
                        "available": False,
                        "reason": f"T1 ou segmentation manquante pour sujet {sid}",
                    }
                )
                continue
            out = runtime_root / "atlas_library" / f"alberts_gousias_{sid}"
            t1_out = _runtime_image(t1_src, out / f"alberts_gousias_{sid}_T1w.nii.gz", force=force, order=1)
            if t2_src is not None:
                t2_out = _runtime_image(t2_src, out / f"alberts_gousias_{sid}_T2w.nii.gz", force=force, order=1)
            else:
                t2_out = None
            label_out = _runtime_image(
                label_src,
                out / f"alberts_gousias_{sid}_label_dseg.nii.gz",
                force=force,
                order=0,
                is_mask=False,
            )
            t1_img = nib.load(str(t1_out))
            label_img = nib.load(str(label_out))
            if label_img.shape[:3] != t1_img.shape[:3] or not np.allclose(label_img.affine, t1_img.affine, atol=1e-4):
                label_img = resample_from_to(label_img, t1_img, order=0, cval=0)
                nib.save(label_img, str(label_out))
            labels = np.rint(label_img.get_fdata(dtype=np.float32)).astype(np.int16)
            brain_mask = _largest_component(ndi.binary_fill_holes(labels > 0))
            _save_mask_from_array(brain_mask, t1_img, out / "brain_mask.nii.gz", force)
            _copy_runtime_template(t1_out, out, force)
            manifest = {
                "atlas": f"ALBERTs/Gousias neonatal atlas {sid}",
                "atlas_id": f"alberts_gousias_{sid}",
                "available": True,
                "source": "ALBERTs-Gousias neonatal atlases images and segmentations",
                "source_urls": {"page": source_page},
                "license": "Brain Development ALBERTs academic license; do not redistribute without permission",
                "population": "neonates; individual manually labelled atlases",
                "raw_files": {
                    "images_archive": str(images_archive),
                    "segmentations_archive": str(seg_archive),
                    "t1": str(t1_src),
                    "t2": None if t2_src is None else str(t2_src),
                    "labels": str(label_src),
                },
                "runtime_files": {
                    "t1": str(out / "MNI152_T1_2mm.nii.gz"),
                    "t2": None if t2_out is None else str(t2_out),
                    "brain_mask": str(out / "brain_mask.nii.gz"),
                    "labels": str(label_out),
                },
                "mask_derivation": "ALBERT segmentation labels > 0, binary_fill_holes, largest component; no evaluated GT used",
                "recommended_for": ["MCRIB", "MCRIB_WM", "MCRIB2", "neonatal cohorts"],
                "validation": {
                    "t1": _validate_nifti(out / "MNI152_T1_2mm.nii.gz"),
                    "brain_mask": _validate_nifti(out / "brain_mask.nii.gz", is_mask=True),
                },
            }
            reports.append(_atlas_manifest(out, manifest))
        return reports
    except Exception as exc:
        out = runtime_root / "atlas_library" / atlas_id
        manifest = _unavailable_manifest(atlas_id, str(exc), manual_root, expected, [source_page])
        out.mkdir(parents=True, exist_ok=True)
        return [_atlas_manifest(out, manifest)]


def prepare_pediatric_templates(
    raw_root: Path,
    runtime_root: Path,
    manual_root: Path,
    template_ids: Sequence[str],
    *,
    force: bool = False,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    wanted = set(template_ids)
    if "fbn125_neonatal" in wanted:
        reports["fbn125_neonatal"] = prepare_fbn125_neonatal(raw_root, runtime_root, manual_root, force)
    if "alberts_gousias" in wanted:
        reports["alberts_gousias"] = prepare_alberts_gousias(raw_root, runtime_root, manual_root, force)
    return reports


def prepare_adult_templates(
    raw_root: Path,
    runtime_root: Path,
    manual_root: Path,
    template_ids: Sequence[str],
    *,
    force: bool = False,
    spacing_mm: float = 1.0,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    wanted = set(template_ids)
    if "oasis30ants" in wanted:
        reports["oasis30ants"] = prepare_oasis30ants(raw_root, runtime_root, force)
    if "mni152_nlin_2009c" in wanted:
        reports["mni152_nlin_2009c"] = prepare_mni152(runtime_root, force)
    if "sri24" in wanted:
        reports["sri24"] = prepare_sri24(raw_root, runtime_root, manual_root, force)
    if "colin27_2008" in wanted:
        reports["colin27_2008"] = prepare_colin27_2008(
            raw_root,
            runtime_root,
            manual_root,
            force=force,
            spacing_mm=spacing_mm,
        )
    if "icbm452" in wanted:
        reports["icbm452"] = prepare_icbm452(raw_root, runtime_root, manual_root, force, spacing_mm)
    if "lpba40_average" in wanted:
        reports["lpba40_average"] = prepare_lpba40_average(raw_root, runtime_root, manual_root, force, spacing_mm)
    if "miitra_older_adult" in wanted:
        reports["miitra_older_adult"] = prepare_miitra_older_adult(raw_root, runtime_root, manual_root, force, spacing_mm)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/external_atlases")
    parser.add_argument("--runtime-root", default="dataset")
    parser.add_argument("--manual-root", default=str(ADULT_TEMPLATE_MANUAL_ROOT))
    parser.add_argument(
        "--adult-templates",
        action="store_true",
        help="Prepare only the adult external template registry used by IBSR18 publication diagnostics.",
    )
    parser.add_argument(
        "--adult-template-ids",
        default=",".join(DEFAULT_ADULT_TEMPLATE_IDS),
        help="Comma-separated adult template ids to prepare.",
    )
    parser.add_argument(
        "--pediatric-templates",
        action="store_true",
        help="Prepare only the pediatric/neonatal external template registry.",
    )
    parser.add_argument(
        "--pediatric-template-ids",
        default=",".join(DEFAULT_PEDIATRIC_TEMPLATE_IDS),
        help="Comma-separated pediatric template ids to prepare.",
    )
    parser.add_argument("--adult-template-spacing-mm", type=float, default=1.0)
    parser.add_argument("--mniinfant-cohorts", default=",".join(str(c) for c in DEFAULT_MNIINFANT_COHORTS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    runtime_root = Path(args.runtime_root)
    manual_root = Path(args.manual_root)
    cohorts = [int(x) for x in str(args.mniinfant_cohorts).replace(" ", "").split(",") if x]
    adult_ids = [x for x in str(args.adult_template_ids).replace(" ", "").split(",") if x]
    pediatric_ids = [x for x in str(args.pediatric_template_ids).replace(" ", "").split(",") if x]

    runtime_layout = prepare_runtime_layout(raw_root, runtime_root, args.force)
    if args.adult_templates:
        report = {
            "runtime_layout": runtime_layout,
            "adult_template_policy": "external_templates_only_no_subject_banks",
            "atlases": prepare_adult_templates(
                raw_root,
                runtime_root,
                manual_root,
                adult_ids,
                force=args.force,
                spacing_mm=float(args.adult_template_spacing_mm),
            ),
        }
        summary = runtime_root / "atlas_library" / "external_adult_template_summary.json"
        _write_manifest(summary, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.pediatric_templates:
        report = {
            "runtime_layout": runtime_layout,
            "pediatric_template_policy": "external_templates_only_no_subject_banks_no_gold",
            "atlases": prepare_pediatric_templates(
                raw_root,
                runtime_root,
                manual_root,
                pediatric_ids,
                force=args.force,
            ),
        }
        summary = runtime_root / "atlas_library" / "external_pediatric_template_summary.json"
        _write_manifest(summary, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    report = {
        "runtime_layout": runtime_layout,
        "atlases": {
            "oasis30ants": prepare_oasis30ants(raw_root, runtime_root, args.force),
            "mni152_nlin_2009c": prepare_mni152(runtime_root, args.force),
            "mniinfant": prepare_mniinfant(raw_root, runtime_root, cohorts, args.force),
            "unc_bcp_06m": prepare_unc_bcp_06m(raw_root, runtime_root, args.force),
            "unc_infant": prepare_unc_infant(raw_root, runtime_root, args.force),
            "pediatric_templates": prepare_pediatric_templates(
                raw_root,
                runtime_root,
                manual_root,
                pediatric_ids,
                force=args.force,
            ),
        },
    }
    summary = runtime_root / "atlas_library" / "external_atlas_summary.json"
    _write_manifest(summary, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
