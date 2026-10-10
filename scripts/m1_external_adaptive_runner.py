"""No-LOO M1 runner for adult external-atlas skull stripping datasets.

The notebook owns the SAM2 classes and preprocessing helpers, so this module is
called with ``globals()`` from a notebook cell.  The runner itself never uses
dataset subjects as atlases: guidance comes only from the external atlas router
and the final adaptive fusion uses no gold masks.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import pandas as pd
from IPython.display import display
from nibabel.processing import resample_from_to, resample_to_output
from scipy import ndimage as ndi

from .adaptive_mask_fusion import apply_adaptive_mask_fusion
from .adult_candidate_adjudicator import apply_adult_candidate_adjudicator
from .adult_external_refinement import apply_adult_external_refinement
from .spatial_coherence_guard import apply_spatial_coherence_guard
from .adult_t1_adaptive_atlas import build_adult_t1_adaptive_guidance_atlas
from .adult_t1_boundary_refinement import apply_adult_t1_boundary_refinement
from .adult_t1_strong_refinement import apply_adult_t1_strong_refinement
from .inflation_guard import apply_inflation_guard
from .adult_intensity_refinement import apply_adult_intensity_refinement
from .external_atlas_config import (
    config_for_dataset_profile,
    default_output_slug,
    external_atlas_files_for_row,
    no_atlas_ablation,
    profile_for_row,
)
from .neuro_utils import largest_component, mask_stats, mask_volume_cm3, save_float, save_mask, write_brain_from_mask


_REQUIRED_NOTEBOOK_SYMBOLS = (
    "run_preprocessing_pipeline",
    "advanced_atlas_registration",
    "SegmentationParams",
    "SAM2BrainExtractor",
)


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


IBSR18_8BIT_FIRST_SLICE = {
    "1_24": -1,
    "2_4": 1,
    "4_8": 7,
    "5_8": 1,
    "16_3": 1,
    "17_3": 1,
    "100_23": 1,
    "191_3": 1,
    "202_3": 1,
    "205_3": 1,
    "6_10": 1,
    "7_8": 1,
    "8_4": 1,
    "11_3": 1,
    "12_3": 1,
    "13_3": 1,
    "15_3": 1,
    "110_3": 0,
    "111_2": 0,
    "112_2": 1,
}


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "oui"}


def _cell(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value)


def _existing_path(row: Mapping[str, Any], keys: list[str]) -> Path | None:
    for key in keys:
        value = _cell(row, key, "")
        if value and value.lower() != "nan":
            path = Path(value)
            if path.exists():
                return path
    return None


def _safe_subject(subject: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(subject)).strip("_")


def _prefix(subject: str) -> str:
    subject = str(subject)
    return subject if subject.startswith("sub-") else f"sub-{subject}"


def _registration_id(subject: str) -> str:
    subject = str(subject)
    return subject[4:] if subject.startswith("sub-") else subject


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if not math.isfinite(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_progress(
    path: str | Path,
    stage: str,
    subject: str | None = None,
    index: int | None = None,
    total: int | None = None,
    rows: list[dict[str, Any]] | None = None,
    start_time: float | None = None,
) -> None:
    rows = rows or []
    updated = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "updated": updated,
        "updated_at": updated,
        "stage": stage,
        "current": subject,
        "subject": subject,
        "index": index,
        "total": total,
        "completed_rows": len(rows),
        "errors": int(sum(1 for row in rows if str(row.get("error", "")).strip())),
    }
    if start_time is not None:
        payload["age_min"] = round((time.time() - start_time) / 60.0, 2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "oui"}


def _set_reproducibility(seed: int, *, deterministic_threads: bool = True) -> dict[str, Any]:
    """Best-effort deterministic setup for notebook-driven M1 runs."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    info: dict[str, Any] = {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_random_seeded": True,
        "torch_seeded": False,
        "ants_seeded": False,
    }
    if deterministic_threads:
        # ITK/ANTs reductions can vary slightly with thread scheduling.  Set this
        # before importing ANTs where possible; if ANTs is already loaded this is
        # still harmless and documents the intended run mode.
        info["previous_itk_threads"] = os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")
        info["previous_omp_threads"] = os.environ.get("OMP_NUM_THREADS")
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        info["deterministic_threads"] = True
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
        info["torch_seeded"] = True
    except Exception as exc:
        info["torch_seed_error"] = str(exc)
    try:
        import ants  # type: ignore

        ants.config._random_seed = seed
        info["ants_seeded"] = True
    except Exception as exc:
        info["ants_seed_error"] = str(exc)
    return info


def _archive_existing_outputs(out_base: Path, *, run_id: str) -> Path | None:
    """Move the previous run aside before force-cleaning output files."""

    if not out_base.exists():
        return None
    items = [p for p in out_base.iterdir() if p.name not in {"_README_ARCHIVES.txt"}]
    if not items:
        return None
    archive_root = out_base.parent / "_m1_archives" / f"{out_base.name}_{run_id}"
    if archive_root.exists():
        base = archive_root
        suffix = 2
        while archive_root.exists():
            archive_root = Path(f"{base}_{suffix}")
            suffix += 1
    archive_root.mkdir(parents=True, exist_ok=False)
    for item in items:
        shutil.move(str(item), str(archive_root / item.name))
    (archive_root / "ARCHIVE_REASON.txt").write_text(
        "Previous M1 outputs archived automatically before force_clean_outputs=True.\n"
        "This prevents a good run from being overwritten by a later diagnostic run.\n"
    )
    return archive_root


def _manifest_path(dataset: str, manifest: str | Path | None) -> Path:
    if manifest:
        return Path(manifest)
    return Path("datasets/manifests") / f"{dataset}.csv"


def _lpba_expected_layout_message() -> str:
    return (
        "LPBA40 brut introuvable. Placer les donnees sous data/LPBA40 avec "
        "des fichiers **/Sxx.native.mri.hdr et **/Sxx.native.brain.mask.hdr "
        "plus leurs sidecars .img ou .img.gz, "
        "puis relancer la cellule."
    )


def _lpba_manifest_runtime_ready(manifest: Path) -> bool:
    if not manifest.exists() or manifest.stat().st_size == 0:
        return False
    try:
        df = pd.read_csv(manifest)
    except Exception:
        return False
    if df.empty:
        return False
    for _, row in df.iterrows():
        if not _bool(row.get("has_gold", True), True) or _bool(row.get("qc_only", False), False):
            continue
        t1_path = Path(str(row.get("input_path") or row.get("t1_path") or ""))
        mask_path = Path(str(row.get("brain_mask_path") or row.get("gold_mask_path") or ""))
        if not t1_path.exists() or not mask_path.exists():
            return False
        if t1_path.suffix.lower() == ".hdr" or mask_path.suffix.lower() == ".hdr":
            return False
        if not (str(t1_path).endswith(".nii") or str(t1_path).endswith(".nii.gz")):
            return False
        if not (str(mask_path).endswith(".nii") or str(mask_path).endswith(".nii.gz")):
            return False
    return True


def ensure_lpba40_prepared(
    *,
    data_root: str | Path = "data",
    output_root: str | Path = "datasets",
    force: bool = True,
) -> Path:
    """Create the LPBA40 standardized manifest when raw Analyze files exist."""

    data_root = Path(data_root)
    output_root = Path(output_root)
    manifest = output_root / "manifests" / "LPBA40.csv"
    if _lpba_manifest_runtime_ready(manifest):
        return manifest
    raw_root = data_root / "LPBA40"
    if not raw_root.exists():
        raise FileNotFoundError(_lpba_expected_layout_message())

    cmd = [
        sys.executable,
        "scripts/standardize_datasets.py",
        "--data-root",
        str(data_root),
        "--output-root",
        str(output_root),
    ]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, check=True)
    if not manifest.exists() or manifest.stat().st_size == 0:
        raise FileNotFoundError(_lpba_expected_layout_message())
    return manifest


def preflight_gold_manifest(
    dataset: str,
    *,
    manifest: str | Path | None = None,
    auto_prepare_lpba40: bool = False,
) -> Path:
    """Validate that a manifest has exploitable gold rows before running M1."""

    if dataset == "LPBA40" and auto_prepare_lpba40:
        manifest_path = ensure_lpba40_prepared()
    else:
        manifest_path = _manifest_path(dataset, manifest)
    if not manifest_path.exists():
        if dataset == "LPBA40":
            raise FileNotFoundError(_lpba_expected_layout_message())
        raise FileNotFoundError(f"Manifest introuvable: {manifest_path}")

    df = pd.read_csv(manifest_path)
    has_gold = (
        df["has_gold"].map(lambda value: _bool(value, True))
        if "has_gold" in df.columns
        else pd.Series(True, index=df.index)
    )
    qc_only = (
        df["qc_only"].map(lambda value: _bool(value, False))
        if "qc_only" in df.columns
        else pd.Series(False, index=df.index)
    )
    gold = df[has_gold & ~qc_only]
    if gold.empty:
        raise RuntimeError(f"Aucun sujet gold exploitable dans {manifest_path}")
    volumes = pd.to_numeric(gold.get("mask_volume_cm3"), errors="coerce").dropna()
    print(f"Preflight {dataset}: {len(gold)} sujets gold | manifest={manifest_path}")
    if len(volumes):
        print(
            "Volumes gold diagnostic: "
            f"min={volumes.min():.1f} median={volumes.median():.1f} max={volumes.max():.1f} cm3"
        )
    return manifest_path


def _load_records(
    dataset: str,
    *,
    manifest: str | Path | None = None,
    profile_filter: str | None = None,
    subjects: list[str] | tuple[str, ...] | None = None,
    n_subjects: int | None = None,
    require_atlas: bool = True,
) -> list[dict[str, Any]]:
    manifest_path = _manifest_path(dataset, manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest introuvable: {manifest_path}")

    df = pd.read_csv(manifest_path)
    subjects_set = set(str(s) for s in subjects) if subjects else None
    records: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        rd = row.to_dict()
        subject = _cell(rd, "subject", "")
        if not subject:
            continue
        if subjects_set and subject not in subjects_set:
            continue
        if profile_filter and _cell(rd, "profile", "") != profile_filter:
            continue
        if not _bool(rd.get("has_gold", True), True) or _bool(rd.get("qc_only", False), False):
            skipped.append((subject, "not_gold_or_qc_only"))
            continue

        t1_path = _existing_path(rd, ["input_path", "t1_path", "source_t1_path"])
        gold_path = _existing_path(rd, ["brain_mask_path", "gold_mask_path", "mask_path"])
        if t1_path is None or gold_path is None:
            skipped.append((subject, "missing_t1_or_gold"))
            continue

        profile = profile_filter or profile_for_row(dataset, rd)
        atlas_paths = external_atlas_files_for_row(dataset, rd, profile_filter=profile)
        if require_atlas and not atlas_paths:
            skipped.append((subject, "missing_external_atlas"))
            continue
        _assert_external_no_loo(dataset, atlas_paths)

        records.append(
            {
                "dataset": dataset,
                "subject": subject,
                "session": _cell(rd, "session", ""),
                "split": _cell(rd, "split", ""),
                "profile": profile,
                "row": rd,
                "t1_path": t1_path,
                "gold_path": gold_path,
                "atlas_paths": atlas_paths,
            }
        )

    if subjects_set:
        found = {record["subject"] for record in records}
        missing = sorted(subjects_set - found)
        if missing:
            raise RuntimeError(f"Sujets demandes absents ou non exploitables dans {manifest_path}: {missing}")

    if n_subjects is not None:
        records = records[: int(n_subjects)]
    if not records:
        reason = f" Skips={skipped[:10]}" if skipped else ""
        raise RuntimeError(f"Aucun sujet gold-standard trouve pour {dataset}/{profile_filter}.{reason}")
    return records


def _assert_external_no_loo(dataset: str, atlas_paths: Mapping[str, Path]) -> None:
    dataset_key = dataset.lower()
    for name, path in atlas_paths.items():
        path_text = str(path).replace("\\", "/").lower()
        text = f"{name} {path_text}".lower()
        parts = [part for part in path_text.split("/") if part]
        atlas_library_tail = ""
        if "atlas_library" in parts:
            atlas_library_tail = "/".join(parts[parts.index("atlas_library") + 1 :])
        dataset_specific = dataset_key in text if dataset_key == "ibsr18" else atlas_library_tail.startswith(dataset_key)
        subject_bank = "/datasets/atlas_library/" in f"/{path_text}"
        if "loo" in text or f"{dataset_key}_loo" in text or dataset_specific or subject_bank or f"/{dataset_key}_loo" in path_text:
            raise RuntimeError(f"Atlas LOO interdit pour {dataset}: {name} -> {path}")


def _subject_paths(anat_dir: str | Path, subject: str) -> dict[str, Path]:
    prefix = _prefix(subject)
    anat_dir = Path(anat_dir)
    return {
        "n4": anat_dir / f"{prefix}_desc-n4_T1w.nii.gz",
        "n4_work": anat_dir / f"{prefix}_desc-n4_res-work_T1w.nii.gz",
        "brain_work": anat_dir / f"{prefix}_desc-brain_res-work_T1w.nii.gz",
        "mask_work": anat_dir / f"{prefix}_desc-brain_res-work_mask.nii.gz",
        "brain": anat_dir / f"{prefix}_desc-brain_T1w.nii.gz",
        "mask": anat_dir / f"{prefix}_desc-brain_mask.nii.gz",
    }


def _repair_nifti_spatial_reference(image_path: str | Path | None, reference_path: str | Path) -> dict[str, Any]:
    """Copy the source image spatial frame onto a preprocessing output.

    The notebook preprocessing functions use SimpleITK for N4/WhiteStripe.
    On LPBA40 Analyze-derived NIfTI inputs, SimpleITK can write an image with
    the same voxel array but a fresh LPS/zero-origin affine.  That makes Dice
    and HD95 effectively compare two different physical spaces.  When shape
    and spacing match, the correct repair is to preserve the reference affine.
    """

    if not image_path:
        return {"checked": False, "changed": False, "reason": "missing_path"}
    image_path = Path(image_path)
    reference_path = Path(reference_path)
    if not image_path.exists():
        return {"checked": False, "changed": False, "reason": "path_not_found", "path": str(image_path)}
    try:
        img = nib.load(str(image_path))
        ref = nib.load(str(reference_path))
    except Exception as exc:
        return {"checked": False, "changed": False, "reason": f"load_failed:{exc}", "path": str(image_path)}

    if img.shape[:3] != ref.shape[:3]:
        return {
            "checked": True,
            "changed": False,
            "reason": "shape_mismatch",
            "path": str(image_path),
            "shape": tuple(int(v) for v in img.shape[:3]),
            "reference_shape": tuple(int(v) for v in ref.shape[:3]),
        }
    img_zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
    ref_zooms = tuple(float(v) for v in ref.header.get_zooms()[:3])
    if not np.allclose(img_zooms, ref_zooms, atol=1e-3):
        return {
            "checked": True,
            "changed": False,
            "reason": "spacing_mismatch",
            "path": str(image_path),
            "zooms": img_zooms,
            "reference_zooms": ref_zooms,
        }
    if np.allclose(img.affine, ref.affine, atol=1e-4):
        return {"checked": True, "changed": False, "reason": "already_aligned", "path": str(image_path)}

    data = np.asanyarray(img.dataobj).copy()
    header = img.header.copy()
    header.set_data_dtype(img.get_data_dtype())
    try:
        zooms = list(header.get_zooms())
        for axis in range(min(3, len(zooms))):
            zooms[axis] = ref_zooms[axis]
        if len(zooms) == data.ndim:
            header.set_zooms(tuple(zooms))
    except Exception:
        pass
    repaired = nib.Nifti1Image(data, ref.affine, header)
    qform_code = int(ref.header["qform_code"]) if int(ref.header["qform_code"]) > 0 else 1
    sform_code = int(ref.header["sform_code"]) if int(ref.header["sform_code"]) > 0 else 1
    repaired.set_qform(ref.affine, code=qform_code)
    repaired.set_sform(ref.affine, code=sform_code)
    repaired.set_data_dtype(img.get_data_dtype())
    nib.save(repaired, str(image_path))
    return {
        "checked": True,
        "changed": True,
        "reason": "affine_repaired",
        "path": str(image_path),
        "old_axcodes": "".join(nib.aff2axcodes(img.affine)),
        "new_axcodes": "".join(nib.aff2axcodes(ref.affine)),
    }


def _work_image_path(native_n4_path: Path, work_path: Path, cfg: Mapping[str, Any]) -> tuple[Path, bool]:
    spacing = cfg.get("work_resample_spacing_mm")
    if not spacing:
        return native_n4_path, False
    spacing = tuple(float(v) for v in spacing)
    native_img = nib.load(str(native_n4_path))
    zooms = tuple(float(v) for v in native_img.header.get_zooms()[:3])
    if len(spacing) != 3 or all(abs(a - b) < 1e-3 for a, b in zip(zooms, spacing)):
        return native_n4_path, False
    if work_path.exists():
        return work_path, True
    work_path.parent.mkdir(parents=True, exist_ok=True)
    resampled = resample_to_output(native_img, voxel_sizes=spacing, order=1)
    resampled.set_data_dtype(np.float32)
    nib.save(resampled, str(work_path))
    return work_path, True


def _resample_mask_to_native(work_mask_path: Path, native_n4_path: Path, native_mask_path: Path, native_brain_path: Path) -> None:
    work_img = nib.load(str(work_mask_path))
    native_img = nib.load(str(native_n4_path))
    if work_img.shape[:3] == native_img.shape[:3] and np.allclose(work_img.affine, native_img.affine, atol=1e-3):
        if work_mask_path != native_mask_path:
            native_mask_path.write_bytes(work_mask_path.read_bytes())
        write_brain_from_mask(native_n4_path, native_mask_path, native_brain_path)
        return
    res = resample_from_to(work_img, native_img, order=0, cval=0)
    mask = res.get_fdata(dtype=np.float32) > 0.5
    save_mask(mask, native_img, native_mask_path)
    write_brain_from_mask(native_n4_path, native_mask_path, native_brain_path)


def _atlas_template_for_mask(mask_path: str | Path) -> Path:
    parent = Path(mask_path).parent
    candidates = [
        parent / "tpl-OASIS30ANTs_res-01_T1w.nii.gz",
        parent / "tpl-MNI152NLin2009cAsym_res-02_T1w.nii.gz",
        parent / "sri24_spgr_unstrip_T1w.nii.gz",
        parent / "colin27_2008_t1_tal_res-1mm_T1w.nii.gz",
        parent / "miitra_older_adult_T1w.nii.gz",
        parent / "MIITRA-T1w-1mm.nii.gz",
        parent / "icbm452_T1w.nii.gz",
        parent / "lpba40_average_T1w.nii.gz",
        parent / "MNI152NLin2009cAsym_T1w_2mm.nii.gz",
        parent / "MNI152_T1_2mm.nii.gz",
        parent / "T1w.nii.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Template T1 introuvable pour atlas {mask_path}")


def _ants_fixed_mask(fixed_img: Any) -> Any | None:
    try:
        fixed_mask = fixed_img.get_mask() if hasattr(fixed_img, "get_mask") else None
    except Exception:
        fixed_mask = None
    if fixed_mask is None:
        try:
            import ants  # type: ignore

            fixed_mask = ants.get_mask(fixed_img)
        except Exception:
            fixed_mask = None
    try:
        if fixed_mask is not None and int(np.count_nonzero(fixed_mask.numpy())) >= 1000:
            return fixed_mask
    except Exception:
        pass
    return None


def _native_ants_syn_atlas_registration(
    subject_n4_path: str | Path,
    atlas_paths: Mapping[str, Path],
    output_dir: str | Path,
    *,
    subject_id: str,
    cfg: Mapping[str, Any],
    verbose: bool = False,
) -> dict[str, Any]:
    """Register external atlas masks directly into the subject native grid."""

    try:
        import ants  # type: ignore
    except Exception as exc:
        raise RuntimeError("ANTsPy est requis pour atlas_registration_backend='native_ants_syn'") from exc

    subject_n4_path = Path(subject_n4_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transforms_dir = output_dir / "_ants_transforms"
    transforms_dir.mkdir(parents=True, exist_ok=True)

    fixed = ants.image_read(str(subject_n4_path))
    fixed_mask = _ants_fixed_mask(fixed)
    transform = str(cfg.get("atlas_registration_transform", "SyN"))
    reg_iterations = tuple(int(v) for v in cfg.get("atlas_registration_syn_iterations", (60, 40, 20, 0)))
    aff_iterations = tuple(int(v) for v in cfg.get("atlas_registration_affine_iterations", (1000, 500, 250, 0)))
    aff_shrink = tuple(int(v) for v in cfg.get("atlas_registration_affine_shrink_factors", (8, 4, 2, 1)))
    aff_sigma = tuple(float(v) for v in cfg.get("atlas_registration_affine_smoothing_sigmas", (4, 2, 1, 0)))
    aff_sampling_rate = float(cfg.get("atlas_registration_aff_random_sampling_rate", 0.10))
    random_seed = cfg.get("atlas_registration_random_seed")
    if random_seed is not None:
        try:
            ants.config._random_seed = int(random_seed)
        except Exception:
            pass
    variants = tuple(str(v) for v in cfg.get("atlas_registration_variants", ("syn_masked", "syn_nomask")))

    results: dict[str, Any] = {"subject_id": subject_id, "registered_atlas_paths": {}, "metrics": {}}
    for name, mask_path in sorted(atlas_paths.items()):
        mask_path = Path(mask_path)
        template_path = _atlas_template_for_mask(mask_path)
        moving_t1 = ants.image_read(str(template_path))
        moving_mask = ants.image_read(str(mask_path))

        for variant in variants:
            use_mask = variant != "syn_nomask" and fixed_mask is not None
            suffix = "syn" if use_mask else "syn_nomask"
            key = name if use_mask else f"{name}_nomask"
            final_path = output_dir / f"sub-{subject_id}_{name}_{suffix}.nii.gz"
            outprefix = str(transforms_dir / f"sub-{subject_id}_{name}_{suffix}_")

            t0 = time.time()
            try:
                reg = ants.registration(
                    fixed=fixed,
                    moving=moving_t1,
                    type_of_transform=transform,
                    outprefix=outprefix,
                    mask=fixed_mask if use_mask else None,
                    mask_all_stages=bool(use_mask),
                    aff_metric="mattes",
                    aff_sampling=32,
                    aff_random_sampling_rate=aff_sampling_rate,
                    syn_metric=str(cfg.get("atlas_registration_syn_metric", "CC")),
                    syn_sampling=int(cfg.get("atlas_registration_syn_sampling", 4)),
                    reg_iterations=reg_iterations,
                    aff_iterations=aff_iterations,
                    aff_shrink_factors=aff_shrink,
                    aff_smoothing_sigmas=aff_sigma,
                    verbose=verbose,
                    **({"random_seed": int(random_seed)} if random_seed is not None else {}),
                )
                warped = ants.apply_transforms(
                    fixed=fixed,
                    moving=moving_mask,
                    transformlist=reg["fwdtransforms"],
                    interpolator="nearestNeighbor",
                    verbose=verbose,
                )
                ants.image_write(warped, str(final_path))
            except Exception as exc:
                results["metrics"][key] = {
                    "registration_backend": "native_ants_syn",
                    "registration_error": str(exc),
                    "variant": variant,
                }
                continue

            elapsed = round(time.time() - t0, 2)
            if not final_path.exists():
                results["metrics"][key] = {
                    "registration_backend": "native_ants_syn",
                    "registration_error": "warped mask missing",
                    "variant": variant,
                }
                continue
            results["registered_atlas_paths"][key] = str(final_path)
            results["metrics"][key] = {
                "registration_backend": "native_ants_syn",
                "transform": transform,
                "variant": variant,
                "used_fixed_mask": bool(use_mask),
                "template_path": str(template_path),
                "registration_sec": elapsed,
                "random_seed": None if random_seed is None else int(random_seed),
                "aff_random_sampling_rate": aff_sampling_rate,
            }
    return results


def _register_external_atlases(
    api: Mapping[str, Any],
    *,
    subject_n4_path: str | Path,
    atlas_paths: Mapping[str, Path],
    output_dir: str | Path,
    subject_id: str,
    cfg: Mapping[str, Any],
    verbose: bool = False,
) -> dict[str, Any]:
    if cfg.get("atlas_registration_backend") == "native_ants_syn":
        return _native_ants_syn_atlas_registration(
            subject_n4_path=subject_n4_path,
            atlas_paths=atlas_paths,
            output_dir=output_dir,
            subject_id=subject_id,
            cfg=cfg,
            verbose=verbose,
        )
    return api["advanced_atlas_registration"](
        subject_n4_path=subject_n4_path,
        atlas_paths=atlas_paths,
        output_dir=output_dir,
        subject_id=subject_id,
        create_brain_mask=True,
        verbose=verbose,
    )


def _cached_registered_atlases(atlas_dir: str | Path, subject: str) -> dict[str, Path]:
    reg_id = _registration_id(subject)
    registered: dict[str, Path] = {}
    for path in sorted(Path(atlas_dir).glob(f"sub-{reg_id}_*.nii.gz")):
        name = path.name
        prefix = f"sub-{reg_id}_"
        if name.startswith(prefix):
            key = name[len(prefix) :].replace(".nii.gz", "")
            registered[key] = path
    return registered


def _atlas_intensity_quality(mask: np.ndarray, image: np.ndarray) -> tuple[float | None, float]:
    if not mask.any() or mask.shape != image.shape:
        return None, 0.0
    finite = np.isfinite(image)
    inside = image[mask & finite]
    shell = ndi.binary_dilation(mask, iterations=3) & ~mask
    outside = image[shell & finite]
    if inside.size < 100 or outside.size < 100:
        return None, 0.0
    inside_med = float(np.median(inside))
    outside_med = float(np.median(outside))
    ratio = inside_med / max(outside_med, 1e-6)
    score = float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))
    return ratio, score


def _load_mask_in_reference(path: str | Path, ref_img) -> np.ndarray:
    img = nib.load(str(path))
    if img.shape[:3] != ref_img.shape[:3] or not np.allclose(img.affine, ref_img.affine, atol=1e-3):
        img = resample_from_to(img, ref_img, order=0, cval=0)
    return img.get_fdata(dtype=np.float32) > 0.5


def _soft_range_score(value: float, hard_min: float, soft_min: float, soft_max: float, hard_max: float) -> float:
    if not math.isfinite(value) or value < hard_min or value > hard_max:
        return 0.0
    if soft_min <= value <= soft_max:
        return 1.0
    if value < soft_min:
        return float(np.clip((value - hard_min) / max(soft_min - hard_min, 1e-6), 0.0, 1.0))
    return float(np.clip((hard_max - value) / max(hard_max - soft_max, 1e-6), 0.0, 1.0))


def _dice_bool(a: np.ndarray, b: np.ndarray) -> float:
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def _path_list_dedup(paths: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _atlas_family(name: str | Path) -> str:
    text = str(name).lower()
    for family in (
        "oasis30ants",
        "miitra_older_adult",
        "miitra",
        "mni152_nlin2009c",
        "mni152",
        "lpba40",
        "sri24",
        "icbm",
        "colin",
        "nki",
    ):
        if family in text:
            return family
    text = Path(str(name)).stem.lower()
    text = text.replace("_syn_nomask", "").replace("_syn", "")
    text = text.replace("_aff", "").replace("_nomask", "")
    return text


def _clean_external_atlas_mask(mask: np.ndarray) -> np.ndarray:
    """Keep the main warped brain component before no-gold atlas scoring."""

    clean = ndi.binary_fill_holes(np.asarray(mask, dtype=bool)).astype(bool)
    return largest_component(clean)


def _atlas_clean_path(out_dir: Path, name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")[:96] or "atlas"
    clean_dir = out_dir / "_cleaned_atlases"
    clean_dir.mkdir(parents=True, exist_ok=True)
    return clean_dir / f"{safe}.nii.gz"


def _extend_unique_paths(paths: list[Path], additions: list[str | Path | None]) -> list[Path]:
    return _path_list_dedup([*paths, *[p for p in additions if p]])


def select_external_atlas_ensemble(
    registered_paths: Mapping[str, Path],
    work_image_path: str | Path,
    cfg: Mapping[str, Any],
    out_dir: str | Path,
) -> dict[str, Any]:
    """Select a no-gold, no-LOO external atlas ensemble in work-image space."""

    if not registered_paths:
        raise RuntimeError("aucun atlas externe recale exploitable")
    for name, path in registered_paths.items():
        registered_text = f"{name} {Path(path).name}".lower()
        if "loo" in registered_text:
            raise RuntimeError(f"Atlas LOO interdit dans les atlas recales: {name} -> {path}")

    ref_img = nib.load(str(work_image_path))
    subject_img = ref_img.get_fdata(dtype=np.float32)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hard_min = float(cfg.get("atlas_qc_min_cm3", cfg.get("min_volume_cm3", 0.0)))
    hard_max = float(cfg.get("atlas_qc_max_cm3", cfg.get("max_volume_cm3", 1e9)))
    soft_min = float(cfg.get("atlas_soft_min_cm3", hard_min + 0.20 * max(hard_max - hard_min, 1.0)))
    soft_max = float(cfg.get("atlas_soft_max_cm3", hard_max - 0.15 * max(hard_max - hard_min, 1.0)))
    if soft_min > soft_max:
        soft_min, soft_max = hard_min, hard_max

    max_components = int(cfg.get("atlas_ensemble_max_components", cfg.get("max_components", 3)))
    max_extent = float(cfg.get("atlas_ensemble_max_extent_ratio", 0.985))
    intensity_weight = float(cfg.get("atlas_intensity_score_weight", 0.0))
    agreement_weight = float(cfg.get("atlas_agreement_weight", 0.35))
    disagreement_penalty = float(cfg.get("atlas_disagreement_penalty", 0.20))
    nomask_penalty = float(cfg.get("atlas_nomask_penalty", 0.02))
    topk = max(1, int(cfg.get("atlas_ensemble_topk", 3)))
    balance_families = bool(cfg.get("atlas_ensemble_balance_families", True))
    consensus_primary_enabled = bool(cfg.get("atlas_consensus_primary_on_disagreement", True))
    consensus_primary_disagreement = float(cfg.get("atlas_consensus_primary_disagreement_min", 0.12))
    consensus_primary_min_ratio = float(cfg.get("atlas_consensus_primary_min_ratio", 0.82))
    consensus_primary_max_ratio = float(cfg.get("atlas_consensus_primary_max_ratio", 1.18))
    consensus_primary_max_intensity = float(cfg.get("atlas_consensus_primary_max_intensity_score", 0.20))
    rescue_enabled = bool(cfg.get("atlas_rescue_tissue_pruned_enabled", True))
    rescue_max_cm3 = float(cfg.get("atlas_rescue_overlarge_max_cm3", max(hard_max * 1.35, hard_max)))
    rescue_percentiles = [float(v) for v in cfg.get("atlas_rescue_tissue_percentiles", (5.0, 8.0, 10.0, 12.0, 15.0, 20.0))]
    low_trust_rescue_enabled = bool(cfg.get("atlas_low_trust_rescue_enabled", False))
    low_trust_intensity_threshold = float(cfg.get("atlas_low_trust_intensity_threshold", 0.10))
    low_trust_min_intensity_gain = float(cfg.get("atlas_low_trust_rescue_min_intensity_gain", 0.25))
    low_trust_primary_bonus = float(cfg.get("atlas_low_trust_rescue_primary_bonus", 0.0))
    low_trust_max_extent = float(cfg.get("atlas_low_trust_rescue_max_extent_ratio", 1.0))

    working: list[dict[str, Any]] = []

    def add_working_item(
        name: str,
        path: Path,
        mask: np.ndarray,
        *,
        original_path: Path,
        rescue_candidate: bool = False,
    ) -> None:
        clean_mask = _clean_external_atlas_mask(mask)
        if not clean_mask.any():
            return
        clean_path = _atlas_clean_path(out_dir, name)
        save_mask(clean_mask, ref_img, clean_path)
        stats = mask_stats(clean_mask, ref_img)
        volume = float(stats["volume_cm3"])
        components = int(stats["components"])
        extent = float(stats.get("extent_ratio", 0.0))
        volume_score = _soft_range_score(volume, hard_min, soft_min, soft_max, hard_max)
        component_score = float(np.clip(1.0 - 0.08 * max(0, components - 1), 0.0, 1.0))
        if components > max_components:
            component_score *= 0.35
        extent_score = 1.0 if extent <= max_extent else float(np.clip((1.0 - extent) / max(1.0 - max_extent, 1e-6), 0.0, 1.0))
        intensity_ratio, intensity_score = _atlas_intensity_quality(clean_mask, subject_img)
        family = _atlas_family(name)
        score = (
            0.46 * volume_score
            + 0.20 * component_score
            + 0.16 * extent_score
            + intensity_weight * intensity_score
        )
        if "nomask" in name.lower():
            score -= nomask_penalty
        if family.startswith("miitra"):
            score += float(cfg.get("atlas_miitra_prior_bonus", 0.0))
        if family.startswith("icbm"):
            score -= float(cfg.get("atlas_icbm_derived_mask_penalty", 0.0))
        if rescue_candidate:
            score -= float(cfg.get("atlas_rescue_candidate_penalty", 0.10))
        plausible = bool(
            clean_mask.any()
            and hard_min <= volume <= hard_max
            and components <= max_components
            and extent <= max_extent
        )
        working.append(
            {
                "name": name,
                "path": clean_path,
                "original_path": str(original_path),
                "mask": clean_mask,
                "score_base": float(score),
                "score": float(score),
                "plausible": plausible,
                "rescue_candidate": bool(rescue_candidate),
                "atlas_volume_cm3": volume,
                "atlas_components": components,
                "atlas_extent_ratio": extent,
                "atlas_volume_score": volume_score,
                "atlas_component_score": component_score,
                "atlas_extent_score": extent_score,
                "atlas_intensity_ratio": intensity_ratio,
                "atlas_intensity_score": intensity_score,
                "atlas_agreement_score": 0.0,
                "atlas_family": family,
            }
        )

    for name, path in sorted(registered_paths.items()):
        path = Path(path)
        if not path.exists():
            continue
        try:
            mask = _load_mask_in_reference(path, ref_img)
            add_working_item(name, path, mask, original_path=path)
            raw_volume = mask_volume_cm3(mask, ref_img)
            if rescue_enabled and hard_max < raw_volume <= rescue_max_cm3 and mask.any():
                values = subject_img[mask & np.isfinite(subject_img)]
                if values.size >= 100:
                    for pct in rescue_percentiles:
                        threshold = float(np.percentile(values, pct))
                        rescued = mask & (subject_img >= threshold)
                        rescue_name = f"{name}_tissue_p{pct:g}"
                        add_working_item(rescue_name, path, rescued, original_path=path, rescue_candidate=True)
        except Exception as exc:
            working.append(
                {
                    "name": name,
                    "path": path,
                    "score_base": -1e9,
                    "score": -1e9,
                    "plausible": False,
                    "error": str(exc),
                }
            )

    if not working:
        raise RuntimeError("aucun atlas externe recale exploitable")

    candidates = [item for item in working if item.get("plausible") and "mask" in item]
    selection_reason = "external_ensemble_plausible"
    if not candidates:
        candidates = [item for item in working if "mask" in item]
        selection_reason = "external_ensemble_fallback_best_scored"
    if not candidates:
        scored = sorted(working, key=lambda item: float(item.get("score", -1e9)), reverse=True)
        raise RuntimeError(f"aucun atlas externe lisible: {scored[0].get('error', 'unknown')}")

    if low_trust_rescue_enabled and candidates:
        provisional = max(candidates, key=lambda item: float(item.get("score_base", item.get("score", -1e9))))
        provisional_intensity = float(provisional.get("atlas_intensity_score") or 0.0)
        if provisional_intensity <= low_trust_intensity_threshold:
            candidate_ids = {id(item) for item in candidates}
            rescue_pool = [
                item
                for item in working
                if id(item) not in candidate_ids
                and "mask" in item
                and "nomask" in str(item.get("name", "")).lower()
                and hard_min <= float(item.get("atlas_volume_cm3", 0.0)) <= hard_max
                and int(item.get("atlas_components", 99)) <= max_components
                and float(item.get("atlas_extent_ratio", 1.0)) <= low_trust_max_extent
                and float(item.get("atlas_intensity_score") or 0.0)
                >= provisional_intensity + low_trust_min_intensity_gain
            ]
            if rescue_pool:
                rescue_item = max(
                    rescue_pool,
                    key=lambda item: (
                        float(item.get("atlas_intensity_score") or 0.0),
                        float(item.get("score_base", item.get("score", -1e9))),
                    ),
                )
                rescue_item["plausible"] = True
                rescue_item["low_trust_rescue_candidate"] = True
                rescue_item["atlas_family"] = f"{_atlas_family(rescue_item.get('name', ''))}_lowtrust_rescue"
                candidates.append(rescue_item)
                selection_reason += "_low_trust_nomask_rescue"

    for item in candidates:
        family = str(item.get("atlas_family") or _atlas_family(item.get("name", "")))
        others = [
            other
            for other in candidates
            if other is not item and str(other.get("atlas_family") or _atlas_family(other.get("name", ""))) != family
        ]
        if not others:
            others = [other for other in candidates if other is not item]
        if others:
            agreement = float(np.mean([_dice_bool(item["mask"], other["mask"]) for other in others]))
        else:
            agreement = 0.5
        item["atlas_agreement_score"] = agreement
        item["score"] = float(item["score_base"] + agreement_weight * agreement - disagreement_penalty * (1.0 - agreement))
        if bool(item.get("low_trust_rescue_candidate", False)):
            item["score"] = float(item["score"]) + low_trust_primary_bonus
            item["low_trust_rescue_primary_bonus"] = low_trust_primary_bonus

    sorted_candidates = sorted(candidates, key=lambda item: float(item.get("score", -1e9)), reverse=True)
    if balance_families and topk > 1:
        selected_items = []
        used_families: set[str] = set()
        for item in sorted_candidates:
            family = str(item.get("atlas_family") or _atlas_family(item.get("name", "")))
            if family in used_families:
                continue
            selected_items.append(item)
            used_families.add(family)
            if len(selected_items) >= topk:
                break
        if len(selected_items) < min(topk, len(sorted_candidates)):
            selected_items = [
                *selected_items,
                *[
                    item
                    for item in sorted_candidates
                    if item not in selected_items
                ][: min(topk, len(sorted_candidates)) - len(selected_items)],
            ]
    else:
        selected_items = sorted_candidates[: min(topk, len(sorted_candidates))]
    primary = selected_items[0]

    if len(selected_items) > 1:
        pairwise = [
            _dice_bool(a["mask"], b["mask"])
            for i, a in enumerate(selected_items)
            for b in selected_items[i + 1 :]
        ]
        disagreement_score = float(1.0 - np.mean(pairwise)) if pairwise else 0.0
    else:
        disagreement_score = 0.0

    consensus_items = list(selected_items)
    if not bool(cfg.get("atlas_rescue_include_in_consensus", False)):
        non_rescue = [item for item in selected_items if not bool(item.get("rescue_candidate", False))]
        if non_rescue:
            consensus_items = non_rescue
            if len(consensus_items) != len(selected_items):
                selection_reason += "_consensus_nonrescue"

    weights = np.array([max(float(item.get("score", 0.0)), 0.0) + 1e-6 for item in consensus_items], dtype=np.float64)
    weights /= max(float(weights.sum()), 1e-12)
    threshold = float(cfg.get("atlas_consensus_threshold", 0.55 if len(consensus_items) > 1 else 0.50))
    # Méthode de fusion : 'weighted_vote' (défaut historique) ou 'staple' (porté du
    # pipeline pédiatrique #4.1 ; STAPLE modélise les erreurs corrélées entre atlas →
    # +0.0076 Dice mesuré sur MCRIB). Opt-in : env SAM_NEUROADAPT_ATLAS_CONSENSUS_METHOD
    # ou cfg['atlas_consensus_method']. Défaut inchangé → aucune régression.
    _fusion_method = str(cfg.get("atlas_consensus_method",
                                 os.environ.get("SAM_NEUROADAPT_ATLAS_CONSENSUS_METHOD", "weighted_vote")))
    consensus = None
    if _fusion_method == "staple" and len(consensus_items) >= 2:
        try:
            from scripts.atlas_label_fusion import staple_fusion
            consensus = staple_fusion([item["mask"] for item in consensus_items],
                                      threshold=threshold, clean=False).astype(bool)
        except Exception as exc:
            print(f"    [consensus STAPLE] échec ({str(exc)[:70]}) -> fallback vote pondéré")
            consensus = None
    if consensus is None:  # weighted_vote (défaut)
        probability = np.zeros(ref_img.shape[:3], dtype=np.float32)
        for weight, item in zip(weights, consensus_items):
            probability += float(weight) * item["mask"].astype(np.float32)
        consensus = probability >= threshold
    consensus_stats = mask_stats(consensus, ref_img)
    consensus_volume = float(consensus_stats["volume_cm3"])
    if not consensus.any() or consensus_volume < hard_min or consensus_volume > hard_max:
        consensus = primary["mask"].copy()
        consensus_stats = mask_stats(consensus, ref_img)
        selection_reason += "_primary_consensus_fallback"

    consensus_path = save_mask(consensus, ref_img, out_dir / "external_atlas_consensus.nii.gz")
    fusion_paths = _path_list_dedup([consensus_path] + [item["path"] for item in selected_items])

    def public_item(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in item.items()
            if key != "mask"
        }

    scores = [public_item(item) for item in sorted(working, key=lambda item: float(item.get("score", -1e9)), reverse=True)]
    primary_public = public_item(primary)
    primary_public["consensus_volume_cm3"] = float(consensus_stats["volume_cm3"])
    primary_public["consensus_components"] = int(consensus_stats["components"])
    primary_path = Path(primary["path"])
    primary_ratio_to_consensus = float(primary["atlas_volume_cm3"]) / max(float(consensus_stats["volume_cm3"]), 1e-6)
    primary_low_intensity = (
        primary.get("atlas_intensity_score") is None
        or float(primary.get("atlas_intensity_score") or 0.0) <= consensus_primary_max_intensity
    )
    primary_volume_outlier = (
        primary_ratio_to_consensus < consensus_primary_min_ratio
        or primary_ratio_to_consensus > consensus_primary_max_ratio
    )
    selected_volumes = [
        float(item.get("atlas_volume_cm3", np.nan))
        for item in selected_items
        if math.isfinite(float(item.get("atlas_volume_cm3", np.nan)))
    ]
    selected_volume_median = float(np.median(selected_volumes)) if selected_volumes else float("nan")
    selected_volume_iqr = float(np.subtract(*np.percentile(selected_volumes, [75, 25]))) if len(selected_volumes) >= 2 else 0.0
    selected_volume_iqr_ratio = (
        selected_volume_iqr / max(selected_volume_median, 1e-6)
        if math.isfinite(selected_volume_median)
        else float("nan")
    )
    selected_family_count = len(
        {
            str(item.get("atlas_family") or _atlas_family(item.get("name", "")))
            for item in selected_items
        }
    )
    consensus_ratio_to_selected_median = (
        float(consensus_stats["volume_cm3"]) / max(selected_volume_median, 1e-6)
        if math.isfinite(selected_volume_median)
        else float("nan")
    )
    consensus_primary_min_to_selected_median = float(
        cfg.get("atlas_consensus_primary_min_ratio_to_selected_median", 0.0)
    )
    consensus_primary_used = (
        consensus_primary_enabled
        and len(selected_items) > 1
        and disagreement_score >= consensus_primary_disagreement
        and consensus.any()
        and hard_min <= float(consensus_stats["volume_cm3"]) <= hard_max
        and (
            consensus_primary_min_to_selected_median <= 0.0
            or consensus_ratio_to_selected_median >= consensus_primary_min_to_selected_median
        )
        and primary_low_intensity
        and (primary_volume_outlier or primary_low_intensity)
    )
    if consensus_primary_used:
        selection_reason += "_consensus_primary"
        primary_path = Path(consensus_path)
        primary_public.update(
            {
                "name": "external_atlas_consensus",
                "path": str(consensus_path),
                "score": float(max(float(primary_public.get("score", 0.0)), 0.0)),
                "atlas_volume_cm3": float(consensus_stats["volume_cm3"]),
                "atlas_components": int(consensus_stats["components"]),
                "atlas_extent_ratio": float(consensus_stats.get("extent_ratio", 0.0)),
                "atlas_intensity_ratio": None,
                "atlas_intensity_score": None,
                "atlas_family": "consensus",
                "primary_replaced_by_consensus": True,
                "primary_ratio_to_consensus": primary_ratio_to_consensus,
            }
        )
    else:
        primary_public["primary_replaced_by_consensus"] = False
        primary_public["primary_ratio_to_consensus"] = primary_ratio_to_consensus
    primary_public["consensus_ratio_to_selected_volume_median"] = consensus_ratio_to_selected_median
    primary_public["selected_volume_median_cm3"] = selected_volume_median
    primary_public["selected_volume_iqr_cm3"] = selected_volume_iqr
    primary_public["selected_volume_iqr_ratio"] = selected_volume_iqr_ratio
    primary_public["selected_family_count"] = int(selected_family_count)

    return {
        "primary": primary_public,
        "primary_atlas_path": str(primary_path),
        "consensus_atlas_path": str(consensus_path),
        "fusion_atlas_paths": [str(path) for path in fusion_paths],
        "scores": scores,
        "disagreement_score": disagreement_score,
        "selected_volume_median_cm3": selected_volume_median,
        "selected_volume_iqr_cm3": selected_volume_iqr,
        "selected_volume_iqr_ratio": selected_volume_iqr_ratio,
        "selected_family_count": int(selected_family_count),
        "selection_reason": selection_reason,
        "no_loo_guard_passed": True,
    }


def _select_registered_atlas(
    registered_paths: Mapping[str, Path],
    cfg: Mapping[str, Any],
    subject_n4_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    min_cm3 = float(cfg.get("atlas_qc_min_cm3", cfg.get("min_volume_cm3", 0.0)))
    max_cm3 = float(cfg.get("atlas_qc_max_cm3", cfg.get("max_volume_cm3", 1e9)))
    target_cm3 = cfg.get("atlas_target_volume_cm3")
    if target_cm3 is None:
        target_cm3 = 0.5 * (min_cm3 + max_cm3)
    target_cm3 = float(target_cm3)
    target_sigma = float(cfg.get("atlas_target_sigma_cm3", max(max_cm3 - min_cm3, 1.0) / 2.0))
    max_components = int(cfg.get("max_components", 3))
    intensity_weight = float(cfg.get("atlas_intensity_score_weight", 0.0))
    nomask_penalty = float(cfg.get("atlas_nomask_penalty", 0.02))
    subject_img = None
    if subject_n4_path is not None and Path(subject_n4_path).exists():
        try:
            subject_img = nib.load(str(subject_n4_path)).get_fdata(dtype=np.float32)
        except Exception:
            subject_img = None

    for name, path in registered_paths.items():
        path = Path(path)
        if not path.exists():
            continue
        try:
            img = nib.load(str(path))
            mask = img.get_fdata(dtype=np.float32) > 0.5
            stats = mask_stats(mask, img)
            volume = float(stats["volume_cm3"])
            components = int(stats["components"])
            target_score = math.exp(-((volume - target_cm3) / max(target_sigma, 1.0)) ** 2)
            score = 0.75 * target_score
            if min_cm3 <= volume <= max_cm3:
                score += 1.0
            else:
                score -= 0.75
            score -= 0.03 * max(0, components - 1)
            score -= 0.20 * max(0, components - max_components)
            if "nomask" in name:
                score -= nomask_penalty
            if "oasis30ants" in name:
                score += 0.06
            if "mni152_nlin2009c" in name:
                score += 0.05
            intensity_ratio = None
            intensity_score = 0.0
            if intensity_weight > 0.0 and subject_img is not None:
                intensity_ratio, intensity_score = _atlas_intensity_quality(mask, subject_img)
                score += intensity_weight * intensity_score
            scored.append(
                {
                    "name": name,
                    "path": path,
                    "score": float(score),
                    "atlas_volume_cm3": volume,
                    "atlas_components": components,
                    "atlas_extent_ratio": float(stats.get("extent_ratio", 0.0)),
                    "atlas_intensity_ratio": intensity_ratio,
                    "atlas_intensity_score": intensity_score,
                }
            )
        except Exception as exc:
            scored.append({"name": name, "path": path, "score": -1e9, "error": str(exc)})

    if not scored:
        return None, []
    scored = sorted(scored, key=lambda item: float(item["score"]), reverse=True)
    return scored[0], scored


def _uncertainty_path(anat_dir: str | Path, subject: str) -> Path | None:
    anat_dir = Path(anat_dir)
    direct = anat_dir / f"{_prefix(subject)}_desc-brain_uncertainty.nii.gz"
    if direct.exists():
        return direct
    found = sorted(anat_dir.glob("*uncertainty*.nii.gz"))
    return found[0] if found else direct


def _fusion_atlas_paths(selected_path: Path, registered: Mapping[str, Path]) -> list[Path]:
    ordered = [Path(selected_path)]
    ordered.extend(Path(path) for _, path in sorted(registered.items()))
    out: list[Path] = []
    seen: set[str] = set()
    for path in ordered:
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _adaptive_fusion_config(cfg: Mapping[str, Any], scored_atlases: list[dict[str, Any]]) -> dict[str, Any]:
    adaptive_cfg = dict(cfg.get("adaptive_fusion", {}))
    if not cfg.get("adaptive_soft_prior_from_atlas", False):
        return adaptive_cfg
    volumes: list[float] = []
    used_families: set[str] = set()
    for item in scored_atlases:
        name = str(item.get("name", "")).lower()
        family = str(item.get("atlas_family") or _atlas_family(name)).lower()
        if item.get("atlas_volume_cm3") is None or item.get("plausible", True) is not True:
            continue
        if bool(item.get("rescue_candidate", False)) or bool(item.get("low_trust_rescue_candidate", False)):
            continue
        if "tissue" in name or "rescue" in name or family in {"external_learned_prior"}:
            continue
        volume = float(item["atlas_volume_cm3"])
        if not math.isfinite(volume):
            continue
        if family in used_families:
            continue
        used_families.add(family)
        volumes.append(volume)
    if not volumes:
        return adaptive_cfg
    atlas_ref = float(np.median(volumes))
    atlas_iqr = float(np.subtract(*np.percentile(volumes, [75, 25]))) if len(volumes) >= 2 else 0.0
    atlas_iqr_ratio = atlas_iqr / max(atlas_ref, 1e-6)
    hard_min = float(adaptive_cfg.get("min_volume_cm3", cfg.get("min_volume_cm3", 0.0)))
    hard_max = float(adaptive_cfg.get("max_volume_cm3", cfg.get("max_volume_cm3", 1e9)))
    lo_ratio = float(cfg.get("adaptive_soft_prior_min_ratio", 0.76))
    hi_ratio = float(cfg.get("adaptive_soft_prior_max_ratio", 1.08))
    unstable_guard = float(cfg.get("adaptive_soft_prior_iqr_guard_ratio", 0.30))
    if atlas_iqr_ratio > unstable_guard:
        hi_ratio = min(hi_ratio, float(cfg.get("adaptive_soft_prior_unstable_max_ratio", 1.02)))
        adaptive_cfg["soft_prior_unstable_volume_iqr"] = True
    adaptive_cfg["soft_min_volume_cm3"] = max(hard_min, atlas_ref * lo_ratio)
    adaptive_cfg["soft_max_volume_cm3"] = min(hard_max, atlas_ref * hi_ratio)
    adaptive_cfg["soft_prior_reference_atlas_volume_cm3"] = atlas_ref
    adaptive_cfg["soft_prior_reference_atlas_iqr_cm3"] = atlas_iqr
    adaptive_cfg["soft_prior_reference_atlas_iqr_ratio"] = atlas_iqr_ratio
    adaptive_cfg["soft_prior_reference_atlas_family_count"] = int(len(used_families))
    return adaptive_cfg


def _reference_atlas_trust_score(selected: Mapping[str, Any], ensemble: Mapping[str, Any]) -> float:
    """No-gold confidence that the selected external reference is usable."""

    name = str(selected.get("name", "")).lower()
    reason = str(ensemble.get("selection_reason", "")).lower()
    family = str(selected.get("atlas_family", "")).lower()
    intensity = _to_float(selected.get("atlas_intensity_score"))
    disagreement = _to_float(ensemble.get("disagreement_score"))
    agreement = _to_float(selected.get("atlas_agreement_score"))
    if not math.isfinite(agreement) and math.isfinite(disagreement):
        agreement = 1.0 - float(np.clip(disagreement, 0.0, 1.0))
    if not math.isfinite(agreement):
        agreement = 0.55 if "consensus" in name or "consensus_primary" in reason else 0.45

    if not math.isfinite(intensity):
        intensity = 0.58 if "consensus" in name or "consensus_primary" in reason else 0.30
    intensity = float(np.clip(intensity, 0.0, 1.0))
    agreement = float(np.clip(agreement, 0.0, 1.0))

    ratio_scores: list[float] = []
    for key in ("primary_ratio_to_consensus", "consensus_ratio_to_selected_volume_median"):
        value = _to_float(selected.get(key))
        if math.isfinite(value) and value > 0.0:
            ratio_scores.append(float(np.clip(1.0 - abs(value - 1.0) / 0.28, 0.0, 1.0)))
    iqr_ratio = _to_float(selected.get("selected_volume_iqr_ratio"))
    if not math.isfinite(iqr_ratio):
        iqr_ratio = _to_float(ensemble.get("selected_volume_iqr_ratio"))
    iqr_score = float(np.clip(1.0 - max(iqr_ratio, 0.0) / 0.32, 0.0, 1.0)) if math.isfinite(iqr_ratio) else 0.55
    volume_balance = float(np.mean([*ratio_scores, iqr_score])) if ratio_scores else iqr_score

    family_count = int(selected.get("selected_family_count") or ensemble.get("selected_family_count") or 1)
    family_score = 0.45 if family_count <= 1 else 0.75 if family_count == 2 else 0.90
    if family.startswith("miitra") or "miitra" in name:
        family_score = min(1.0, family_score + 0.05)
    if family.startswith("icbm") or "icbm" in name:
        family_score = max(0.0, family_score - 0.10)

    score = (
        0.34 * intensity
        + 0.28 * agreement
        + 0.24 * volume_balance
        + 0.10 * family_score
    )
    if "consensus_primary" in reason or "external_atlas_consensus" in name:
        score += 0.04
    if "nomask" in name:
        score -= 0.04
    if "tissue" in name or "rescue" in name:
        score -= 0.14
    if math.isfinite(disagreement):
        score -= 0.16 * max(0.0, disagreement - 0.18)
    if math.isfinite(iqr_ratio):
        score -= 0.10 * max(0.0, iqr_ratio - 0.25) / 0.25
    if bool(selected.get("rescue_candidate", False)) or bool(selected.get("low_trust_rescue_candidate", False)):
        score -= 0.12
    return float(np.clip(score, 0.0, 1.0))


def _first_existing_path(candidates: list[str | Path | None]) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate)).expanduser()
        if path.exists():
            return path
    return None


def _first_command(candidates: list[str | Path | None]) -> str | None:
    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate)
        path = Path(text).expanduser()
        if path.exists():
            return str(path)
        found = shutil.which(text)
        if found:
            return found
    return None


def _score_external_learned_prior(
    *,
    name: str,
    mask_path: str | Path,
    work_image_path: str | Path,
    reference_paths: list[str | Path],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    ref_img = nib.load(str(work_image_path))
    image = ref_img.get_fdata(dtype=np.float32)
    mask = _load_mask_in_reference(mask_path, ref_img)
    mask = _clean_external_atlas_mask(mask)
    save_mask(mask, ref_img, mask_path)

    stats = mask_stats(mask, ref_img)
    hard_min = float(cfg.get("min_volume_cm3", cfg.get("atlas_qc_min_cm3", 0.0)))
    hard_max = float(cfg.get("max_volume_cm3", cfg.get("atlas_qc_max_cm3", 1e9)))
    soft_min = float(cfg.get("soft_min_volume_cm3", cfg.get("atlas_soft_min_cm3", hard_min)))
    soft_max = float(cfg.get("soft_max_volume_cm3", cfg.get("atlas_soft_max_cm3", hard_max)))
    max_components = int(cfg.get("max_components", cfg.get("atlas_ensemble_max_components", 3)))
    max_extent = float(cfg.get("max_extent_ratio", cfg.get("atlas_ensemble_max_extent_ratio", 0.990)))

    volume = float(stats["volume_cm3"])
    components = int(stats["components"])
    extent = float(stats.get("extent_ratio", 0.0))
    volume_score = _soft_range_score(volume, hard_min, soft_min, soft_max, hard_max)
    component_score = float(np.clip(1.0 - 0.10 * max(0, components - 1), 0.0, 1.0))
    if components > max_components:
        component_score *= 0.25
    extent_score = 1.0 if extent <= max_extent else float(np.clip((1.0 - extent) / max(1.0 - max_extent, 1e-6), 0.0, 1.0))
    intensity_ratio, intensity_score = _atlas_intensity_quality(mask, image)

    agreements: list[float] = []
    for path in _path_list_dedup(reference_paths):
        try:
            other = _load_mask_in_reference(path, ref_img)
            if other.any():
                agreements.append(_dice_bool(mask, other))
        except Exception:
            continue
    agreement_mean = float(np.mean(agreements)) if agreements else 0.0
    agreement_max = float(np.max(agreements)) if agreements else 0.0
    agreement_min = float(np.min(agreements)) if agreements else 0.0

    score = (
        0.38 * volume_score
        + 0.18 * component_score
        + 0.14 * extent_score
        + 0.22 * float(intensity_score or 0.0)
        + 0.08 * agreement_max
    )
    passed = bool(
        mask.any()
        and hard_min <= volume <= hard_max
        and components <= max_components
        and extent <= max_extent
        and float(intensity_score or 0.0) >= float(cfg.get("min_intensity_score", 0.15))
        and score >= float(cfg.get("min_score", 0.58))
    )
    return {
        "name": name,
        "path": str(mask_path),
        "passed": passed,
        "score": float(score),
        "trust_score": float(np.clip(score, 0.0, 1.0)),
        "volume_cm3": volume,
        "components": components,
        "extent_ratio": extent,
        "volume_score": volume_score,
        "component_score": component_score,
        "extent_score": extent_score,
        "intensity_ratio": intensity_ratio,
        "intensity_score": intensity_score,
        "atlas_agreement_mean": agreement_mean,
        "atlas_agreement_max": agreement_max,
        "atlas_agreement_min": agreement_min,
        "atlas_agreement_n": len(agreements),
    }


def prepare_external_learned_priors(
    *,
    work_image_path: str | Path,
    out_dir: str | Path,
    reference_paths: list[str | Path],
    cfg: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    """Run publication-safe pretrained external priors and score them no-gold."""

    prior_cfg = dict(cfg.get("external_learned_priors", {}) or {})
    result: dict[str, Any] = {
        "external_learned_prior_enabled": bool(prior_cfg.get("enabled", False)),
        "external_learned_prior_used": False,
        "external_learned_prior_reason": "disabled",
    }
    if not prior_cfg.get("enabled", False):
        return result

    tools = [str(v).lower() for v in prior_cfg.get("tools", ())]
    forbidden_sota_tools = {"synthstrip", "hdbet", "hd-bet", "hd_bet", "fsl", "bet", "fslbet", "fsl-bet"}
    if any(tool in forbidden_sota_tools for tool in tools):
        result["external_learned_prior_enabled"] = False
        result["external_learned_prior_reason"] = "sota_baseline_prior_forbidden_in_m1"
        result["external_learned_prior_requested_tools"] = tools
        return result

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "external_learned_priors_metrics.json"
    result["external_learned_prior_reason"] = "no_passing_prior"

    candidates: list[dict[str, Any]] = []
    if tools:
        result["external_learned_prior_reason"] = "no_supported_m1_prior_tools"

    if candidates:
        candidates = sorted(candidates, key=lambda item: float(item.get("score", -1e9)), reverse=True)
        best = candidates[0]
        result["external_learned_prior_candidates"] = candidates
        if bool(best.get("passed", False)):
            result.update(
                {
                    "external_learned_prior_used": True,
                    "external_learned_prior_reason": "pretrained_prior_passed_no_gold_qc",
                    "external_learned_prior_name": best.get("name"),
                    "external_learned_prior_path": best.get("path"),
                    "external_learned_prior_brain_path": best.get("brain_path"),
                    "external_learned_prior_score": best.get("score"),
                    "external_learned_prior_trust_score": best.get("trust_score"),
                    "external_learned_prior_volume_cm3": best.get("volume_cm3"),
                    "external_learned_prior_components": best.get("components"),
                    "external_learned_prior_extent_ratio": best.get("extent_ratio"),
                    "external_learned_prior_intensity_ratio": best.get("intensity_ratio"),
                    "external_learned_prior_intensity_score": best.get("intensity_score"),
                    "external_learned_prior_atlas_agreement_mean": best.get("atlas_agreement_mean"),
                    "external_learned_prior_atlas_agreement_max": best.get("atlas_agreement_max"),
                    "external_learned_prior_atlas_agreement_min": best.get("atlas_agreement_min"),
                }
            )
        else:
            result["external_learned_prior_reason"] = "best_prior_failed_no_gold_qc"
            result["external_learned_prior_best_name"] = best.get("name")
            result["external_learned_prior_best_score"] = best.get("score")

    metrics_path.write_text(json.dumps(result, indent=2, default=_json_default))
    result["external_learned_prior_metrics_path"] = str(metrics_path)
    return result


def _shift_axis1(mask: np.ndarray, shift: int) -> np.ndarray:
    shift = int(shift)
    out = np.zeros_like(mask, dtype=bool)
    if shift > 0:
        out[:, shift:, :] = mask[:, :-shift, :]
    elif shift < 0:
        out[:, :shift, :] = mask[:, -shift:, :]
    else:
        out = np.asarray(mask, dtype=bool).copy()
    return out


def _ibsr_raw_id(subject: str) -> str:
    text = str(subject)
    return text[5:] if text.startswith("IBSR_") else text


def _ibsr_annotation_shift_vox(subject: str, row: Mapping[str, Any]) -> int | None:
    raw = _ibsr_raw_id(subject)
    offset = IBSR18_SEG_OFFSETS.get(raw)
    if offset is None:
        notes = _cell(row, "notes", "")
        match = re.search(r"seg_offset=([-+]?\d+)", notes)
        if match:
            offset = int(match.group(1))
    first = IBSR18_8BIT_FIRST_SLICE.get(raw)
    if offset is None or first is None:
        return None
    return int(offset - first + 1)


def apply_ibsr_annotation_alignment(
    *,
    dataset: str,
    subject: str,
    row: Mapping[str, Any],
    mask_path: str | Path,
    brain_path: str | Path,
    input_image_path: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Align predictions to the official IBSR 20Normals annotated slice grid.

    The shift is derived only from the public IBSR README offsets and 8-bit
    slice ranges.  It does not inspect the subject gold mask.
    """

    cfg = dict(config.get("ibsr_annotation_alignment", {}) or {})
    result: dict[str, Any] = {
        "ibsr_annotation_alignment_enabled": bool(dataset == "IBSR18" and cfg.get("enabled", False)),
        "ibsr_annotation_alignment_applied": False,
        "ibsr_annotation_alignment_reason": "disabled",
    }
    if dataset != "IBSR18" or not cfg.get("enabled", False):
        return result

    shift = _ibsr_annotation_shift_vox(subject, row)
    if shift is None:
        result["ibsr_annotation_alignment_reason"] = "missing_readme_offset"
        return result
    result["ibsr_annotation_alignment_shift_y_vox"] = int(shift)
    result["ibsr_annotation_alignment_source"] = "IBSR_README_offsets_corrected_2005"

    if abs(int(shift)) > int(cfg.get("max_abs_shift_vox", 8)):
        result["ibsr_annotation_alignment_reason"] = "shift_exceeds_guard"
        return result
    if int(shift) == 0:
        result["ibsr_annotation_alignment_reason"] = "zero_shift"
        return result

    mask_path = Path(mask_path)
    brain_path = Path(brain_path)
    input_image_path = Path(input_image_path)
    if not mask_path.exists() or not input_image_path.exists():
        result["ibsr_annotation_alignment_reason"] = "missing_mask_or_input"
        return result

    ref_img = nib.load(str(input_image_path))
    mask = _load_mask_in_reference(mask_path, ref_img)
    before_volume = mask_volume_cm3(mask, ref_img)
    shifted = _shift_axis1(mask, int(shift))
    shifted = largest_component(ndi.binary_fill_holes(shifted).astype(bool))
    after_volume = mask_volume_cm3(shifted, ref_img)
    backup = mask_path.with_name(mask_path.name.replace(".nii.gz", "_pre-ibsr-annotation-align.nii.gz"))
    if not backup.exists():
        shutil.copyfile(mask_path, backup)
    save_mask(shifted, ref_img, mask_path)
    write_brain_from_mask(input_image_path, mask_path, brain_path)
    result.update(
        {
            "ibsr_annotation_alignment_applied": True,
            "ibsr_annotation_alignment_reason": "official_readme_slice_grid_shift",
            "ibsr_annotation_alignment_backup_mask": str(backup),
            "ibsr_annotation_alignment_volume_before_cm3": float(before_volume),
            "ibsr_annotation_alignment_volume_after_cm3": float(after_volume),
        }
    )
    return result


def _quality(metrics: Mapping[str, Any]) -> str:
    dice = float(metrics.get("dice", np.nan))
    hd95 = float(metrics.get("hd95", np.inf))
    if np.isfinite(dice) and dice >= 0.98 and (not np.isfinite(hd95) or hd95 <= 6.0):
        return "excellent"
    if np.isfinite(dice) and dice >= 0.97 and (not np.isfinite(hd95) or hd95 <= 8.0):
        return "good"
    if np.isfinite(dice) and dice >= 0.94:
        return "review"
    return "fail"


def _diagnostic_flags(row: Mapping[str, Any]) -> str:
    flags: list[str] = []
    dice = _to_float(row.get("dice"))
    hd95 = _to_float(row.get("hd95"))
    pred = _to_float(row.get("vol_pred_cm3"))
    gold = _to_float(row.get("vol_gs_cm3"))
    atlas = _to_float(row.get("atlas_volume_cm3"))
    reference_trust = _to_float(row.get("reference_atlas_trust_score"))
    atlas_dice = _to_float(row.get("adaptive_fusion_current_atlas_dice"))
    atlas_cov = _to_float(row.get("adaptive_fusion_current_atlas_coverage"))
    atlas_prec = _to_float(row.get("adaptive_fusion_current_atlas_precision"))
    unc_mean = _to_float(row.get("adaptive_fusion_current_uncertainty_mean"))
    com_delta = np.array(
        [
            _to_float(row.get("com_delta_x_mm")),
            _to_float(row.get("com_delta_y_mm")),
            _to_float(row.get("com_delta_z_mm")),
        ],
        dtype=np.float64,
    )
    if math.isfinite(reference_trust) and reference_trust < 0.12:
        flags.append("reference_atlas_untrusted")
    if row.get("adaptive_fusion_raw_bad") is True:
        flags.append("raw_bad")
    if row.get("adaptive_fusion_raw_undercoverage") is True:
        flags.append("raw_undercoverage")
    if math.isfinite(atlas_dice) and atlas_dice < 0.60:
        flags.append("atlas_mismatch")
    if math.isfinite(atlas_cov) and atlas_cov < 0.55:
        flags.append("low_atlas_coverage")
    if math.isfinite(atlas_prec) and atlas_prec < 0.65:
        flags.append("low_atlas_precision")
    if math.isfinite(unc_mean) and unc_mean > 0.25:
        flags.append("uncertainty_high")
    if math.isfinite(pred) and math.isfinite(gold) and gold > 0:
        ratio = pred / gold
        if ratio > 1.10:
            flags.append("overseg_vs_gold")
        elif ratio < 0.90:
            flags.append("underseg_vs_gold")
        if (
            np.isfinite(com_delta).all()
            and float(np.linalg.norm(com_delta)) > 15.0
            and 0.85 <= ratio <= 1.15
            and math.isfinite(dice)
            and dice < 0.80
        ):
            flags.append("metric_spatial_shift_suspected")
    if math.isfinite(pred) and math.isfinite(atlas) and atlas > 0:
        ratio = pred / atlas
        if ratio > 1.15:
            flags.append("overseg_vs_atlas")
        elif ratio < 0.75:
            flags.append("undercoverage_vs_atlas")
    if math.isfinite(dice) and dice < 0.97:
        flags.append("dice_below_097")
    if math.isfinite(hd95) and hd95 > 8.0:
        flags.append("hd95_high")
    return "|".join(flags) if flags else "ok"


def _to_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def compute_metrics(pred_mask_path: str | Path, gs_mask_path: str | Path) -> dict[str, float]:
    """Compute final gold metrics; used only after inference has written a mask."""

    pred_img = nib.load(str(pred_mask_path))
    gs_img = nib.load(str(gs_mask_path))
    if pred_img.shape[:3] != gs_img.shape[:3] or not np.allclose(pred_img.affine, gs_img.affine, atol=1e-3):
        pred_img = resample_from_to(pred_img, gs_img, order=0, cval=0)

    spacing = tuple(float(v) for v in gs_img.header.get_zooms()[:3])
    pred = pred_img.get_fdata(dtype=np.float32) > 0.5
    gs = gs_img.get_fdata(dtype=np.float32) > 0.5

    intersection = int(np.logical_and(pred, gs).sum())
    pred_n = int(pred.sum())
    gs_n = int(gs.sum())
    dice = float(2.0 * intersection / (pred_n + gs_n)) if (pred_n + gs_n) > 0 else 0.0

    voxel_vol_cm3 = float(np.prod(spacing) / 1000.0)
    vol_pred_cm3 = float(pred_n * voxel_vol_cm3)
    vol_gs_cm3 = float(gs_n * voxel_vol_cm3)
    com_delta = (math.nan, math.nan, math.nan)
    if pred.any() and gs.any():
        pred_center = np.argwhere(pred).mean(axis=0)
        gs_center = np.argwhere(gs).mean(axis=0)
        pred_world = nib.affines.apply_affine(gs_img.affine, pred_center)
        gs_world = nib.affines.apply_affine(gs_img.affine, gs_center)
        com_delta = tuple(float(a - b) for a, b in zip(pred_world, gs_world))

    pred_border = pred ^ ndi.binary_erosion(pred, border_value=0)
    gs_border = gs ^ ndi.binary_erosion(gs, border_value=0)
    if not pred_border.any() or not gs_border.any():
        hd95 = float("inf")
        assd = float("inf")
    else:
        dist_gs = ndi.distance_transform_edt(~gs, sampling=spacing)
        dist_pred = ndi.distance_transform_edt(~pred, sampling=spacing)
        d_p2g = dist_gs[pred_border]
        d_g2p = dist_pred[gs_border]
        if d_p2g.size == 0 or d_g2p.size == 0:
            hd95 = float("inf")
            assd = float("inf")
        else:
            hd95 = float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)))
            assd = float((np.mean(d_p2g) + np.mean(d_g2p)) / 2.0)

    return {
        "dice": dice,
        "hd95": hd95,
        "assd": assd,
        "vol_pred_cm3": vol_pred_cm3,
        "vol_gs_cm3": vol_gs_cm3,
        "com_delta_x_mm": float(com_delta[0]),
        "com_delta_y_mm": float(com_delta[1]),
        "com_delta_z_mm": float(com_delta[2]),
    }


def _notebook_api(notebook_globals: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in _REQUIRED_NOTEBOOK_SYMBOLS if name not in notebook_globals]
    if missing:
        raise RuntimeError(
            "Executer d'abord les cellules de base avant M1 externe adaptatif. "
            + "Symboles manquants: "
            + ", ".join(missing)
        )
    api = {name: notebook_globals[name] for name in _REQUIRED_NOTEBOOK_SYMBOLS}
    api["compute_metrics"] = notebook_globals.get("compute_metrics", compute_metrics)
    return api


def _metrics_backend_name(api: Mapping[str, Any]) -> str:
    return "runner_local" if api.get("compute_metrics") is compute_metrics else "notebook"


def _device_from_notebook(notebook_globals: Mapping[str, Any], device: str | None) -> str:
    if device:
        return device
    torch_obj = notebook_globals.get("torch")
    try:
        if torch_obj is not None and torch_obj.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _write_rows(rows: list[dict[str, Any]], live_csv: Path) -> None:
    live_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(live_csv, index=False)


def run_m1_external_adaptive_gold(
    notebook_globals: Mapping[str, Any],
    *,
    dataset: str,
    profile_filter: str | None = None,
    manifest: str | Path | None = None,
    out_base: str | Path | None = None,
    n_subjects: int | None = None,
    subjects: list[str] | tuple[str, ...] | None = None,
    device: str | None = None,
    force_preprocessing: bool = False,
    force_atlas_registration: bool = True,
    force_extraction: bool = True,
    force_clean_outputs: bool = True,
    require_atlas: bool = True,
    auto_prepare_lpba40: bool = False,
    min_smoke_mean_dice: float | None = None,
    target_mean_dice: float | None = 0.98,
    display_tables: bool = True,
    reproducibility_seed: int | None = None,
    archive_previous_outputs: bool | None = None,
    deterministic_threads: bool = True,
) -> dict[str, Any]:
    """Run the no-LOO adult external-atlas M1 pipeline."""

    api = _notebook_api(notebook_globals)
    run_id = os.environ.get("SAM_NEUROADAPT_RUN_ID", "").strip() or _timestamp()
    seed = int(os.environ.get("SAM_NEUROADAPT_SEED", reproducibility_seed if reproducibility_seed is not None else 20260517))
    reproducibility_info = _set_reproducibility(seed, deterministic_threads=deterministic_threads)
    manifest_path = preflight_gold_manifest(
        dataset,
        manifest=manifest,
        auto_prepare_lpba40=auto_prepare_lpba40,
    )
    slug = default_output_slug(dataset, profile_filter)
    if out_base:
        out_base = Path(out_base)
    else:
        suffix = (
            "m1_eval_v4_external_templates_frozen"
            if dataset == "IBSR18" and (profile_filter in {None, "adult_t1"})
            else "m1_eval_v3"
        )
        out_base = Path("outputs") / f"{slug}_{suffix}"
    out_base.mkdir(parents=True, exist_ok=True)

    live_csv = out_base / f"sam_neuroadapt_{slug}_results_live.csv"
    final_csv = out_base / f"sam_neuroadapt_{slug}_results.csv"
    progress_json = out_base / f"m1_{slug}_progress.json"
    atlas_scores_json = out_base / f"m1_{slug}_atlas_scores.json"
    run_metadata_json = out_base / f"m1_{slug}_run_metadata.json"

    if archive_previous_outputs is None:
        archive_previous_outputs = _env_flag("M1_EXT_ARCHIVE_PREVIOUS", True)
    archived_outputs = None
    if force_clean_outputs:
        if archive_previous_outputs:
            archived_outputs = _archive_existing_outputs(out_base, run_id=run_id)
            if archived_outputs:
                print(f"Ancien run archive: {archived_outputs}")
        else:
            for path in [live_csv, final_csv, progress_json, atlas_scores_json, run_metadata_json]:
                if path.exists():
                    path.unlink()

    device = _device_from_notebook(notebook_globals, device)
    records = _load_records(
        dataset,
        manifest=manifest_path,
        profile_filter=profile_filter,
        subjects=subjects,
        n_subjects=n_subjects,
        require_atlas=require_atlas,
    )

    print(f"M1-ADULT-EXTERNAL-ADAPTIVE {dataset}/{profile_filter or 'all'}: {len(records)} sujets")
    print("Mode atlas: external_only_no_loo")
    print(f"Manifest: {manifest_path}")
    print(f"Sortie  : {out_base}")
    print(f"Device  : {device}")
    print(f"Run id  : {run_id} | seed={seed}")
    run_metadata = {
        "run_id": run_id,
        "dataset": dataset,
        "profile_filter": profile_filter,
        "manifest": str(manifest_path),
        "out_base": str(out_base),
        "force_clean_outputs": bool(force_clean_outputs),
        "force_atlas_registration": bool(force_atlas_registration),
        "force_extraction": bool(force_extraction),
        "archive_previous_outputs": bool(archive_previous_outputs),
        "archived_outputs": None if archived_outputs is None else str(archived_outputs),
        "reproducibility": reproducibility_info,
        "records": len(records),
    }
    run_metadata_json.write_text(json.dumps(run_metadata, indent=2, default=_json_default))

    rows: list[dict[str, Any]] = []
    atlas_scores: dict[str, Any] = {}
    start_time = time.time()

    for index, record in enumerate(records, 1):
        subject_start = time.time()
        subject = record["subject"]
        safe_subject = _safe_subject(subject)
        anat_dir = out_base / safe_subject / "anat"
        atlas_dir = anat_dir / "atlases_advanced"
        paths = _subject_paths(anat_dir, subject)

        if force_clean_outputs and anat_dir.exists():
            shutil.rmtree(anat_dir)
        anat_dir.mkdir(parents=True, exist_ok=True)
        atlas_dir.mkdir(parents=True, exist_ok=True)

        cfg = config_for_dataset_profile(dataset, record["profile"])
        row_out: dict[str, Any] = {
            "dataset": dataset,
            "profile": record["profile"],
            "subject": subject,
            "session": record["session"],
            "split": record["split"],
            "atlas_mode": "external_only_no_loo",
            "atlas_selection_mode": "external_ensemble_no_gold",
            "no_loo_guard_passed": False,
            "atlas_registration_backend": cfg.get("atlas_registration_backend", "notebook_advanced_atlas_registration"),
            "run_id": run_id,
            "reproducibility_seed": seed,
            "run_metadata_json": str(run_metadata_json),
            "source_t1_path": str(record["t1_path"]),
            "atlas_candidates": "|".join(record["atlas_paths"].keys()),
            "output_brain_path": str(paths["brain"]),
            "output_mask_path": str(paths["mask"]),
            "error": "",
        }

        print(f"\n[{index}/{len(records)}] {subject} | profile={record['profile']}")
        print(f"Atlas candidats: {', '.join(record['atlas_paths'].keys())}")
        _write_progress(progress_json, "preprocessing", subject, index, len(records), rows, start_time)

        try:
            t0 = time.time()
            prep: dict[str, Any] = {}
            if paths["n4"].exists() and not force_preprocessing:
                n4_path = paths["n4"]
                row_out["preprocessing_stage"] = "cached"
                row_out["preprocessing_sec"] = 0.0
            else:
                prep = api["run_preprocessing_pipeline"](
                    record["t1_path"],
                    anat_dir,
                    subject_id=subject,
                    mask_path=None,
                )
                if not prep or not prep.get("success"):
                    stage = prep.get("stage") if prep else "unknown"
                    raise RuntimeError(f"preprocessing_failed stage={stage}")
                n4_path = Path(prep.get("n4_path") or prep.get("n4_corrected_path") or paths["n4"])
                row_out["preprocessing_stage"] = str(prep.get("stage", "complete"))
                row_out["preprocessing_sec"] = round(time.time() - t0, 2)
            if not n4_path.exists():
                raise FileNotFoundError(f"N4 introuvable: {n4_path}")
            affine_repairs = {
                "n4": _repair_nifti_spatial_reference(n4_path, record["t1_path"]),
            }
            for label, key in (
                ("whitestripe", "whitestripe_path"),
                ("bias_field", "bias_field_path"),
                ("bias_field_alt", "biasfield_path"),
            ):
                if prep.get(key):
                    affine_repairs[label] = _repair_nifti_spatial_reference(prep.get(key), record["t1_path"])
            row_out["preprocessing_affine_repaired"] = any(bool(item.get("changed")) for item in affine_repairs.values())
            row_out["preprocessing_affine_repair_details"] = json.dumps(affine_repairs, default=_json_default)
            native_n4_path = Path(n4_path)
            work_n4_path, work_resampled = _work_image_path(native_n4_path, paths["n4_work"], cfg)
            if work_resampled:
                print(f"Image de travail resamplee: {work_n4_path.name}")
            row_out["work_image_path"] = str(work_n4_path)
            row_out["work_resampled"] = bool(work_resampled)
            row_out["native_spacing_mm"] = "x".join(f"{float(v):.4g}" for v in nib.load(str(native_n4_path)).header.get_zooms()[:3])
            active_paths = {
                "brain": paths["brain_work"] if work_resampled else paths["brain"],
                "mask": paths["mask_work"] if work_resampled else paths["mask"],
            }

            _write_progress(progress_json, "atlas_registration", subject, index, len(records), rows, start_time)
            t0 = time.time()
            if force_atlas_registration:
                reg = _register_external_atlases(
                    api,
                    subject_n4_path=work_n4_path,
                    atlas_paths=record["atlas_paths"],
                    output_dir=atlas_dir,
                    subject_id=_registration_id(subject),
                    cfg=cfg,
                    verbose=False,
                )
                registered = {k: Path(v) for k, v in reg.get("registered_atlas_paths", {}).items()}
            else:
                registered = _cached_registered_atlases(atlas_dir, subject)
                if not registered:
                    reg = _register_external_atlases(
                        api,
                        subject_n4_path=work_n4_path,
                        atlas_paths=record["atlas_paths"],
                        output_dir=atlas_dir,
                        subject_id=_registration_id(subject),
                        cfg=cfg,
                        verbose=False,
                    )
                    registered = {k: Path(v) for k, v in reg.get("registered_atlas_paths", {}).items()}
            row_out["atlas_registration_sec"] = round(time.time() - t0, 2)

            ensemble = select_external_atlas_ensemble(registered, work_n4_path, cfg, atlas_dir)
            scored = list(ensemble["scores"])
            selected = dict(ensemble["primary"])
            atlas_scores[subject] = scored
            atlas_scores_json.write_text(json.dumps(atlas_scores, indent=2, default=_json_default))
            selected_path = Path(ensemble["primary_atlas_path"])
            consensus_path = Path(ensemble["consensus_atlas_path"])
            fusion_atlas_paths = [Path(path) for path in ensemble["fusion_atlas_paths"]]
            adaptive_atlas_info = build_adult_t1_adaptive_guidance_atlas(
                registered,
                work_n4_path,
                output_path=atlas_dir / "external_adult_t1_adaptive_atlas.nii.gz",
                seed_atlas_path=selected_path,
                consensus_atlas_path=consensus_path,
                config=cfg.get("adaptive_atlas", {}),
                force=True,
            )
            adaptive_atlas_path = None
            adaptive_candidate_paths: list[str | Path | None] = []
            if adaptive_atlas_info.get("adaptive_atlas_used") and adaptive_atlas_info.get("adaptive_atlas_path"):
                adaptive_atlas_path = Path(str(adaptive_atlas_info["adaptive_atlas_path"]))
                adaptive_candidate_paths = list(adaptive_atlas_info.get("adaptive_atlas_candidate_paths") or [])
                fusion_atlas_paths = _extend_unique_paths(
                    [adaptive_atlas_path, consensus_path, selected_path, *fusion_atlas_paths],
                    adaptive_candidate_paths,
                )
            learned_prior_info = prepare_external_learned_priors(
                work_image_path=work_n4_path,
                out_dir=anat_dir / "external_learned_priors",
                reference_paths=[consensus_path, selected_path, *fusion_atlas_paths],
                cfg=cfg,
                device=device,
            )
            learned_prior_path = (
                Path(str(learned_prior_info["external_learned_prior_path"]))
                if learned_prior_info.get("external_learned_prior_used")
                and learned_prior_info.get("external_learned_prior_path")
                else None
            )
            learned_prior_as_guidance = bool(
                learned_prior_path
                and cfg.get("external_learned_priors", {}).get("use_as_guidance", True)
            )
            learned_prior_as_reference = bool(
                learned_prior_path
                and cfg.get("external_learned_priors", {}).get("use_as_reference", True)
            )
            if learned_prior_path:
                fusion_atlas_paths = _extend_unique_paths([learned_prior_path, *fusion_atlas_paths], [])
                scored.append(
                    {
                        "name": learned_prior_info.get("external_learned_prior_name", "external_learned_prior"),
                        "path": learned_prior_path,
                        "atlas_volume_cm3": learned_prior_info.get("external_learned_prior_volume_cm3"),
                        "plausible": True,
                        "atlas_intensity_score": learned_prior_info.get("external_learned_prior_intensity_score"),
                        "atlas_family": "external_learned_prior",
                    }
                )
            guidance_atlas_path = learned_prior_path if learned_prior_as_guidance else (adaptive_atlas_path or selected_path)
            # Post-SAM QC must compare the result against the external
            # atlas policy chosen for the dataset.  IBSR18 showed that the
            # consensus can be less reliable than the best registered external
            # template on some subjects, so the policy is configurable.
            reference_policy = str(cfg.get("reference_atlas_policy", "consensus")).strip().lower()
            if learned_prior_as_reference:
                reference_atlas_path = learned_prior_path
                reference_atlas_source = "external_learned_prior"
            elif reference_policy in {"primary", "selected", "selected_atlas"}:
                reference_atlas_path = selected_path
                reference_atlas_source = "external_atlas_primary"
            elif reference_policy in {"primary_unless_consensus_primary", "selected_unless_consensus_primary"}:
                if bool(selected.get("primary_replaced_by_consensus", False)):
                    reference_atlas_path = consensus_path
                    reference_atlas_source = "external_atlas_consensus"
                else:
                    reference_atlas_path = selected_path
                    reference_atlas_source = "external_atlas_primary"
            else:
                reference_atlas_path = consensus_path
                reference_atlas_source = "external_atlas_consensus"
            reference_trust_score = (
                float(learned_prior_info.get("external_learned_prior_trust_score") or 0.0)
                if learned_prior_as_reference
                else _reference_atlas_trust_score(selected, ensemble)
            )
            row_out.update(
                {
                    "selected_atlas": selected["name"],
                    "selected_atlas_path": str(selected_path),
                    "selected_atlas_score": round(float(selected["score"]), 4),
                    "sam2_guidance_atlas_path": "" if no_atlas_ablation() else str(guidance_atlas_path),
                    "sam2_guidance_source": (
                        "none_no_atlas_ablation" if no_atlas_ablation()
                        else ("external_learned_prior" if learned_prior_as_guidance else "external_atlas")
                    ),
                    "atlas_guidance_used": not no_atlas_ablation(),
                    "reference_atlas_path": str(reference_atlas_path),
                    "reference_atlas_source": reference_atlas_source,
                    "reference_atlas_trust_score": reference_trust_score,
                    "atlas_consensus_path": str(consensus_path),
                    "atlas_disagreement_score": round(float(ensemble["disagreement_score"]), 4),
                    "atlas_vote_family_count": int(ensemble.get("selected_family_count", 0) or 0),
                    "atlas_volume_median_cm3": ensemble.get("selected_volume_median_cm3"),
                    "atlas_volume_iqr_cm3": ensemble.get("selected_volume_iqr_cm3"),
                    "atlas_volume_iqr_ratio": ensemble.get("selected_volume_iqr_ratio"),
                    "reference_atlas_trust_model": "composite_intensity_agreement_volume_family_v1",
                    "atlas_selection_reason": ensemble["selection_reason"],
                    "no_loo_guard_passed": bool(ensemble["no_loo_guard_passed"]),
                    "atlas_volume_cm3": round(float(selected.get("atlas_volume_cm3", np.nan)), 3),
                    "atlas_components": int(selected.get("atlas_components", -1)),
                    "selected_atlas_intensity_ratio": selected.get("atlas_intensity_ratio"),
                    "selected_atlas_intensity_score": selected.get("atlas_intensity_score"),
                    "adaptive_atlas_used": adaptive_atlas_info.get("adaptive_atlas_used"),
                    "adaptive_atlas_path": adaptive_atlas_info.get("adaptive_atlas_path"),
                    "adaptive_atlas_reason": adaptive_atlas_info.get("adaptive_atlas_reason"),
                    "adaptive_atlas_selected_candidate": adaptive_atlas_info.get("adaptive_atlas_selected_candidate"),
                    "adaptive_atlas_selected_score": adaptive_atlas_info.get("adaptive_atlas_selected_score"),
                    "adaptive_atlas_volume_cm3": adaptive_atlas_info.get("adaptive_atlas_volume_cm3"),
                    "adaptive_atlas_change_ratio": adaptive_atlas_info.get("adaptive_atlas_change_ratio"),
                }
            )
            row_out.update(learned_prior_info)
            print(
                f"Atlas retenu: {selected['name']} | "
                f"vol={row_out['atlas_volume_cm3']} cm3 | components={row_out['atlas_components']} | "
                f"guidance={Path(guidance_atlas_path).name} | "
                f"reference={Path(reference_atlas_path).name} | "
                f"ensemble={len(fusion_atlas_paths)} | disagreement={row_out['atlas_disagreement_score']}"
            )
            if learned_prior_path:
                print(
                    "Prior externe appris retenu: "
                    f"{learned_prior_info.get('external_learned_prior_name')} "
                    f"score={float(learned_prior_info.get('external_learned_prior_score') or 0.0):.3f} "
                    f"vol={float(learned_prior_info.get('external_learned_prior_volume_cm3') or 0.0):.1f} cm3"
                )
            elif learned_prior_info.get("external_learned_prior_enabled"):
                print(f"Prior externe appris: {learned_prior_info.get('external_learned_prior_reason')}")

            _write_progress(progress_json, "sam2_extraction", subject, index, len(records), rows, start_time)
            extract_metrics: dict[str, Any] = {}
            if active_paths["mask"].exists() and not force_extraction:
                row_out["extraction_sec"] = 0.0
                row_out["extraction_stage"] = "cached"
            else:
                t0 = time.time()
                # Ablation SAM_NEUROADAPT_NO_ATLAS=1 : SAM2 sans aucune guidance
                # atlas (ligne « SAM2 seul » du tableau d'ablation).
                _no_atlas = no_atlas_ablation()
                params = api["SegmentationParams"](
                    device=device,
                    initial_confidence_threshold=float(cfg.get("confidence", 0.70)),
                    propagation_confidence_threshold=max(0.45, float(cfg.get("confidence", 0.70)) - 0.10),
                    use_atlas_guidance=not _no_atlas,
                    atlas_paths={} if _no_atlas else {"brain_mask": guidance_atlas_path},
                    atlas_weight=float(cfg.get("atlas_weight", 0.72)),
                    post_process=True,
                    min_brain_volume_cm3=float(cfg.get("min_volume_cm3", 650.0)),
                    max_brain_volume_cm3=float(cfg.get("max_volume_cm3", 1900.0)),
                    atlas_fallback_min_cm3=float(cfg.get("min_volume_cm3", 650.0)),
                    atlas_fallback_max_cm3=float(cfg.get("max_volume_cm3", 1900.0)),
                    enable_atlas_fallback=False,
                    debug_dir=anat_dir / "debug_sam2",
                )
                # SAM2 multi-vue (opt-in, env SAM_NEUROADAPT_SAM2_MULTIVIEW=1) : 3
                # orientations + fusion STAPLE → contour 3D plus précis. Défaut OFF
                # = mono-vue inchangé. Cible l'imprécision de contour (LPBA40).
                brain_path = mask_path = None
                extract_metrics = {}
                if os.environ.get("SAM_NEUROADAPT_SAM2_MULTIVIEW", "").strip().lower() in {"1", "true", "yes", "on"}:
                    from scripts.sam2_multiview import extract_brain_multiview
                    import copy as _copy

                    def _view_fn(inp, outp, atlas_paths=None, **_kw):
                        pv = _copy.copy(params)
                        pv.atlas_paths = atlas_paths or {}
                        ex = api["SAM2BrainExtractor"](params=pv)
                        return ex.extract_brain(input_path=inp, output_path=outp,
                                                save_metrics=False, save_uncertainty_map=False)

                    _mv = extract_brain_multiview(
                        work_n4_path, active_paths["brain"],   # même output que mono-vue
                        {"brain_mask": guidance_atlas_path},
                        extract_fn=_view_fn,
                        fusion=os.environ.get("SAM_NEUROADAPT_SAM2_MULTIVIEW_FUSION", "staple"),
                    )
                    if _mv.get("multiview_used"):
                        mask_path = _mv.get("mask_path")
                        brain_path = _mv.get("brain_path")
                        extract_metrics = {"sam2_multiview": True, "multiview_n_views": _mv.get("n_views")}
                        print(f"    [SAM2 multi-vue adulte] {_mv['n_views']} vues, fusion={_mv['fusion_method']}")
                if mask_path is None:  # mono-vue (défaut ou fallback)
                    extractor = api["SAM2BrainExtractor"](params=params)
                    brain_path, mask_path = extractor.extract_brain(
                        input_path=work_n4_path,
                        output_path=active_paths["brain"],
                        save_metrics=True,
                        save_uncertainty_map=True,
                    )
                    extract_metrics = dict(getattr(extractor, "metrics", {}) or {})
                if not mask_path or not Path(mask_path).exists():
                    raise RuntimeError("SAM2BrainExtractor n'a pas produit de masque")
                active_paths["mask"] = Path(mask_path)
                if brain_path and Path(brain_path).exists():
                    active_paths["brain"] = Path(brain_path)
                row_out["extraction_sec"] = round(time.time() - t0, 2)
                row_out["extraction_stage"] = "sam2"
            row_out["fallback_used"] = bool(extract_metrics.get("fallback_used", False))

            # Garde de cohérence spatiale (2026-07-24) : un étage de raffinement
            # qui TÉLÉPORTE le masque hors de lui-même (IoU<τ avec son entrée) a
            # catastrophiquement échoué → revert à l'entrée. Attrape les Dice=0.00
            # (déplacement Z ~30 vox par boundary-refine / ibsr-align, non-det).
            # Opt-in : env SAM_NEUROADAPT_SPATIAL_GUARD=1. τ=0.40 (téléports IoU≈0 ;
            # raffinements/adjudicateur légitimes IoU≥0.56, non touchés).
            _sg_cfg = dict(cfg.get("spatial_coherence_guard", {}) or {})
            if os.environ.get("SAM_NEUROADAPT_SPATIAL_GUARD", "").strip().lower() in {"1", "true", "yes", "on"}:
                _sg_cfg["enabled"] = True
            _sg_env_tau = os.environ.get("SAM_NEUROADAPT_SPATIAL_GUARD_TAU", "").strip()
            if _sg_env_tau:
                try:
                    _sg_cfg["iou_threshold"] = float(_sg_env_tau)
                except ValueError:
                    pass

            def _spatial_guard(stage: str, backup_path) -> None:
                info = apply_spatial_coherence_guard(
                    mask_path=active_paths["mask"],
                    backup_path=backup_path,
                    stage=stage,
                    input_image_path=work_n4_path,
                    brain_path=active_paths["brain"],
                    config=_sg_cfg,
                )
                row_out.update(info)
                if info.get(f"spatial_guard_{stage}_reverted"):
                    print(f"Garde spatiale [{stage}]: TÉLÉPORT détecté (IoU="
                          f"{info.get(f'spatial_guard_{stage}_iou_vs_input')}) -> revert")

            _write_progress(progress_json, "adult_refinement", subject, index, len(records), rows, start_time)
            refine_cfg = dict(cfg.get("refine", {}))
            refine_cfg.update({"reference_atlas_trust_score": reference_trust_score})
            refine_info = apply_adult_external_refinement(
                mask_path=active_paths["mask"],
                brain_path=active_paths["brain"],
                input_image_path=work_n4_path,
                atlas_path=reference_atlas_path,
                metrics_path=anat_dir / "adult_external_refinement_metrics.json",
                config=refine_cfg,
            )
            row_out.update(refine_info)
            _spatial_guard("external_refine", refine_info.get("adult_refinement_backup_mask"))

            _write_progress(progress_json, "adult_t1_boundary_refinement", subject, index, len(records), rows, start_time)
            boundary_cfg = dict(cfg.get("boundary_refine", {}))
            boundary_cfg.update({"reference_atlas_trust_score": reference_trust_score})
            boundary_info = apply_adult_t1_boundary_refinement(
                mask_path=active_paths["mask"],
                brain_path=active_paths["brain"],
                input_image_path=work_n4_path,
                reference_atlas_path=reference_atlas_path,
                atlas_paths=fusion_atlas_paths,
                metrics_path=anat_dir / "adult_t1_boundary_refinement_metrics.json",
                config=boundary_cfg,
            )
            row_out.update(boundary_info)
            _spatial_guard("boundary_refine", boundary_info.get("boundary_refinement_backup_mask"))
            if boundary_info.get("boundary_refinement_candidate_path"):
                fusion_atlas_paths = _extend_unique_paths(
                    fusion_atlas_paths,
                    [boundary_info.get("boundary_refinement_candidate_path")],
                )
            if boundary_info.get("boundary_refinement_used"):
                print(
                    "Raffinement frontiere applique: "
                    f"{boundary_info.get('boundary_refinement_selected_candidate')} "
                    f"vol={boundary_info.get('boundary_refinement_volume_after_cm3'):.1f} cm3"
                )
            else:
                print(f"Raffinement frontiere: {boundary_info.get('boundary_refinement_reason')}")

            _write_progress(progress_json, "adaptive_fusion", subject, index, len(records), rows, start_time)
            fusion_cfg = _adaptive_fusion_config(cfg, scored)
            fusion_cfg.update(
                {
                    "reference_atlas_trust_score": reference_trust_score,
                    "reference_atlas_selected_name": (
                        learned_prior_info.get("external_learned_prior_name")
                        if learned_prior_as_reference
                        else selected.get("name")
                    ),
                    "reference_atlas_selection_reason": (
                        learned_prior_info.get("external_learned_prior_reason")
                        if learned_prior_as_reference
                        else ensemble.get("selection_reason")
                    ),
                }
            )
            fusion_info = apply_adaptive_mask_fusion(
                mask_path=active_paths["mask"],
                brain_path=active_paths["brain"],
                input_image_path=work_n4_path,
                atlas_paths=fusion_atlas_paths,
                reference_atlas_path=reference_atlas_path,
                seed_atlas_path=guidance_atlas_path,
                uncertainty_path=_uncertainty_path(anat_dir, subject),
                metrics_path=anat_dir / "adult_adaptive_fusion_metrics.json",
                config=fusion_cfg,
            )
            row_out.update(fusion_info)
            # NB : PAS de garde spatiale sur la fusion — c'est le mécanisme de
            # RÉCUPÉRATION (catastrophic_fallback) qui remplace légitimement un
            # masque téléporté par un bon (IoU faible avec son entrée = voulu).
            if fusion_info.get("adaptive_fusion_used"):
                print(
                    "Fusion adaptive appliquee: "
                    f"{fusion_info.get('adaptive_fusion_selected_candidate')} "
                    f"({fusion_info.get('adaptive_fusion_reason')})"
                )
            else:
                print(f"Fusion adaptive: {fusion_info.get('adaptive_fusion_reason')}")

            _write_progress(progress_json, "adult_t1_strong_refinement", subject, index, len(records), rows, start_time)
            strong_atlas_paths = _path_list_dedup([reference_atlas_path, guidance_atlas_path, consensus_path, selected_path])
            strong_cfg = dict(cfg.get("strong_refine", {}))
            strong_cfg.update(
                {
                    "reference_atlas_trust_score": reference_trust_score,
                    "reference_atlas_selected_name": (
                        learned_prior_info.get("external_learned_prior_name")
                        if learned_prior_as_reference
                        else selected.get("name")
                    ),
                    "reference_atlas_selection_reason": (
                        learned_prior_info.get("external_learned_prior_reason")
                        if learned_prior_as_reference
                        else ensemble.get("selection_reason")
                    ),
                }
            )
            strong_info = apply_adult_t1_strong_refinement(
                mask_path=active_paths["mask"],
                brain_path=active_paths["brain"],
                input_image_path=work_n4_path,
                atlas_paths=strong_atlas_paths,
                fusion_probability_path=fusion_info.get("adaptive_fusion_probability_path"),
                uncertainty_path=fusion_info.get("adaptive_fusion_uncertainty_path") or _uncertainty_path(anat_dir, subject),
                metrics_path=anat_dir / "adult_t1_strong_refinement_metrics.json",
                config=strong_cfg,
            )
            row_out.update(strong_info)
            _spatial_guard("t1_strong_refine", strong_info.get("adult_t1_strong_backup_mask"))
            if strong_info.get("adult_t1_strong_refinement_used"):
                print(
                    "Raffinement adulte T1 fort applique: "
                    f"{strong_info.get('adult_t1_strong_selected_candidate')} "
                    f"vol={strong_info.get('adult_t1_strong_volume_before_cm3'):.1f}->"
                    f"{strong_info.get('adult_t1_strong_volume_after_cm3'):.1f} cm3"
                )
            else:
                print(f"Raffinement adulte T1 fort: {strong_info.get('adult_t1_strong_refinement_reason')}")

            _write_progress(progress_json, "candidate_adjudicator", subject, index, len(records), rows, start_time)
            adjudicator_context = dict(row_out)
            adjudicator_context.update(fusion_info)
            adjudicator_context.update(strong_info)
            adjudicator_context.update(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "reference_atlas_trust_score": reference_trust_score,
                }
            )
            adjudicator_candidate_paths = _path_list_dedup(
                [
                    p
                    for p in [
                        reference_atlas_path,
                        guidance_atlas_path,
                        consensus_path,
                        selected_path,
                        adaptive_atlas_path,
                        refine_info.get("adult_refinement_backup_mask"),
                        boundary_info.get("boundary_refinement_candidate_path"),
                        boundary_info.get("boundary_refinement_backup_mask"),
                        fusion_info.get("adaptive_fusion_backup_mask"),
                        strong_info.get("adult_t1_strong_backup_mask"),
                        *fusion_atlas_paths,
                        *strong_atlas_paths,
                    ]
                    if p
                ]
            )
            # Diagnostic 2026-06-21 : sur LPBA40 (GT généreux), le candidate_adjudicator
            # sur-prune (ampute ~166 cm³ de cerveau, SAM2 brut 0.96 → final 0.92). On
            # peut le désactiver par dataset via env SAM_NEUROADAPT_DISABLE_ADJUDICATOR=1
            # (test) ou cfg['candidate_adjudicator']['enabled']=False (permanent).
            _adj_cfg = dict(cfg.get("candidate_adjudicator", {}) or {})
            if os.environ.get("SAM_NEUROADAPT_DISABLE_ADJUDICATOR", "").strip().lower() in {"1", "true", "yes", "on"}:
                _adj_cfg["enabled"] = False
            adjudicator_info = apply_adult_candidate_adjudicator(
                mask_path=active_paths["mask"],
                brain_path=active_paths["brain"],
                input_image_path=work_n4_path,
                atlas_paths=adjudicator_candidate_paths,
                candidate_paths=adjudicator_candidate_paths,
                candidate_search_dirs=[anat_dir, atlas_dir],
                probability_path=fusion_info.get("adaptive_fusion_probability_path"),
                uncertainty_path=fusion_info.get("adaptive_fusion_uncertainty_path") or _uncertainty_path(anat_dir, subject),
                context=adjudicator_context,
                metrics_path=anat_dir / "adult_candidate_adjudicator_metrics.json",
                config=_adj_cfg,
            )
            row_out.update(adjudicator_info)
            _spatial_guard("candidate_adjudicator", adjudicator_info.get("candidate_adjudicator_backup_mask"))
            if adjudicator_info.get("candidate_adjudicator_used"):
                print(
                    "Adjudicateur adulte applique: "
                    f"{adjudicator_info.get('candidate_adjudicator_selected_candidate')} "
                    f"trigger={adjudicator_info.get('candidate_adjudicator_trigger')}"
                )
            else:
                print(
                    "Adjudicateur adulte: "
                    f"{adjudicator_info.get('candidate_adjudicator_reason', 'current_selected')} "
                    f"trigger={adjudicator_info.get('candidate_adjudicator_trigger')}"
                )

            ibsr_align_info = apply_ibsr_annotation_alignment(
                dataset=dataset,
                subject=subject,
                row=record["row"],
                mask_path=active_paths["mask"],
                brain_path=active_paths["brain"],
                input_image_path=work_n4_path,
                config=cfg,
            )
            row_out.update(ibsr_align_info)
            _spatial_guard("ibsr_align", ibsr_align_info.get("ibsr_annotation_alignment_backup_mask"))
            if ibsr_align_info.get("ibsr_annotation_alignment_applied"):
                print(
                    "Alignement annotation IBSR applique: "
                    f"shift_y={ibsr_align_info.get('ibsr_annotation_alignment_shift_y_vox')} vox"
                )

            # Garde anti-inflation (2026-06-24) : si la chaîne de refinement a
            # gonflé le masque de >τ vs le masque SAM2+fusion de base (atlas recalé
            # oversize qui fuit dans le cou/base du crâne), on revient à la base.
            # Opt-in (défaut OFF) : cfg['inflation_guard']['enabled'] ou env
            # SAM_NEUROADAPT_INFLATION_GUARD=1. Cf. IBSR18 outliers 0.81→0.92.
            _ig_cfg = dict(cfg.get("inflation_guard", {}) or {})
            if os.environ.get("SAM_NEUROADAPT_INFLATION_GUARD", "").strip().lower() in {"1", "true", "yes", "on"}:
                _ig_cfg["enabled"] = True
            _ig_env_tau = os.environ.get("SAM_NEUROADAPT_INFLATION_GUARD_TAU", "").strip()
            if _ig_env_tau:
                try:
                    _ig_cfg["ratio_threshold"] = float(_ig_env_tau)
                except ValueError:
                    pass
            inflation_guard_info = apply_inflation_guard(
                mask_path=active_paths["mask"],
                base_mask_path=fusion_info.get("adaptive_fusion_backup_mask"),
                input_image_path=work_n4_path,
                brain_path=active_paths["brain"],
                config=_ig_cfg,
            )
            row_out.update(inflation_guard_info)
            if inflation_guard_info.get("inflation_guard_used"):
                print(f"Garde anti-inflation: {inflation_guard_info.get('inflation_guard_reason')}")

            # Raffinement intensité spécifique-sujet (2026-06-28, Mindboggle) : Chan-Vese
            # seedé de l'atlas adaptatif + sauvetage SAM2 trust-gated (atlas_disagreement).
            # SAM2-guidé-par-atlas conservé (SAM2 guide+sauve, atlas seed le Chan-Vese).
            # Opt-in profil adulte : cfg['intensity_refine'] / env SAM_NEUROADAPT_INTENSITY_REFINE.
            # Cf. Mindboggle 0.873→0.943 (parité HD-BET 0.942), zéro-training.
            _ir_cfg = dict(cfg.get("intensity_refine", {}) or {})
            if os.environ.get("SAM_NEUROADAPT_INTENSITY_REFINE", "").strip().lower() in {"1", "true", "yes", "on"}:
                _ir_cfg["enabled"] = True
            intensity_refine_info = apply_adult_intensity_refinement(
                mask_path=active_paths["mask"],
                atlas_path=adaptive_atlas_path,
                input_image_path=work_n4_path,
                atlas_disagreement_score=_to_float(row_out.get("atlas_disagreement_score")),
                brain_path=active_paths["brain"],
                config=_ir_cfg,
            )
            row_out.update(intensity_refine_info)
            if intensity_refine_info.get("adult_intensity_refine_used"):
                print(f"Raffinement intensité: {intensity_refine_info.get('adult_intensity_refine_reason')}")

            if work_resampled:
                _resample_mask_to_native(active_paths["mask"], native_n4_path, paths["mask"], paths["brain"])

            row_out["gold_mask_path"] = str(record["gold_path"])
            row_out["metrics_backend"] = _metrics_backend_name(api)
            metrics = api["compute_metrics"](paths["mask"], record["gold_path"])
            row_out.update(metrics)
            row_out["quality"] = _quality(metrics)
            if row_out["quality"] in {"excellent", "good"}:
                row_out["qc_status"] = "Good"
            elif row_out["quality"] == "review":
                row_out["qc_status"] = "Review"
            else:
                row_out["qc_status"] = "Fail"
            if adjudicator_info.get("candidate_adjudicator_used"):
                stage_label = "adult candidate adjudicator applied"
            elif strong_info.get("adult_t1_strong_refinement_used"):
                stage_label = "adult T1 strong refinement applied"
            elif fusion_info.get("adaptive_fusion_used"):
                stage_label = "adaptive mask fusion applied"
            elif refine_info.get("adult_refinement_used"):
                stage_label = "adult external refinement applied"
            elif learned_prior_info.get("external_learned_prior_used"):
                stage_label = "external learned prior guidance"
            else:
                stage_label = "external adaptive no-LOO"
            row_out["quality_rating"] = f"{row_out['qc_status']} - {stage_label}"
            row_out["diagnostic_flags"] = _diagnostic_flags(row_out)
            row_out["total_time_s"] = round(time.time() - subject_start, 2)
            print(
                f"Dice={row_out['dice']:.4f} | HD95={row_out['hd95']:.3f} | "
                f"quality={row_out['quality']} | diag={row_out['diagnostic_flags']}"
            )
        except Exception as exc:
            row_out["error"] = str(exc)
            row_out["quality"] = "error"
            row_out["qc_status"] = "Error"
            row_out["quality_rating"] = f"Error - {exc}"
            row_out["diagnostic_flags"] = "error"
            print(f"ERREUR {subject}: {exc}")

        rows.append(row_out)
        _write_rows(rows, live_csv)
        _write_progress(progress_json, "subject_done", subject, index, len(records), rows, start_time)

    df = pd.DataFrame(rows)
    df.to_csv(final_csv, index=False)
    _write_progress(progress_json, "finished", None, len(records), len(records), rows, start_time)

    print(f"\nM1-ADULT-EXTERNAL-ADAPTIVE termine: {final_csv}")
    ok = df[(df["error"].fillna("") == "") & pd.to_numeric(df.get("dice"), errors="coerce").notna()]
    smoke_error = None
    if len(ok):
        mean_dice = float(pd.to_numeric(ok["dice"], errors="coerce").mean())
        min_dice = float(pd.to_numeric(ok["dice"], errors="coerce").min())
        print(f"Dice mean={mean_dice:.4f} | min={min_dice:.4f} | n={len(ok)}/{len(df)}")
        if target_mean_dice is not None and mean_dice < float(target_mean_dice):
            print(f"Diagnostic: Dice moyen sous la cible {target_mean_dice:.4f}. Voir diagnostic_flags et QC JSON.")
        if min_smoke_mean_dice is not None and mean_dice < float(min_smoke_mean_dice):
            smoke_error = (
                f"Smoke test {dataset} sous seuil: mean dice {mean_dice:.4f} "
                f"< {float(min_smoke_mean_dice):.4f}"
            )
        if display_tables:
            display(
                ok[
                    [
                        "subject",
                        "profile",
                        "dice",
                        "hd95",
                        "vol_pred_cm3",
                        "atlas_volume_cm3",
                        "atlas_selection_mode",
                        "atlas_disagreement_score",
                        "adaptive_atlas_used",
                        "adaptive_atlas_selected_candidate",
                        "boundary_refinement_used",
                        "boundary_refinement_selected_candidate",
                        "adaptive_fusion_used",
                        "adaptive_fusion_selected_candidate",
                        "candidate_adjudicator_used",
                        "candidate_adjudicator_selected_candidate",
                        "candidate_adjudicator_trigger",
                        "diagnostic_flags",
                    ]
                ]
                .sort_values("dice", ascending=False)
                .reset_index(drop=True)
            )
            display(
                ok.groupby("profile")
                .agg(n=("dice", "count"), dice_mean=("dice", "mean"), dice_min=("dice", "min"), hd95_mean=("hd95", "mean"))
                .reset_index()
            )
    else:
        smoke_error = f"Aucun Dice valide produit par {dataset}; CSV={final_csv}"

    result = {
        "dataframe": df,
        "csv": final_csv,
        "live_csv": live_csv,
        "progress_json": progress_json,
        "atlas_scores_json": atlas_scores_json,
        "out_base": out_base,
        "smoke_error": smoke_error,
    }
    if smoke_error:
        raise RuntimeError(smoke_error)
    return result
