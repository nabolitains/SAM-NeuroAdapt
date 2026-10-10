"""Pipeline M1 iSeg-style portable - factorisation de la cellule 39 du notebook.

Ce module porte la logique de la cellule M1-iSeg-2017 (qui atteint 0.9845 mean
Dice sur iSeg-2017 grâce à l'atlas UNC-BCP 06M et un pipeline de refinements
calibres). Il permet de l'appliquer a d'autres datasets pediatriques (BOBs,
MCRIB, MCRIB_WM) avec exactement le meme code-path et les memes seuils
refinement, en parametrant uniquement les priors de volume par dataset et le
resolver d'atlas.

Point d'entree: `run_iseg_style_pipeline(notebook_globals, spec, ...)`.

Aucune dependance LOO. Tous les atlases proviennent de sources externes
(UNC-BCP, UNC Infant Database, MNI-Infant) deja preparees sous
`dataset/atlas_library/`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage as ndi

from scripts.iseg2017_robust_refinement import (
    apply_pediatric_t2_refinement,
    robust_atlas_image_alignment_score,
)
from scripts.adult_external_refinement import apply_adult_external_refinement
from scripts.iseg2017_adaptive_atlas import build_iseg_adaptive_guidance_atlas
from scripts.adaptive_mask_fusion import apply_adaptive_mask_fusion
from scripts.external_atlas_config import month_from_session, no_atlas_ablation as _no_atlas_ablation

warnings.filterwarnings("ignore")


# ============================================================
# Dataclasses
# ============================================================

@dataclass(frozen=True)
class IsegStyleVolumePriors:
    """Per-dataset priors and dataset-specific thresholds.

    Defaults reproduce the iSeg-2017 constants exactly (cell 39 lines 56-164).
    Subclassed for BOBs / MCRIB / MCRIB_WM with volume scaling per cohort age.
    """
    # Mask QC thresholds
    mask_min_cm3: float = 550.0
    mask_max_cm3: float = 1100.0
    mask_max_components: int = 4
    mask_max_extent: float = 0.96
    # Atlas QC thresholds
    atlas_qc_min_cm3: float = 550.0
    atlas_qc_max_cm3: float = 1100.0
    atlas_max_components: int = 1
    atlas_nomask_bonus: float = 0.03
    min_atlas_image_score: float = 0.05
    min_atlas_selection_score: float = 0.25
    # SAM2 brain volume bounds
    min_brain_volume_cm3: float = 550.0
    max_brain_volume_cm3: float = 1100.0
    # SAM2 contrainte atlas per-slice (eap_tight) — pédiatrique: 0.12, adulte: 0.40
    sam2_eap_tight_ratio_threshold: float = 0.40
    # IDÉE 1 — masquage hybride du volume d'entrée SAM2 (hard + transition gaussienne)
    sam2_enable_atlas_input_masking: bool = False
    sam2_atlas_mask_hard_radius_mm: float = 12.0
    sam2_atlas_mask_soft_radius_mm: float = 3.0
    sam2_atlas_mask_min_volume_cm3: float = 200.0
    sam2_atlas_mask_background_value: float = 0.0
    # IDÉE 2 — contrainte eap_tight inconditionnelle sur slice initiale
    sam2_eap_tight_always_initial: bool = False
    sam2_eap_tight_min_preservation: float = 0.20
    # IDÉE 4 — érosion atlas adaptative (atlas trop large vs âge)
    sam2_adaptive_atlas_shrink_enabled: bool = False
    sam2_adaptive_atlas_shrink_target_ratio: float = 1.15
    sam2_adaptive_atlas_shrink_trigger_ratio: float = 1.40
    sam2_adaptive_atlas_shrink_max_iters: int = 10
    # IDÉE 3 — atlas comme mask_input prompt SAM2 (vague 2)
    sam2_enable_atlas_mask_input_prompt: bool = False
    sam2_mask_input_strength: float = 10.0
    sam2_mask_input_use_eap: bool = True
    # Vague 2 — early stopping volume-driven
    sam2_enable_volume_early_stopping: bool = False
    sam2_volume_early_stop_ratio: float = 1.20
    # Vague 3 — bidirectional consensus
    sam2_enable_bidirectional_consensus: bool = False
    sam2_bidirectional_consensus_min_dice: float = 0.80
    sam2_bidirectional_consensus_min_range: int = 6
    # Atlas fallback (off in iSeg)
    atlas_fallback_min_cm3: float = 550.0
    atlas_fallback_max_cm3: float = 1100.0
    atlas_min_coverage: float = 0.85
    atlas_min_precision: float = 0.90
    enable_atlas_fallback: bool = False
    # Atlas refinement
    enable_atlas_refinement: bool = True
    refinement_atlas_dilation: int = 1
    refinement_min_keep_ratio: float = 0.85
    refinement_min_atlas_coverage: float = 0.85
    refinement_min_precision_gain: float = 0.01
    # Boundary refinement
    enable_boundary_refinement: bool = True
    boundary_refinement_width: int = 1
    boundary_atlas_dilation: int = 1
    boundary_intensity_percentile: float = 8.0
    boundary_max_added_cm3: float = 35.0
    boundary_max_volume_ratio: float = 1.06
    boundary_min_atlas_precision: float = 0.92
    # Pediatric T1/T2 refinement
    enable_pediatric_t2_refinement: bool = True
    pediatric_t2_shell_width: int = 20
    pediatric_t2_atlas_dilation: int = 15
    pediatric_t2_t1_percentile: float = 3.0
    pediatric_t2_t2_percentile: float = 3.0
    pediatric_t2_max_added_cm3: float = 500.0
    pediatric_t2_min_atlas_precision: float = 0.75
    # Wide refinement (post-hoc adult-style)
    enable_wide_refinement: bool = True
    wide_refinement_config: dict = field(default_factory=lambda: {
        "enabled": True,
        "min_volume_cm3": 550.0,
        "max_volume_cm3": 1100.0,
        "prune_enabled": True,
        "prune_atlas_dilation": 2,
        "prune_if_atlas_precision_below": 0.78,
        "prune_if_volume_ratio_above": 1.22,
        "prune_min_keep_ratio": 0.82,
        "recover_enabled": True,
        "recover_shell_width": 4,
        "recover_atlas_dilation": 4,
        "recover_intensity_percentile": 3.0,
        "recover_max_added_cm3": 160.0,
        "recover_max_volume_ratio": 1.16,
        "recover_min_atlas_precision": 0.74,
        "min_atlas_dice_gain": 0.001,
        "backup_suffix": "_pre-iseg-wide-refine",
    })
    # Adaptive guidance atlas
    enable_adaptive_guidance_atlas: bool = True
    adaptive_guidance_config: dict = field(default_factory=lambda: {
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
    })
    # Adaptive mask fusion
    enable_adaptive_mask_fusion: bool = True
    adaptive_mask_fusion_config: dict = field(default_factory=lambda: {
        "enabled": True,
        "min_volume_cm3": 550.0,
        "soft_min_volume_cm3": 720.0,
        "soft_max_volume_cm3": 900.0,
        "max_volume_cm3": 1100.0,
        "tissue_percentiles": [8.0, 10.0, 12.0, 15.0, 18.0],
        "use_t2_gate": False,
        "coverage_target": 0.84,
        "coverage_sigma": 0.13,
        "catastrophic_min_precision": 0.50,
        "catastrophic_min_coverage": 0.45,
        "catastrophic_min_atlas_dice": 0.55,
        "catastrophic_max_uncertainty_mean": 0.25,
        "undercoverage_min_precision": 0.96,
        "undercoverage_max_coverage": 0.82,
        "undercoverage_max_volume_ratio": 0.86,
        "current_mask_bonus": 0.08,
        "tissue_undercoverage_bonus": 0.07,
        "selection_margin": 0.015,
        "backup_suffix": "_pre-adaptive-fusion",
    })
    # Atlas weight for SAM2 guidance
    atlas_weight: float = 0.75
    # Spacing override (None for native, else (sx, sy, sz))
    spacing_override: tuple | None = None


@dataclass(frozen=True)
class IsegStyleDatasetSpec:
    """Specification for running the iSeg-style pipeline on a dataset."""
    dataset: str
    manifest_csv: Path
    out_base: Path
    selected_records_fn: Callable[[], list[dict]]
    atlas_resolver: Callable[[Mapping[str, Any]], Mapping[str, Path]]
    volume_priors: IsegStyleVolumePriors


# ============================================================
# Pre-tuned priors per dataset
# ============================================================

ISEG_DEFAULT_PRIORS = IsegStyleVolumePriors(sam2_eap_tight_ratio_threshold=0.12, enable_atlas_fallback=True)  # iSeg defaults + pediatric SAM2 fix

BOBS_DEFAULT_PRIORS = IsegStyleVolumePriors(
    # Aligned with m1_pediatric_iseg_style_runner.py _base_config_for_record for month < 8.
    mask_min_cm3=300.0, mask_max_cm3=1250.0,
    mask_max_extent=0.97,
    atlas_qc_min_cm3=240.0, atlas_qc_max_cm3=1300.0,
    atlas_max_components=3,
    # Relaxed atlas acceptance — MNIInfant coverage for young infants is structurally
    # lower than UNC-BCP-06M on 6-month iSeg brains; using iSeg defaults (0.85) blocks
    # all refinements for BOBs subjects where atlas_coverage lands at ~0.80-0.83.
    atlas_min_coverage=0.80,
    atlas_min_precision=0.88,
    min_atlas_image_score=0.04,
    min_atlas_selection_score=0.20,
    min_brain_volume_cm3=300.0, max_brain_volume_cm3=1250.0,
    sam2_eap_tight_ratio_threshold=0.12,
    # IDÉE 1/2/4 — vague 1 activée pour BOBs (sujets pédiatriques difficiles)
    sam2_enable_atlas_input_masking=True,
    sam2_atlas_mask_hard_radius_mm=12.0,
    sam2_atlas_mask_soft_radius_mm=3.0,
    sam2_atlas_mask_min_volume_cm3=200.0,
    sam2_eap_tight_always_initial=True,
    sam2_eap_tight_min_preservation=0.20,
    sam2_adaptive_atlas_shrink_enabled=True,
    sam2_adaptive_atlas_shrink_target_ratio=1.15,
    sam2_adaptive_atlas_shrink_trigger_ratio=1.40,
    # Vague 2 — IDÉE 3 (mask_input atlas) + early stopping volume-driven
    sam2_enable_atlas_mask_input_prompt=True,
    sam2_mask_input_strength=10.0,
    sam2_mask_input_use_eap=True,
    sam2_enable_volume_early_stopping=True,
    sam2_volume_early_stop_ratio=1.20,
    # Vague 3 — bidirectional consensus : DÉSACTIVÉ par défaut.
    # Tests sur BOBs ont montré une régression (sub-402408: -0.12 Dice, sub-116056: -0.03)
    # à cause de la pollution de l'état SAM2 predictor par la 2e passe.
    # Conservé comme feature opt-in pour debug/recherche future.
    sam2_enable_bidirectional_consensus=False,
    sam2_bidirectional_consensus_min_dice=0.80,
    sam2_bidirectional_consensus_min_range=6,
    atlas_fallback_min_cm3=300.0, atlas_fallback_max_cm3=1250.0,
    enable_atlas_fallback=True,
    # Refinement thresholds relaxed to match young_infant tier in the runner.
    refinement_min_keep_ratio=0.81,
    refinement_min_atlas_coverage=0.75,
    refinement_min_precision_gain=0.006,
    boundary_min_atlas_precision=0.88,
    atlas_nomask_bonus=0.01,
    pediatric_t2_max_added_cm3=260.0,
    boundary_max_added_cm3=35.0,
    wide_refinement_config={
        "enabled": True,
        "min_volume_cm3": 300.0, "max_volume_cm3": 1250.0,
        "prune_enabled": True, "prune_atlas_dilation": 2,
        "prune_if_atlas_precision_below": 0.78,
        "prune_if_volume_ratio_above": 1.22,
        "prune_min_keep_ratio": 0.82,
        "recover_enabled": True, "recover_shell_width": 4,
        "recover_atlas_dilation": 4, "recover_intensity_percentile": 3.0,
        "recover_max_added_cm3": 160.0, "recover_max_volume_ratio": 1.16,
        "recover_min_atlas_precision": 0.74, "min_atlas_dice_gain": 0.001,
        "backup_suffix": "_pre-pediatric-wide-refine",
    },
    adaptive_guidance_config={
        "min_volume_cm3": 240.0, "max_volume_cm3": 1300.0,
        "seed_prefer": "aff", "shell_width": 24, "envelope_dilation": 3,
        "t1_percentile": 3.0, "t2_percentile": 3.0,
        "target_from_envelope_ratio": 0.90, "target_seed_gain_ratio": 1.12,
        "target_max_cm3": 1050.0, "max_added_cm3": 260.0, "min_added_cm3": 5.0,
    },
    adaptive_mask_fusion_config={
        "enabled": True,
        "min_volume_cm3": 300.0,
        "soft_min_volume_cm3": 520.0, "soft_max_volume_cm3": 980.0,
        "max_volume_cm3": 1250.0,
        "tissue_percentiles": [6.0, 8.0, 10.0, 12.0, 15.0, 18.0],
        "use_t2_gate": False,
        "coverage_target": 0.84, "coverage_sigma": 0.13,
        "catastrophic_min_precision": 0.48, "catastrophic_min_coverage": 0.45,
        "catastrophic_min_atlas_dice": 0.55,
        "catastrophic_max_uncertainty_mean": 0.28,
        "undercoverage_min_precision": 0.96, "undercoverage_max_coverage": 0.82,
        "undercoverage_max_volume_ratio": 0.86,
        "current_mask_bonus": 0.08, "tissue_undercoverage_bonus": 0.07,
        "selection_margin": 0.015,
        "backup_suffix": "_pre-adaptive-fusion",
    },
)

MCRIB_DEFAULT_PRIORS = IsegStyleVolumePriors(
    # Aligned with m1_pediatric_iseg_style_runner.py neonatal config (month <= 2).
    mask_min_cm3=220.0, mask_max_cm3=850.0,
    mask_max_extent=0.97,
    atlas_qc_min_cm3=180.0, atlas_qc_max_cm3=900.0,
    atlas_max_components=3,
    atlas_min_coverage=0.78,
    atlas_min_precision=0.88,
    min_atlas_image_score=0.03,
    min_atlas_selection_score=0.18,
    min_brain_volume_cm3=220.0, max_brain_volume_cm3=850.0,
    sam2_eap_tight_ratio_threshold=0.12,
    atlas_fallback_min_cm3=220.0, atlas_fallback_max_cm3=850.0,
    enable_atlas_fallback=True,
    refinement_min_keep_ratio=0.78,
    refinement_min_atlas_coverage=0.70,
    refinement_min_precision_gain=0.004,
    boundary_intensity_percentile=6.0,
    boundary_max_added_cm3=25.0,
    boundary_max_volume_ratio=1.08,
    boundary_min_atlas_precision=0.86,
    pediatric_t2_shell_width=16,
    pediatric_t2_atlas_dilation=12,
    pediatric_t2_max_added_cm3=180.0,
    pediatric_t2_min_atlas_precision=0.70,
    wide_refinement_config={
        "enabled": True,
        "min_volume_cm3": 220.0, "max_volume_cm3": 850.0,
        "prune_enabled": True, "prune_atlas_dilation": 2,
        "prune_if_atlas_precision_below": 0.74,
        "prune_if_volume_ratio_above": 1.18,
        "prune_min_keep_ratio": 0.76,
        "recover_enabled": True, "recover_shell_width": 4,
        "recover_atlas_dilation": 4, "recover_intensity_percentile": 3.0,
        "recover_max_added_cm3": 80.0, "recover_max_volume_ratio": 1.15,
        "recover_min_atlas_precision": 0.70, "min_atlas_dice_gain": 0.001,
        "backup_suffix": "_pre-pediatric-wide-refine",
    },
    adaptive_guidance_config={
        "min_volume_cm3": 180.0, "max_volume_cm3": 900.0,
        "seed_prefer": "aff", "shell_width": 18, "envelope_dilation": 3,
        "t1_percentile": 3.0, "t2_percentile": 3.0,
        "target_from_envelope_ratio": 0.88, "target_seed_gain_ratio": 1.10,
        "target_max_cm3": 650.0, "max_added_cm3": 180.0, "min_added_cm3": 3.0,
    },
    adaptive_mask_fusion_config={
        "enabled": True,
        "min_volume_cm3": 220.0,
        "soft_min_volume_cm3": 340.0, "soft_max_volume_cm3": 620.0,
        "max_volume_cm3": 850.0,
        "tissue_percentiles": [6.0, 8.0, 10.0, 12.0, 15.0, 18.0],
        "use_t2_gate": False,
        "coverage_target": 0.78, "coverage_sigma": 0.15,
        "catastrophic_min_precision": 0.48, "catastrophic_min_coverage": 0.40,
        "catastrophic_min_atlas_dice": 0.48,
        "catastrophic_max_uncertainty_mean": 0.28,
        "undercoverage_min_precision": 0.94, "undercoverage_max_coverage": 0.80,
        "undercoverage_max_volume_ratio": 0.86,
        "current_mask_bonus": 0.08, "tissue_undercoverage_bonus": 0.07,
        "selection_margin": 0.015,
        "backup_suffix": "_pre-adaptive-fusion",
    },
)

MCRIB_WM_DEFAULT_PRIORS = MCRIB_DEFAULT_PRIORS  # identical


# ============================================================
# Atlas resolvers
# ============================================================

ATLAS_BY_AGE_INFANT = [
    # (month_low_inclusive, month_high_exclusive, atlas_dir_name)
    # Validated empirically (benchmark Step 1): unc_bcp_06m best for 0-8mo
    (0, 9, "unc_bcp_06m"),
    (9, 16, "unc_infant_1yr"),
    (16, 99, "unc_infant_2yr"),
]


def fixed_atlas_resolver(t1: Path, brain_mask: Path) -> Callable[[Mapping], dict]:
    """Constant atlas pair for every record (used by iSeg)."""
    t1, brain_mask = Path(t1), Path(brain_mask)

    def _resolve(record):
        return {"t1": t1, "brain_mask": brain_mask}
    return _resolve


def external_atlas_by_age_resolver(
    atlas_root: Path = Path("dataset/atlas_library"),
) -> Callable[[Mapping], dict]:
    """Per-subject atlas mapping by age (no LOO, all external)."""
    atlas_root = Path(atlas_root)

    def _resolve(record):
        session = record.get("session") or ""
        month = month_from_session(session)
        if month is None:
            month = 6
        atlas_name = next(
            (name for lo, hi, name in ATLAS_BY_AGE_INFANT if lo <= month < hi),
            "unc_bcp_06m",
        )
        atlas_dir = atlas_root / atlas_name
        return {
            "t1": atlas_dir / "MNI152_T1_2mm.nii.gz",
            "brain_mask": atlas_dir / "brain_mask.nii.gz",
        }
    return _resolve


def loo_atlas_resolver(
    atlas_root: Path,
    dataset_tag: str,
) -> Callable[[Mapping], dict]:
    """Per-subject consolidated LOO atlas resolver (analogue de UNC_BCP_06M).

    Reads pre-built consolidated atlases from build_iseg_style_atlas.py output:
        atlas_root / sub-XXX / tpl-<TAG><sid>_T1w.nii.gz
        atlas_root / sub-XXX / tpl-<TAG><sid>_desc-brain_mask.nii.gz
        atlas_root / sub-XXX / manifest.json (anti-leakage audit)
        atlas_root / sub-XXX / MNI152_T1_2mm.nii.gz (alias for advanced_atlas_registration)

    Runtime asserts that manifest.json.anti_leakage.excluded_subject_id matches
    the requested subject, raising RuntimeError on any mismatch.
    """
    atlas_root = Path(atlas_root)

    def _resolve(record):
        subject = record["subject"]
        subject_sid = subject.replace("sub-", "", 1)
        out_dir = atlas_root / subject
        t1 = out_dir / f"tpl-{dataset_tag}{subject_sid}_T1w.nii.gz"
        brain_mask = out_dir / f"tpl-{dataset_tag}{subject_sid}_desc-brain_mask.nii.gz"
        manifest = out_dir / "manifest.json"
        if not t1.exists():
            raise FileNotFoundError(f"LOO atlas T1 manquant: {t1}")
        if not brain_mask.exists():
            raise FileNotFoundError(f"LOO atlas brain_mask manquant: {brain_mask}")
        if not manifest.exists():
            raise FileNotFoundError(f"LOO atlas manifest manquant: {manifest}")
        # Anti-leakage runtime assertion
        try:
            payload = json.loads(manifest.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Manifest LOO illisible: {manifest}: {exc}")
        excluded = payload.get("anti_leakage", {}).get("excluded_subject_id")
        if excluded != subject_sid:
            raise RuntimeError(
                f"Anti-leakage violation: atlas {manifest} excludes '{excluded}' "
                f"but subject is '{subject_sid}'"
            )
        return {
            "t1": t1,
            "brain_mask": brain_mask,
        }
    return _resolve


# ============================================================
# Records resolvers
# ============================================================

def _bool_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _natural_key(value: Any) -> list:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(value))]


def iseg_records_resolver(
    manifest: Path,
    split_filter: str = "train_labeled",
    subjects: list[str] | None = None,
    n_subjects: int | None = None,
) -> list[dict]:
    """Records resolver for iSeg-2017 training_manifest.csv format."""
    manifest = Path(manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest iSeg2017 introuvable: {manifest}")
    df = pd.read_csv(manifest)
    required = {"subject", "split", "has_gold", "t1_path", "t2_path", "brain_mask_path", "label_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Manifest iSeg2017 colonnes manquantes: {missing}")
    df = df[(df["split"].astype(str) == split_filter) & df["has_gold"].map(_bool_truthy)].copy()
    if len(df) == 0:
        raise RuntimeError("Aucun sujet iSeg2017 labellise disponible pour les metriques")
    records: list[dict] = []
    for _, row in df.sort_values("subject", key=lambda s: s.map(_natural_key)).iterrows():
        subject = str(row["subject"])
        subject_raw = subject.replace("sub-", "", 1)
        rec = {
            "subject": subject,
            "subject_raw": subject_raw,
            "subject_id": f"{subject_raw}_T1w",
            "session": "ses-6mo",
            "t1_path": Path(str(row["t1_path"])),
            "t2_path": Path(str(row["t2_path"])),
            "gs_mask_path": Path(str(row["brain_mask_path"])),
            "label_path": Path(str(row["label_path"])),
            "split": str(row["split"]),
        }
        for key in ["t1_path", "t2_path", "gs_mask_path", "label_path"]:
            if not rec[key].exists():
                raise FileNotFoundError(f"{subject}: fichier {key} introuvable: {rec[key]}")
        records.append(rec)
    if subjects:
        wanted = [str(s).replace("sub-", "", 1) for s in subjects]
        by_key: dict[str, dict] = {}
        for r in records:
            for k in {r["subject"], r["subject_raw"], r["subject_id"]}:
                by_key[k] = r
                by_key[k.replace("sub-", "", 1)] = r
        missing_s = [s for s in wanted if s not in by_key]
        if missing_s:
            raise RuntimeError(f"Sujets demandes introuvables: {missing_s}")
        records = [by_key[s] for s in wanted]
    if n_subjects is not None:
        records = records[: int(n_subjects)]
    return records


def bids_gold_records_resolver(
    manifest: Path,
    subjects: list[str] | None = None,
    n_subjects: int | None = None,
) -> list[dict]:
    """Records resolver for BOBs/MCRIB/MCRIB_WM standardized BIDS manifests."""
    manifest = Path(manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest introuvable: {manifest}")
    df = pd.read_csv(manifest)
    required = {"dataset", "subject", "session", "has_gold", "qc_only",
                "t1_path", "t2_path", "brain_mask_path", "label_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Manifest colonnes manquantes: {missing}")
    df = df[df["has_gold"].map(_bool_truthy) & ~df["qc_only"].map(_bool_truthy)].copy()
    if len(df) == 0:
        raise RuntimeError(f"Aucun sujet gold disponible dans {manifest}")
    records: list[dict] = []
    for _, row in df.sort_values("subject", key=lambda s: s.map(_natural_key)).iterrows():
        subject = str(row["subject"])
        subject_raw = subject.replace("sub-", "", 1)
        rec = {
            "subject": subject,
            "subject_raw": subject_raw,
            "subject_id": f"{subject_raw}_T1w",
            "session": str(row.get("session", "")),
            "t1_path": Path(str(row["t1_path"])),
            "t2_path": Path(str(row["t2_path"])),
            "gs_mask_path": Path(str(row["brain_mask_path"])),
            "label_path": Path(str(row["label_path"])),
            "split": "gold",
            "profile": str(row.get("profile", "")),
        }
        for key in ["t1_path", "t2_path", "gs_mask_path", "label_path"]:
            if not rec[key].exists():
                raise FileNotFoundError(f"{subject}: fichier {key} introuvable: {rec[key]}")
        records.append(rec)
    if subjects:
        wanted = [str(s).replace("sub-", "", 1) for s in subjects]
        records = [r for r in records if r["subject_raw"] in wanted or r["subject"] in subjects]
    if n_subjects is not None:
        records = records[: int(n_subjects)]
    return records


# ============================================================
# IO utilities (atomic write, progress)
# ============================================================

def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _write_progress(progress_json: Path, live_csv: Path, dataset: str,
                    stage: str, subject=None, idx=None, total=None,
                    rows=None) -> None:
    payload: dict[str, Any] = {
        "dataset": dataset,
        "stage": stage,
        "subject": subject,
        "idx": idx,
        "total": total,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if rows is not None:
        payload["n_rows"] = len(rows)
        payload["n_errors"] = int(sum(1 for r in rows if str(r.get("error", ""))))
        payload["n_dice"] = int(sum(1 for r in rows if pd.notna(r.get("dice", np.nan))))
        try:
            pd.DataFrame(rows).to_csv(live_csv, index=False)
        except Exception as exc:
            payload["live_csv_error"] = str(exc)
    _atomic_write_text(progress_json, json.dumps(payload, indent=2, ensure_ascii=False))


def _save_nifti_3d(src_path: Path, dst_path: Path, *, is_mask: bool = False,
                   spacing_override=None, force: bool = True) -> Path:
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    if dst_path.exists() and not force:
        return dst_path
    img = nib.load(str(src_path))
    data = np.asanyarray(img.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise RuntimeError(f"Image non 3D apres squeeze: {src_path} shape={data.shape}")
    affine = np.array(img.affine, dtype=float)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if spacing_override is not None:
        sx, sy, sz = [float(v) for v in spacing_override]
        affine = np.diag([sx, sy, sz, 1.0])
        zooms = (sx, sy, sz)
    if is_mask:
        out_data = (data > 0).astype(np.uint8)
        out_dtype = np.uint8
    elif np.issubdtype(data.dtype, np.integer):
        out_data = np.rint(data).astype(np.int16)
        out_dtype = np.int16
    else:
        out_data = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        out_dtype = np.float32
    out_img = nib.Nifti1Image(out_data, affine)
    out_img.header.set_zooms(zooms)
    out_img.set_data_dtype(out_dtype)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(dst_path))
    return dst_path


def _prepare_source_images(rec: Mapping, anat_dir: Path, *,
                            spacing_override=None, force: bool = True) -> tuple:
    source_dir = Path(anat_dir) / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    subject_id = rec["subject_id"]
    t1_out = source_dir / f"sub-{subject_id}.nii.gz"
    t2_out = source_dir / f"sub-{subject_id}_T2w.nii.gz"
    mask_out = source_dir / f"sub-{subject_id}_gold-brain_mask.nii.gz"
    label_out = source_dir / f"sub-{subject_id}_desc-label_seg.nii.gz"
    t1_out = _save_nifti_3d(rec["t1_path"], t1_out, is_mask=False,
                             spacing_override=spacing_override, force=force)
    t2_out = _save_nifti_3d(rec["t2_path"], t2_out, is_mask=False,
                             spacing_override=spacing_override, force=force)
    mask_out = _save_nifti_3d(rec["gs_mask_path"], mask_out, is_mask=True,
                               spacing_override=spacing_override, force=force)
    label_out = _save_nifti_3d(rec["label_path"], label_out, is_mask=False,
                                spacing_override=spacing_override, force=force)
    return t1_out, mask_out, t2_out, label_out


def _image_info(path: Path) -> dict:
    img = nib.load(str(path))
    return {
        "shape": "x".join(str(int(v)) for v in img.shape[:3]),
        "zooms": "x".join(f"{float(v):g}" for v in img.header.get_zooms()[:3]),
        "orientation": "".join(nib.aff2axcodes(img.affine)),
    }


# ============================================================
# Mask quality / atlas selection / refinement primitives
# ============================================================

def _mask_quality(path: Path, priors: IsegStyleVolumePriors, *,
                   min_cm3=None, max_cm3=None,
                   max_components=None, max_extent_ratio=None):
    min_cm3 = priors.mask_min_cm3 if min_cm3 is None else float(min_cm3)
    max_cm3 = priors.mask_max_cm3 if max_cm3 is None else float(max_cm3)
    max_components = priors.mask_max_components if max_components is None else int(max_components)
    max_extent_ratio = priors.mask_max_extent if max_extent_ratio is None else float(max_extent_ratio)
    try:
        img = nib.load(str(path))
        mask = img.get_fdata() > 0.5
        vox = float(np.prod(img.header.get_zooms()[:3]))
        vol_cm3 = float(np.sum(mask) * vox / 1000.0)
        lbl, ncomp = ndi.label(mask)
        reason: list[str] = []
        if vol_cm3 < min_cm3 or vol_cm3 > max_cm3:
            reason.append(f"volume={vol_cm3:.1f} cm3 hors [{min_cm3:.0f}, {max_cm3:.0f}]")
        if ncomp > max_components:
            reason.append(f"composantes={ncomp} > {max_components}")
        if mask.any():
            pts = np.where(mask)
            bbox = [(int(p.min()), int(p.max())) for p in pts]
            extent_ratios = [(b - a + 1) / s for (a, b), s in zip(bbox, mask.shape)]
            extent_ratio = max(extent_ratios)
            centroid = tuple(float(p.mean() / s) for p, s in zip(pts, mask.shape))
            if extent_ratio > max_extent_ratio:
                reason.append(f"bbox trop etendue={extent_ratio:.1%} > {max_extent_ratio:.0%}")
        else:
            bbox = None
            extent_ratio = 0.0
            centroid = None
            reason.append("masque vide")
        volume_center = (min_cm3 + max_cm3) / 2.0
        volume_half_range = max(1.0, (max_cm3 - min_cm3) / 2.0)
        volume_score = max(0.0, 1.0 - abs(vol_cm3 - volume_center) / volume_half_range)
        extent_score = max(0.0, 1.0 - max(0.0, extent_ratio - 0.72) / max(0.01, max_extent_ratio - 0.72))
        component_score = 1.0 if ncomp == 1 else max(0.0, 1.0 - (ncomp - 1) / max_components)
        score = 0.60 * volume_score + 0.20 * extent_score + 0.20 * component_score
        return len(reason) == 0, {
            "volume_cm3": vol_cm3,
            "components": int(ncomp),
            "bbox": bbox,
            "max_extent_ratio": float(extent_ratio),
            "centroid_rel": centroid,
            "qc_score": float(score),
        }, "; ".join(reason)
    except Exception as e:
        return False, {}, f"erreur lecture masque: {e}"


def _clean_subject_artifacts(anat_dir: Path, subject_id: str, *,
                              clean_stale: bool, force_atlas_reg: bool,
                              force_extraction: bool, delete_short_atlases: bool) -> None:
    if not clean_stale:
        return
    anat_dir = Path(anat_dir)
    deleted: list[str] = []
    patterns = [
        f"sub-{subject_id}_desc-brain*.nii.gz",
        f"sub-{subject_id}_desc-uncertainty*.nii.gz",
        f"sub-{subject_id}_desc-brain_metrics.json",
    ]
    atlas_dir = anat_dir / "atlases_advanced"
    if force_atlas_reg and atlas_dir.exists():
        patterns += [f"atlases_advanced/sub-{subject_id}_brain_mask_*.nii.gz"]
    if force_extraction:
        for pattern in patterns:
            for p in anat_dir.glob(pattern):
                try:
                    p.unlink()
                    deleted.append(p.name)
                except FileNotFoundError:
                    pass
    if delete_short_atlases and atlas_dir.exists():
        for p in atlas_dir.glob(f"sub-{subject_id}_brain_mask_*.nii.gz"):
            try:
                if p.stat().st_size < 4096:
                    p.unlink()
                    deleted.append(p.name)
            except FileNotFoundError:
                pass
    if deleted:
        print(f"    Nettoyage anti-stale: {len(deleted)} fichier(s) supprime(s)")


def _select_preproc_mask(anat_dir: Path, subject_id: str,
                          priors: IsegStyleVolumePriors) -> Path | None:
    candidates: list[Path] = []
    adv_dir = Path(anat_dir) / "atlases_advanced"
    if adv_dir.exists():
        for suffix in ["aff_nomask", "aff", "syn"]:
            candidates.append(adv_dir / f"sub-{subject_id}_brain_mask_{suffix}.nii.gz")
    rejected, seen = [], set()
    for cand in candidates:
        if not cand.exists() or str(cand) in seen:
            continue
        seen.add(str(cand))
        ok, stats, why = _mask_quality(cand, priors)
        if ok:
            print(f"    Masque preprocessing valide: {cand.name} ({stats['volume_cm3']:.1f} cm3, composantes={stats['components']})")
            return cand
        rejected.append((cand.name, why))
    if rejected:
        print("    Masques preprocessing rejetes:")
        for name, why in rejected[:5]:
            print(f"      - {name}: {why}")
    return None


def _registered_atlas_candidates(anat_dir: Path, subject_id: str) -> list[Path]:
    adv_dir = Path(anat_dir) / "atlases_advanced"
    paths = [adv_dir / f"sub-{subject_id}_brain_mask_adaptive.nii.gz"]
    paths.extend(sorted(adv_dir.glob(f"sub-{subject_id}_brain_mask_*_adaptive.nii.gz")))
    for suffix in ["aff_nomask", "aff", "syn"]:
        paths.append(adv_dir / f"sub-{subject_id}_brain_mask_{suffix}.nii.gz")
    out, seen = [], set()
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _adaptive_fusion_atlas_paths(anat_dir: Path, subject_id: str,
                                  *, selected_atlas: Path | None = None,
                                  adaptive_path: Path | None = None) -> list[Path]:
    paths: list[Path] = [selected_atlas, adaptive_path]  # type: ignore
    adv_dir = Path(anat_dir) / "atlases_advanced"
    for suffix in ["adaptive", "aff", "aff_nomask", "syn"]:
        paths.append(adv_dir / f"sub-{subject_id}_brain_mask_{suffix}.nii.gz")
    out, seen = [], set()
    for path in paths:
        if not path:
            continue
        path = Path(path)
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _atlas_variant_bonus(atlas_path: Path, priors: IsegStyleVolumePriors) -> float:
    name = Path(atlas_path).name
    if "_adaptive" in name:
        return 0.12
    if "_aff_nomask" in name:
        return float(priors.atlas_nomask_bonus)
    return 0.0


def _select_registered_atlas(anat_dir: Path, subject_id: str,
                              priors: IsegStyleVolumePriors,
                              image_path: Path | None = None):
    valid, rejected = [], []
    for cand in _registered_atlas_candidates(anat_dir, subject_id):
        if not cand.exists():
            continue
        ok, stats, why = _mask_quality(
            cand, priors,
            min_cm3=priors.atlas_qc_min_cm3,
            max_cm3=priors.atlas_qc_max_cm3,
            max_components=priors.atlas_max_components,
        )
        if ok:
            if image_path is not None:
                img_score, img_stats = robust_atlas_image_alignment_score(cand, image_path)
            else:
                img_score, img_stats = 0.0, {}
            stats.update(img_stats)
            stats["image_alignment_score"] = float(img_score)
            variant_bonus = _atlas_variant_bonus(cand, priors)
            stats["variant_bonus"] = float(variant_bonus)
            stats["selection_score_base"] = float(0.55 * stats.get("qc_score", 0.0) + 0.45 * img_score)
            stats["selection_score"] = float(np.clip(stats["selection_score_base"] + variant_bonus, 0.0, 1.0))
            weak = []
            if image_path is not None and img_score < priors.min_atlas_image_score:
                weak.append(f"score_image={img_score:.3f} < {priors.min_atlas_image_score:.3f}")
            if stats["selection_score"] < priors.min_atlas_selection_score:
                weak.append(f"score_global={stats['selection_score']:.3f} < {priors.min_atlas_selection_score:.3f}")
            if weak:
                rejected.append((cand.name, "atlas recale trop faible: " + "; ".join(weak)))
                continue
            valid.append((cand, stats))
        else:
            rejected.append((cand.name, why))
    if valid:
        valid.sort(key=lambda item: item[1].get("selection_score", item[1].get("qc_score", 0.0)), reverse=True)
        cand, stats = valid[0]
        print(f"    Atlas valide: {cand.name} ({stats['volume_cm3']:.1f} cm3, composantes={stats['components']}, "
              f"score={stats.get('selection_score', stats.get('qc_score', 0.0)):.3f})")
        if len(valid) > 1:
            print("    Autres atlas plausibles:")
            for alt, alt_stats in valid[1:3]:
                print(f"      - {alt.name}: {alt_stats['volume_cm3']:.1f} cm3, "
                      f"score={alt_stats.get('selection_score', alt_stats.get('qc_score', 0.0)):.3f}")
        return cand, stats, rejected
    if rejected:
        print("    Atlas rejetes par QC anatomique/alignement:")
        for name, why in rejected[:8]:
            print(f"      - {name}: {why}")
    else:
        print(f"    Aucun atlas valide trouve pour sub-{subject_id}")
    return None, {}, rejected


def _confidence_from_audit(anat_dir: Path, subject_id: str, pp=None) -> float:
    try:
        audit = None
        if isinstance(pp, dict) and pp.get("visual_metrics"):
            audit = pp["visual_metrics"]
        recommended = audit.get("recommended_threshold") if isinstance(audit, dict) else None
        if recommended is not None:
            confidence = float(np.clip(recommended, 0.65, 0.85))
        else:
            confidence = 0.72
        if isinstance(audit, dict):
            edge_ws = float(audit.get("edge_degradation_ws", 0.0) or 0.0)
            if edge_ws > 0.15:
                print(f"    QC WhiteStripe: bords degrades ({edge_ws*100:.1f}%) -> N4 conservee pour SAM2")
        print(f"    Seuil confiance={confidence:.2f}")
        return confidence
    except Exception:
        print("    Pas d'audit exploitable -> seuil confiance=0.72")
        return 0.72


def _extract_brain(*, notebook_globals: Mapping, input_path: Path, output_path: Path,
                    priors: IsegStyleVolumePriors,
                    confidence_threshold: float | None = None,
                    use_atlas_guidance: bool = True,
                    atlas_paths=None,
                    post_process: bool = True,
                    save_metrics: bool = True,
                    save_uncertainty_map: bool = True,
                    debug_dir: Path | None = None,
                    age_months: int | None = None):
    """Wrap notebook SAM2BrainExtractor with priors-driven SegmentationParams."""
    SegmentationParams = notebook_globals["SegmentationParams"]
    SAM2BrainExtractor = notebook_globals["SAM2BrainExtractor"]
    params = SegmentationParams(
        debug_dir=debug_dir,
        use_atlas_guidance=use_atlas_guidance,
        atlas_weight=priors.atlas_weight,
        post_process=post_process,
    )
    params.min_brain_volume_cm3 = priors.min_brain_volume_cm3
    params.max_brain_volume_cm3 = priors.max_brain_volume_cm3
    params.atlas_fallback_min_cm3 = priors.atlas_fallback_min_cm3
    params.atlas_fallback_max_cm3 = priors.atlas_fallback_max_cm3
    params.atlas_min_coverage = priors.atlas_min_coverage
    params.atlas_min_precision = priors.atlas_min_precision
    params.enable_atlas_fallback = bool(atlas_paths) and priors.enable_atlas_fallback
    if hasattr(params, 'eap_tight_ratio_threshold'):
        params.eap_tight_ratio_threshold = float(priors.sam2_eap_tight_ratio_threshold)
    # IDÉE 1 / 2 / 4 — propager les nouveaux flags du SAM2BrainExtractor amélioré.
    # Les hasattr permettent la compatibilité avec un notebook non-mis-à-jour.
    if hasattr(params, 'enable_atlas_input_masking'):
        params.enable_atlas_input_masking = bool(priors.sam2_enable_atlas_input_masking)
        params.atlas_mask_hard_radius_mm = float(priors.sam2_atlas_mask_hard_radius_mm)
        params.atlas_mask_soft_radius_mm = float(priors.sam2_atlas_mask_soft_radius_mm)
        params.atlas_mask_min_volume_cm3 = float(priors.sam2_atlas_mask_min_volume_cm3)
        params.atlas_mask_background_value = float(priors.sam2_atlas_mask_background_value)
    if hasattr(params, 'eap_tight_always_initial'):
        params.eap_tight_always_initial = bool(priors.sam2_eap_tight_always_initial)
        params.eap_tight_min_preservation = float(priors.sam2_eap_tight_min_preservation)
    if hasattr(params, 'adaptive_atlas_shrink_enabled'):
        params.adaptive_atlas_shrink_enabled = bool(priors.sam2_adaptive_atlas_shrink_enabled)
        params.adaptive_atlas_shrink_target_ratio = float(priors.sam2_adaptive_atlas_shrink_target_ratio)
        params.adaptive_atlas_shrink_trigger_ratio = float(priors.sam2_adaptive_atlas_shrink_trigger_ratio)
        params.adaptive_atlas_shrink_max_iters = int(priors.sam2_adaptive_atlas_shrink_max_iters)
        if age_months is not None:
            params.adaptive_atlas_age_months = int(age_months)
    # Vague 2 — IDÉE 3 + early stopping
    if hasattr(params, 'enable_atlas_mask_input_prompt'):
        params.enable_atlas_mask_input_prompt = bool(priors.sam2_enable_atlas_mask_input_prompt)
        params.mask_input_strength = float(priors.sam2_mask_input_strength)
        params.mask_input_use_eap = bool(priors.sam2_mask_input_use_eap)
    if hasattr(params, 'enable_volume_early_stopping'):
        params.enable_volume_early_stopping = bool(priors.sam2_enable_volume_early_stopping)
        params.volume_early_stop_ratio = float(priors.sam2_volume_early_stop_ratio)
    # Vague 3 — bidirectional consensus
    if hasattr(params, 'enable_bidirectional_consensus'):
        params.enable_bidirectional_consensus = bool(priors.sam2_enable_bidirectional_consensus)
        params.bidirectional_consensus_min_dice = float(priors.sam2_bidirectional_consensus_min_dice)
        params.bidirectional_consensus_min_range = int(priors.sam2_bidirectional_consensus_min_range)
    if atlas_paths:
        params.atlas_paths = atlas_paths
    if confidence_threshold is not None:
        params.initial_confidence_threshold = float(confidence_threshold)
        params.propagation_confidence_threshold = float(confidence_threshold) * 0.9
    try:
        extractor = SAM2BrainExtractor(params=params)
        return extractor.extract_brain(
            input_path=input_path,
            output_path=output_path,
            save_metrics=save_metrics,
            save_uncertainty_map=save_uncertainty_map,
        )
    except Exception as e:
        print(f"Erreur lors de l'extraction du cerveau: {e}")
        traceback.print_exc()
        return None, None


def _nii_with_suffix(path: Path, suffix: str) -> Path:
    path = Path(path)
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name[:-7] + suffix + ".nii.gz")
    return path.with_name(path.stem + suffix + path.suffix)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    lbl, ncomp = ndi.label(mask)
    if ncomp <= 1:
        return mask
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    return lbl == int(np.argmax(counts))


def _mask_overlap_stats(mask: np.ndarray, atlas: np.ndarray, zooms: tuple) -> dict:
    mask = np.asarray(mask).astype(bool)
    atlas = np.asarray(atlas).astype(bool)
    inter = int(np.logical_and(mask, atlas).sum())
    mask_n = int(mask.sum())
    atlas_n = int(atlas.sum())
    voxel_vol = float(np.prod(zooms))
    return {
        "volume_cm3": float(mask_n * voxel_vol / 1000.0),
        "atlas_coverage": float(inter / (atlas_n + 1e-8)),
        "atlas_precision": float(inter / (mask_n + 1e-8)),
        "atlas_dice": float((2.0 * inter) / (mask_n + atlas_n + 1e-8)),
        "voxels": mask_n,
    }


def _apply_local_boundary_refinement(mask, atlas, mask_img, input_image_path,
                                      priors: IsegStyleVolumePriors):
    from nibabel.processing import resample_from_to
    stats: dict = {
        "boundary_refinement_enabled": bool(priors.enable_boundary_refinement),
        "boundary_refinement_used": False,
        "boundary_refinement_reason": None,
    }
    if not priors.enable_boundary_refinement:
        stats["boundary_refinement_reason"] = "disabled"
        return mask, stats
    try:
        input_image_path = Path(input_image_path)
        if not input_image_path.exists():
            stats["boundary_refinement_reason"] = "input_missing"
            return mask, stats
        mask = np.asarray(mask).astype(bool)
        atlas = np.asarray(atlas).astype(bool)
        if mask.sum() == 0:
            stats["boundary_refinement_reason"] = "empty_mask"
            return mask, stats
        input_img = nib.load(str(input_image_path))
        input_for_mask = input_img
        if input_img.shape[:3] != mask_img.shape[:3] or not np.allclose(input_img.affine, mask_img.affine, atol=1e-3):
            input_for_mask = resample_from_to(input_img, mask_img, order=1, cval=0)
        data = np.nan_to_num(input_for_mask.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        structure = ndi.generate_binary_structure(3, 1)
        atlas_allowed = atlas
        if int(priors.boundary_atlas_dilation) > 0:
            atlas_allowed = ndi.binary_dilation(atlas, structure=structure,
                                                 iterations=int(priors.boundary_atlas_dilation))
        shell = ndi.binary_dilation(mask, structure=structure,
                                     iterations=int(priors.boundary_refinement_width)) & (~mask) & atlas_allowed
        in_values = data[mask]
        in_values = in_values[np.isfinite(in_values)]
        if in_values.size == 0:
            stats["boundary_refinement_reason"] = "no_finite_intensity"
            return mask, stats
        pct = float(priors.boundary_intensity_percentile)
        threshold = float(np.percentile(in_values, pct))
        additions = shell & (data >= threshold)
        if int(additions.sum()) == 0:
            stats.update({
                "boundary_refinement_reason": "no_candidate_voxels",
                "boundary_refinement_intensity_threshold": threshold,
                "boundary_refinement_candidate_voxels": int(shell.sum()),
            })
            return mask, stats
        proposal = _largest_component(mask | additions)
        proposal = ndi.binary_fill_holes(proposal).astype(bool)
        zooms = tuple(float(v) for v in mask_img.header.get_zooms()[:3])
        voxel_vol_cm3 = float(np.prod(zooms) / 1000.0)
        before = _mask_overlap_stats(mask, atlas, zooms)
        after = _mask_overlap_stats(proposal, atlas, zooms)
        added_voxels = int(np.logical_and(proposal, ~mask).sum())
        added_cm3 = float(added_voxels * voxel_vol_cm3)
        volume_ratio = float(after["volume_cm3"] / max(before["volume_cm3"], 1e-8))
        stats.update({
            "boundary_refinement_intensity_percentile": pct,
            "boundary_refinement_intensity_threshold": threshold,
            "boundary_refinement_candidate_voxels": int(shell.sum()),
            "boundary_refinement_added_voxels": added_voxels,
            "boundary_refinement_added_cm3": added_cm3,
            "boundary_refinement_volume_before_cm3": before["volume_cm3"],
            "boundary_refinement_volume_after_cm3": after["volume_cm3"],
            "boundary_refinement_volume_ratio": volume_ratio,
            "boundary_refinement_atlas_precision_before": before["atlas_precision"],
            "boundary_refinement_atlas_precision_after": after["atlas_precision"],
            "boundary_refinement_atlas_coverage_before": before["atlas_coverage"],
            "boundary_refinement_atlas_coverage_after": after["atlas_coverage"],
        })
        reject = []
        if after["volume_cm3"] < priors.min_brain_volume_cm3:
            reject.append(f"volume_after={after['volume_cm3']:.1f} cm3 < {priors.min_brain_volume_cm3:.0f}")
        if after["volume_cm3"] > priors.max_brain_volume_cm3:
            reject.append(f"volume_after={after['volume_cm3']:.1f} cm3 > {priors.max_brain_volume_cm3:.0f}")
        if added_cm3 > float(priors.boundary_max_added_cm3):
            reject.append(f"added_cm3={added_cm3:.1f} > {priors.boundary_max_added_cm3:.1f}")
        if volume_ratio > float(priors.boundary_max_volume_ratio):
            reject.append(f"volume_ratio={volume_ratio:.3f} > {priors.boundary_max_volume_ratio:.3f}")
        if after["atlas_precision"] < float(priors.boundary_min_atlas_precision):
            reject.append(f"atlas_precision_after={after['atlas_precision']:.3f} < {priors.boundary_min_atlas_precision:.3f}")
        if reject:
            stats["boundary_refinement_reason"] = "; ".join(reject)
            return mask, stats
        stats["boundary_refinement_used"] = True
        stats["boundary_refinement_reason"] = "applied"
        return proposal, stats
    except Exception as e:
        stats["boundary_refinement_reason"] = f"erreur boundary refinement: {e}"
        return mask, stats


def _refine_mask_with_atlas(mask_path: Path, brain_path: Path | None,
                              input_image_path: Path, atlas_path: Path,
                              priors: IsegStyleVolumePriors,
                              metrics_path: Path | None = None,
                              *, force: bool = True) -> dict:
    stats: dict = {
        "refinement_enabled": bool(priors.enable_atlas_refinement),
        "refinement_used": False,
        "refinement_reason": None,
    }
    if not priors.enable_atlas_refinement:
        stats["refinement_reason"] = "disabled"
        return stats
    if atlas_path is None or not Path(atlas_path).exists():
        stats["refinement_reason"] = "atlas_missing"
        return stats
    try:
        from nibabel.processing import resample_from_to
        mask_path = Path(mask_path)
        brain_path = Path(brain_path) if brain_path is not None else None
        input_image_path = Path(input_image_path)
        atlas_path = Path(atlas_path)
        mask_img = nib.load(str(mask_path))
        mask = mask_img.get_fdata() > 0.5
        atlas_img = nib.load(str(atlas_path))
        if atlas_img.shape[:3] != mask_img.shape[:3] or not np.allclose(atlas_img.affine, mask_img.affine, atol=1e-3):
            atlas_img = resample_from_to(atlas_img, mask_img, order=0, cval=0)
        atlas = atlas_img.get_fdata() > 0.5
        zooms = tuple(float(v) for v in mask_img.header.get_zooms()[:3])
        before = _mask_overlap_stats(mask, atlas, zooms)
        structure = ndi.generate_binary_structure(3, 1)
        allowed = atlas
        if int(priors.refinement_atlas_dilation) > 0:
            allowed = ndi.binary_dilation(atlas, structure=structure,
                                           iterations=int(priors.refinement_atlas_dilation))
        refined = _largest_component(mask & allowed)
        refined = ndi.binary_fill_holes(refined).astype(bool)
        strict_after = _mask_overlap_stats(refined, atlas, zooms)
        after = strict_after
        reject = []
        if after["voxels"] == 0:
            reject.append("masque raffine vide")
        if after["volume_cm3"] < priors.min_brain_volume_cm3:
            reject.append(f"volume_after={after['volume_cm3']:.1f} cm3 < {priors.min_brain_volume_cm3:.0f}")
        if after["volume_cm3"] > priors.max_brain_volume_cm3:
            reject.append(f"volume_after={after['volume_cm3']:.1f} cm3 > {priors.max_brain_volume_cm3:.0f}")
        keep_ratio = after["voxels"] / max(1, before["voxels"])
        if keep_ratio < float(priors.refinement_min_keep_ratio):
            reject.append(f"keep_ratio={keep_ratio:.3f} < {priors.refinement_min_keep_ratio:.3f}")
        if after["atlas_coverage"] < float(priors.refinement_min_atlas_coverage):
            reject.append(f"atlas_coverage_after={after['atlas_coverage']:.3f} < {priors.refinement_min_atlas_coverage:.3f}")
        precision_gain = after["atlas_precision"] - before["atlas_precision"]
        already_atlas_constrained = before["atlas_precision"] >= float(priors.boundary_min_atlas_precision)
        if precision_gain < float(priors.refinement_min_precision_gain) and not already_atlas_constrained:
            reject.append(f"precision_gain={precision_gain:.3f} < {priors.refinement_min_precision_gain:.3f}")
        stats.update({
            "refinement_dilation": int(priors.refinement_atlas_dilation),
            "refinement_volume_before_cm3": before["volume_cm3"],
            "atlas_refinement_volume_after_cm3": strict_after["volume_cm3"],
            "atlas_refinement_keep_ratio": float(keep_ratio),
            "refinement_atlas_coverage_before": before["atlas_coverage"],
            "atlas_refinement_atlas_coverage_after": strict_after["atlas_coverage"],
            "refinement_atlas_precision_before": before["atlas_precision"],
            "atlas_refinement_atlas_precision_after": strict_after["atlas_precision"],
            "refinement_atlas_dice_before": before["atlas_dice"],
            "atlas_refinement_atlas_dice_after": strict_after["atlas_dice"],
            "refinement_precision_gain": float(precision_gain),
        })
        if reject:
            stats["refinement_reason"] = "; ".join(reject)
            print(f"    Raffinement atlas non applique: {stats['refinement_reason']}")
            return stats
        refined, boundary_stats = _apply_local_boundary_refinement(refined, atlas, mask_img,
                                                                     input_image_path, priors)
        stats.update(boundary_stats)
        after = _mask_overlap_stats(refined, atlas, zooms)
        stats.update({
            "refinement_volume_after_cm3": after["volume_cm3"],
            "refinement_keep_ratio": float(after["voxels"] / max(1, before["voxels"])),
            "refinement_atlas_coverage_after": after["atlas_coverage"],
            "refinement_atlas_precision_after": after["atlas_precision"],
            "refinement_atlas_dice_after": after["atlas_dice"],
        })
        backup_mask = _nii_with_suffix(mask_path, "_pre-refine")
        if not backup_mask.exists() or force:
            nib.save(mask_img, str(backup_mask))
        out_hdr = mask_img.header.copy()
        out_img = nib.Nifti1Image(refined.astype(np.uint8), mask_img.affine, out_hdr)
        out_img.header.set_zooms(zooms)
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, str(mask_path))
        if brain_path is not None and input_image_path.exists():
            input_img = nib.load(str(input_image_path))
            refine_img_for_input = out_img
            if input_img.shape[:3] != out_img.shape[:3] or not np.allclose(input_img.affine, out_img.affine, atol=1e-3):
                refine_img_for_input = resample_from_to(out_img, input_img, order=0, cval=0)
            refined_input = refine_img_for_input.get_fdata() > 0.5
            brain_data = np.nan_to_num(input_img.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0) * refined_input.astype(np.float32)
            brain_hdr = input_img.header.copy()
            brain_img = nib.Nifti1Image(brain_data.astype(np.float32), input_img.affine, brain_hdr)
            brain_img.set_data_dtype(np.float32)
            nib.save(brain_img, str(brain_path))
        stats["refinement_used"] = True
        stats["refinement_reason"] = "applied"
        stats["refinement_backup_mask"] = str(backup_mask)
        label = "Raffinement atlas+frontiere applique" if stats.get("boundary_refinement_used") else "Raffinement atlas applique"
        print(f"    {label}: volume {before['volume_cm3']:.1f}->{after['volume_cm3']:.1f} cm3, "
              f"atlas_precision {before['atlas_precision']:.3f}->{after['atlas_precision']:.3f}, coverage={after['atlas_coverage']:.3f}")
        if metrics_path is not None and Path(metrics_path).exists():
            try:
                payload = json.loads(Path(metrics_path).read_text())
                payload.update(stats)
                Path(metrics_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            except Exception as exc:
                print(f"    Impossible de mettre a jour les metriques de raffinement: {exc}")
        return stats
    except Exception as e:
        stats["refinement_reason"] = f"erreur raffinement atlas: {e}"
        print(f"    {stats['refinement_reason']}")
        traceback.print_exc()
        return stats


# ============================================================
# Entry point: run_iseg_style_pipeline
# ============================================================

def _ensure_notebook_globals(notebook_globals: Mapping) -> None:
    required = [
        "run_preprocessing_pipeline",
        "advanced_atlas_registration",
        "SegmentationParams",
        "SAM2BrainExtractor",
        "compute_metrics",
    ]
    missing = [name for name in required if name not in notebook_globals]
    if missing:
        raise RuntimeError(
            "Symbole(s) notebook manquant(s): "
            + ", ".join(missing)
            + ". Executer d'abord les cellules d'initialisation du notebook."
        )


def run_iseg_style_pipeline(
    notebook_globals: Mapping[str, Any],
    spec: IsegStyleDatasetSpec,
    *,
    n_subjects: int | None = None,
    subjects: Iterable[str] | None = None,
    force_source_conversion: bool = True,
    force_preprocessing: bool = True,
    force_atlas_registration: bool = True,
    force_extraction: bool = True,
    clean_stale_outputs: bool = True,
    delete_short_stale_atlases: bool = True,
    stop_if_no_valid_atlas: bool = True,
    reproducibility_seed: int | None = None,
) -> dict[str, Any]:
    """Run the iSeg-style M1 pipeline on a dataset.

    Identical code-path to cell 39 of the notebook. Priors and atlas resolver
    differ per dataset; refinement thresholds are invariant.
    """
    _ensure_notebook_globals(notebook_globals)
    run_preprocessing_pipeline = notebook_globals["run_preprocessing_pipeline"]
    advanced_atlas_registration = notebook_globals["advanced_atlas_registration"]
    compute_metrics = notebook_globals["compute_metrics"]

    priors = spec.volume_priors
    out_base = Path(spec.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    final_csv = out_base / f"sam_neuroadapt_{spec.dataset.lower()}_results.csv"
    live_csv = out_base / f"sam_neuroadapt_{spec.dataset.lower()}_results_live.csv"
    progress_json = out_base / f"m1_{spec.dataset.lower()}_progress.json"
    selection_json = out_base / f"{spec.dataset.lower()}_selection.json"

    if reproducibility_seed is not None:
        np.random.seed(int(reproducibility_seed))
        os.environ["SAM_NEUROADAPT_SEED"] = str(int(reproducibility_seed))

    # 1. Discover & select records
    all_records = spec.selected_records_fn()
    if subjects:
        wanted = {str(s).replace("sub-", "", 1) for s in subjects}
        all_records = [r for r in all_records if r.get("subject_raw") in wanted or r.get("subject") in subjects]
    selected = all_records[: int(n_subjects)] if n_subjects is not None else all_records
    if not selected:
        raise RuntimeError(f"Aucun sujet selectionne pour {spec.dataset}")

    # 2. Persist selection.json (audit + traceability)
    payload = {
        "dataset": spec.dataset,
        "manifest": str(spec.manifest_csv),
        "n_requested": n_subjects,
        "n_available": len(all_records),
        "n_selected": len(selected),
        "subjects": [
            {k: (str(v) if isinstance(v, Path) else v) for k, v in rec.items()}
            for rec in selected
        ],
    }
    # Capture atlas resolved per subject for the selection record.
    selection_atlases = {}
    for rec in selected:
        try:
            atlas_pair = spec.atlas_resolver(rec)
            selection_atlases[rec["subject"]] = {k: str(v) for k, v in atlas_pair.items()}
        except Exception as exc:
            selection_atlases[rec["subject"]] = {"error": str(exc)}
    payload["atlas_resolver"] = selection_atlases
    selection_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"{spec.dataset} sujets disponibles: {len(all_records)} | selectionnes: {len(selected)}")
    print(f"Selection sauvegardee: {selection_json}")

    print(f"\n=== M1-{spec.dataset} - Pipeline SAM-NeuroAdapt (iSeg-style) ===\n")
    _write_progress(progress_json, live_csv, spec.dataset, "started",
                    total=len(selected), rows=[])

    results: list[dict] = []

    for idx, rec in enumerate(selected, 1):
        sid = rec["subject"]
        subject_id = rec["subject_id"]
        subject_bids = f"sub-{subject_id}"
        out_dir = out_base / sid
        anat_dir = out_dir / "anat"
        out_dir.mkdir(parents=True, exist_ok=True)
        anat_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        print(f"\n[{idx:02d}/{len(selected)}] {sid}")
        _write_progress(progress_json, live_csv, spec.dataset, "subject_started",
                        sid, idx, len(selected), results)

        try:
            # 3. Resolve atlas for this subject (no LOO; external only)
            atlas_pair = spec.atlas_resolver(rec)
            atlas_t1 = Path(atlas_pair["t1"])
            atlas_mask = Path(atlas_pair["brain_mask"])
            if not atlas_t1.exists():
                raise FileNotFoundError(f"Atlas T1 introuvable: {atlas_t1}")
            if not atlas_mask.exists():
                raise FileNotFoundError(f"Atlas brain_mask introuvable: {atlas_mask}")
            atlas_paths_for_reg = {"brain_mask": atlas_mask}

            # 4. Prepare source images (with optional spacing override)
            source_t1, gs_mask, source_t2, source_label = _prepare_source_images(
                rec, anat_dir,
                spacing_override=priors.spacing_override,
                force=force_source_conversion,
            )
            if not source_t1.exists() or not gs_mask.exists():
                raise FileNotFoundError(f"T1 ou gold-standard manquant pour {sid}")
            img_info = _image_info(source_t1)
            gs_info = _image_info(gs_mask)

            _clean_subject_artifacts(
                anat_dir, subject_id,
                clean_stale=clean_stale_outputs,
                force_atlas_reg=force_atlas_registration,
                force_extraction=force_extraction,
                delete_short_atlases=delete_short_stale_atlases,
            )

            # 5. Preprocessing N4 + WhiteStripe
            print(f"  [{sid}] Etape 1/3 : N4 + WhiteStripe...")
            _write_progress(progress_json, live_csv, spec.dataset, "preprocessing",
                            sid, idx, len(selected), results)
            n4_path = anat_dir / f"{subject_bids}_desc-n4_T1w.nii.gz"
            ws_path = anat_dir / f"{subject_bids}_desc-n4_desc-ws_T1w.nii.gz"
            pp = None
            if force_preprocessing or not n4_path.exists() or not ws_path.exists():
                preproc_mask = _select_preproc_mask(anat_dir, subject_id, priors)
                pp = run_preprocessing_pipeline(
                    t1_path=source_t1,
                    output_dir=anat_dir,
                    subject_id=subject_id,
                    mask_path=preproc_mask if preproc_mask is not None and preproc_mask.exists() else None,
                )
                if not pp.get("success"):
                    raise RuntimeError(f"Preprocessing failed: {pp}")
                n4_path = Path(pp.get("n4_path")) if pp.get("n4_path") else n4_path
                ws_path = Path(pp.get("whitestripe_path")) if pp.get("whitestripe_path") else ws_path
            else:
                print(f"    Cache preprocessing reutilise: {n4_path.name}, {ws_path.name}")

            if n4_path.exists():
                input_image_path = n4_path
                print(f"    Image SAM2: N4 ({n4_path.name})")
            elif ws_path.exists():
                input_image_path = ws_path
                print(f"    Image SAM2 fallback: WhiteStripe ({ws_path.name})")
            else:
                raise RuntimeError("Preprocessing failed - N4/WS introuvable")

            # 6. Atlas registration + adaptive guidance atlas
            print(f"  [{sid}] Etape 2/3 : Recalage atlas...")
            _write_progress(progress_json, live_csv, spec.dataset, "atlas_registration",
                            sid, idx, len(selected), results)
            atlas_adv_dir = anat_dir / "atlases_advanced"
            atlas_adv_dir.mkdir(parents=True, exist_ok=True)
            adaptive_atlas_stats: dict = {}
            if priors.enable_adaptive_guidance_atlas:
                try:
                    adaptive_atlas_stats = build_iseg_adaptive_guidance_atlas(
                        anat_dir=anat_dir,
                        subject_id=subject_id,
                        input_image_path=n4_path,
                        t2_path=source_t2,
                        force=False,
                        config=priors.adaptive_guidance_config,
                    )
                except Exception as e_adapt:
                    print(f"    Atlas adaptatif non disponible avant recalage: {e_adapt}")
                    adaptive_atlas_stats = {"adaptive_atlas_reason": str(e_adapt)}

            selected_atlas, atlas_stats, rejected_atlases = _select_registered_atlas(
                anat_dir, subject_id, priors, n4_path
            )
            if force_atlas_registration or selected_atlas is None:
                try:
                    advanced_atlas_registration(
                        subject_n4_path=n4_path,
                        atlas_paths=atlas_paths_for_reg,
                        output_dir=atlas_adv_dir,
                        subject_id=subject_id,
                        create_brain_mask=True,
                        verbose=False,
                    )
                except Exception as e_reg:
                    print(f"    Recalage atlas echoue: {e_reg}")
                if priors.enable_adaptive_guidance_atlas:
                    try:
                        adaptive_atlas_stats = build_iseg_adaptive_guidance_atlas(
                            anat_dir=anat_dir,
                            subject_id=subject_id,
                            input_image_path=n4_path,
                            t2_path=source_t2,
                            force=True,
                            config=priors.adaptive_guidance_config,
                        )
                        if adaptive_atlas_stats.get("adaptive_atlas_used"):
                            print("    Atlas adaptatif pret: "
                                  f"{Path(adaptive_atlas_stats['output_path']).name} "
                                  f"vol={adaptive_atlas_stats.get('adaptive_volume_cm3', float('nan')):.1f} cm3 "
                                  f"(+{adaptive_atlas_stats.get('adaptive_added_cm3', 0.0):.1f})")
                    except Exception as e_adapt:
                        print(f"    Construction atlas adaptatif echouee: {e_adapt}")
                        adaptive_atlas_stats = {"adaptive_atlas_reason": str(e_adapt)}
                selected_atlas, atlas_stats, rejected_atlases = _select_registered_atlas(
                    anat_dir, subject_id, priors, n4_path
                )

            if selected_atlas is None:
                msg = "Aucun atlas recale valide apres recalage"
                if stop_if_no_valid_atlas:
                    raise RuntimeError(msg + " - extraction SAM2 non lancee pour eviter une sur-segmentation")
                print(f"    {msg} -> extraction SAM2 sans fallback atlas")
                atlas_paths_subj = {}
            else:
                atlas_paths_subj = {"brain_mask": selected_atlas}

            confidence = _confidence_from_audit(anat_dir, subject_id, pp)

            # 7. SAM2 extraction
            print(f"  [{sid}] Etape 3/3 : Extraction SAM-NeuroAdapt...")
            _write_progress(progress_json, live_csv, spec.dataset, "sam2_extraction",
                            sid, idx, len(selected), results)
            brain_out = anat_dir / f"{subject_bids}_desc-brain_T1w.nii.gz"
            mask_out = anat_dir / f"{subject_bids}_desc-brain_mask.nii.gz"
            debug_dir = anat_dir / "debug_sam"
            debug_dir.mkdir(parents=True, exist_ok=True)

            if force_extraction or not mask_out.exists():
                _subj_age_months = month_from_session(rec.get("session", ""))
                b_path, m_path = _extract_brain(
                    notebook_globals=notebook_globals,
                    input_path=input_image_path,
                    output_path=brain_out,
                    priors=priors,
                    confidence_threshold=confidence,
                    use_atlas_guidance=bool(atlas_paths_subj) and not _no_atlas_ablation(),
                    atlas_paths={} if _no_atlas_ablation() else atlas_paths_subj,
                    post_process=True,
                    save_metrics=True,
                    save_uncertainty_map=True,
                    debug_dir=debug_dir,
                    age_months=_subj_age_months,
                )
            else:
                b_path, m_path = brain_out, mask_out
                print(f"    Cache extraction reutilise: {mask_out.name}")

            if m_path is None or not Path(str(m_path)).exists():
                raise RuntimeError("Extraction echouee - masque introuvable")

            metrics_path = anat_dir / f"{subject_bids}_desc-brain_metrics.json"
            refinement_stats: dict = {}
            if selected_atlas is not None:
                refinement_stats = _refine_mask_with_atlas(
                    mask_path=m_path,
                    brain_path=b_path,
                    input_image_path=input_image_path,
                    atlas_path=selected_atlas,
                    priors=priors,
                    metrics_path=metrics_path,
                    force=force_extraction,
                )

            pediatric_refinement_stats: dict = {}
            if selected_atlas is not None:
                pediatric_refinement_stats = apply_pediatric_t2_refinement(
                    mask_path=m_path,
                    brain_path=b_path,
                    input_image_path=input_image_path,
                    t2_path=source_t2,
                    atlas_path=selected_atlas,
                    metrics_path=metrics_path,
                    config={
                        "enabled": priors.enable_pediatric_t2_refinement,
                        "shell_width": priors.pediatric_t2_shell_width,
                        "atlas_dilation": priors.pediatric_t2_atlas_dilation,
                        "t1_percentile": priors.pediatric_t2_t1_percentile,
                        "t2_percentile": priors.pediatric_t2_t2_percentile,
                        "max_added_cm3": priors.pediatric_t2_max_added_cm3,
                        "max_volume_cm3": priors.max_brain_volume_cm3,
                        "min_atlas_precision": priors.pediatric_t2_min_atlas_precision,
                        "allow_atlas_dice_decrease_on_undercoverage": True,
                        "target_min_atlas_volume_ratio": 0.93,
                        "min_recovery_added_cm3": 25.0,
                        "max_atlas_dice_drop": 0.14,
                    },
                )
                if pediatric_refinement_stats.get("pediatric_refinement_used"):
                    print(f"    Raffinement pediatrique T1/T2 applique: +"
                          f"{pediatric_refinement_stats.get('pediatric_refinement_added_cm3', 0.0):.1f} cm3")
                elif pediatric_refinement_stats:
                    print(f"    Raffinement pediatrique T1/T2 non applique: "
                          f"{pediatric_refinement_stats.get('pediatric_refinement_reason')}")

            wide_refinement_stats: dict = {}
            if priors.enable_wide_refinement and selected_atlas is not None:
                wide_refinement_stats = apply_adult_external_refinement(
                    mask_path=m_path,
                    brain_path=b_path,
                    input_image_path=input_image_path,
                    atlas_path=selected_atlas,
                    metrics_path=metrics_path,
                    config=priors.wide_refinement_config,
                )
                if wide_refinement_stats.get("adult_refinement_used"):
                    print(f"    Raffinement elargi atlas/T1 applique: "
                          f"{wide_refinement_stats.get('adult_refinement_reason')}")
                elif wide_refinement_stats:
                    print(f"    Raffinement elargi atlas/T1 non applique: "
                          f"{wide_refinement_stats.get('adult_refinement_reason')}")

            adaptive_fusion_stats: dict = {}
            if priors.enable_adaptive_mask_fusion and selected_atlas is not None:
                # Enrichir la config avec les stats WhiteStripe (Niveau 1) et age_months (Niveau 2).
                # IsegStyleVolumePriors est frozen=True → copie muable indispensable.
                _fusion_config = dict(priors.adaptive_mask_fusion_config)
                if isinstance(pp, dict):
                    _ws = pp.get("ws_stats") or {}
                    if isinstance(_ws, dict):
                        for _k in ("wm_mean_raw", "gm_mean_raw", "csf_mean_raw", "gm_wm_contrast"):
                            _v = _ws.get(_k)
                            if _v is not None:
                                try:
                                    _fusion_config[_k] = float(_v)
                                except (TypeError, ValueError):
                                    pass
                _age = month_from_session(rec.get("session", ""))
                if _age is not None:
                    _fusion_config["age_months"] = int(_age)

                # Propager le score de sélection atlas comme trust score.
                # Sans cela, reference_atlas_trust_score=0.0 (défaut) déclenche
                # toujours low_trust_fail_clean et bloque la fusion quelle que soit
                # la qualité des candidats tissue.
                _atlas_trust = float(atlas_stats.get("selection_score", 0.0)
                                     if isinstance(atlas_stats, dict) else 0.0)
                if _atlas_trust > 0.0:
                    _fusion_config["reference_atlas_trust_score"] = _atlas_trust

                try:
                    # Si un backup pre-fusion existe (run précédent), restaurer le masque SAM2
                    # original avant de relancer la fusion pour éviter une dérive cumulative.
                    _backup_path = m_path.with_name(
                        m_path.name.replace(".nii.gz", "_pre-adaptive-fusion.nii.gz")
                    )
                    if _backup_path.exists() and not force_extraction:
                        shutil.copyfile(_backup_path, m_path)

                    fusion_atlas_paths = _adaptive_fusion_atlas_paths(
                        anat_dir, subject_id,
                        selected_atlas=selected_atlas,
                        adaptive_path=adaptive_atlas_stats.get("output_path"),
                    )
                    adaptive_ref = adaptive_atlas_stats.get("output_path")
                    if not adaptive_ref or not Path(str(adaptive_ref)).exists():
                        adaptive_ref = selected_atlas
                    adaptive_fusion_stats = apply_adaptive_mask_fusion(
                        mask_path=m_path,
                        brain_path=b_path,
                        input_image_path=input_image_path,
                        atlas_paths=fusion_atlas_paths,
                        reference_atlas_path=adaptive_ref,
                        seed_atlas_path=anat_dir / "atlases_advanced" / f"sub-{subject_id}_brain_mask_aff.nii.gz",
                        uncertainty_path=anat_dir / f"{subject_bids}_desc-brain_uncertainty.nii.gz",
                        t2_path=source_t2,
                        metrics_path=metrics_path,
                        config=_fusion_config,
                    )
                    if adaptive_fusion_stats.get("adaptive_fusion_used"):
                        print(f"    Fusion adaptive finale appliquee: "
                              f"{adaptive_fusion_stats.get('adaptive_fusion_selected_candidate')} "
                              f"({adaptive_fusion_stats.get('adaptive_fusion_reason')})")
                    elif adaptive_fusion_stats:
                        print(f"    Fusion adaptive finale non appliquee: "
                              f"{adaptive_fusion_stats.get('adaptive_fusion_reason')}")
                except Exception as e_fusion:
                    print(f"    Fusion adaptive finale echouee: {e_fusion}")
                    adaptive_fusion_stats = {"adaptive_fusion_reason": str(e_fusion)}

            m = compute_metrics(m_path, gs_mask)
            extract_metrics: dict = {}
            if metrics_path.exists():
                with open(metrics_path) as fm:
                    extract_metrics = json.load(fm)
            quality_rating = extract_metrics.get("quality_rating")
            if pediatric_refinement_stats.get("pediatric_refinement_used"):
                quality_rating = "Good - pediatric T1/T2 refinement applied"
            if wide_refinement_stats.get("adult_refinement_used"):
                quality_rating = "Good - pediatric T1/T2 + wide atlas/T1 refinement applied"
            if adaptive_fusion_stats.get("adaptive_fusion_used"):
                quality_rating = "Good - adaptive mask fusion applied"

            m.update({
                "subject": sid,
                "dataset": spec.dataset,
                "split": rec.get("split"),
                "session": rec.get("session", ""),
                "source_t1_path": str(source_t1),
                "source_t2_path": str(source_t2),
                "gold_mask_path": str(gs_mask),
                "label_path": str(source_label),
                "t2_path": str(source_t2),
                "original_t1_path": str(rec.get("t1_path", "")),
                "original_t2_path": str(rec.get("t2_path", "")),
                "original_gold_mask_path": str(rec.get("gs_mask_path", "")),
                "original_label_path": str(rec.get("label_path", "")),
                "input_shape": img_info.get("shape"),
                "input_zooms": img_info.get("zooms"),
                "input_orientation": img_info.get("orientation"),
                "gold_shape": gs_info.get("shape"),
                "gold_zooms": gs_info.get("zooms"),
                "gold_orientation": gs_info.get("orientation"),
                "input_image_used": str(input_image_path),
                "mask_path": str(m_path),
                "brain_path": str(b_path),
                "atlas_path": str(selected_atlas) if selected_atlas is not None else "",
                "atlas_resolved_t1": str(atlas_t1),
                "atlas_resolved_mask": str(atlas_mask),
                "atlas_guidance_used": bool(atlas_paths_subj),
                "atlas_rejected_count": len(rejected_atlases),
                "atlas_volume_cm3": atlas_stats.get("volume_cm3"),
                "atlas_components": atlas_stats.get("components"),
                "atlas_qc_score": atlas_stats.get("qc_score"),
                "atlas_selection_score": atlas_stats.get("selection_score"),
                "atlas_image_alignment_score": atlas_stats.get("image_alignment_score"),
                "atlas_intensity_ratio": atlas_stats.get("intensity_ratio"),
                "adaptive_atlas_path": adaptive_atlas_stats.get("output_path"),
                "adaptive_atlas_used": adaptive_atlas_stats.get("adaptive_atlas_used"),
                "adaptive_atlas_reason": adaptive_atlas_stats.get("adaptive_atlas_reason"),
                "adaptive_atlas_seed_variant": adaptive_atlas_stats.get("seed_variant"),
                "adaptive_atlas_seed_volume_cm3": adaptive_atlas_stats.get("seed_volume_cm3"),
                "adaptive_atlas_envelope_volume_cm3": adaptive_atlas_stats.get("envelope_volume_cm3"),
                "adaptive_atlas_volume_cm3": adaptive_atlas_stats.get("adaptive_volume_cm3"),
                "adaptive_atlas_added_cm3": adaptive_atlas_stats.get("adaptive_added_cm3"),
                "dice_with_atlas": extract_metrics.get("dice_with_atlas"),
                "atlas_coverage": extract_metrics.get("atlas_coverage"),
                "atlas_precision": extract_metrics.get("atlas_precision"),
                "pre_fallback_dice_with_atlas": extract_metrics.get("pre_fallback_dice_with_atlas"),
                "pre_fallback_atlas_coverage": extract_metrics.get("pre_fallback_atlas_coverage"),
                "fallback_used": extract_metrics.get("fallback_used"),
                "fallback_reason": extract_metrics.get("fallback_reason"),
                "refinement_used": extract_metrics.get("refinement_used", refinement_stats.get("refinement_used")),
                "refinement_reason": extract_metrics.get("refinement_reason", refinement_stats.get("refinement_reason")),
                "boundary_refinement_used": extract_metrics.get("boundary_refinement_used", refinement_stats.get("boundary_refinement_used")),
                "boundary_refinement_reason": extract_metrics.get("boundary_refinement_reason", refinement_stats.get("boundary_refinement_reason")),
                "boundary_refinement_added_cm3": extract_metrics.get("boundary_refinement_added_cm3", refinement_stats.get("boundary_refinement_added_cm3")),
                "pediatric_refinement_used": extract_metrics.get("pediatric_refinement_used", pediatric_refinement_stats.get("pediatric_refinement_used")),
                "pediatric_refinement_reason": extract_metrics.get("pediatric_refinement_reason", pediatric_refinement_stats.get("pediatric_refinement_reason")),
                "pediatric_refinement_added_cm3": extract_metrics.get("pediatric_refinement_added_cm3", pediatric_refinement_stats.get("pediatric_refinement_added_cm3")),
                "iseg_wide_refinement_used": extract_metrics.get("adult_refinement_used", wide_refinement_stats.get("adult_refinement_used")),
                "iseg_wide_refinement_reason": extract_metrics.get("adult_refinement_reason", wide_refinement_stats.get("adult_refinement_reason")),
                "adaptive_fusion_used": extract_metrics.get("adaptive_fusion_used", adaptive_fusion_stats.get("adaptive_fusion_used")),
                "adaptive_fusion_reason": extract_metrics.get("adaptive_fusion_reason", adaptive_fusion_stats.get("adaptive_fusion_reason")),
                "adaptive_fusion_selected_candidate": extract_metrics.get("adaptive_fusion_selected_candidate", adaptive_fusion_stats.get("adaptive_fusion_selected_candidate")),
                "quality_rating": quality_rating,
                "mask_volume_cm3_from_extractor": extract_metrics.get("mask_volume_cm3"),
                "total_time_s": round(time.time() - t0, 1),
                "error": "",
            })

        except Exception as e:
            traceback.print_exc()
            m = {
                "subject": sid,
                "dataset": spec.dataset,
                "split": rec.get("split"),
                "session": rec.get("session", ""),
                "source_t1_path": str(rec.get("t1_path", "")),
                "gold_mask_path": str(rec.get("gs_mask_path", "")),
                "t2_path": str(rec.get("t2_path", "")),
                "label_path": str(rec.get("label_path", "")),
                "error": str(e),
                "dice": float("nan"),
                "hd95": float("nan"),
                "assd": float("nan"),
                "total_time_s": round(time.time() - t0, 1),
            }

        results.append(m)
        _write_progress(progress_json, live_csv, spec.dataset, "subject_finished",
                        sid, idx, len(selected), results)
        d = m.get("dice", float("nan"))
        h = m.get("hd95", float("nan"))
        err = m.get("error", "")
        if err:
            print(f"  -> ERREUR: {err}")
        else:
            print(f"  -> Dice={d:.4f}  HD95={h:.2f} mm  Vol={m.get('vol_pred_cm3', float('nan')):.1f} cm3  "
                  f"GS={m.get('vol_gs_cm3', float('nan')):.1f} cm3  AtlasVol={m.get('atlas_volume_cm3', float('nan')):.1f} cm3  "
                  f"PedAdd={m.get('pediatric_refinement_added_cm3', 0) or 0:.1f} cm3  "
                  f"Wide={m.get('iseg_wide_refinement_used', False)}  t={m.get('total_time_s', 0):.0f}s")

    df = pd.DataFrame(results)
    df.to_csv(final_csv, index=False)
    _write_progress(progress_json, live_csv, spec.dataset, "finished",
                    total=len(selected), rows=results)

    df_ok = df[df.get("error", "").fillna("").astype(str).str.len() == 0].copy() if "error" in df.columns else df.copy()
    df_ok["dice"] = pd.to_numeric(df_ok.get("dice"), errors="coerce")
    df_ok = df_ok.dropna(subset=["dice"])
    print(f"\n=== Resultats M1-{spec.dataset} (N={len(df_ok)}/{len(selected)} sujets) ===")
    if len(df_ok):
        cols = [c for c in ["subject", "dice", "hd95", "assd", "vol_pred_cm3", "vol_gs_cm3",
                            "atlas_volume_cm3", "atlas_selection_score", "refinement_used",
                            "pediatric_refinement_used", "pediatric_refinement_added_cm3",
                            "iseg_wide_refinement_used", "fallback_used", "adaptive_fusion_used",
                            "adaptive_fusion_selected_candidate", "quality_rating"]
                if c in df_ok.columns]
        print(df_ok[cols].to_string(index=False))
        print(f"\nDice  : {df_ok.dice.mean():.4f} +- {df_ok.dice.std():.4f}")
        print(f"HD95  : {df_ok.hd95.mean():.2f} +- {df_ok.hd95.std():.2f} mm")
        print(f"ASSD  : {df_ok.assd.mean():.2f} +- {df_ok.assd.std():.2f} mm")
        print(f"Volume predit : {df_ok.vol_pred_cm3.mean():.1f} +- {df_ok.vol_pred_cm3.std():.1f} cm3")
        print(f"Volume gold   : {df_ok.vol_gs_cm3.mean():.1f} +- {df_ok.vol_gs_cm3.std():.1f} cm3")
    print(f"\nResultats sauvegardes: {final_csv}")
    print(f"Suivi live: {live_csv} | {progress_json}")

    return {
        "dataset": spec.dataset,
        "out_base": str(out_base),
        "final_csv": str(final_csv),
        "results": results,
        "df": df,
        "df_ok": df_ok,
        "n_selected": len(selected),
        "mean_dice": float(df_ok.dice.mean()) if len(df_ok) else float("nan"),
    }


__all__ = [
    "IsegStyleVolumePriors",
    "IsegStyleDatasetSpec",
    "ISEG_DEFAULT_PRIORS",
    "BOBS_DEFAULT_PRIORS",
    "MCRIB_DEFAULT_PRIORS",
    "MCRIB_WM_DEFAULT_PRIORS",
    "ATLAS_BY_AGE_INFANT",
    "fixed_atlas_resolver",
    "external_atlas_by_age_resolver",
    "loo_atlas_resolver",
    "iseg_records_resolver",
    "bids_gold_records_resolver",
    "run_iseg_style_pipeline",
]
