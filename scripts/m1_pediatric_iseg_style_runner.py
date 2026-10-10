"""Scalable pediatric M1 runner that mirrors the M1-iSeg2017 architecture.

The runner is intentionally called from the notebook with ``globals()``.  It
uses the same notebook primitives as the iSeg2017 cell:
preprocessing, advanced atlas registration, SegmentationParams,
SAM2BrainExtractor, and compute_metrics.  Gold masks are carried only for final
post-run metrics and are never passed to preprocessing, atlas registration,
SAM2 guidance, fallback, refinement, or fusion.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi

from .adaptive_mask_fusion import apply_adaptive_mask_fusion, _NORMATIVE_BRAIN_VOLUME_CM3
from .adult_external_refinement import apply_adult_external_refinement
from .external_atlas_config import default_output_slug, external_atlas_files_for_row, month_from_session, no_atlas_ablation, profile_for_row
from .iseg2017_adaptive_atlas import build_iseg_adaptive_guidance_atlas
from .iseg2017_robust_refinement import apply_pediatric_t2_refinement, robust_atlas_image_alignment_score
from .m1_external_adaptive_runner import compute_metrics as _fallback_compute_metrics


REQUIRED_NOTEBOOK_SYMBOLS = (
    "run_preprocessing_pipeline",
    "advanced_atlas_registration",
    "SegmentationParams",
    "SAM2BrainExtractor",
)


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
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "oui"}


def _cell(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value)


def _safe_subject(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _natural_key(value: str | Path) -> list[Any]:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(value))]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


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


def _set_seed(seed: int | None, *, deterministic_threads: bool | None = None) -> None:
    """Setup déterministe du runner pédiatrique.

    Auparavant ce helper ne posait que les graines (manual_seed) SANS forcer le
    déterminisme cuDNN. Résultat : un bruit run-to-run ~0.05 Dice (SAM2 cuDNN +
    réductions multi-thread d'ANTs sur ~20 recalages/sujet) qui DÉPASSE les marges
    SOTA marginales → comparaisons non régressives faussées.

    On force désormais le déterminisme **cuDNN** (coût quasi-nul, supprime la
    variance GPU de SAM2). Le mono-threading ITK/OMP (déterminisme ANTs) est
    OPT-IN (`deterministic_threads=True` ou env `SAM_NEUROADAPT_DETERMINISTIC_THREADS=1`)
    car il ralentit ANTs de façon prohibitive (~4-9×, parfois pathologique).
    """
    if seed is None:
        return
    if deterministic_threads is None:
        deterministic_threads = os.environ.get(
            "SAM_NEUROADAPT_DETERMINISTIC_THREADS", ""
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    os.environ.setdefault("PYTHONHASHSEED", str(int(seed)))
    # ANTs : la métrique (Mattes MI) échantillonne des voxels au hasard → sans
    # graine fixe, chaque registration diffère → sélection d'atlas différente →
    # gros swing Dice (jusqu'à 0.14 observé). antspyx IGNORE la variable d'env
    # ANTS_RANDOM_SEED ; le bon mécanisme (utilisé par le runner adulte) est le
    # réglage GLOBAL `ants.config._random_seed`. On pose les deux par robustesse.
    os.environ["ANTS_RANDOM_SEED"] = str(int(seed))
    try:
        import ants  # type: ignore

        ants.config._random_seed = int(seed)
    except Exception:
        pass
    if deterministic_threads:
        # ANTs/ITK : réductions sensibles à l'ordonnancement des threads. Mono-
        # threader rend la registration reproductible MAIS très lent → opt-in.
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        # Déterminisme cuDNN pour SAM2 (coût quasi-nul, supprime la variance GPU).
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
    except Exception:
        pass


def _notebook_api(notebook_globals: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in REQUIRED_NOTEBOOK_SYMBOLS if name not in notebook_globals]
    if missing:
        raise RuntimeError("Notebook prerequisites missing: " + ", ".join(missing))
    compute = notebook_globals.get("compute_metrics") or _fallback_compute_metrics
    return {
        "run_preprocessing_pipeline": notebook_globals["run_preprocessing_pipeline"],
        "advanced_atlas_registration": notebook_globals["advanced_atlas_registration"],
        "SegmentationParams": notebook_globals["SegmentationParams"],
        "SAM2BrainExtractor": notebook_globals["SAM2BrainExtractor"],
        "compute_metrics": compute,
    }


def _existing_path(row: Mapping[str, Any], keys: tuple[str, ...]) -> Path | None:
    for key in keys:
        value = _cell(row, key, "")
        if value and value.lower() != "nan":
            path = Path(value)
            if path.exists():
                return path
    return None


def _manifest_path(dataset: str, manifest: str | Path | None) -> Path:
    if manifest:
        return Path(manifest)
    path = Path("datasets/manifests") / f"{dataset}.csv"
    if not path.exists() and str(dataset).upper() == "MCRIB2":
        alias = Path("datasets/manifests/MCRIB_WM.csv")
        if alias.exists():
            return alias
    return path


def _session_month(row: Mapping[str, Any]) -> int | None:
    month = month_from_session(_cell(row, "session", ""))
    if month is not None:
        return int(month)
    notes = _cell(row, "notes", "")
    match = re.search(r"(\d+)\s*mo", notes, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _apply_normative_volume_bounds(cfg_out: dict[str, Any], age_months: float | int | None) -> None:
    """PATCH P12 — Dérive les bornes de volume de la fusion adaptative depuis la
    table normative pédiatrique, par interpolation mensuelle.

    Réutilisable : appelé (a) à la construction de la config quand l'âge vient de
    la session, et (b) APRÈS inférence d'âge par atlas pour les datasets sans âge
    (ex. SynthStripInfant) afin de réactiver les priors volumétriques.
    """
    if age_months is None:
        return
    ages = sorted(_NORMATIVE_BRAIN_VOLUME_CM3.keys())
    lo = max((a for a in ages if a <= age_months), default=ages[0])
    hi = min((a for a in ages if a >= age_months), default=ages[-1])
    if lo == hi:
        nmin, nmax = _NORMATIVE_BRAIN_VOLUME_CM3[lo]
    else:
        t = (age_months - lo) / (hi - lo)
        lo_min, lo_max = _NORMATIVE_BRAIN_VOLUME_CM3[lo]
        hi_min, hi_max = _NORMATIVE_BRAIN_VOLUME_CM3[hi]
        nmin = lo_min + t * (hi_min - lo_min)
        nmax = lo_max + t * (hi_max - lo_max)
    af = cfg_out.get("adaptive_fusion")
    if not isinstance(af, dict):
        return
    af["soft_min_volume_cm3"] = nmin * 0.90
    af["soft_max_volume_cm3"] = nmax * 1.10
    af["min_volume_cm3"]      = nmin * 0.70
    af["max_volume_cm3"]      = nmax * 1.35


def _base_config_for_record(dataset: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """No-gold config from dataset identity/session only, not GT values."""

    month = _session_month(row)
    ds = str(dataset)
    fetal = ds == "FetalFMRI" or _cell(row, "profile") == "fetal_fmri"
    if fetal:
        # Fetal functional EPI (96x96x37 BOLD): tiny brains, bornes calibrées sur
        # la distribution réelle des masques (min 48, p5 140, médiane 292, p95 429,
        # max 572 cm3). Guidage par atlas néonatal, dégradation gracieuse si recalage raté.
        min_cm3, soft_min, soft_max, max_cm3 = 40.0, 120.0, 450.0, 700.0
        atlas_min, atlas_max = 30.0, 750.0
        target_max, max_added, ped_added = 500.0, 120.0, 120.0
    elif ds == "iSeg2017":
        min_cm3, soft_min, soft_max, max_cm3 = 550.0, 720.0, 900.0, 1100.0
        atlas_min, atlas_max = 550.0, 1100.0
        target_max, max_added, ped_added = 980.0, 260.0, 500.0
    elif ds in {"MCRIB", "MCRIB_WM", "MCRIB2", "ALBERTs", "NeoBrainS12", "dHCPNeonatal"} or (month is not None and month <= 2):
        min_cm3, soft_min, soft_max, max_cm3 = 220.0, 340.0, 620.0, 850.0
        atlas_min, atlas_max = 180.0, 900.0
        target_max, max_added, ped_added = 650.0, 180.0, 180.0
    elif month is not None and month < 8:
        min_cm3, soft_min, soft_max, max_cm3 = 300.0, 520.0, 980.0, 1250.0
        atlas_min, atlas_max = 240.0, 1300.0
        target_max, max_added, ped_added = 1050.0, 260.0, 260.0
    elif month is not None and month < 18:
        min_cm3, soft_min, soft_max, max_cm3 = 450.0, 700.0, 1200.0, 1500.0
        atlas_min, atlas_max = 350.0, 1550.0
        target_max, max_added, ped_added = 1300.0, 300.0, 300.0
    else:
        min_cm3, soft_min, soft_max, max_cm3 = 250.0, 550.0, 1150.0, 1500.0
        atlas_min, atlas_max = 220.0, 1550.0
        target_max, max_added, ped_added = 1250.0, 320.0, 320.0

    neonatal = fetal or ds in {"MCRIB", "MCRIB_WM", "MCRIB2", "ALBERTs", "NeoBrainS12", "dHCPNeonatal"} or (
        month is not None and month <= 2
    )
    young_infant = not neonatal and (month is not None and month < 8)
    _cfg_out = {
        "dataset": ds,
        "month": month,
        # Réutilisation cache pour économiser ~2-3 min/sujet sur les re-runs
        # (preprocessing N4/WhiteStripe + recalage atlas multi-candidats sont coûteux)
        "force_source_conversion": False,
        "force_preprocessing": False,
        "force_atlas_registration": False,
        "force_extraction": True,  # SAM2 doit toujours refaire (paramètres changent)
        # Idempotence des refinements amont : snapshot/restore du masque SAM2 brut et
        # des états intermédiaires pour que des runs successifs soient déterministes
        # (cf. plan idempotence-des-refinements-amont). False = ancien comportement.
        "refinement_idempotent_mode": True,
        "clean_stale_outputs": False,
        "delete_short_stale_atlases": True,
        # Fœtal : si aucun atlas néonatal ne se recale sur l'EPI basse résolution,
        # ne pas lever d'erreur — basculer en SAM2-seul pour cette frame (zéro crash).
        "stop_if_no_valid_atlas": not fetal,
        "spacing_override": None,
        "mask_min_cm3": min_cm3,
        "mask_max_cm3": max_cm3,
        "mask_max_components": 4,
        "mask_max_extent": 0.97,
        "atlas_qc_min_cm3": atlas_min,
        "atlas_qc_max_cm3": atlas_max,
        "atlas_max_components": 3,
        "atlas_nomask_bonus": 0.03 if not (neonatal or young_infant) else 0.01,
        "atlas_family_bonus": (
            {
                "unc_bcp_06m": 0.06,
                "alberts_gousias": 0.05,  # ~348 cm³, age-matched for neonatal
                "fbn125_neonatal": -0.04,  # 756 cm³, 2× too large for neonatal subjects
            }
            if neonatal
            else {}
        ),
        "atlas_nomask_penalty_families": {"fbn125_neonatal": 0.18} if neonatal else {},
        "min_atlas_image_score": 0.03 if neonatal else 0.05,
        "min_atlas_selection_score": 0.18 if neonatal else 0.25,
        "min_brain_volume_cm3": min_cm3,
        "max_brain_volume_cm3": max_cm3,
        "atlas_fallback_min_cm3": min_cm3,
        "atlas_fallback_max_cm3": max_cm3,
        "atlas_min_coverage": 0.78 if neonatal else (0.80 if young_infant else 0.85),
        "atlas_min_precision": 0.88 if neonatal else 0.90,
        "enable_atlas_fallback": False,
        "atlas_weight": 0.75,
        "confidence_default": 0.70 if neonatal else 0.72,
        # SAM2 IDÉE 1/2/4 — activées pour pédiatrique (neonatal + young_infant)
        # pour éliminer oversegmentation skull/scalp via masquage atlas amont
        "sam2_enable_atlas_input_masking": bool(neonatal or young_infant),
        "sam2_atlas_mask_hard_radius_mm": 10.0 if neonatal else 12.0,
        "sam2_atlas_mask_soft_radius_mm": 2.5 if neonatal else 3.0,
        "sam2_atlas_mask_min_volume_cm3": 150.0 if neonatal else 200.0,
        "sam2_atlas_mask_background_value": 0.0,
        "sam2_eap_tight_always_initial": bool(neonatal or young_infant),
        "sam2_eap_tight_min_preservation": 0.20,
        "sam2_adaptive_atlas_shrink_enabled": bool(neonatal or young_infant),
        # Trigger 1.40 / target 1.15 — valeurs validées vague 1+2.
        # Test trigger 1.05 / target 1.00 a causé sur-érosion (sub-116056 -0.08,
        # sub-284149 -0.07) car expected_volume table moyenne ≠ GS individuel.
        "sam2_adaptive_atlas_shrink_target_ratio": 1.15,
        "sam2_adaptive_atlas_shrink_trigger_ratio": 1.40,
        "sam2_adaptive_atlas_shrink_max_iters": 10,
        # Vague 2 — IDÉE 3 (mask_input atlas) + early stopping
        # IDÉE 3 active sur pédiatrique : guide SAM2 explicitement avec prior atlas low-res
        "sam2_enable_atlas_mask_input_prompt": bool(neonatal or young_infant),
        "sam2_mask_input_strength": 10.0,
        "sam2_mask_input_use_eap": True,
        # Early stopping : prévient dérive de propagation vers scalp
        "sam2_enable_volume_early_stopping": bool(neonatal or young_infant),
        "sam2_volume_early_stop_ratio": 1.20,
        # Vague 3 — bidirectional consensus : DÉSACTIVÉ par défaut (régression observée).
        # Re-propagation inverse pollue l'état SAM2 predictor et coûte 3× le temps.
        # Conservé comme feature opt-in pour debug/recherche future.
        "sam2_enable_bidirectional_consensus": False,
        "sam2_bidirectional_consensus_min_dice": 0.80,
        "sam2_bidirectional_consensus_min_range": 6,
        # P1.1 — Multi-point SAM2 prompting (points pos+neg depuis atlas adaptatif)
        # Activé pour néonataux et jeunes nourrissons (gain attendu +3–6% Dice vs bbox seul)
        "sam2_multipoint_enabled": bool(neonatal or young_infant),
        "sam2_multipoint_n_positive": 8,
        "sam2_multipoint_n_negative": 5,
        "sam2_multipoint_seed": 42,
        "enable_atlas_refinement": True,
        "refinement_atlas_dilation": 1,
        "refinement_min_keep_ratio": 0.78 if neonatal else (0.81 if young_infant else 0.85),
        "refinement_min_atlas_coverage": 0.70 if neonatal else (0.75 if young_infant else 0.85),
        "refinement_min_precision_gain": 0.004 if neonatal else (0.006 if young_infant else 0.01),
        "enable_boundary_refinement": True,
        "boundary_refinement_width": 1,
        "boundary_atlas_dilation": 1,
        "boundary_intensity_percentile": 6.0 if neonatal else 8.0,
        "boundary_max_added_cm3": 25.0 if neonatal else 35.0,
        "boundary_max_volume_ratio": 1.08 if neonatal else 1.06,
        "boundary_min_atlas_precision": 0.86 if neonatal else (0.88 if young_infant else 0.92),
        "enable_pediatric_t2_refinement": True,
        "pediatric_t2_shell_width": 16 if neonatal else 20,
        "pediatric_t2_atlas_dilation": 12 if neonatal else 15,
        "pediatric_t2_t1_percentile": 3.0,
        "pediatric_t2_t2_percentile": 3.0,
        "pediatric_t2_max_added_cm3": ped_added,
        "pediatric_t2_min_atlas_precision": 0.70 if neonatal else 0.75,
        "enable_wide_refinement": True,
        "wide_refinement": {
            "enabled": True,
            "min_volume_cm3": min_cm3,
            "max_volume_cm3": max_cm3,
            "prune_enabled": True,
            "prune_atlas_dilation": 2,
            "prune_if_atlas_precision_below": 0.74 if neonatal else 0.78,
            "prune_if_volume_ratio_above": 1.18 if neonatal else 1.22,
            "prune_min_keep_ratio": 0.76 if neonatal else 0.82,
            "recover_enabled": True,
            "recover_shell_width": 4,
            "recover_atlas_dilation": 4,
            "recover_intensity_percentile": 3.0,
            "recover_max_added_cm3": 80.0 if neonatal else 160.0,
            "recover_max_volume_ratio": 1.15 if neonatal else 1.16,
            "recover_min_atlas_precision": 0.70 if neonatal else 0.74,
            "min_atlas_dice_gain": 0.001,
            "backup_suffix": "_pre-pediatric-wide-refine",
        },
        "enable_adaptive_guidance_atlas": True,
        "adaptive_guidance": {
            "min_volume_cm3": atlas_min,
            "max_volume_cm3": atlas_max,
            "seed_prefer": "aff",
            "shell_width": 18 if neonatal else 24,
            "envelope_dilation": 3,
            "t1_percentile": 3.0,
            "t2_percentile": 3.0,
            "target_from_envelope_ratio": 0.88 if neonatal else 0.90,
            "target_seed_gain_ratio": 1.10 if neonatal else 1.12,
            "target_max_cm3": target_max,
            "max_added_cm3": max_added,
            "min_added_cm3": 3.0 if neonatal else 5.0,
        },
        # PATCH P4 — la fusion est réactivée partout: elle peut guider la sortie via
        # current_pruned_to_atlas / current_shell_recovery (toujours SAM2-derived).
        # La sortie finale ne peut JAMAIS être un candidat atlas pur (atlas_tissue_p*,
        # atlas_majority, reference_atlas) grâce à sam2_only_output=True dans
        # adaptive_mask_fusion._cfg().
        "enable_adaptive_mask_fusion": True,
        "adaptive_fusion": {
            "enabled": True,
            "min_volume_cm3": min_cm3,
            "soft_min_volume_cm3": soft_min,
            "soft_max_volume_cm3": soft_max,
            "max_volume_cm3": max_cm3,
            "tissue_percentiles": [6.0, 8.0, 10.0, 12.0, 15.0, 18.0],
            "use_t2_gate": False,
            "coverage_target": 0.78 if neonatal else 0.84,
            "coverage_sigma": 0.15 if neonatal else 0.13,
            "catastrophic_min_precision": 0.48,
            "catastrophic_min_coverage": 0.40 if neonatal else 0.45,
            "catastrophic_min_atlas_dice": 0.48 if neonatal else 0.55,
            "catastrophic_max_uncertainty_mean": 0.28,
            "undercoverage_min_precision": 0.94 if neonatal else 0.96,
            "undercoverage_max_coverage": 0.80 if neonatal else 0.82,
            "undercoverage_max_volume_ratio": 0.86,
            # PATCH P1 — bonus 0.08→0.20 : la fusion privilégiait atlas_tissue_p* (aligné atlas)
            # alors que current (SAM2 direct) matche mieux le GT. Analyse MCRIB : 14/19 sujets
            # remplaçaient current par atlas_tissue → Dice 0.92 au lieu de 0.95. Patch global.
            "current_mask_bonus": 0.20,
            "tissue_undercoverage_bonus": 0.07,
            "selection_margin": 0.015,
            "backup_suffix": "_pre-adaptive-fusion",
            # PATCH P9 — prior volumétrique normatif pédiatrique.
            # Activé pour tous les sujets pédiatriques ; l'âge est transmis
            # au moment de l'appel (subject_age_months = cfg["month"]).
            "normative_volume_prior_enabled": True,
            # PATCH P15 — pénalité sigmoïde renforcée pour battre current_mask_bonus=0.20
            # dès un excès volume normatif ≥ 12 %.
            "normative_volume_prior_max_penalty": 0.40,
            # PATCH P9 / P1.3 — intensité T2 informative pour les nourrissons.
            # En T2 pédiatrique, le ratio de contraste frontière (intérieur/coque)
            # est un bon discriminateur entre masque correct et surestimation
            # (scalp/LCR brillant inclus). Poids modéré (0.03) pour éviter
            # tout impact fort sur l'iSeg-2017 où le T1 est la modalité primaire.
            "intensity_weight": 0.03 if (neonatal or young_infant) else 0.0,
            # PATCH P14 — contraste frontière : pénalise les masques qui incluent
            # du scalp/LCR externe (ratio intérieur/coque ≈ 1). Activé sur nourrissons.
            "boundary_contrast_weight": 0.05 if (neonatal or young_infant) else 0.0,
            # PATCH P16 — random walker T1+T2 (Grady IEEE TPAMI 2006) comme candidat
            # supplémentaire. Active uniquement profil pédiatrique avec T2 disponible
            # (BOBs, MCRIB). iSeg-2017 saute automatiquement (pas de T2).
            "rw_candidate_enabled": bool(neonatal or young_infant),
            "rw_candidate_require_t2": True,
            # PATCH P17 — Chan-Vese vectoriel : actif pédiatrique. Complète RW
            # sur les zones isointenses 6-9 mois (gradient faible).
            "cv_candidate_enabled": bool(neonatal or young_infant),
            "cv_candidate_require_t2": False,
        },
    }
    _cfg_out["is_fetal"] = fetal
    if fetal:
        # PATCH P19 — prior volumétrique fœtal explicite : l'atlas néonatal (le plus
        # petit disponible) reste ~1.5× trop grand pour le cerveau fœtal (GT médian
        # ~292 cm³, p95 ~429). Ces bornes activent la pénalité de sur-segmentation
        # de la fusion indépendamment de l'âge (l'atlas néonatal donnerait 310-540).
        _aff = _cfg_out["adaptive_fusion"]
        _aff["normative_volume_prior_bounds_cm3"] = (180.0, 420.0)
        _aff["soft_max_volume_cm3"] = min(float(_aff.get("soft_max_volume_cm3", 450.0)), 400.0)
    # PATCH P12 — bornes volumétriques normatives (interpolation mensuelle).
    # Extrait en helper réutilisable pour pouvoir le ré-appliquer après inférence
    # d'âge par atlas (datasets sans âge de session). No-op si month is None.
    _apply_normative_volume_bounds(_cfg_out, month)
    return _cfg_out


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
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    wanted = set(str(s) for s in subjects) if subjects else None
    records: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        rd = row.to_dict()
        subject = _cell(rd, "subject", "")
        if not subject:
            continue
        if wanted and subject not in wanted:
            continue
        if profile_filter and _cell(rd, "profile", "") != profile_filter:
            continue
        if not _bool(rd.get("has_gold", True), True) or _bool(rd.get("qc_only", False), False):
            skipped.append((subject, "not_gold_or_qc_only"))
            continue
        t1_path = _existing_path(rd, ("input_path", "t1_path", "source_t1_path"))
        t2_path = _existing_path(rd, ("t2_path", "source_t2_path"))
        gold_path = _existing_path(rd, ("brain_mask_path", "gold_mask_path", "mask_path"))
        label_path = _existing_path(rd, ("label_path",))
        if t1_path is None or gold_path is None:
            skipped.append((subject, "missing_t1_or_gold"))
            continue
        profile = profile_filter or profile_for_row(dataset, rd)
        atlas_paths = external_atlas_files_for_row(dataset, rd, profile_filter=profile)
        if require_atlas and not atlas_paths:
            skipped.append((subject, "missing_external_atlas"))
            continue
        records.append(
            {
                "dataset": dataset,
                "subject": subject,
                "session": _cell(rd, "session", ""),
                "split": _cell(rd, "split", ""),
                "profile": profile,
                "row": rd,
                "t1_path": t1_path,
                "t2_path": t2_path,
                "gold_path": gold_path,
                "label_path": label_path,
                "atlas_paths": atlas_paths,
                "config": _base_config_for_record(dataset, rd),
            }
        )
    if wanted:
        found = {record["subject"] for record in records}
        missing = sorted(wanted - found)
        if missing:
            raise RuntimeError(f"Requested subjects absent or unusable in {manifest_path}: {missing}")
    if n_subjects is not None:
        records = records[: int(n_subjects)]
    if not records:
        raise RuntimeError(f"No gold records for {dataset}. Skips={skipped[:10]}")
    return records


def _save_nifti_3d(src_path: Path, dst_path: Path, *, is_mask: bool = False, spacing_override=None, force: bool = True) -> Path:
    if dst_path.exists() and not force:
        return dst_path
    img = nib.load(str(src_path))
    data = np.asanyarray(img.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise RuntimeError(f"Not 3D after squeeze: {src_path} shape={data.shape}")
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


def _prepare_source_images(record: Mapping[str, Any], anat_dir: Path, subject_id: str, cfg: Mapping[str, Any]):
    source_dir = anat_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    force = bool(cfg.get("force_source_conversion", True))
    spacing = cfg.get("spacing_override")
    t1_out = _save_nifti_3d(Path(record["t1_path"]), source_dir / f"sub-{subject_id}.nii.gz", is_mask=False, spacing_override=spacing, force=force)
    t2_path = record.get("t2_path")
    t2_out = None
    if t2_path:
        t2_out = _save_nifti_3d(Path(t2_path), source_dir / f"sub-{subject_id}_T2w.nii.gz", is_mask=False, spacing_override=spacing, force=force)
    mask_out = _save_nifti_3d(
        Path(record["gold_path"]),
        source_dir / f"sub-{subject_id}_gold-brain_mask.nii.gz",
        is_mask=True,
        spacing_override=spacing,
        force=force,
    )
    label_out = None
    if record.get("label_path"):
        label_out = _save_nifti_3d(
            Path(record["label_path"]),
            source_dir / f"sub-{subject_id}_desc-label_seg.nii.gz",
            is_mask=False,
            spacing_override=spacing,
            force=force,
        )
    return t1_out, mask_out, t2_out, label_out


def _image_info(path: Path) -> dict[str, Any]:
    img = nib.load(str(path))
    return {
        "shape": "x".join(str(int(v)) for v in img.shape[:3]),
        "zooms": "x".join(f"{float(v):g}" for v in img.header.get_zooms()[:3]),
        "orientation": "".join(nib.aff2axcodes(img.affine)),
    }


def _mask_quality(path: Path, cfg: Mapping[str, Any], *, atlas: bool = False):
    min_cm3 = float(cfg["atlas_qc_min_cm3"] if atlas else cfg["mask_min_cm3"])
    max_cm3 = float(cfg["atlas_qc_max_cm3"] if atlas else cfg["mask_max_cm3"])
    max_components = int(cfg["atlas_max_components"] if atlas else cfg["mask_max_components"])
    max_extent_ratio = float(cfg.get("mask_max_extent", 0.97))
    try:
        img = nib.load(str(path))
        mask = img.get_fdata(dtype=np.float32) > 0.5
        vox = float(np.prod(img.header.get_zooms()[:3]))
        vol_cm3 = float(mask.sum() * vox / 1000.0)
        lbl, ncomp = ndi.label(mask)
        reason = []
        if vol_cm3 < min_cm3 or vol_cm3 > max_cm3:
            reason.append(f"volume={vol_cm3:.1f} cm3 outside [{min_cm3:.0f}, {max_cm3:.0f}]")
        if ncomp > max_components:
            reason.append(f"components={ncomp} > {max_components}")
        if mask.any():
            pts = np.where(mask)
            bbox = [(int(p.min()), int(p.max())) for p in pts]
            extent_ratio = max((b - a + 1) / s for (a, b), s in zip(bbox, mask.shape))
            centroid = tuple(float(p.mean() / s) for p, s in zip(pts, mask.shape))
            if extent_ratio > max_extent_ratio:
                reason.append(f"bbox_extent={extent_ratio:.1%} > {max_extent_ratio:.0%}")
        else:
            bbox = None
            extent_ratio = 0.0
            centroid = None
            reason.append("empty_mask")
        center = (min_cm3 + max_cm3) / 2.0
        half = max(1.0, (max_cm3 - min_cm3) / 2.0)
        volume_score = max(0.0, 1.0 - abs(vol_cm3 - center) / half)
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
    except Exception as exc:
        return False, {}, f"mask_read_error={exc}"


def _clean_subject_artifacts(anat_dir: Path, subject_id: str, cfg: Mapping[str, Any]) -> None:
    if not cfg.get("clean_stale_outputs", True):
        return
    patterns = [
        f"sub-{subject_id}_desc-brain*.nii.gz",
        f"sub-{subject_id}_desc-uncertainty*.nii.gz",
        f"sub-{subject_id}_desc-brain_metrics.json",
    ]
    atlas_dir = anat_dir / "atlases_advanced"
    if cfg.get("force_atlas_registration", True) and atlas_dir.exists():
        patterns += [f"atlases_advanced/sub-{subject_id}_brain_mask*.nii.gz"]
    deleted = []
    if cfg.get("force_extraction", True):
        for pattern in patterns:
            for path in anat_dir.glob(pattern):
                try:
                    path.unlink()
                    deleted.append(path.name)
                except FileNotFoundError:
                    pass
    if cfg.get("delete_short_stale_atlases", True) and atlas_dir.exists():
        for path in atlas_dir.glob(f"sub-{subject_id}_brain_mask*.nii.gz"):
            try:
                if path.stat().st_size < 4096:
                    path.unlink()
                    deleted.append(path.name)
            except FileNotFoundError:
                pass
    if deleted:
        print(f"    anti-stale cleanup: {len(deleted)} files removed")


def _select_preproc_mask(anat_dir: Path, subject_id: str, cfg: Mapping[str, Any]) -> Path | None:
    candidates = []
    adv_dir = anat_dir / "atlases_advanced"
    if adv_dir.exists():
        candidates.extend(sorted(adv_dir.glob(f"sub-{subject_id}_brain_mask*.nii.gz"), key=_natural_key))
    rejected = []
    for cand in candidates:
        ok, stats, why = _mask_quality(cand, cfg, atlas=False)
        if ok:
            print(f"    preprocessing mask accepted: {cand.name} ({stats['volume_cm3']:.1f} cm3)")
            return cand
        rejected.append((cand.name, why))
    if rejected:
        print("    preprocessing masks rejected:")
        for name, why in rejected[:5]:
            print(f"      - {name}: {why}")
    return None


def _registered_atlas_candidates(anat_dir: Path, subject_id: str) -> list[Path]:
    adv_dir = anat_dir / "atlases_advanced"
    paths = [adv_dir / f"sub-{subject_id}_brain_mask_adaptive.nii.gz"]
    paths.extend(sorted(adv_dir.glob(f"sub-{subject_id}_brain_mask*_adaptive.nii.gz"), key=_natural_key))
    for suffix in ("aff_nomask", "aff", "syn"):
        paths.extend(sorted(adv_dir.glob(f"sub-{subject_id}_brain_mask*_{suffix}.nii.gz"), key=_natural_key))
    out, seen = [], set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _adaptive_fusion_atlas_paths(anat_dir: Path, subject_id: str, selected_atlas=None, adaptive_path=None) -> list[Path]:
    paths = [selected_atlas, adaptive_path]
    paths.extend(_registered_atlas_candidates(anat_dir, subject_id))
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


_MNIINFANT_COHORT_MIDPOINTS: dict[int, float] = {
    1: 1.0, 2: 3.5, 3: 6.5, 4: 9.5, 5: 12.5,
    6: 15.5, 7: 19.0, 8: 24.0, 9: 30.0, 10: 38.5, 11: 52.0,
}


def _atlas_age_months(atlas_path: Path) -> float | None:
    """Extraire l'âge nominal (mois postnatals) depuis le nom de l'atlas.

    Reconnaît les âges réels de chaque banque pour l'appariement atlas↔sujet par
    âge (P1.2). dHCP par semaine de gestation → mois postnatals (terme 40 SA = 0,
    prématuré < 40 SA = âge négatif).
    """
    name = atlas_path.name.lower()
    m = re.search(r"mniinfant_c(\d+)", name)
    if m:
        return _MNIINFANT_COHORT_MIDPOINTS.get(int(m.group(1)))
    # UNC-BCP bins mensuels (unc_bcp_01m … unc_bcp_24m) — AVANT le générique unc_bcp.
    m = re.search(r"unc_bcp_(\d+)m", name)
    if m:
        return float(int(m.group(1)))
    # dHCP neonatal par semaine de gestation (ga36 … ga44).
    m = re.search(r"dhcp_neonatal_ga(\d+)", name)
    if m:
        return (float(int(m.group(1))) - 40.0) / 4.345
    if "neonatal" in name or "_neo" in name:
        return 0.0
    if "unc_bcp" in name:
        return 6.0
    if "1yr" in name:
        return 12.0
    if "2yr" in name:
        return 24.0
    return None


def _atlas_age_weight(atlas_age: float | None, subject_age: float | None) -> float:
    """Poids gaussien P1.2 : exp(-(diff_age)^2 / (2*sigma^2)).

    sigma = 1.5 mois pour 0-6 mois (grand écart = atlas mal calibré),
             3.0 mois pour 6-18 mois, 6.0 mois au-delà.
    Borné à [0.60, 1.0] pour ne pas éliminer des atlases autrement bons.
    """
    if atlas_age is None or subject_age is None:
        return 1.0
    diff = abs(float(atlas_age) - float(subject_age))
    age_s = float(subject_age)
    sigma = 1.5 if age_s <= 6.0 else (3.0 if age_s <= 18.0 else 6.0)
    w = math.exp(-(diff ** 2) / (2.0 * sigma ** 2))
    return float(max(0.60, w))


def _atlas_variant_bonus(atlas_path: Path, cfg: Mapping[str, Any]) -> float:
    name = atlas_path.name.lower()
    bonus = 0.0
    if "_adaptive" in name:
        bonus += 0.12
    if "_aff_nomask" in name:
        bonus += float(cfg.get("atlas_nomask_bonus", 0.03))
    for token, value in dict(cfg.get("atlas_family_bonus", {})).items():
        if str(token).lower() in name:
            bonus += float(value)
    if "_nomask" in name:
        for token, value in dict(cfg.get("atlas_nomask_penalty_families", {})).items():
            if str(token).lower() in name:
                bonus -= float(value)
    return bonus


def _build_consensus_selected_atlas(
    anat_dir: Path,
    subject_id: str,
    valid: list,
    topk: int,
    cfg: Mapping[str, Any],
):
    """LEVER #4.1 — Fusion multi-atlas : consensus pondéré des top-K atlas recalés.

    Remplace le winner-take-all (instable : bascule entre atlas de scores proches
    sous le bruit de registration → swing Dice) par un masque-atlas CONSENSUS,
    plus stable (déterminisme) et plus précis (corrige les erreurs d'un atlas
    isolé). Le consensus reste une GUIDANCE pour SAM2 (rôle identique à l'atlas
    unique) — SAM2 demeure le segmenteur. Fallback sûr vers valid[0] si échec.
    """
    top = valid[: max(2, int(topk))]
    try:
        from .atlas_label_fusion import weighted_vote_fusion, staple_fusion

        ref = nib.load(str(top[0][0]))
        ref_shape = ref.shape
        masks, weights = [], []
        for path, st in top:
            arr = np.asarray(nib.load(str(path)).dataobj) > 0
            if arr.shape != ref_shape:
                raise ValueError(f"forme atlas incohérente {arr.shape} vs {ref_shape}")
            masks.append(arr)
            weights.append(float(st.get("selection_score", 0.0)))
        method = str(cfg.get("atlas_consensus_method",
                            os.environ.get("SAM_NEUROADAPT_ATLAS_CONSENSUS_METHOD", "weighted_vote")))
        thr = float(cfg.get("atlas_consensus_threshold", 0.5))
        if method == "staple":
            fused = staple_fusion(masks, threshold=thr)
        else:
            fused = weighted_vote_fusion(masks, weights, threshold=thr)
        out = anat_dir / "atlases_advanced" / f"sub-{subject_id}_brain_mask_consensus.nii.gz"
        nib.save(nib.Nifti1Image(fused.astype(np.uint8), ref.affine, ref.header), str(out))
        ok, stats, _why = _mask_quality(out, cfg, atlas=True)
        if not isinstance(stats, dict):
            stats = {}
        stats["selection_score"] = float(np.mean(weights)) if weights else 0.0
        stats["consensus_topk"] = len(top)
        stats["consensus_method"] = method
        stats["consensus_members"] = [p.name for p, _ in top]
        # Âge de cohorte = meilleure cohorte NOMMÉE du top-K (pour P19).
        for path, _st in top:
            _a = _atlas_age_months(path)
            if _a is not None:
                stats["cohort_age_months"] = _a
                break
        print(f"    [consensus #4.1] {method} top-{len(top)} -> "
              f"{stats.get('volume_cm3', 0.0):.1f} cm3, comp={stats.get('components', '?')}")
        return out, stats
    except Exception as exc:
        print(f"    [consensus #4.1] échec ({str(exc)[:80]}) -> fallback winner-take-all")
        return valid[0]


def _select_registered_atlas(
    anat_dir: Path,
    subject_id: str,
    image_path: Path,
    cfg: Mapping[str, Any],
    subject_age_months: int | None = None,
):
    valid, rejected = [], []
    for cand in _registered_atlas_candidates(anat_dir, subject_id):
        if not cand.exists():
            continue
        ok, stats, why = _mask_quality(cand, cfg, atlas=True)
        if not ok:
            rejected.append((cand.name, why))
            continue
        img_score, img_stats = robust_atlas_image_alignment_score(cand, image_path)
        stats.update(img_stats)
        stats["image_alignment_score"] = float(img_score)
        stats["variant_bonus"] = _atlas_variant_bonus(cand, cfg)
        stats["selection_score_base"] = float(0.55 * stats.get("qc_score", 0.0) + 0.45 * img_score)
        # P1.2 — Pondération par âge : favorise les atlas calibrés pour l'âge du sujet
        _age_w = _atlas_age_weight(_atlas_age_months(cand), subject_age_months)
        stats["atlas_age_weight"] = _age_w
        stats["selection_score"] = float(np.clip(
            (stats["selection_score_base"] + stats["variant_bonus"]) * _age_w, 0.0, 1.0))
        # PATCH P13 — Pénaliser les atlas dont le volume dépasse de >15 % la borne
        # normative supérieure pour l'âge du sujet. Un atlas trop grand guide SAM2
        # vers une sur-segmentation systématique (ex. sub-726893 à 5 mois).
        if subject_age_months is not None:
            _p13_ages = sorted(_NORMATIVE_BRAIN_VOLUME_CM3.keys())
            _p13_lo = max((a for a in _p13_ages if a <= subject_age_months), default=_p13_ages[0])
            _p13_hi = min((a for a in _p13_ages if a >= subject_age_months), default=_p13_ages[-1])
            if _p13_lo == _p13_hi:
                _, _p13_nmax = _NORMATIVE_BRAIN_VOLUME_CM3[_p13_lo]
            else:
                _p13_t = (subject_age_months - _p13_lo) / (_p13_hi - _p13_lo)
                _, _p13_lo_max = _NORMATIVE_BRAIN_VOLUME_CM3[_p13_lo]
                _, _p13_hi_max = _NORMATIVE_BRAIN_VOLUME_CM3[_p13_hi]
                _p13_nmax = _p13_lo_max + _p13_t * (_p13_hi_max - _p13_lo_max)
            _p13_ratio = stats.get("volume_cm3", 0.0) / max(_p13_nmax, 1.0)
            if _p13_ratio > 1.15:
                _p13_pen = float(min(0.20, (_p13_ratio - 1.15) / 0.30 * 0.20))
                stats["selection_score"] = float(max(0.0, stats["selection_score"] - _p13_pen))
                stats["atlas_normative_penalty"] = _p13_pen
        # VISION — pénalité de volume ASYMÉTRIQUE (anti-effondrement). Pour les
        # datasets sans âge, on n'utilise PAS l'âge estimé (bruité) pour pondérer la
        # sélection (P1.2 désactivé) : on pénalise SEULEMENT les atlas dont le volume
        # warpé dépasse nettement le volume cérébral ESTIMÉ du sujet (atlas trop
        # grand = sur-segmentation garantie). Conservateur : aucune pénalité pour les
        # atlas plus petits ou proches → l'alignement reste le décideur principal,
        # une estimation imparfaite ne force jamais un mauvais choix (≠ régressions
        # de P1.2). Seuil 1.5 = ne mord que sur les cas clairement sur-dimensionnés.
        _est_vol = cfg.get("estimated_brain_volume_cm3")
        if _est_vol and float(_est_vol) > 0:
            _vol_ratio = stats.get("volume_cm3", 0.0) / float(_est_vol)
            _osz_thr = float(cfg.get("atlas_volume_oversize_threshold", 1.5))
            if _vol_ratio > _osz_thr:
                _vpen = float(min(0.45, (_vol_ratio - _osz_thr) / 0.5 * 0.45))
                stats["selection_score"] = float(max(0.0, stats["selection_score"] - _vpen))
                stats["atlas_volume_oversize_penalty"] = _vpen
                stats["atlas_volume_ratio_to_subject"] = float(_vol_ratio)
        weak = []
        if img_score < float(cfg.get("min_atlas_image_score", 0.05)):
            weak.append(f"image_score={img_score:.3f}")
        if stats["selection_score"] < float(cfg.get("min_atlas_selection_score", 0.25)):
            weak.append(f"global_score={stats['selection_score']:.3f}")
        if weak:
            rejected.append((cand.name, "weak_registered_atlas: " + "; ".join(weak)))
            continue
        valid.append((cand, stats))
    if valid:
        valid.sort(key=lambda item: item[1].get("selection_score", item[1].get("qc_score", 0.0)), reverse=True)
        # LEVER #4.1 — Consensus multi-atlas (opt-in via atlas_consensus_topk≥2
        # ou env SAM_NEUROADAPT_ATLAS_CONSENSUS_TOPK). Défaut 0 = winner-take-all
        # historique (aucune régression).
        _topk = int(cfg.get("atlas_consensus_topk",
                            os.environ.get("SAM_NEUROADAPT_ATLAS_CONSENSUS_TOPK", 0)) or 0)
        if _topk >= 2 and len(valid) >= 2:
            cand, stats = _build_consensus_selected_atlas(anat_dir, subject_id, valid, _topk, cfg)
        else:
            cand, stats = valid[0]
        # PATCH P19 — âge de cohorte pour l'inférence d'âge (datasets sans âge de
        # session). Le gagnant peut être l'atlas adaptatif (nom basé sujet, sans
        # cohorte) : dans ce cas on prend l'âge de la meilleure cohorte NOMMÉE parmi
        # les candidats valides (déjà triés par score d'alignement décroissant).
        _coh_age = _atlas_age_months(cand)
        if _coh_age is None:
            for _alt_cand, _alt_stats in valid:
                _alt_age = _atlas_age_months(_alt_cand)
                if _alt_age is not None:
                    _coh_age = _alt_age
                    break
        stats["cohort_age_months"] = _coh_age
        print(
            f"    atlas accepted: {cand.name} "
            f"({stats['volume_cm3']:.1f} cm3, comp={stats['components']}, score={stats.get('selection_score', 0.0):.3f})"
        )
        if len(valid) > 1:
            for alt, alt_stats in valid[1:4]:
                print(f"      alt: {alt.name}: {alt_stats['volume_cm3']:.1f} cm3, score={alt_stats.get('selection_score', 0.0):.3f}")
        return cand, stats, rejected
    if rejected:
        print("    registered atlases rejected:")
        for name, why in rejected[:10]:
            print(f"      - {name}: {why}")
    return None, {}, rejected


def _canonicalize_selected_atlas_variants(anat_dir: Path, subject_id: str, selected_atlas: Path, *, force: bool) -> None:
    name = selected_atlas.name
    suffixes = ("aff_nomask", "aff", "syn")
    selected_suffix = next((suffix for suffix in suffixes if name.endswith(f"_{suffix}.nii.gz")), None)
    if not selected_suffix:
        return
    base = selected_atlas.with_name(name[: -len(f"_{selected_suffix}.nii.gz")])
    canonical_base = anat_dir / "atlases_advanced" / f"sub-{subject_id}_brain_mask"
    if base.name == canonical_base.name:
        return
    for suffix in suffixes:
        src = base.with_name(base.name + f"_{suffix}.nii.gz")
        dst = canonical_base.with_name(canonical_base.name + f"_{suffix}.nii.gz")
        if src.exists() and (force or not dst.exists()):
            shutil.copy2(src, dst)


def _confidence_from_audit(anat_dir: Path, subject_id: str, cfg: Mapping[str, Any], pp=None) -> float:
    try:
        audit = None
        if isinstance(pp, dict) and pp.get("visual_metrics"):
            audit = pp["visual_metrics"]
        recommended = audit.get("recommended_threshold") if isinstance(audit, dict) else None
        if recommended is not None:
            confidence = float(np.clip(recommended, 0.62, 0.85))
        else:
            confidence = float(cfg.get("confidence_default", 0.72))
        if isinstance(audit, dict):
            edge_ws = float(audit.get("edge_degradation_ws", 0.0) or 0.0)
            if edge_ws > 0.15:
                print(f"    WhiteStripe QC: degraded edges ({edge_ws*100:.1f}%), using N4 for SAM2")
        print(f"    confidence threshold={confidence:.2f}")
        return confidence
    except Exception:
        confidence = float(cfg.get("confidence_default", 0.72))
        print(f"    no usable audit, confidence={confidence:.2f}")
        return confidence


def _extract_brain(api: Mapping[str, Any], input_path: Path, output_path: Path, confidence_threshold: float, atlas_paths: dict[str, Path], cfg: Mapping[str, Any], debug_dir: Path):
    params = api["SegmentationParams"](
        debug_dir=debug_dir,
        use_atlas_guidance=bool(atlas_paths),
        atlas_weight=float(cfg.get("atlas_weight", 0.75)),
        post_process=True,
    )
    params.min_brain_volume_cm3 = float(cfg["min_brain_volume_cm3"])
    params.max_brain_volume_cm3 = float(cfg["max_brain_volume_cm3"])
    params.atlas_fallback_min_cm3 = float(cfg["atlas_fallback_min_cm3"])
    params.atlas_fallback_max_cm3 = float(cfg["atlas_fallback_max_cm3"])
    params.atlas_min_coverage = float(cfg["atlas_min_coverage"])
    params.atlas_min_precision = float(cfg["atlas_min_precision"])
    params.enable_atlas_fallback = bool(atlas_paths) and bool(cfg.get("enable_atlas_fallback", False))
    params.atlas_paths = atlas_paths or {}
    params.initial_confidence_threshold = float(confidence_threshold)
    params.propagation_confidence_threshold = float(confidence_threshold) * 0.9
    # IDÉE 1/2/4 — propagation flags SAM2 améliorés. hasattr() = compat backward
    # avec notebook non-réexécuté (les nouveaux attrs sont silencieusement ignorés).
    if hasattr(params, 'enable_atlas_input_masking'):
        params.enable_atlas_input_masking = bool(cfg.get("sam2_enable_atlas_input_masking", False))
        params.atlas_mask_hard_radius_mm = float(cfg.get("sam2_atlas_mask_hard_radius_mm", 12.0))
        params.atlas_mask_soft_radius_mm = float(cfg.get("sam2_atlas_mask_soft_radius_mm", 3.0))
        params.atlas_mask_min_volume_cm3 = float(cfg.get("sam2_atlas_mask_min_volume_cm3", 200.0))
        params.atlas_mask_background_value = float(cfg.get("sam2_atlas_mask_background_value", 0.0))
    if hasattr(params, 'eap_tight_always_initial'):
        params.eap_tight_always_initial = bool(cfg.get("sam2_eap_tight_always_initial", False))
        params.eap_tight_min_preservation = float(cfg.get("sam2_eap_tight_min_preservation", 0.20))
    if hasattr(params, 'adaptive_atlas_shrink_enabled'):
        params.adaptive_atlas_shrink_enabled = bool(cfg.get("sam2_adaptive_atlas_shrink_enabled", False))
        params.adaptive_atlas_shrink_target_ratio = float(cfg.get("sam2_adaptive_atlas_shrink_target_ratio", 1.15))
        params.adaptive_atlas_shrink_trigger_ratio = float(cfg.get("sam2_adaptive_atlas_shrink_trigger_ratio", 1.40))
        params.adaptive_atlas_shrink_max_iters = int(cfg.get("sam2_adaptive_atlas_shrink_max_iters", 10))
        _age_m = cfg.get("month")
        if _age_m is not None:
            params.adaptive_atlas_age_months = int(_age_m)
    # Vague 2 — IDÉE 3 (mask_input atlas) + early stopping volume-driven
    if hasattr(params, 'enable_atlas_mask_input_prompt'):
        params.enable_atlas_mask_input_prompt = bool(cfg.get("sam2_enable_atlas_mask_input_prompt", False))
        params.mask_input_strength = float(cfg.get("sam2_mask_input_strength", 10.0))
        params.mask_input_use_eap = bool(cfg.get("sam2_mask_input_use_eap", True))
    if hasattr(params, 'enable_volume_early_stopping'):
        params.enable_volume_early_stopping = bool(cfg.get("sam2_enable_volume_early_stopping", False))
        params.volume_early_stop_ratio = float(cfg.get("sam2_volume_early_stop_ratio", 1.20))
    # Vague 3 — bidirectional consensus
    if hasattr(params, 'enable_bidirectional_consensus'):
        params.enable_bidirectional_consensus = bool(cfg.get("sam2_enable_bidirectional_consensus", False))
        params.bidirectional_consensus_min_dice = float(cfg.get("sam2_bidirectional_consensus_min_dice", 0.80))
        params.bidirectional_consensus_min_range = int(cfg.get("sam2_bidirectional_consensus_min_range", 6))
    # P1.1 — Multi-point SAM2 prompting
    if hasattr(params, 'multipoint_enabled'):
        params.multipoint_enabled = bool(cfg.get("sam2_multipoint_enabled", False))
        params.multipoint_n_positive = int(cfg.get("sam2_multipoint_n_positive", 8))
        params.multipoint_n_negative = int(cfg.get("sam2_multipoint_n_negative", 5))
        params.multipoint_seed = int(cfg.get("sam2_multipoint_seed", 42))
    # SAM2 multi-vue (opt-in) : présente le volume dans les 3 orientations à SAM2
    # (qui ne propage que sur l'axe Z), puis fusionne par STAPLE → contour 3D plus
    # précis. Flag cfg['sam2_multiview'] ou env SAM_NEUROADAPT_SAM2_MULTIVIEW.
    # Défaut OFF → comportement mono-vue inchangé (aucune régression).
    _multiview = bool(cfg.get("sam2_multiview",
                              os.environ.get("SAM_NEUROADAPT_SAM2_MULTIVIEW", "").strip().lower()
                              in {"1", "true", "yes", "on"}))
    if _multiview:
        from .sam2_multiview import extract_brain_multiview

        def _view_extract_fn(inp, outp, atlas_paths=None, **_kw):
            import copy
            p_view = copy.copy(params)
            p_view.atlas_paths = atlas_paths or {}
            ext = api["SAM2BrainExtractor"](params=p_view)
            return ext.extract_brain(input_path=inp, output_path=outp,
                                     save_metrics=False, save_uncertainty_map=False)

        rep = extract_brain_multiview(
            input_path, output_path, atlas_paths or None,
            extract_fn=_view_extract_fn,
            fusion=str(cfg.get("sam2_multiview_fusion", "staple")),
            axes=tuple(cfg.get("sam2_multiview_axes", (0, 1, 2))),
        )
        if rep.get("multiview_used"):
            print(f"    [SAM2 multi-vue] {rep['n_views']} vues, fusion={rep['fusion_method']} "
                  f"-> {rep.get('volume_voxels', 0)} vox")
            return rep.get("brain_path"), rep.get("mask_path", str(output_path))
        print(f"    [SAM2 multi-vue] échec ({rep.get('reason')}) -> mono-vue")

    extractor = api["SAM2BrainExtractor"](params=params)
    return extractor.extract_brain(input_path=input_path, output_path=output_path, save_metrics=True, save_uncertainty_map=True)


def _nii_with_suffix(path: Path, suffix: str) -> Path:
    name = path.name
    if name.endswith(".nii.gz"):
        return path.with_name(name[:-7] + suffix + ".nii.gz")
    return path.with_name(path.stem + suffix + path.suffix)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lbl, ncomp = ndi.label(np.asarray(mask).astype(bool))
    if ncomp <= 1:
        return np.asarray(mask).astype(bool)
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    return lbl == int(np.argmax(counts))


def _mask_overlap_stats(mask: np.ndarray, atlas: np.ndarray, zooms) -> dict[str, Any]:
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


def _apply_local_boundary_refinement(mask, atlas, mask_img, input_image_path: Path, cfg: Mapping[str, Any]):
    stats = {"boundary_refinement_enabled": bool(cfg.get("enable_boundary_refinement", True)), "boundary_refinement_used": False}
    if not cfg.get("enable_boundary_refinement", True):
        stats["boundary_refinement_reason"] = "disabled"
        return mask, stats
    try:
        if not input_image_path.exists() or mask.sum() == 0:
            stats["boundary_refinement_reason"] = "input_missing_or_empty_mask"
            return mask, stats
        input_img = nib.load(str(input_image_path))
        input_for_mask = input_img
        if input_img.shape[:3] != mask_img.shape[:3] or not np.allclose(input_img.affine, mask_img.affine, atol=1e-3):
            input_for_mask = resample_from_to(input_img, mask_img, order=1, cval=0)
        data = np.nan_to_num(input_for_mask.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        structure = ndi.generate_binary_structure(3, 1)
        atlas_allowed = atlas
        if int(cfg.get("boundary_atlas_dilation", 1)) > 0:
            atlas_allowed = ndi.binary_dilation(atlas, structure=structure, iterations=int(cfg.get("boundary_atlas_dilation", 1)))
        shell = ndi.binary_dilation(mask, structure=structure, iterations=int(cfg.get("boundary_refinement_width", 1))) & (~mask) & atlas_allowed
        values = data[mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            stats["boundary_refinement_reason"] = "no_finite_intensity"
            return mask, stats
        threshold = float(np.percentile(values, float(cfg.get("boundary_intensity_percentile", 8.0))))
        additions = shell & (data >= threshold)
        if additions.sum() == 0:
            stats.update({"boundary_refinement_reason": "no_candidate_voxels", "boundary_refinement_intensity_threshold": threshold})
            return mask, stats
        proposal = ndi.binary_fill_holes(_largest_component(mask | additions)).astype(bool)
        zooms = tuple(float(v) for v in mask_img.header.get_zooms()[:3])
        before = _mask_overlap_stats(mask, atlas, zooms)
        after = _mask_overlap_stats(proposal, atlas, zooms)
        added_cm3 = float(np.logical_and(proposal, ~mask).sum() * np.prod(zooms) / 1000.0)
        volume_ratio = float(after["volume_cm3"] / max(before["volume_cm3"], 1e-8))
        stats.update(
            {
                "boundary_refinement_intensity_threshold": threshold,
                "boundary_refinement_added_cm3": added_cm3,
                "boundary_refinement_volume_before_cm3": before["volume_cm3"],
                "boundary_refinement_volume_after_cm3": after["volume_cm3"],
                "boundary_refinement_volume_ratio": volume_ratio,
                "boundary_refinement_atlas_precision_after": after["atlas_precision"],
                "boundary_refinement_atlas_coverage_after": after["atlas_coverage"],
            }
        )
        reject = []
        if after["volume_cm3"] < float(cfg["min_brain_volume_cm3"]) or after["volume_cm3"] > float(cfg["max_brain_volume_cm3"]):
            reject.append("volume_outside_bounds")
        if added_cm3 > float(cfg.get("boundary_max_added_cm3", 35.0)):
            reject.append("added_cm3_too_large")
        if volume_ratio > float(cfg.get("boundary_max_volume_ratio", 1.06)):
            reject.append("volume_ratio_too_large")
        if after["atlas_precision"] < float(cfg.get("boundary_min_atlas_precision", 0.92)):
            reject.append("atlas_precision_too_low")
        if reject:
            stats["boundary_refinement_reason"] = ";".join(reject)
            return mask, stats
        stats["boundary_refinement_used"] = True
        stats["boundary_refinement_reason"] = "applied"
        return proposal, stats
    except Exception as exc:
        stats["boundary_refinement_reason"] = f"error={exc}"
        return mask, stats


def _refine_mask_with_atlas(mask_path: Path, brain_path: Path, input_image_path: Path, atlas_path: Path, metrics_path: Path, cfg: Mapping[str, Any]):
    stats = {"refinement_enabled": bool(cfg.get("enable_atlas_refinement", True)), "refinement_used": False}
    if not cfg.get("enable_atlas_refinement", True):
        stats["refinement_reason"] = "disabled"
        return stats
    if not atlas_path or not Path(atlas_path).exists():
        stats["refinement_reason"] = "atlas_missing"
        return stats
    try:
        mask_img = nib.load(str(mask_path))
        mask = mask_img.get_fdata(dtype=np.float32) > 0.5
        atlas_img = nib.load(str(atlas_path))
        if atlas_img.shape[:3] != mask_img.shape[:3] or not np.allclose(atlas_img.affine, mask_img.affine, atol=1e-3):
            atlas_img = resample_from_to(atlas_img, mask_img, order=0, cval=0)
        atlas = atlas_img.get_fdata(dtype=np.float32) > 0.5
        zooms = tuple(float(v) for v in mask_img.header.get_zooms()[:3])
        before = _mask_overlap_stats(mask, atlas, zooms)
        structure = ndi.generate_binary_structure(3, 1)
        allowed = atlas
        if int(cfg.get("refinement_atlas_dilation", 1)) > 0:
            allowed = ndi.binary_dilation(atlas, structure=structure, iterations=int(cfg.get("refinement_atlas_dilation", 1)))
        refined = ndi.binary_fill_holes(_largest_component(mask & allowed)).astype(bool)
        strict_after = _mask_overlap_stats(refined, atlas, zooms)
        reject = []
        keep_ratio = strict_after["voxels"] / max(1, before["voxels"])
        precision_gain = strict_after["atlas_precision"] - before["atlas_precision"]
        if strict_after["voxels"] == 0:
            reject.append("empty_refined_mask")
        if strict_after["volume_cm3"] < float(cfg["min_brain_volume_cm3"]) or strict_after["volume_cm3"] > float(cfg["max_brain_volume_cm3"]):
            reject.append("volume_outside_bounds")
        if keep_ratio < float(cfg.get("refinement_min_keep_ratio", 0.85)):
            reject.append("keep_ratio_too_low")
        if strict_after["atlas_coverage"] < float(cfg.get("refinement_min_atlas_coverage", 0.85)):
            reject.append("atlas_coverage_too_low")
        if precision_gain < float(cfg.get("refinement_min_precision_gain", 0.01)) and before["atlas_precision"] < float(cfg.get("boundary_min_atlas_precision", 0.92)):
            reject.append("precision_gain_too_low")
        stats.update(
            {
                "refinement_volume_before_cm3": before["volume_cm3"],
                "atlas_refinement_volume_after_cm3": strict_after["volume_cm3"],
                "atlas_refinement_keep_ratio": float(keep_ratio),
                "refinement_atlas_coverage_before": before["atlas_coverage"],
                "atlas_refinement_atlas_coverage_after": strict_after["atlas_coverage"],
                "refinement_atlas_precision_before": before["atlas_precision"],
                "atlas_refinement_atlas_precision_after": strict_after["atlas_precision"],
                "refinement_precision_gain": float(precision_gain),
            }
        )
        if reject:
            stats["refinement_reason"] = ";".join(reject)
            print(f"    atlas refinement not applied: {stats['refinement_reason']}")
            return stats
        refined, boundary_stats = _apply_local_boundary_refinement(refined, atlas, mask_img, input_image_path, cfg)
        stats.update(boundary_stats)
        after = _mask_overlap_stats(refined, atlas, zooms)
        stats.update(
            {
                "refinement_volume_after_cm3": after["volume_cm3"],
                "refinement_keep_ratio": float(after["voxels"] / max(1, before["voxels"])),
                "refinement_atlas_coverage_after": after["atlas_coverage"],
                "refinement_atlas_precision_after": after["atlas_precision"],
                "refinement_atlas_dice_after": after["atlas_dice"],
            }
        )
        backup = _nii_with_suffix(mask_path, "_pre-refine")
        if not backup.exists() or cfg.get("force_extraction", True):
            shutil.copyfile(mask_path, backup)
        out_img = nib.Nifti1Image(refined.astype(np.uint8), mask_img.affine, mask_img.header)
        out_img.set_data_dtype(np.uint8)
        nib.save(out_img, str(mask_path))
        if brain_path and input_image_path.exists():
            input_img = nib.load(str(input_image_path))
            refine_for_input = out_img
            if input_img.shape[:3] != out_img.shape[:3] or not np.allclose(input_img.affine, out_img.affine, atol=1e-3):
                refine_for_input = resample_from_to(out_img, input_img, order=0, cval=0)
            data = input_img.get_fdata(dtype=np.float32) * (refine_for_input.get_fdata(dtype=np.float32) > 0.5).astype(np.float32)
            brain_img = nib.Nifti1Image(data.astype(np.float32), input_img.affine, input_img.header)
            brain_img.set_data_dtype(np.float32)
            nib.save(brain_img, str(brain_path))
        stats["refinement_used"] = True
        stats["refinement_reason"] = "applied"
        stats["refinement_backup_mask"] = str(backup)
        print(
            f"    atlas/boundary refinement applied: {before['volume_cm3']:.1f}->{after['volume_cm3']:.1f} cm3, "
            f"precision {before['atlas_precision']:.3f}->{after['atlas_precision']:.3f}"
        )
        if metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text())
                payload.update(stats)
                metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            except Exception:
                pass
        return stats
    except Exception as exc:
        stats["refinement_reason"] = f"error={exc}"
        traceback.print_exc()
        return stats


def _write_progress(progress_json: Path, dataset: str, stage: str, subject=None, idx=None, total=None, rows=None) -> None:
    payload = {
        "dataset": dataset,
        "stage": stage,
        "subject": subject,
        "idx": idx,
        "index": idx,
        "total": total,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if rows is not None:
        payload["n_rows"] = len(rows)
        payload["n_errors"] = int(sum(1 for r in rows if str(r.get("error", "")).strip()))
        dices = [float(r["dice"]) for r in rows if str(r.get("dice", "")).strip() and not pd.isna(r.get("dice"))]
        payload["n_dice"] = len(dices)
        payload["dice_mean"] = float(np.mean(dices)) if dices else None
    _atomic_write_text(progress_json, json.dumps(payload, indent=2, ensure_ascii=False))


def run_m1_pediatric_iseg_style_gold(
    notebook_globals: Mapping[str, Any],
    *,
    dataset: str,
    manifest: str | Path | None = None,
    out_base: str | Path | None = None,
    profile_filter: str | None = None,
    n_subjects: int | None = None,
    subjects: list[str] | tuple[str, ...] | None = None,
    require_atlas: bool = True,
    force_clean_outputs: bool = True,
    force_extraction: bool = True,
    refinement_idempotent_mode: bool | None = None,
    display_tables: bool = True,
    target_mean_dice: float | None = None,
    reproducibility_seed: int | None = None,
) -> dict[str, Any]:
    """Run the iSeg2017-style pediatric architecture on any gold manifest."""

    api = _notebook_api(notebook_globals)
    seed = int(os.environ.get("SAM_NEUROADAPT_SEED", reproducibility_seed if reproducibility_seed is not None else 20260517))
    _set_seed(seed)
    records = _load_records(
        dataset,
        manifest=manifest,
        profile_filter=profile_filter,
        subjects=subjects,
        n_subjects=n_subjects,
        require_atlas=require_atlas,
    )
    slug = default_output_slug(dataset, profile_filter)
    out_base = Path(out_base) if out_base else Path("outputs") / f"{slug}_m1_pediatric_isegstyle_eval"
    out_base.mkdir(parents=True, exist_ok=True)
    final_csv = out_base / f"sam_neuroadapt_{slug}_results.csv"
    live_csv = out_base / f"sam_neuroadapt_{slug}_results_live.csv"
    progress_json = out_base / f"m1_{slug}_progress.json"
    selection_json = out_base / f"{slug}_selection.json"

    if force_clean_outputs:
        for path in (final_csv, live_csv, progress_json, selection_json):
            if path.exists():
                path.unlink()

    selection_payload = {
        "dataset": dataset,
        "manifest": str(_manifest_path(dataset, manifest)),
        "n_requested": n_subjects,
        "n_selected": len(records),
        "seed": seed,
        "subjects": [
            {
                "subject": rec["subject"],
                "session": rec["session"],
                "profile": rec["profile"],
                "atlas_files": {k: str(v) for k, v in rec["atlas_paths"].items()},
                "config_no_gold": {
                    k: v
                    for k, v in rec["config"].items()
                    if k
                    in {
                        "month",
                        "mask_min_cm3",
                        "mask_max_cm3",
                        "atlas_qc_min_cm3",
                        "atlas_qc_max_cm3",
                        "min_brain_volume_cm3",
                        "max_brain_volume_cm3",
                    }
                },
            }
            for rec in records
        ],
    }
    selection_json.write_text(json.dumps(selection_payload, indent=2, default=_json_default, ensure_ascii=False))

    print(f"M1-PEDIATRIC-iSegStyle {dataset}: {len(records)} subjects")
    print(f"Manifest: {_manifest_path(dataset, manifest)}")
    print(f"Output:   {out_base}")
    print(f"Selection:{selection_json}")
    _write_progress(progress_json, dataset, "started", total=len(records), rows=[])

    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(records, 1):
        sid = rec["subject"]
        safe_sid = _safe_subject(sid)
        raw_subject = safe_sid.replace("sub-", "", 1)
        subject_id = f"{raw_subject}_T1w"
        subject_bids = f"sub-{subject_id}"
        out_dir = out_base / safe_sid
        anat_dir = out_dir / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(rec["config"])
        if not force_extraction:
            cfg["force_extraction"] = False
        if refinement_idempotent_mode is not None:
            cfg["refinement_idempotent_mode"] = bool(refinement_idempotent_mode)
        t0 = time.time()

        print(f"\n[{idx:02d}/{len(records)}] {sid} | session={rec['session']} | profile={rec['profile']}")
        print(f"  atlas candidates: {', '.join(rec['atlas_paths'].keys())}")
        _write_progress(progress_json, dataset, "subject_started", sid, idx, len(records), rows)
        try:
            source_t1, gs_mask, source_t2, source_label = _prepare_source_images(rec, anat_dir, subject_id, cfg)
            img_info = _image_info(source_t1)
            gs_info = _image_info(gs_mask)
            _clean_subject_artifacts(anat_dir, subject_id, cfg)

            print(f"  [{sid}] step 1/3: N4 + WhiteStripe")
            _write_progress(progress_json, dataset, "preprocessing", sid, idx, len(records), rows)
            n4_path = anat_dir / f"{subject_bids}_desc-n4_T1w.nii.gz"
            ws_path = anat_dir / f"{subject_bids}_desc-n4_desc-ws_T1w.nii.gz"
            pp = None
            if cfg.get("force_preprocessing", True) or not n4_path.exists() or not ws_path.exists():
                preproc_mask = _select_preproc_mask(anat_dir, subject_id, cfg)
                pp = api["run_preprocessing_pipeline"](
                    t1_path=source_t1,
                    output_dir=anat_dir,
                    subject_id=subject_id,
                    mask_path=preproc_mask if preproc_mask is not None and preproc_mask.exists() else None,
                )
                if not pp or not pp.get("success"):
                    raise RuntimeError(f"preprocessing_failed: {pp}")
                n4_path = Path(pp.get("n4_path")) if pp.get("n4_path") else n4_path
                ws_path = Path(pp.get("whitestripe_path")) if pp.get("whitestripe_path") else ws_path
            else:
                print(f"    preprocessing cache: {n4_path.name}, {ws_path.name}")
            input_image_path = n4_path if n4_path.exists() else ws_path
            if not input_image_path.exists():
                raise RuntimeError("preprocessing did not produce N4/WhiteStripe")
            print(f"    SAM2 image: {input_image_path.name}")

            print(f"  [{sid}] step 2/3: external pediatric atlas registration")
            _write_progress(progress_json, dataset, "atlas_registration", sid, idx, len(records), rows)
            atlas_dir = anat_dir / "atlases_advanced"
            atlas_dir.mkdir(parents=True, exist_ok=True)
            adaptive_atlas_stats = {}
            # VISION appariement atlas↔sujet par TAILLE : pour les datasets sans âge
            # de session, estimer l'âge depuis le VOLUME cérébral réel (skull-strip
            # grossier) AVANT la sélection → apparie un atlas bien dimensionné dès le
            # départ, au lieu de recaler un atlas trop grand puis d'inférer un âge
            # circulaire (P19). N'affecte PAS les datasets à âge connu (month != None).
            if cfg.get("month") is None and cfg.get("estimated_age_months") is None:
                try:
                    from .subject_volume_estimator import (
                        estimate_age_months_from_image, estimate_brain_volume_cm3)
                    _vol_est = estimate_brain_volume_cm3(n4_path)
                    _age_est = estimate_age_months_from_image(n4_path)
                    if _age_est is not None:
                        cfg["estimated_age_months"] = float(_age_est)
                        cfg["estimated_brain_volume_cm3"] = float(_vol_est)
                        _apply_normative_volume_bounds(cfg, _age_est)
                        print(f"    [taille→âge] volume estimé {_vol_est:.0f} cm³ → "
                              f"âge≈{_age_est:.1f} mois (appariement atlas par taille)")
                except Exception as _exc:
                    print(f"    [taille→âge] estimation échouée ({str(_exc)[:60]})")
            selected_atlas, atlas_stats, rejected_atlases = _select_registered_atlas(
                anat_dir, subject_id, n4_path, cfg,
                subject_age_months=cfg.get("month"))  # âge RÉEL seul ; l'âge estimé (bruité) pilote l'anti-sur-dim. par volume, pas P1.2
            if cfg.get("force_atlas_registration", True) or selected_atlas is None:
                try:
                    api["advanced_atlas_registration"](
                        subject_n4_path=n4_path,
                        atlas_paths=rec["atlas_paths"],
                        output_dir=atlas_dir,
                        subject_id=subject_id,
                        create_brain_mask=True,
                        verbose=False,
                    )
                except Exception as exc:
                    print(f"    atlas registration failed: {exc}")
                selected_atlas, atlas_stats, rejected_atlases = _select_registered_atlas(
                anat_dir, subject_id, n4_path, cfg,
                subject_age_months=cfg.get("month"))  # âge RÉEL seul ; l'âge estimé (bruité) pilote l'anti-sur-dim. par volume, pas P1.2
            if selected_atlas is None:
                msg = "no valid registered external pediatric atlas"
                if cfg.get("stop_if_no_valid_atlas", True):
                    raise RuntimeError(msg)
                atlas_paths_subj = {}
            else:
                _canonicalize_selected_atlas_variants(anat_dir, subject_id, selected_atlas, force=True)
                # PATCH P19 — Âge inféré par atlas pour les datasets sans âge de session
                # (ex. SynthStripInfant). Le cohort MNIInfant le mieux aligné (retenu par
                # _select_registered_atlas sur le score image-intensité) fournit une
                # estimation d'âge qui réactive les priors volumétriques (P12/P9) et la
                # pondération d'âge (P1.2/P13). 100 % guidé-atlas, sans GT.
                if cfg.get("month") is None and not cfg.get("is_fetal") and cfg.get("estimated_age_months") is None:
                    # L'âge vient des stats de sélection (cohort_age_months), robuste au
                    # cas où le gagnant est l'atlas adaptatif (nom sujet, sans cohorte).
                    _est_age = (atlas_stats or {}).get("cohort_age_months")
                    if _est_age is None:
                        _est_age = _atlas_age_months(Path(selected_atlas))
                    if _est_age is not None:
                        cfg["estimated_age_months"] = float(_est_age)
                        _apply_normative_volume_bounds(cfg, _est_age)
                        print(f"    [age-infer] age estime {float(_est_age):.1f} mois (cohorte la mieux "
                              f"alignee) -> priors volumetriques reactives")
                if cfg.get("enable_adaptive_guidance_atlas", True):
                    try:
                        adaptive_atlas_stats = build_iseg_adaptive_guidance_atlas(
                            anat_dir=anat_dir,
                            subject_id=subject_id,
                            input_image_path=n4_path,
                            t2_path=source_t2,
                            force=True,
                            config=cfg["adaptive_guidance"],
                        )
                        if adaptive_atlas_stats.get("adaptive_atlas_used"):
                            print(
                                f"    adaptive atlas ready: {Path(adaptive_atlas_stats['output_path']).name} "
                                f"vol={adaptive_atlas_stats.get('adaptive_volume_cm3', float('nan')):.1f} cm3"
                            )
                    except Exception as exc:
                        print(f"    adaptive atlas failed: {exc}")
                        adaptive_atlas_stats = {"adaptive_atlas_reason": str(exc)}
                selected_atlas, atlas_stats, rejected_atlases = _select_registered_atlas(
                anat_dir, subject_id, n4_path, cfg,
                subject_age_months=cfg.get("month"))  # âge RÉEL seul ; l'âge estimé (bruité) pilote l'anti-sur-dim. par volume, pas P1.2
                if selected_atlas is None:
                    raise RuntimeError("no valid atlas after adaptive guidance build")
                atlas_paths_subj = {"brain_mask": selected_atlas}

            confidence = _confidence_from_audit(anat_dir, subject_id, cfg, pp)

            print(f"  [{sid}] step 3/3: SAM-NeuroAdapt extraction")
            _write_progress(progress_json, dataset, "sam2_extraction", sid, idx, len(records), rows)
            brain_out = anat_dir / f"{subject_bids}_desc-brain_T1w.nii.gz"
            mask_out = anat_dir / f"{subject_bids}_desc-brain_mask.nii.gz"
            debug_dir = anat_dir / "debug_sam"
            debug_dir.mkdir(parents=True, exist_ok=True)
            # Ablation SAM_NEUROADAPT_NO_ATLAS=1 : SAM2 sans guidance atlas.
            # _extract_brain calcule use_atlas_guidance=bool(atlas_paths) et
            # enable_atlas_fallback and bool(atlas_paths) → un dict vide suffit
            # à couper prompts, mask_input, masquage d'entrée ET fallback.
            _sam2_atlas_paths = {} if no_atlas_ablation() else atlas_paths_subj
            if cfg.get("force_extraction", True) or not mask_out.exists():
                brain_path, mask_path = _extract_brain(api, input_image_path, brain_out, confidence, _sam2_atlas_paths, cfg, debug_dir)
            else:
                brain_path, mask_path = brain_out, mask_out
                print(f"    extraction cache: {mask_out.name}")
            if mask_path is None or not Path(str(mask_path)).exists():
                raise RuntimeError("SAM2 extraction did not produce a mask")
            brain_path = Path(brain_path)
            mask_path = Path(mask_path)
            metrics_path = anat_dir / f"{subject_bids}_desc-brain_metrics.json"

            # Propagate the atlas image-alignment score as reference trust so that
            # apply_adult_external_refinement and apply_adaptive_mask_fusion can
            # distinguish a well-registered atlas (trust > 0.18) from a poor one.
            # The pediatric runner never called _reference_atlas_trust_score, so
            # both functions always received the default 0.0 → low_trust_fail_clean
            # was permanently active, silently blocking all pruning and fusion.
            _reference_trust = float((atlas_stats or {}).get("image_alignment_score", 0.0))

            # Idempotence des refinements amont. Les fonctions de refinement lisent
            # et écrasent mask_path ; sans garde, un run --no-force-extraction part
            # du masque final du run précédent → accumulation. On snapshot le masque
            # SAM2 brut + chaque état intermédiaire pour que chaque étape lise
            # toujours le même masque d'entrée canonique (runs déterministes).
            _idempotent = bool(cfg.get("refinement_idempotent_mode", True))
            _bk_refine = _nii_with_suffix(mask_path, "_pre-refine")
            _bk_t2 = _nii_with_suffix(mask_path, "_pre-pediatric-t2-refine")
            _bk_wide = _nii_with_suffix(mask_path, "_pre-pediatric-wide-refine")
            _bk_fusion = _nii_with_suffix(mask_path, "_pre-adaptive-fusion")
            if _idempotent:
                if cfg.get("force_extraction", True):
                    # Run frais : le masque SAM2 sur disque fait foi. On purge la
                    # chaîne de backups aval pour qu'elle soit recréée depuis ce run
                    # (les paramètres SAM2 peuvent avoir changé entre sessions).
                    for _bk in (_bk_refine, _bk_t2, _bk_wide, _bk_fusion):
                        if _bk.exists():
                            _bk.unlink()
                    shutil.copyfile(mask_path, _bk_refine)
                    print(f"  [IDEMPOTENT] fresh SAM2 raw snapshot -> {_bk_refine.name}")
                elif _bk_refine.exists():
                    shutil.copyfile(_bk_refine, mask_path)
                    print(f"  [IDEMPOTENT] restored SAM2 raw mask from {_bk_refine.name}")
                else:
                    shutil.copyfile(mask_path, _bk_refine)

            refinement_stats = {}
            if selected_atlas is not None:
                refinement_stats = _refine_mask_with_atlas(mask_path, brain_path, input_image_path, selected_atlas, metrics_path, cfg)

            # Snapshot/restore de l'entrée de l'étape pediatric-T2 (= post-étape 1).
            if _idempotent:
                if _bk_t2.exists():
                    shutil.copyfile(_bk_t2, mask_path)
                else:
                    shutil.copyfile(mask_path, _bk_t2)

            pediatric_refinement_stats = {}
            if selected_atlas is not None:
                pediatric_refinement_stats = apply_pediatric_t2_refinement(
                    mask_path=mask_path,
                    brain_path=brain_path,
                    input_image_path=input_image_path,
                    t2_path=source_t2,
                    atlas_path=selected_atlas,
                    metrics_path=metrics_path,
                    config={
                        "enabled": cfg.get("enable_pediatric_t2_refinement", True),
                        "shell_width": cfg["pediatric_t2_shell_width"],
                        "atlas_dilation": cfg["pediatric_t2_atlas_dilation"],
                        "t1_percentile": cfg["pediatric_t2_t1_percentile"],
                        "t2_percentile": cfg["pediatric_t2_t2_percentile"],
                        "max_added_cm3": cfg["pediatric_t2_max_added_cm3"],
                        "max_volume_cm3": cfg["max_brain_volume_cm3"],
                        "min_atlas_precision": cfg["pediatric_t2_min_atlas_precision"],
                        "allow_atlas_dice_decrease_on_undercoverage": True,
                        "target_min_atlas_volume_ratio": 0.90 if cfg.get("month") is not None and cfg.get("month") <= 2 else 0.93,
                        "min_recovery_added_cm3": 12.0 if cfg.get("month") is not None and cfg.get("month") <= 2 else 25.0,
                        "max_atlas_dice_drop": 0.14,
                    },
                )
                if pediatric_refinement_stats.get("pediatric_refinement_used"):
                    print(f"    pediatric T1/T2 refinement applied: +{pediatric_refinement_stats.get('pediatric_refinement_added_cm3', 0.0):.1f} cm3")
                elif pediatric_refinement_stats:
                    print(f"    pediatric T1/T2 refinement not applied: {pediatric_refinement_stats.get('pediatric_refinement_reason')}")

            # Snapshot/restore de l'entrée de l'étape wide (= post-étape pediatric-T2).
            if _idempotent:
                if _bk_wide.exists():
                    shutil.copyfile(_bk_wide, mask_path)
                else:
                    shutil.copyfile(mask_path, _bk_wide)

            wide_refinement_stats = {}
            if cfg.get("enable_wide_refinement", True) and selected_atlas is not None:
                wide_refinement_stats = apply_adult_external_refinement(
                    mask_path=mask_path,
                    brain_path=brain_path,
                    input_image_path=input_image_path,
                    atlas_path=selected_atlas,
                    metrics_path=metrics_path,
                    config={**cfg["wide_refinement"], "reference_atlas_trust_score": _reference_trust},
                )
                if wide_refinement_stats.get("adult_refinement_used"):
                    print(f"    wide pediatric atlas/T1 refinement applied: {wide_refinement_stats.get('adult_refinement_reason')}")
                elif wide_refinement_stats:
                    print(f"    wide pediatric atlas/T1 refinement not applied: {wide_refinement_stats.get('adult_refinement_reason')}")

            # Snapshot/restore de l'entrée de la fusion (= post-étape wide). La fusion
            # écrit mask_path ; garantir une entrée canonique la rend déterministe.
            if _idempotent:
                if _bk_fusion.exists():
                    shutil.copyfile(_bk_fusion, mask_path)
                else:
                    shutil.copyfile(mask_path, _bk_fusion)

            adaptive_fusion_stats = {}
            if cfg.get("enable_adaptive_mask_fusion", True) and selected_atlas is not None:
                try:
                    fusion_atlas_paths = _adaptive_fusion_atlas_paths(
                        anat_dir,
                        subject_id,
                        selected_atlas=selected_atlas,
                        adaptive_path=(adaptive_atlas_stats or {}).get("output_path"),
                    )
                    adaptive_ref = (adaptive_atlas_stats or {}).get("output_path")
                    if not adaptive_ref or not Path(str(adaptive_ref)).exists():
                        adaptive_ref = selected_atlas
                    adaptive_fusion_stats = apply_adaptive_mask_fusion(
                        mask_path=mask_path,
                        brain_path=brain_path,
                        input_image_path=input_image_path,
                        atlas_paths=fusion_atlas_paths,
                        reference_atlas_path=adaptive_ref,
                        seed_atlas_path=anat_dir / "atlases_advanced" / f"sub-{subject_id}_brain_mask_aff.nii.gz",
                        uncertainty_path=anat_dir / f"{subject_bids}_desc-brain_uncertainty.nii.gz",
                        t2_path=source_t2,
                        metrics_path=metrics_path,
                        config={
                            **cfg["adaptive_fusion"],
                            "reference_atlas_trust_score": _reference_trust,
                            # PATCH P9 — transmet l'âge en mois pour le prior normatif.
                            # PATCH P19 — utilise l'âge inféré par atlas si la session
                            # ne fournit pas d'âge (ex. SynthStripInfant).
                            "subject_age_months": (cfg.get("month") if cfg.get("month") is not None
                                                   else cfg.get("estimated_age_months")),
                        },
                    )
                    if adaptive_fusion_stats.get("adaptive_fusion_used"):
                        print(
                            "    final adaptive fusion applied: "
                            f"{adaptive_fusion_stats.get('adaptive_fusion_selected_candidate')} "
                            f"({adaptive_fusion_stats.get('adaptive_fusion_reason')})"
                        )
                    elif adaptive_fusion_stats:
                        print(f"    final adaptive fusion not applied: {adaptive_fusion_stats.get('adaptive_fusion_reason')}")
                except Exception as exc:
                    print(f"    final adaptive fusion failed: {exc}")
                    adaptive_fusion_stats = {"adaptive_fusion_reason": str(exc)}

                # PATCH P6 — QC assertion: la sortie finale DOIT être SAM2-derived.
                # Exception PATCH P10 : Emergency Atlas Rescue (SAM2 hors-cerveau) est
                # autorisé et ne constitue pas une violation — il est tracé séparément.
                if "adaptive_fusion_final_is_sam2_derived" in adaptive_fusion_stats:
                    _is_sam2 = bool(adaptive_fusion_stats.get("adaptive_fusion_final_is_sam2_derived"))
                    _sel = adaptive_fusion_stats.get("adaptive_fusion_selected_candidate", "?")
                    _p10 = bool(adaptive_fusion_stats.get("adaptive_fusion_p10_emergency_atlas_rescue", False))
                    if not _is_sam2 and not _p10:
                        print(
                            f"    [PATCH P4 VIOLATION] subject={sid} selected='{_sel}' "
                            f"final_is_sam2_derived=False — sortie atlas-pur indésirable"
                        )
                        adaptive_fusion_stats["adaptive_fusion_qc_violation"] = True
                    elif not _is_sam2 and _p10:
                        print(
                            f"    [PATCH P10] Emergency Atlas Rescue: '{_sel}' — SAM2 hors-cerveau, atlas autorisé"
                        )
                        adaptive_fusion_stats["adaptive_fusion_qc_violation"] = False
                    else:
                        adaptive_fusion_stats["adaptive_fusion_qc_violation"] = False

            try:
                metrics = api["compute_metrics"](mask_path, gs_mask)
            except Exception as e_metrics:
                print(f"    [WARN] compute_metrics failed: {e_metrics}")
                try:
                    _pred_nib = nib.load(str(mask_path))
                    _pred_d = _pred_nib.get_fdata() > 0.5
                    _gt_d   = nib.load(str(gs_mask)).get_fdata() > 0.5
                    _vox_v  = float(np.prod(_pred_nib.header.get_zooms()[:3])) / 1000.0
                    _inter  = (_pred_d & _gt_d).sum()
                    _dice   = 2.0 * _inter / max(_pred_d.sum() + _gt_d.sum(), 1)
                    metrics = {
                        "dice": float(_dice), "hd95": None, "assd": None,
                        "vol_pred_cm3": float(_pred_d.sum()) * _vox_v,
                        "vol_gs_cm3":   float(_gt_d.sum())  * _vox_v,
                    }
                    print(f"    [RESCUE] dice recalculated from output: {_dice:.4f}")
                except Exception as e2:
                    print(f"    [ERROR] rescue metrics failed: {e2}")
                    metrics = {"dice": None, "hd95": None, "assd": None}
            extract_metrics = {}
            if metrics_path.exists():
                try:
                    extract_metrics = json.loads(metrics_path.read_text())
                except Exception:
                    extract_metrics = {}
            quality = extract_metrics.get("quality_rating")
            if pediatric_refinement_stats.get("pediatric_refinement_used"):
                quality = "Good - pediatric T1/T2 refinement applied"
            if wide_refinement_stats.get("adult_refinement_used"):
                quality = "Good - pediatric T1/T2 + wide atlas/T1 refinement applied"
            if adaptive_fusion_stats.get("adaptive_fusion_used"):
                quality = "Good - adaptive mask fusion applied"

            metrics.update(
                {
                    "subject": sid,
                    "dataset": dataset,
                    "split": rec.get("split"),
                    "session": rec.get("session"),
                    "profile": rec.get("profile"),
                    "source_t1_path": str(source_t1),
                    "source_t2_path": str(source_t2 or ""),
                    "gold_mask_path": str(gs_mask),
                    "label_path": str(source_label or ""),
                    "original_t1_path": str(rec["t1_path"]),
                    "original_t2_path": str(rec.get("t2_path") or ""),
                    "original_gold_mask_path": str(rec["gold_path"]),
                    "original_label_path": str(rec.get("label_path") or ""),
                    "input_shape": img_info.get("shape"),
                    "input_zooms": img_info.get("zooms"),
                    "input_orientation": img_info.get("orientation"),
                    "gold_shape": gs_info.get("shape"),
                    "gold_zooms": gs_info.get("zooms"),
                    "gold_orientation": gs_info.get("orientation"),
                    "input_image_used": str(input_image_path),
                    "mask_path": str(mask_path),
                    "brain_path": str(brain_path),
                    "atlas_path": str(selected_atlas) if selected_atlas is not None else "",
                    "atlas_guidance_used": bool(_sam2_atlas_paths),
                    "atlas_rejected_count": len(rejected_atlases),
                    "atlas_volume_cm3": atlas_stats.get("volume_cm3"),
                    "atlas_components": atlas_stats.get("components"),
                    "atlas_qc_score": atlas_stats.get("qc_score"),
                    "atlas_selection_score": atlas_stats.get("selection_score"),
                    "atlas_image_alignment_score": atlas_stats.get("image_alignment_score"),
                    "atlas_intensity_ratio": atlas_stats.get("intensity_ratio"),
                    "adaptive_atlas_path": (adaptive_atlas_stats or {}).get("output_path"),
                    "adaptive_atlas_used": (adaptive_atlas_stats or {}).get("adaptive_atlas_used"),
                    "adaptive_atlas_reason": (adaptive_atlas_stats or {}).get("adaptive_atlas_reason"),
                    "adaptive_atlas_seed_variant": (adaptive_atlas_stats or {}).get("seed_variant"),
                    "adaptive_atlas_seed_volume_cm3": (adaptive_atlas_stats or {}).get("seed_volume_cm3"),
                    "adaptive_atlas_volume_cm3": (adaptive_atlas_stats or {}).get("adaptive_volume_cm3"),
                    "adaptive_atlas_added_cm3": (adaptive_atlas_stats or {}).get("adaptive_added_cm3"),
                    "dice_with_atlas": extract_metrics.get("dice_with_atlas"),
                    "atlas_coverage": extract_metrics.get("atlas_coverage"),
                    "atlas_precision": extract_metrics.get("atlas_precision"),
                    "fallback_used": extract_metrics.get("fallback_used"),
                    "fallback_reason": extract_metrics.get("fallback_reason"),
                    "refinement_used": extract_metrics.get("refinement_used", refinement_stats.get("refinement_used")),
                    "refinement_reason": extract_metrics.get("refinement_reason", refinement_stats.get("refinement_reason")),
                    "boundary_refinement_used": extract_metrics.get("boundary_refinement_used", refinement_stats.get("boundary_refinement_used")),
                    "boundary_refinement_reason": extract_metrics.get("boundary_refinement_reason", refinement_stats.get("boundary_refinement_reason")),
                    "pediatric_refinement_used": extract_metrics.get("pediatric_refinement_used", pediatric_refinement_stats.get("pediatric_refinement_used")),
                    "pediatric_refinement_reason": extract_metrics.get("pediatric_refinement_reason", pediatric_refinement_stats.get("pediatric_refinement_reason")),
                    "pediatric_refinement_added_cm3": extract_metrics.get("pediatric_refinement_added_cm3", pediatric_refinement_stats.get("pediatric_refinement_added_cm3")),
                    "iseg_wide_refinement_used": extract_metrics.get("adult_refinement_used", wide_refinement_stats.get("adult_refinement_used")),
                    "iseg_wide_refinement_reason": extract_metrics.get("adult_refinement_reason", wide_refinement_stats.get("adult_refinement_reason")),
                    "adaptive_fusion_used": extract_metrics.get("adaptive_fusion_used", adaptive_fusion_stats.get("adaptive_fusion_used")),
                    "adaptive_fusion_reason": extract_metrics.get("adaptive_fusion_reason", adaptive_fusion_stats.get("adaptive_fusion_reason")),
                    "adaptive_fusion_selected_candidate": extract_metrics.get(
                        "adaptive_fusion_selected_candidate", adaptive_fusion_stats.get("adaptive_fusion_selected_candidate")
                    ),
                    "quality_rating": quality,
                    "mask_volume_cm3_from_extractor": extract_metrics.get("mask_volume_cm3"),
                    "total_time_s": round(time.time() - t0, 1),
                    "error": "",
                }
            )
        except Exception as exc:
            traceback.print_exc()
            metrics = {
                "subject": sid,
                "dataset": dataset,
                "split": rec.get("split"),
                "session": rec.get("session"),
                "profile": rec.get("profile"),
                "source_t1_path": str(rec.get("t1_path", "")),
                "gold_mask_path": str(rec.get("gold_path", "")),
                "t2_path": str(rec.get("t2_path", "") or ""),
                "label_path": str(rec.get("label_path", "") or ""),
                "error": str(exc),
                "dice": float("nan"),
                "hd95": float("nan"),
                "assd": float("nan"),
                "total_time_s": round(time.time() - t0, 1),
            }

        rows.append(metrics)
        pd.DataFrame(rows).to_csv(live_csv, index=False)
        _write_progress(progress_json, dataset, "subject_finished", sid, idx, len(records), rows)
        if str(metrics.get("error", "")).strip():
            print(f"  -> ERROR: {metrics['error']}")
        else:
            print(
                f"  -> Dice={metrics.get('dice', float('nan')):.4f}  "
                f"HD95={metrics.get('hd95', float('nan')):.2f} mm  "
                f"Vol={metrics.get('vol_pred_cm3', float('nan')):.1f} cm3  "
                f"GS={metrics.get('vol_gs_cm3', float('nan')):.1f} cm3  "
                f"AtlasVol={metrics.get('atlas_volume_cm3', float('nan')):.1f} cm3  "
                f"t={metrics.get('total_time_s', 0):.0f}s"
            )

    df = pd.DataFrame(rows)
    df.to_csv(final_csv, index=False)
    _write_progress(progress_json, dataset, "finished", total=len(records), rows=rows)
    ok = df[df.get("error", "").fillna("").astype(str).str.len() == 0].copy() if "error" in df.columns else df.copy()
    ok["dice"] = pd.to_numeric(ok.get("dice"), errors="coerce")
    ok = ok.dropna(subset=["dice"])
    print(f"\nM1-PEDIATRIC-iSegStyle results {dataset}: N={len(ok)}/{len(records)}")
    if len(ok):
        cols = [
            c
            for c in [
                "subject",
                "session",
                "dice",
                "hd95",
                "assd",
                "vol_pred_cm3",
                "vol_gs_cm3",
                "atlas_volume_cm3",
                "atlas_selection_score",
                "pediatric_refinement_used",
                "iseg_wide_refinement_used",
                "adaptive_fusion_used",
                "adaptive_fusion_selected_candidate",
                "quality_rating",
            ]
            if c in ok.columns
        ]
        print(ok[cols].to_string(index=False))
        print(f"\nDice  : {ok.dice.mean():.4f} +/- {ok.dice.std():.4f}")
        print(f"HD95  : {ok.hd95.mean():.2f} +/- {ok.hd95.std():.2f} mm")
        print(f"ASSD  : {ok.assd.mean():.2f} +/- {ok.assd.std():.2f} mm")
        if target_mean_dice is not None and float(ok.dice.mean()) < float(target_mean_dice):
            print(f"Diagnostic: mean Dice below target {float(target_mean_dice):.4f}")
    print(f"\nFinal CSV: {final_csv}")
    print(f"Live CSV : {live_csv}")
    print(f"Progress : {progress_json}")
    return {
        "dataframe": df,
        "csv": final_csv,
        "live_csv": live_csv,
        "progress_json": progress_json,
        "selection_json": selection_json,
        "out_base": out_base,
    }
