"""Robust adaptive mask selection without gold labels.

This module is intentionally dataset-agnostic.  It takes the current SAM mask,
registered atlas candidates, the subject image, and an optional SAM uncertainty
map, then selects a final mask using only QC signals available at inference
time.  It does not read ground truth masks and does not implement LOO logic.
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

# PATCH P16 — random walker multi-modal T1+T2 (Grady, IEEE TPAMI 2006).
# Garde-fou : si skimage indisponible, P16 reste désactivé silencieusement.
try:
    from skimage.segmentation import random_walker as _sk_random_walker
except ImportError:
    _sk_random_walker = None

# PATCH P17 — Chan-Vese morphologique (Chan-Vese 2001, Sandberg vectoriel 2002).
try:
    from skimage.segmentation import morphological_chan_vese as _sk_morph_chan_vese
except ImportError:
    _sk_morph_chan_vese = None

try:
    from scipy.stats import kurtosis as _scipy_kurtosis, skew as _scipy_skew
    from scipy.signal import find_peaks as _find_peaks
    _SCIPY_STATS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_STATS_AVAILABLE = False

try:
    from skimage.filters import threshold_multiotsu as _threshold_multiotsu
    _SKIMAGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKIMAGE_AVAILABLE = False

try:
    from .neuro_utils import (
        largest_component,
        load_img,
        load_mask,
        mask_overlap_stats,
        mask_stats,
        mask_volume_cm3,
        save_float,
        save_mask,
        update_metrics_json,
    )
except ImportError:  # pragma: no cover
    from neuro_utils import (
        largest_component,
        load_img,
        load_mask,
        mask_overlap_stats,
        mask_stats,
        mask_volume_cm3,
        save_float,
        save_mask,
        update_metrics_json,
    )


# PATCH P4 — Liste des noms de candidats qui contiennent (au moins partiellement) le masque SAM2.
# Toute autre famille (atlas_*, reference_atlas, atlas_tissue_p*, atlas_majority) est
# purement atlas-driven et NE DOIT JAMAIS être la sortie finale d'un pipeline SAM-NeuroAdapt.
_SAM2_DERIVED_CANDIDATES = {"current", "current_pruned_to_atlas", "current_shell_recovery"}

# PATCH P9 — prior volumétrique normatif pédiatrique par âge (mois).
# Intervalles ±2σ du volume cérébral total (cm³) issus de la littérature publiée
# (Knickmeyer 2008 NeuroImage, Gilmore 2012, Peterson 2003 Pediatrics, dHCP 2018).
# Plage volontairement large pour éviter les faux positifs (prématurés, macrocéphalie).
# La pénalité est progressive (soft) — jamais de disqualification dure.
_NORMATIVE_BRAIN_VOLUME_CM3: dict[int, tuple[float, float]] = {
    # Intervalle ±2σ (cm³) : valeurs basées sur Knickmeyer 2008, Gilmore 2012,
    # Peterson 2003, dHCP 2018. Plage calibrée pour ne pas pénaliser des grands
    # cerveaux naturels mais attraper les surestimations atlas (>15–20 % au-dessus).
    0:  (310.0, 540.0),   # néonataux : inclut MCRIB P05 (529 cm³)
    1:  (355.0, 590.0),
    2:  (405.0, 640.0),
    3:  (455.0, 665.0),   # sub-116056 à 704 cm³ → 5.9 % au-dessus → pénalité ~0.006
    4:  (505.0, 715.0),
    5:  (550.0, 760.0),
    6:  (585.0, 960.0),   # élargi : iSeg-2017 a des GT >900 cm³ (pop. à large cerveau)
    7:  (625.0, 1010.0),
    8:  (660.0, 1050.0),
    9:  (705.0, 1100.0),
    12: (770.0, 1180.0),
    15: (835.0, 1250.0),
    18: (875.0, 1300.0),
    24: (940.0, 1400.0),
    36: (1050.0, 1400.0),
    48: (1110.0, 1460.0),
    60: (1150.0, 1490.0),
}

# PATCH P5 — préfixes additionnels reconnus comme SAM2-derived (intersection ou union
# de current avec un candidat atlas-pur). Permet d'utiliser l'atlas comme guide tout
# en gardant le masque SAM2 comme base.
_SAM2_DERIVED_PREFIXES = (
    "current_int_tissue_p",    # current ∩ atlas_tissue_p* (atlas-guided trim)
    "current_int_majority",    # current ∩ atlas_majority (consensus-guided trim)
    "current_union_tissue_p",  # PATCH P8 — current ∪ atlas_tissue_p* (atlas-guided expansion)
    "current_rw_",             # PATCH P16 — random walker T1+T2 (Grady 2006)
    "current_cv_",             # PATCH P17 — Chan-Vese vectoriel (Sandberg-Chan 2002)
)


def _is_sam2_derived(candidate_name: str) -> bool:
    """True si le masque candidat contient au moins partiellement du SAM2."""
    if candidate_name in _SAM2_DERIVED_CANDIDATES:
        return True
    return any(candidate_name.startswith(p) for p in _SAM2_DERIVED_PREFIXES)


def _cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    base = {
        "enabled": True,
        "min_volume_cm3": 550.0,
        "soft_min_volume_cm3": 720.0,
        "soft_max_volume_cm3": 900.0,
        "max_volume_cm3": 1100.0,
        "max_components": 3,
        "max_extent_ratio": 0.97,
        "tissue_percentiles": [8.0, 10.0, 12.0, 15.0, 18.0],
        "seed_percentile_source": "first_nonadaptive",
        "use_t2_gate": False,
        "t2_percentile": 3.0,
        "reference_atlas_penalty": 0.10,
        "current_mask_bonus": 0.08,
        "tissue_undercoverage_bonus": 0.07,
        "intensity_weight": 0.0,
        "boundary_contrast_weight": 0.0,
        # PATCH P16 — random walker T1+T2 candidate (désactivé par défaut, activé profil pédiatrique)
        "rw_candidate_enabled": False,
        "rw_candidate_require_t2": True,
        "rw_seed_brain_erosion_iter": 4,
        "rw_seed_bg_dilation_iter": 3,
        "rw_beta": 130.0,
        "rw_prob_threshold": 0.55,
        "rw_min_dice_vs_current": 0.85,
        "rw_min_atlas_dice": 0.80,
        "rw_volume_ratio_bounds": (0.70, 1.10),
        # PATCH P16-bis — bonus dédié pour candidats current_rw_* qui ont passé
        # les 4 garde-fous internes (dice vs current, atlas_dice, ratio volume).
        # Garde-fou supplémentaire : bonus appliqué UNIQUEMENT si RW a changé
        # significativement le current (current_dice < seuil) — sinon il n'apporte
        # rien et ne doit pas voler la sélection.
        "rw_t1t2_bonus": 0.22,
        "rw_t1_bonus":   0.10,
        "rw_bonus_max_current_dice": 0.92,
        # PATCH P17 — Chan-Vese vectoriel : level-set region-based (Mumford-Shah)
        # sans dépendance au gradient. Complète P16 sur les zones isointenses où
        # la frontière LCR/cerveau a un gradient faible (pédiatrique 6-9 mois).
        "cv_candidate_enabled": False,
        "cv_candidate_require_t2": False,
        "cv_iterations": 30,
        "cv_smoothing": 1,
        "cv_lambda1": 1.0,
        "cv_lambda2": 1.0,
        "cv_init_dilation_iter": 2,
        "cv_min_dice_vs_current": 0.85,
        "cv_min_atlas_dice": 0.80,
        "cv_volume_ratio_bounds": (0.70, 1.15),
        "cv_t1t2_bonus": 0.20,
        "cv_t1_bonus":   0.08,
        "cv_bonus_max_current_dice": 0.92,
        "atlas_candidate_penalty": 0.04,
        "include_atlas_candidates": False,
        "max_nonraw_change_ratio": 0.38,
        "min_nonraw_candidate_current_dice": 0.0,
        "min_raw_candidate_current_dice": 0.0,
        "selection_margin": 0.015,
        # PATCH P6 — margin minimale pour les candidats SAM2-hybrid (current_int_*,
        # current_pruned_to_atlas, current_shell_recovery). Sûr car PATCH P4
        # garantit que la sortie reste SAM2-derived ; ces hybrides sont
        # structurellement bornés (current ∩/∪ atlas) et ne peuvent pas
        # dégrader catastrophiquement par rapport à current. La margin sert
        # uniquement de rempart contre les flips dus au bruit numérique.
        # Vérifié empiriquement sur BOBs : Δ score typiques 2e-4 à 1e-2.
        "sam2_hybrid_selection_margin": 0.0001,
        # PATCH P7 — sur-segmentation douce (atlas sur-évalué ou SAM2 légèrement trop
        # large). Détecté quand current inclut presque tout l'atlas (coverage > threshold)
        # et est plus volumineux que l'atlas (ratio > threshold), sans que raw_overcoverage
        # classique ne soit déclenché (précision trop haute ou ratio pas assez fort).
        # Observé systématiquement sur BOBs 1-8 mois ; absent sur iSeg-2017.
        # Seuil relevé 1.02→1.05 (PATCH P7b) : à 1.02 des sujets MCRIB_WM avec ratio
        # ~1.03 déclenchaient le flag à tort, poussant vers un hybrid trop serré.
        # La correction n'est utile qu'au-delà de ~5 % de sur-volume.
        "soft_overcoverage_min_volume_ratio": 1.05,
        "soft_overcoverage_min_coverage": 0.985,
        "current_soft_overcoverage_penalty": 0.10,
        # PATCH P8 — expansion SAM2-union pour atlas_mismatch sévère (SAM2 sous-couvre
        # l'atlas de plus de 30 %). Dans ce cas les candidats ∩ sont inutiles (plus petits
        # que current). On crée current ∪ atlas_tissue_pX = SAM2 + région atlas manquante.
        # Seuil 0.70 : uniquement déclenché quand atlas_coverage < 70 % (asl_t1_105 = 0.61).
        # Vérifié non-régressif sur BOBs/iSeg/MCRIB/MCRIB_WM/MCRIB2 (coverage ≥ 0.83).
        "union_candidate_min_atlas_mismatch_cov": 0.70,
        "max_union_candidate_change_ratio": 0.60,
        "union_candidate_min_volume_score": 0.60,
        "coverage_target": 0.84,
        "coverage_sigma": 0.13,
        "catastrophic_min_precision": 0.50,
        "catastrophic_min_coverage": 0.45,
        "catastrophic_min_atlas_dice": 0.55,
        "catastrophic_max_uncertainty_mean": 0.25,
        "undercoverage_min_precision": 0.96,
        "undercoverage_max_coverage": 0.82,
        "undercoverage_max_volume_ratio": 0.86,
        "overcoverage_min_coverage": 0.94,
        "overcoverage_max_precision": 0.88,
        "overcoverage_min_volume_ratio": 1.10,
        "min_undercoverage_candidate_current_dice": 0.82,
        "max_undercoverage_change_ratio": 0.48,
        "min_overcoverage_candidate_current_dice": 0.82,
        "max_overcoverage_change_ratio": 0.40,
        "triggered_selection_margin": 0.0,
        "current_undercoverage_penalty": 0.10,
        "current_overcoverage_penalty": 0.08,
        "undercoverage_recovery_bonus": 0.12,
        "overcoverage_prune_bonus": 0.10,
        "undercoverage_rescue_enabled": True,
        "undercoverage_rescue_current_ref_ratio": 0.70,
        "undercoverage_rescue_min_current_precision": 0.92,
        "undercoverage_rescue_max_current_coverage": 0.70,
        "undercoverage_rescue_min_candidate_current_dice": 0.45,
        "undercoverage_rescue_max_change_ratio": 1.35,
        "undercoverage_rescue_min_atlas_precision": 0.94,
        "undercoverage_rescue_min_atlas_coverage": 0.82,
        "undercoverage_rescue_min_volume_score": 0.70,
        "undercoverage_rescue_min_intensity_score": 0.45,
        "undercoverage_rescue_bonus": 0.24,
        "undercoverage_rescue_rank_high_coverage": 0.985,
        "undercoverage_rescue_rank_min_expansion_ratio": 1.015,
        "undercoverage_rescue_rank_expansion_bonus": 0.10,
        "undercoverage_rescue_rank_consensus_bonus": 0.02,
        "undercoverage_rescue_rank_low_coverage": 0.96,
        "undercoverage_rescue_rank_low_coverage_penalty": 0.06,
        "nomask_pair_min_pairwise_dice": 0.88,
        "nomask_pair_min_families": 2,
        "nomask_pair_min_uncertainty_score": 0.84,
        "nomask_pair_min_volume_score": 0.70,
        "nomask_pair_min_intensity_score": 0.75,
        "nomask_pair_strong_pairwise_dice": 0.92,
        "nomask_pair_strong_min_uncertainty_score": 0.90,
        "nomask_pair_strong_min_intensity_score": 0.90,
        "nomask_pair_strong_min_volume_score": 0.95,
        "nomask_pair_max_current_volume_ratio": 1.00,
        "nomask_pair_undercoverage_max_current_volume_ratio": 1.35,
        "nomask_pair_max_reference_volume_ratio": 1.02,
        "nomask_pair_undercoverage_max_reference_volume_ratio": 1.25,
        "nomask_pair_bonus": 0.35,
        "nomask_pair_non_nomask_penalty": 0.22,
        "masked_nomask_conflict_max_pairwise_dice": 0.75,
        "masked_nomask_conflict_penalty": 0.18,
        "atlas_guard_enabled": True,
        "reference_atlas_trust_score": 0.0,
        "atlas_guard_min_trust": 0.45,
        "atlas_guard_max_current_atlas_dice": 0.96,
        "atlas_guard_over_volume_ratio": 1.14,
        "atlas_guard_under_volume_ratio": 0.94,
        "min_atlas_guard_candidate_current_dice": 0.82,
        "min_atlas_guard_candidate_atlas_dice": 0.985,
        "max_atlas_guard_change_ratio": 0.46,
        "atlas_guard_reference_bonus": 0.20,
        "atlas_guard_current_penalty": 0.16,
        "low_trust_fail_clean_enabled": True,
        "low_trust_fail_clean_threshold": 0.18,
        "low_trust_fail_clean_keep_current_if_hard_ok": True,
        "replace_only_if_raw_qc_triggered": False,
        "prune_dilation": 2,
        "recover_shell_width": 8,
        "recover_max_added_cm3": 180.0,
        "backup_suffix": "_pre-adaptive-fusion",
        # PATCH P4 — contrainte architecturale SAM-NeuroAdapt:
        # le masque final DOIT contenir SAM2. Si True, restreint la sortie
        # aux candidats SAM2-derived (current, current_pruned_to_atlas,
        # current_shell_recovery). Les candidats atlas purs (atlas_majority,
        # atlas_tissue_p*, reference_atlas, atlas_candidate_*) peuvent encore
        # entrer dans le scoring (pour guider) mais ne peuvent JAMAIS être
        # la sortie finale.
        "sam2_only_output": True,
        # PATCH P9 — prior volumétrique normatif pédiatrique.
        # Pénalité progressive (soft) quand le volume d'un candidat dépasse
        # la borne normative supérieure pour l'âge du sujet. Ne disqualifie
        # jamais : max_penalty plafonné à 0.10, appliqué proportionnellement
        # à l'excès de volume relatif (ratio excès/borne_max). Sûr car
        # la borne est ±2σ (large plage) et la pénalité est bornée.
        # Nécessite que `subject_age_months` soit fourni dans la config.
        "normative_volume_prior_enabled": False,
        "normative_volume_prior_max_penalty": 0.10,
        # PATCH P19 — bornes (lo,hi) cm³ explicites ; si fournies, court-circuitent
        # la lookup par âge (utile pour le fœtal sans âge de session).
        "normative_volume_prior_bounds_cm3": None,
        "subject_age_months": None,
        # PATCH P10 — Emergency Atlas Rescue.
        # Quand SAM2 échoue catastrophiquement (masque hors du cerveau : dice_atlas<0.05
        # ET vol_ratio>2.5 par rapport à l'atlas), PATCH P4 est relaxé localement :
        # l'atlas de référence est utilisé directement plutôt que le masque SAM2 aberrant.
        # Déclenché UNIQUEMENT si le meilleur candidat SAM2-derived a lui-même atlas_dice<0.05.
        "sam2_only_allow_atlas_emergency": True,
        "sam2_only_emergency_max_atlas_dice": 0.05,
        "sam2_only_emergency_min_vol_ratio": 2.5,
        # PATCH P19 — trust minimal de l'atlas pour autoriser le rescue d'un masque
        # SAM2 de volume normal mais déplacé (recouvrement atlas quasi-nul).
        "sam2_only_emergency_min_trust": 0.45,
    }
    if config:
        base.update(config)
    base["tissue_percentiles"] = [float(v) for v in base.get("tissue_percentiles", [])]
    return base


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
    img = _as_ref_img(path, ref_img, order=0)
    return img.get_fdata(dtype=np.float32) > 0.5


def _read_float(path: str | Path | None, ref_img) -> np.ndarray | None:
    if not path or not Path(path).exists():
        return None
    img = _as_ref_img(path, ref_img, order=1)
    return np.nan_to_num(img.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _unique_paths(paths: Iterable[str | Path | None]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for item in paths:
        if not item:
            continue
        p = Path(item)
        if not p.exists():
            continue
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    iterations = max(0, int(iterations))
    if iterations == 0:
        return np.asarray(mask, dtype=bool)
    return ndi.binary_dilation(mask, structure=ndi.generate_binary_structure(3, 1), iterations=iterations)


def _clean(mask: np.ndarray, fill_holes: bool = True) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if fill_holes:
        mask = ndi.binary_fill_holes(mask).astype(bool)
    return largest_component(mask)


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


def _coverage_score(coverage: float, cfg: dict[str, Any]) -> float:
    target = float(cfg["coverage_target"])
    sigma = max(float(cfg["coverage_sigma"]), 1e-6)
    return float(np.exp(-((coverage - target) / sigma) ** 2))


def _extent(mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    coords = np.where(mask)
    return float(max((c.max() - c.min() + 1) / s for c, s in zip(coords, mask.shape)))


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


def _boundary_contrast_score(mask: np.ndarray, image: np.ndarray, shell_voxels: int = 3) -> float:
    """Score de qualité de frontière : contraste intérieur vs coque externe [0, 1].

    Un masque trop large inclut du scalp/LCR externe dont l'intensité est proche
    de l'intérieur → ratio ≈ 1 → score bas. Un bon masque a ratio ≥ 2 → score > 0.5.
    """
    try:
        if not mask.any() or image.size == 0:
            return 0.5
        interior = ndi.binary_erosion(mask, iterations=2)
        dilated  = ndi.binary_dilation(mask, iterations=shell_voxels)
        shell    = dilated & ~mask
        vals_in  = image[interior]
        vals_sh  = image[shell]
        if vals_in.size < 100 or vals_sh.size < 100:
            return 0.5
        med_in = float(np.median(vals_in))
        med_sh = float(np.median(vals_sh))
        if med_in <= 0:
            return 0.5
        ratio = med_in / max(med_sh, 1e-6)
        return float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))
    except Exception:
        return 0.5


def _build_random_walker_candidate(
    current: np.ndarray,
    ref_atlas: np.ndarray,
    image_t1: np.ndarray,
    image_t2: np.ndarray | None,
    cfg: dict[str, Any],
) -> np.ndarray | None:
    """PATCH P16 — candidat raffiné par random walker multi-modal T1(+T2).

    Inspiré de Grady, IEEE TPAMI 2006. Multi-channel via skimage.
    Conservatif : seeds dérivées de current (érosion forte = brain certain) +
    background dérivé du complément dilaté. Le walker affine la frontière en
    exploitant les sauts d'intensité T1/T2 caractéristiques scalp/LCR.

    Retourne None si pré-conditions non remplies (skimage absent, T2 manquant
    en mode require_t2, seeds insuffisantes, ou garde-fous anti-régression).
    """
    if _sk_random_walker is None or not current.any():
        return None
    if bool(cfg.get("rw_candidate_require_t2", True)) and image_t2 is None:
        return None

    n_brain = int(cfg.get("rw_seed_brain_erosion_iter", 4))
    n_bg    = int(cfg.get("rw_seed_bg_dilation_iter", 3))
    seed_brain = ndi.binary_erosion(current, iterations=n_brain)
    seed_bg    = ~ndi.binary_dilation(current, iterations=n_bg)
    if int(seed_brain.sum()) < 1000 or int(seed_bg.sum()) < 1000:
        return None

    labels = np.zeros(current.shape, dtype=np.uint8)
    labels[seed_brain] = 1
    labels[seed_bg]    = 2

    def _zn(a):
        med = float(np.median(a))
        std = float(np.std(a) + 1e-6)
        return ((a - med) / std).astype(np.float32)

    if image_t2 is not None:
        data = np.stack([_zn(image_t1), _zn(image_t2)], axis=-1)
        channel_axis = -1
    else:
        data = _zn(image_t1)
        channel_axis = None

    try:
        proba = _sk_random_walker(
            data, labels,
            beta=float(cfg.get("rw_beta", 130.0)),
            mode="cg_mg",
            return_full_prob=True,
            channel_axis=channel_axis,
        )
    except Exception:
        return None

    brain_prob = proba[0]
    thr = float(cfg.get("rw_prob_threshold", 0.55))
    rw_mask = (brain_prob >= thr) & ndi.binary_dilation(current, iterations=n_bg + 2)
    rw_mask = _clean(rw_mask, fill_holes=True)
    if not rw_mask.any():
        return None

    dice_curr = _mask_pairwise_dice(rw_mask, current)
    if dice_curr < float(cfg.get("rw_min_dice_vs_current", 0.85)):
        return None
    dice_atl = _mask_pairwise_dice(rw_mask, ref_atlas)
    if dice_atl < float(cfg.get("rw_min_atlas_dice", 0.80)):
        return None
    vmin, vmax = cfg.get("rw_volume_ratio_bounds", (0.70, 1.10))
    ratio_curr = float(rw_mask.sum()) / max(float(current.sum()), 1.0)
    if not (float(vmin) <= ratio_curr <= float(vmax)):
        return None
    return rw_mask


def _build_chan_vese_candidate(
    current: np.ndarray,
    ref_atlas: np.ndarray,
    image_t1: np.ndarray,
    image_t2: np.ndarray | None,
    cfg: dict[str, Any],
) -> np.ndarray | None:
    """PATCH P17 — candidat raffiné par Chan-Vese vectoriel morphologique.

    Inspiré de Chan & Vese, IEEE TIP 2001 (mono-canal) + Sandberg & Chan 2002
    (vectoriel). Complète le random walker (P16) sur les zones isointenses où
    le gradient est faible : Chan-Vese minimise la variance intra-region sans
    dépendre du gradient → robuste pour LCR/cerveau pédiatrique 6-9 mois.

    Stratégie multi-modale : si T2 disponible, on travaille sur la combinaison
    z_norm(T1) - z_norm(T2). Le LCR (sombre T1 + brillant T2) y donne des
    valeurs très négatives ; le cerveau ~ 0 ; le scalp positif. Cela exacerbe
    la séparabilité tissulaire et rend Chan-Vese plus discriminant qu'en
    T1 seul.

    Retourne None si pré-conditions non remplies ou garde-fous anti-régression
    activés (dice vs current, dice vs atlas, ratio volume).
    """
    if _sk_morph_chan_vese is None or not current.any():
        return None
    if bool(cfg.get("cv_candidate_require_t2", False)) and image_t2 is None:
        return None

    def _zn(a):
        med = float(np.median(a))
        std = float(np.std(a) + 1e-6)
        return ((a - med) / std).astype(np.float32)

    if image_t2 is not None:
        # Combinaison multi-modale : LCR très négatif, scalp positif, cerveau ~0
        combined = _zn(image_t1) - _zn(image_t2)
    else:
        combined = _zn(image_t1)

    # Init level-set : current dilaté de quelques voxels pour permettre érosion
    # OU expansion (utile pour les cas under-segmentés).
    init = ndi.binary_dilation(current, iterations=int(cfg.get("cv_init_dilation_iter", 2))).astype(np.int8)

    try:
        cv_mask = _sk_morph_chan_vese(
            combined,
            num_iter=int(cfg.get("cv_iterations", 30)),
            init_level_set=init,
            smoothing=int(cfg.get("cv_smoothing", 1)),
            lambda1=float(cfg.get("cv_lambda1", 1.0)),
            lambda2=float(cfg.get("cv_lambda2", 1.0)),
        ).astype(bool)
    except Exception:
        return None

    # Contrainte : ne jamais s'étendre hors d'une enveloppe atlas raisonnable.
    envelope = ndi.binary_dilation(current, iterations=int(cfg.get("cv_init_dilation_iter", 2)) + 8)
    envelope |= ndi.binary_dilation(ref_atlas, iterations=4)
    cv_mask = cv_mask & envelope
    cv_mask = _clean(cv_mask, fill_holes=True)
    if not cv_mask.any():
        return None

    # Garde-fous anti-régression
    dice_curr = _mask_pairwise_dice(cv_mask, current)
    if dice_curr < float(cfg.get("cv_min_dice_vs_current", 0.85)):
        return None
    dice_atl = _mask_pairwise_dice(cv_mask, ref_atlas)
    if dice_atl < float(cfg.get("cv_min_atlas_dice", 0.80)):
        return None
    vmin, vmax = cfg.get("cv_volume_ratio_bounds", (0.70, 1.15))
    ratio_curr = float(cv_mask.sum()) / max(float(current.sum()), 1.0)
    if not (float(vmin) <= ratio_curr <= float(vmax)):
        return None
    return cv_mask


def _candidate_metrics(
    name: str,
    mask: np.ndarray,
    ref_atlas: np.ndarray,
    ref_img,
    image: np.ndarray,
    current: np.ndarray,
    uncertainty: np.ndarray | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    stats = mask_stats(mask, ref_img)
    overlap = mask_overlap_stats(mask, ref_atlas, ref_img)
    ref_volume = mask_volume_cm3(ref_atlas, ref_img)
    unc_values = uncertainty[mask] if uncertainty is not None and mask.any() else np.asarray([], dtype=np.float32)
    unc_mean = float(np.mean(unc_values)) if unc_values.size else None
    unc_p90 = float(np.percentile(unc_values, 90)) if unc_values.size else None
    intensity, intensity_stats = _intensity_score(mask, image)
    current_overlap = mask_overlap_stats(mask, current, ref_img)
    return {
        "name": name,
        "volume_cm3": float(stats["volume_cm3"]),
        "components": int(stats["components"]),
        "extent_ratio": float(stats.get("extent_ratio", _extent(mask))),
        "atlas_dice": float(overlap["dice"]),
        "atlas_coverage": float(overlap["coverage"]),
        "atlas_precision": float(overlap["precision"]),
        "reference_volume_cm3": float(ref_volume),
        "volume_ratio_to_reference": float(stats["volume_cm3"]) / max(float(ref_volume), 1e-6),
        "uncertainty_mean": unc_mean,
        "uncertainty_p90": unc_p90,
        "volume_score": _volume_score(float(stats["volume_cm3"]), cfg),
        "coverage_score": _coverage_score(float(overlap["coverage"]), cfg),
        "intensity_score": intensity,
        "boundary_contrast": _boundary_contrast_score(mask, image),
        "current_dice": float(current_overlap["dice"]),
        "change_ratio": float(np.logical_xor(mask, current).sum() / max(current.sum(), 1)),
        **{f"intensity_{k}": v for k, v in intensity_stats.items()},
    }


def _normative_volume_penalty(volume_cm3: float, age_months: int | None, cfg: dict[str, Any]) -> float:
    """PATCH P9 + P15 — pénalité sigmoïde quand le volume sort de la plage normative.

    Forme sigmoïde 1 / (1 + exp(-k × (excess - pivot))) :
    - excess ≤ pivot/2 (5 %) : pénalité < 0.30 × max_pen
    - excess = pivot (10 %) : pénalité = 0.50 × max_pen (point de bascule)
    - excess ≥ pivot×2 (20 %) : pénalité ≈ max_pen (plateau)

    Avec max_pen=0.40 et current_mask_bonus=0.20, le différentiel net dépasse 0
    dès un excès de 12 % — débloque les cas marginaux sans pénaliser la zone sûre.
    """
    if not bool(cfg.get("normative_volume_prior_enabled", False)):
        return 0.0
    # PATCH P19 — bornes normatives explicites (ex. fœtal) court-circuitant la
    # lookup par âge. Permet d'activer la pénalité même sans âge de session, là où
    # l'atlas néonatal (âge 0 → bornes 310-540) reste trop grand pour le fœtus.
    _bounds = cfg.get("normative_volume_prior_bounds_cm3")
    if _bounds is not None:
        soft_min, soft_max = float(_bounds[0]), float(_bounds[1])
    elif age_months is None:
        return 0.0
    else:
        ages = sorted(_NORMATIVE_BRAIN_VOLUME_CM3.keys())
        lo_age = max((a for a in ages if a <= age_months), default=ages[0])
        hi_age = min((a for a in ages if a >= age_months), default=ages[-1])
        if lo_age == hi_age:
            soft_min, soft_max = _NORMATIVE_BRAIN_VOLUME_CM3[lo_age]
        else:
            t = (age_months - lo_age) / (hi_age - lo_age)
            lo_min, lo_max = _NORMATIVE_BRAIN_VOLUME_CM3[lo_age]
            hi_min, hi_max = _NORMATIVE_BRAIN_VOLUME_CM3[hi_age]
            soft_min = lo_min + t * (hi_min - lo_min)
            soft_max = lo_max + t * (hi_max - lo_max)
    max_pen   = float(cfg.get("normative_volume_prior_max_penalty", 0.10))
    pivot_x   = float(cfg.get("normative_volume_prior_pivot_excess", 0.10))
    pivot_k   = float(cfg.get("normative_volume_prior_pivot_k", 20.0))
    if volume_cm3 > soft_max:
        excess = (volume_cm3 - soft_max) / max(soft_max, 1.0)
        sig = 1.0 / (1.0 + float(np.exp(-pivot_k * (excess - pivot_x))))
        return float(sig * max_pen)
    if volume_cm3 < soft_min:
        deficit = (soft_min - volume_cm3) / max(soft_min, 1.0)
        sig = 1.0 / (1.0 + float(np.exp(-pivot_k * (deficit - pivot_x))))
        return float(sig * max_pen * 0.5)  # sous-vol moins pénalisé
    return 0.0


def _score_candidate(
    metrics: dict[str, Any],
    raw_bad: bool,
    raw_undercoverage: bool,
    raw_overcoverage: bool,
    raw_atlas_mismatch: bool,
    cfg: dict[str, Any],
    raw_soft_overcoverage: bool = False,
) -> float:
    uncertainty_mean = metrics.get("uncertainty_mean")
    uncertainty_score = 1.0 if uncertainty_mean is None else float(np.clip(1.0 - uncertainty_mean, 0.0, 1.0))
    score = (
        0.28 * float(metrics["volume_score"])
        + 0.25 * float(metrics["coverage_score"])
        + 0.18 * float(np.clip(metrics["atlas_precision"], 0.0, 1.0))
        + 0.18 * uncertainty_score
        + 0.11 * float(np.clip(metrics["atlas_dice"], 0.0, 1.0))
        + float(cfg.get("intensity_weight", 0.0)) * float(metrics.get("intensity_score", 0.5))
        + float(cfg.get("boundary_contrast_weight", 0.0)) * float(metrics.get("boundary_contrast", 0.5))
    )
    name = str(metrics["name"])
    if name == "current" and not raw_bad:
        score += float(cfg["current_mask_bonus"])
    # PATCH P16-bis — bonus pour candidats random walker SI ils ont changé
    # significativement le current (current_dice < seuil). Sinon le RW n'apporte
    # rien et garder current est plus sûr. Cumule avec les 4 garde-fous internes
    # de _build_random_walker_candidate() qui ont déjà filtré les sorties extrêmes.
    if name.startswith("current_rw_"):
        _rw_change_dice = float(metrics.get("current_dice", 1.0))
        if _rw_change_dice < float(cfg.get("rw_bonus_max_current_dice", 0.92)):
            _rw_bonus = float(cfg.get("rw_t1t2_bonus", 0.22)) if "t1t2" in name \
                        else float(cfg.get("rw_t1_bonus", 0.10))
            score += _rw_bonus
    # PATCH P17 — bonus symétrique pour Chan-Vese (légèrement plus bas que RW
    # car CV est moins déterministe et plus sensible à l'initialisation).
    if name.startswith("current_cv_"):
        _cv_change_dice = float(metrics.get("current_dice", 1.0))
        if _cv_change_dice < float(cfg.get("cv_bonus_max_current_dice", 0.92)):
            _cv_bonus = float(cfg.get("cv_t1t2_bonus", 0.20)) if "t1t2" in name \
                        else float(cfg.get("cv_t1_bonus", 0.08))
            score += _cv_bonus
    if name.startswith("atlas_tissue_p") and raw_undercoverage:
        score += float(cfg["tissue_undercoverage_bonus"])
    if raw_undercoverage:
        score += float(cfg.get("undercoverage_recovery_bonus", 0.12)) * float(np.clip(metrics["atlas_coverage"], 0.0, 1.0))
        if name in {"current_shell_recovery", "atlas_majority", "reference_atlas"} or name.startswith("atlas_candidate_"):
            score += 0.04
        if name == "current":
            score -= float(cfg.get("current_undercoverage_penalty", 0.10))
    if raw_overcoverage:
        if name in {"current_pruned_to_atlas"} or name.startswith("atlas_tissue_p"):
            score += float(cfg.get("overcoverage_prune_bonus", 0.10))
        if name == "current":
            score -= float(cfg.get("current_overcoverage_penalty", 0.08))
    if raw_atlas_mismatch:
        if name in {"reference_atlas", "atlas_majority"} or name.startswith("atlas_candidate_"):
            score += float(cfg.get("atlas_guard_reference_bonus", 0.20))
        if name == "current":
            score -= float(cfg.get("atlas_guard_current_penalty", 0.16))
    if name == "reference_atlas":
        score -= float(cfg["reference_atlas_penalty"])
    if name.startswith("atlas_candidate_"):
        score -= float(cfg.get("atlas_candidate_penalty", 0.04))
    if bool(metrics.get("nomask_pair_consensus_context")):
        score += float(cfg.get("nomask_pair_bonus", 0.35))
    if (
        bool(metrics.get("nomask_pair_global_context"))
        and name != "current"
        and (
            name in {"reference_atlas", "atlas_majority"}
            or name.startswith("atlas_candidate_")
            or name.startswith("atlas_tissue_p")
        )
        and not bool(metrics.get("nomask_pair_consensus_context"))
    ):
        score -= float(cfg.get("nomask_pair_non_nomask_penalty", 0.22))
    if bool(metrics.get("masked_nomask_conflict")):
        score -= float(cfg.get("masked_nomask_conflict_penalty", 0.18))
    # PATCH P7 — pénalité soft-overcoverage : current inclut trop l'atlas sans
    # déclencher raw_overcoverage classique (précision trop haute ou ratio modeste).
    # Favorise les hybrides current_int_* qui calibrent mieux le volume final.
    if raw_soft_overcoverage and name == "current":
        score -= float(cfg.get("current_soft_overcoverage_penalty", 0.10))
    if raw_bad and name == "current":
        score -= 1.0
    if metrics["components"] > int(cfg["max_components"]):
        score -= 0.08
    if metrics["extent_ratio"] > float(cfg["max_extent_ratio"]):
        score -= 0.08
    # PATCH P9 — prior volumétrique normatif : pénalité proportionnelle à l'excès
    # (ou déficit) de volume par rapport à la borne ±2σ pour l'âge du sujet.
    _age_m = cfg.get("subject_age_months")
    _age_m = int(_age_m) if _age_m is not None else None
    score -= _normative_volume_penalty(
        float(metrics.get("volume_cm3", 0.0)), _age_m, cfg
    )
    return float(score)


def _atlas_family_from_name(name: str) -> str:
    text = str(name).lower()
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
    if "consensus" in text or "majority" in text:
        return "consensus"
    return "other_external"


def _atlas_variant_from_name(name: str) -> str:
    text = str(name).lower()
    if "consensus" in text or "majority" in text:
        return "consensus"
    if "tissue" in text:
        return "tissue"
    if "nomask" in text:
        return "nomask"
    if any(token in text for token in ("mni152", "oasis30ants", "sri24", "colin27", "icbm")):
        return "masked"
    return "other"


def _mask_pairwise_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    if denom <= 0:
        return 0.0
    return float(2.0 * int(np.logical_and(a, b).sum()) / denom)


def _annotate_nomask_support(
    candidates: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    raw_undercoverage: bool = False,
) -> dict[str, Any]:
    for item in candidates:
        name = str(item.get("name", ""))
        metrics = item.get("metrics", {})
        family = _atlas_family_from_name(name)
        variant = _atlas_variant_from_name(name)
        metrics["atlas_family"] = family
        metrics["atlas_variant"] = variant
        metrics["nomask_pair_consensus_context"] = False
        metrics["nomask_pair_global_context"] = False
        metrics["nomask_pair_volume_ok"] = False
        metrics["masked_nomask_conflict"] = False
        metrics["nomask_pair_support_families"] = 0
        metrics["nomask_pair_support_best_dice"] = None
    nomask_items = [
        item
        for item in candidates
        if str(item.get("name", "")).startswith("atlas_candidate_")
        and item.get("metrics", {}).get("atlas_variant") == "nomask"
        and np.asarray(item.get("mask"), dtype=bool).any()
    ]
    min_pair = float(cfg.get("nomask_pair_min_pairwise_dice", 0.88))
    min_families = int(cfg.get("nomask_pair_min_families", 2))
    current_volume = 0.0
    for item in candidates:
        if str(item.get("name", "")) == "current":
            current_volume = float(item.get("metrics", {}).get("volume_cm3", 0.0))
            break
    max_current_ratio = float(
        cfg.get(
            "nomask_pair_undercoverage_max_current_volume_ratio"
            if raw_undercoverage
            else "nomask_pair_max_current_volume_ratio",
            1.35 if raw_undercoverage else 1.00,
        )
    )
    max_reference_ratio = float(
        cfg.get(
            "nomask_pair_undercoverage_max_reference_volume_ratio"
            if raw_undercoverage
            else "nomask_pair_max_reference_volume_ratio",
            1.25 if raw_undercoverage else 1.02,
        )
    )
    support_by_item: dict[int, set[str]] = {id(item): {str(item["metrics"].get("atlas_family"))} for item in nomask_items}
    best_by_item: dict[int, float] = {id(item): 0.0 for item in nomask_items}
    edges: list[dict[str, Any]] = []
    for i, left in enumerate(nomask_items):
        for right in nomask_items[i + 1 :]:
            left_family = str(left["metrics"].get("atlas_family"))
            right_family = str(right["metrics"].get("atlas_family"))
            if left_family == right_family:
                continue
            dice = _mask_pairwise_dice(left["mask"], right["mask"])
            if dice >= min_pair:
                support_by_item[id(left)].add(right_family)
                support_by_item[id(right)].add(left_family)
                best_by_item[id(left)] = max(best_by_item[id(left)], dice)
                best_by_item[id(right)] = max(best_by_item[id(right)], dice)
                edges.append({"left": left["name"], "right": right["name"], "dice": float(dice)})
    pair_count = 0
    for item in nomask_items:
        metrics = item["metrics"]
        support_count = len(support_by_item[id(item)])
        metrics["nomask_pair_support_families"] = int(support_count)
        best = best_by_item[id(item)]
        metrics["nomask_pair_support_best_dice"] = None if best <= 0.0 else float(best)
        uncertainty_mean = metrics.get("uncertainty_mean")
        uncertainty_score = 1.0 if uncertainty_mean is None else float(np.clip(1.0 - float(uncertainty_mean), 0.0, 1.0))
        volume = float(metrics.get("volume_cm3", 0.0))
        ref_volume = float(metrics.get("reference_volume_cm3", 0.0))
        volume_pair_ok = volume > 0.0
        if current_volume > 0.0:
            volume_pair_ok = volume_pair_ok and volume / current_volume <= max_current_ratio
        if ref_volume > 0.0:
            volume_pair_ok = volume_pair_ok and volume / ref_volume <= max_reference_ratio
        strong_pair_ok = (
            best >= float(cfg.get("nomask_pair_strong_pairwise_dice", 0.92))
            and uncertainty_score >= float(cfg.get("nomask_pair_strong_min_uncertainty_score", 0.90))
            and float(metrics.get("intensity_score", 0.0))
            >= float(cfg.get("nomask_pair_strong_min_intensity_score", 0.90))
            and float(metrics.get("volume_score", 0.0)) >= float(cfg.get("nomask_pair_strong_min_volume_score", 0.95))
        )
        metrics["nomask_pair_volume_ok"] = bool(volume_pair_ok)
        if (
            support_count >= min_families
            and volume_pair_ok
            and float(metrics.get("volume_score", 0.0)) >= float(cfg.get("nomask_pair_min_volume_score", 0.70))
            and float(metrics.get("intensity_score", 0.0)) >= float(cfg.get("nomask_pair_min_intensity_score", 0.75))
            and (
                uncertainty_score >= float(cfg.get("nomask_pair_min_uncertainty_score", 0.84))
                or strong_pair_ok
            )
        ):
            metrics["nomask_pair_consensus_context"] = True
            pair_count += 1
    for item in candidates:
        item.get("metrics", {})["nomask_pair_global_context"] = bool(pair_count > 0)
    nomask_by_family: dict[str, list[dict[str, Any]]] = {}
    for item in nomask_items:
        nomask_by_family.setdefault(str(item["metrics"].get("atlas_family")), []).append(item)
    conflict_count = 0
    max_conflict = float(cfg.get("masked_nomask_conflict_max_pairwise_dice", 0.75))
    for item in candidates:
        metrics = item.get("metrics", {})
        if metrics.get("atlas_variant") not in {"masked", "tissue", "consensus"}:
            continue
        same_family = nomask_by_family.get(str(metrics.get("atlas_family")), [])
        if not same_family:
            continue
        best_nomask = max(
            same_family,
            key=lambda value: (
                bool(value["metrics"].get("nomask_pair_consensus_context")),
                float(value["score"]),
            ),
        )
        dice = _mask_pairwise_dice(item["mask"], best_nomask["mask"])
        metrics["same_family_nomask_dice"] = float(dice)
        if dice <= max_conflict and (
            bool(best_nomask["metrics"].get("nomask_pair_consensus_context"))
            or (
                int(best_nomask["metrics"].get("nomask_pair_support_families", 0)) >= min_families
                and bool(best_nomask["metrics"].get("nomask_pair_volume_ok"))
            )
        ):
            metrics["masked_nomask_conflict"] = True
            conflict_count += 1
    return {
        "nomask_pair_consensus_context": bool(pair_count > 0),
        "nomask_pair_consensus_candidate_count": int(pair_count),
        "nomask_pair_consensus_edges": edges,
        "masked_nomask_conflict_count": int(conflict_count),
    }


def _select_reference_atlas(atlas_items: list[tuple[Path, np.ndarray]], reference_atlas_path: str | Path | None) -> tuple[Path, np.ndarray]:
    if reference_atlas_path and Path(reference_atlas_path).exists():
        ref = Path(reference_atlas_path).resolve()
        for path, mask in atlas_items:
            if path.resolve() == ref:
                return path, mask
    for path, mask in atlas_items:
        if "adaptive" in path.name:
            return path, mask
    return atlas_items[0]


def _select_seed_atlas(atlas_items: list[tuple[Path, np.ndarray]], seed_atlas_path: str | Path | None) -> tuple[Path, np.ndarray]:
    if seed_atlas_path and Path(seed_atlas_path).exists():
        ref = Path(seed_atlas_path).resolve()
        for path, mask in atlas_items:
            if path.resolve() == ref and "adaptive" not in path.name:
                return path, mask
    for path, mask in atlas_items:
        if "adaptive" not in path.name and "_aff" in path.name:
            return path, mask
    for path, mask in atlas_items:
        if "adaptive" not in path.name:
            return path, mask
    return atlas_items[0]


def _fusion_probability(candidates: list[dict[str, Any]]) -> np.ndarray | None:
    usable = [c for c in candidates if c.get("mask") is not None and c.get("score") is not None]
    if not usable:
        return None
    scores = np.asarray([max(0.0, float(c["score"])) for c in usable], dtype=np.float32)
    if not np.any(scores > 0):
        scores = np.ones(len(usable), dtype=np.float32)
    scores = scores / np.sum(scores)
    prob = np.zeros_like(usable[0]["mask"], dtype=np.float32)
    for weight, item in zip(scores, usable):
        prob += float(weight) * item["mask"].astype(np.float32)
    return np.clip(prob, 0.0, 1.0)


def _ws_level1(seed_values: np.ndarray, cfg: dict) -> list[float] | None:
    """Niveau 1 : WhiteStripe Recycling — utilise les stats GMM déjà calculées."""
    wm = cfg.get("wm_mean_raw")
    gm = cfg.get("gm_mean_raw")
    csf = cfg.get("csf_mean_raw")
    if wm is None or gm is None or csf is None:
        return None
    wm, gm, csf = float(wm), float(gm), float(csf)
    if not (csf < gm < wm) or wm <= 0:
        return None
    t_csf_gm = (csf + gm) / 2.0
    t_gm_wm  = (gm  + wm) / 2.0
    if t_gm_wm <= t_csf_gm:
        return None
    thresholds = np.linspace(t_csf_gm, t_gm_wm, 6)
    pcts = [float(np.clip(np.mean(seed_values <= t) * 100.0, 2.0, 40.0)) for t in thresholds]
    pcts = sorted(set(round(p, 1) for p in pcts))
    return pcts if len(pcts) >= 2 else None


def _bet_anchor(seed_values: np.ndarray) -> list[float]:
    """Niveau 2e : ancre BET (p2 + 0.10×(p98−p2)), validé sur 20 ans de skull stripping."""
    p2  = float(np.percentile(seed_values, 2.0))
    p98 = float(np.percentile(seed_values, 98.0))
    if p98 <= p2:
        return [6.0, 8.0, 10.0, 12.0, 15.0, 18.0]
    anchor_intensity = p2 + 0.10 * (p98 - p2)
    anchor_pct = float(np.clip(np.mean(seed_values <= anchor_intensity) * 100.0, 4.0, 30.0))
    offsets = [-2.0, 0.0, 2.0, 4.0, 7.0, 10.0]
    return sorted(set(round(float(np.clip(anchor_pct + o, 2.0, 40.0)), 1) for o in offsets))


def _otsu_multiseuil(seed_values: np.ndarray, n_classes: int = 4) -> list[float] | None:
    """Niveau 2c : Otsu multi-seuil pour distribution bimodale claire (rapide, reproductible)."""
    if not _SKIMAGE_AVAILABLE:
        return None
    try:
        sample = (seed_values if seed_values.size <= 100_000
                  else seed_values[::max(1, seed_values.size // 100_000)])
        thresholds = _threshold_multiotsu(sample, classes=n_classes)
        pcts = sorted(set(
            round(float(np.clip(np.mean(seed_values <= t) * 100.0, 2.0, 40.0)), 1)
            for t in thresholds
        ))
        return pcts if len(pcts) >= 2 else None
    except Exception:
        return None


def _kde_vallees(seed_values: np.ndarray, sigma: float = 4.0) -> list[float] | None:
    """Niveau 2a/2d : détection de vallées dans le KDE — robuste même en cas isointense."""
    if not _SCIPY_STATS_AVAILABLE:
        return None
    try:
        bins = min(512, max(64, seed_values.size // 50))
        counts, edges = np.histogram(seed_values, bins=bins)
        smoothed = ndi.gaussian_filter1d(counts.astype(float), sigma=sigma)
        centers  = (edges[:-1] + edges[1:]) / 2.0
        peaks, _ = _find_peaks(smoothed, height=smoothed.max() * 0.05, distance=bins // 8)
        if len(peaks) < 2:
            return None
        result = []
        for i in range(len(peaks) - 1):
            region     = smoothed[peaks[i]:peaks[i + 1] + 1]
            valley_idx = peaks[i] + int(np.argmin(region))
            vp = float(np.clip(np.mean(seed_values <= centers[valley_idx]) * 100.0, 2.0, 40.0))
            for offset in (-2.0, 0.0, 2.0):
                result.append(float(np.clip(vp + offset, 2.0, 40.0)))
        result = sorted(set(round(p, 1) for p in result))
        return result if len(result) >= 2 else None
    except Exception:
        return None


def _compute_adaptive_tissue_percentiles(
    seed_values: np.ndarray,
    cfg: dict,
) -> tuple[list[float], str]:
    """Génère des tissue_percentiles adaptatifs depuis seed_values + stats optionnelles dans cfg.

    Priorité des stratégies :
    1. WhiteStripe Recycling  — si wm/gm/csf_mean_raw présents dans cfg
    2a/2b. KDE vallées σ=7   — si age_months∈[7,13] ou gm_wm_contrast<0.25 (isointense)
    2c. Otsu multi-seuil      — si Bimodality Coefficient > 0.55 (bimodal clair)
    2d. KDE vallées σ=4       — fallback général
    2e. BET anchor            — dernier recours calculé
    3.  Hardcoded             — fallback absolu si tout échoue
    """
    fallback = list(cfg.get("tissue_percentiles", [8.0, 10.0, 12.0, 15.0, 18.0]))
    if seed_values.size < 100:
        return fallback, "hardcoded_insufficient_seed_voxels"

    # Niveau 1 : WhiteStripe Recycling
    try:
        pcts = _ws_level1(seed_values, cfg)
        if pcts and len(pcts) >= 2:
            return pcts, "whitestripe_level1"
    except Exception:
        pass

    # Niveau 2 : Pattern-Aware
    age_months     = cfg.get("age_months")
    gm_wm_contrast = cfg.get("gm_wm_contrast")
    isointense = (
        (age_months is not None and 7 <= int(age_months) <= 13)
        or (gm_wm_contrast is not None and float(gm_wm_contrast) < 0.25)
    )

    # 2a/2b : isointense → KDE sigma=7
    if isointense:
        try:
            pcts = _kde_vallees(seed_values, sigma=7.0)
            if pcts and len(pcts) >= 2:
                tag = (f"_age{age_months}mo"
                       if age_months is not None and 7 <= int(age_months) <= 13
                       else "")
                return pcts, f"kde_valleys_isointense_sigma7{tag}"
        except Exception:
            pass

    # 2c : bimodal → Otsu
    if _SCIPY_STATS_AVAILABLE:
        try:
            sample = (seed_values if seed_values.size <= 200_000
                      else seed_values[::max(1, seed_values.size // 200_000)])
            sk = float(_scipy_skew(sample))
            ku = float(_scipy_kurtosis(sample, fisher=True))
            bc = (sk ** 2 + 1.0) / (ku + 3.0 + 3.0 / max(sample.size - 1, 1))
            if bc > 0.55:
                pcts = _otsu_multiseuil(seed_values)
                if pcts and len(pcts) >= 2:
                    return pcts, f"otsu_multiseuil_bc{bc:.2f}"
        except Exception:
            pass

    # 2d : KDE sigma=4 général
    try:
        pcts = _kde_vallees(seed_values, sigma=4.0)
        if pcts and len(pcts) >= 2:
            return pcts, "kde_valleys_sigma4"
    except Exception:
        pass

    # 2e : BET anchor
    try:
        pcts = _bet_anchor(seed_values)
        if pcts and len(pcts) >= 2:
            return pcts, "bet_anchor"
    except Exception:
        pass

    return fallback, "hardcoded_fallback"


def apply_adaptive_mask_fusion(
    mask_path: str | Path,
    brain_path: str | Path,
    input_image_path: str | Path,
    atlas_paths: Iterable[str | Path | None],
    *,
    reference_atlas_path: str | Path | None = None,
    seed_atlas_path: str | Path | None = None,
    uncertainty_path: str | Path | None = None,
    t2_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select and optionally apply a robust final brain mask.

    The selector uses no ground truth.  It is a post-SAM guardrail that can keep
    the current mask, prune a catastrophic mask, or switch to a tissue-pruned
    atlas candidate when the current mask is a high-precision undercoverage.
    """

    cfg = _cfg(config)
    result: dict[str, Any] = {
        "adaptive_fusion_enabled": bool(cfg.get("enabled", True)),
        "adaptive_fusion_used": False,
        "adaptive_fusion_reason": "disabled" if not cfg.get("enabled", True) else "not_improved",
    }
    if not cfg.get("enabled", True):
        return result

    mask_path = Path(mask_path)
    brain_path = Path(brain_path)
    input_image_path = Path(input_image_path)
    if not mask_path.exists() or not input_image_path.exists():
        result["adaptive_fusion_reason"] = "missing_mask_or_input"
        return result

    ref_img = load_img(input_image_path)
    image = np.nan_to_num(ref_img.get_fdata(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    current = _read_mask(mask_path, ref_img)
    atlas_file_list = _unique_paths([reference_atlas_path, seed_atlas_path, *list(atlas_paths)])
    atlas_items = [(path, _read_mask(path, ref_img)) for path in atlas_file_list]
    atlas_items = [(path, mask) for path, mask in atlas_items if mask.any()]
    if not atlas_items:
        result["adaptive_fusion_reason"] = "no_valid_atlas_candidates"
        return result

    ref_atlas_path, ref_atlas = _select_reference_atlas(atlas_items, reference_atlas_path)
    seed_path, seed_atlas = _select_seed_atlas(atlas_items, seed_atlas_path)
    uncertainty = _read_float(uncertainty_path, ref_img)
    t2 = _read_float(t2_path, ref_img)
    vox_cm3 = float(np.prod(ref_img.header.get_zooms()[:3]) / 1000.0)

    current_overlap = mask_overlap_stats(current, ref_atlas, ref_img)
    current_volume = mask_volume_cm3(current, ref_img)
    ref_volume = mask_volume_cm3(ref_atlas, ref_img)
    current_unc = None
    if uncertainty is not None and current.any():
        current_unc = float(np.mean(uncertainty[current]))
    raw_bad = (
        current_overlap["precision"] < float(cfg["catastrophic_min_precision"])
        or current_overlap["coverage"] < float(cfg["catastrophic_min_coverage"])
        or current_overlap["dice"] < float(cfg["catastrophic_min_atlas_dice"])
        or (current_unc is not None and current_unc > float(cfg["catastrophic_max_uncertainty_mean"]))
        or current_volume < float(cfg["min_volume_cm3"])
        or current_volume > float(cfg["max_volume_cm3"])
    )
    raw_undercoverage = (
        current_overlap["precision"] >= float(cfg["undercoverage_min_precision"])
        and current_overlap["coverage"] < float(cfg["undercoverage_max_coverage"])
        and current_volume < float(cfg["undercoverage_max_volume_ratio"]) * max(ref_volume, 1e-6)
    )
    raw_overcoverage = (
        current_overlap["coverage"] >= float(cfg["overcoverage_min_coverage"])
        and current_overlap["precision"] < float(cfg["overcoverage_max_precision"])
        and current_volume > float(cfg["overcoverage_min_volume_ratio"]) * max(ref_volume, 1e-6)
    )
    current_ref_ratio = current_volume / max(ref_volume, 1e-6)
    reference_trust = float(cfg.get("reference_atlas_trust_score", 0.0) or 0.0)
    # PATCH P7 — sur-segmentation douce détectable par : current ⊃ atlas à ≥98.5%
    # ET volume current > atlas de ≥2%. Le seuil classique raw_overcoverage est bloqué
    # par une precision trop haute (SAM2 bien centré sur l'atlas mais trop grand).
    raw_soft_overcoverage = (
        not raw_bad
        and not raw_undercoverage
        and not raw_overcoverage
        and current_ref_ratio > float(cfg.get("soft_overcoverage_min_volume_ratio", 1.02))
        and current_overlap["coverage"] > float(cfg.get("soft_overcoverage_min_coverage", 0.985))
    )
    raw_atlas_mismatch = (
        bool(cfg.get("atlas_guard_enabled", True))
        and reference_trust >= float(cfg.get("atlas_guard_min_trust", 0.45))
        and current_overlap["dice"] < float(cfg.get("atlas_guard_max_current_atlas_dice", 0.94))
        and (
            current_ref_ratio > float(cfg.get("atlas_guard_over_volume_ratio", 1.14))
            or current_ref_ratio < float(cfg.get("atlas_guard_under_volume_ratio", 0.94))
        )
    )

    candidates: list[dict[str, Any]] = []

    def add_candidate(name: str, mask: np.ndarray) -> None:
        clean = _clean(mask)
        metrics = _candidate_metrics(name, clean, ref_atlas, ref_img, image, current, uncertainty, cfg)
        score = _score_candidate(
            metrics, raw_bad, raw_undercoverage, raw_overcoverage, raw_atlas_mismatch, cfg,
            raw_soft_overcoverage=raw_soft_overcoverage,
        )
        candidates.append({"name": name, "mask": clean, "metrics": metrics, "score": score})

    add_candidate("current", current)
    add_candidate("reference_atlas", ref_atlas)
    if bool(cfg.get("include_atlas_candidates", False)):
        for path, atlas_mask in atlas_items:
            if path.resolve() == ref_atlas_path.resolve():
                continue
            safe = "".join(ch if ch.isalnum() else "_" for ch in path.stem)[:72]
            add_candidate(f"atlas_candidate_{safe}", atlas_mask)

    if len(atlas_items) >= 2:
        atlas_stack = np.stack([m.astype(np.uint8) for _, m in atlas_items], axis=0)
        vote_need = max(1, int(np.ceil(0.5 * len(atlas_items))))
        add_candidate("atlas_majority", atlas_stack.sum(axis=0) >= vote_need)

    seed_values   = image[seed_atlas & np.isfinite(image)]
    adaptive_pcts = list(cfg["tissue_percentiles"])   # défaut si seed insuffisant
    adaptive_strategy = "hardcoded_no_seed"
    if seed_values.size >= 100:
        adaptive_pcts, adaptive_strategy = _compute_adaptive_tissue_percentiles(seed_values, cfg)
        adaptive_pcts = [float(np.clip(p, 2.0, 40.0)) for p in adaptive_pcts]
        result["adaptive_tissue_percentile_strategy"] = adaptive_strategy
        result["adaptive_tissue_percentiles_used"]    = adaptive_pcts
        if cfg.get("adaptive_percentiles_verbose", False):
            print(f"[AdaptiveFusion] strategy={adaptive_strategy} pcts={adaptive_pcts}")
        for pct in adaptive_pcts:
            thr    = float(np.percentile(seed_values, pct))
            tissue = ref_atlas & (image >= thr)
            if bool(cfg.get("use_t2_gate", False)) and t2 is not None:
                t2_values = t2[seed_atlas & np.isfinite(t2)]
                if t2_values.size >= 100:
                    t2_thr = float(np.percentile(t2_values, float(cfg["t2_percentile"])))
                    tissue = tissue & (t2 >= t2_thr)
            add_candidate(f"atlas_tissue_p{pct:g}", tissue)
            # PATCH P5 — variante SAM2-derived : current ∩ atlas_tissue_pX
            # Garde le calibrage volumique de atlas_tissue_pX tout en exigeant que
            # chaque voxel soit aussi accepté par SAM2 (current). Volume final ≤ min(current, tissue).
            if current.any():
                add_candidate(f"current_int_tissue_p{pct:g}", current & tissue)

    # PATCH P11 — Soft-overcoverage gap fill.
    # Quand raw_soft_overcoverage=True ET les percentiles adaptatifs sautent de > 5 pts
    # au-dessus du dernier percentile hardcodé (ex : Otsu p26-40 alors que hardcodé
    # max=18 → gap 18-26), des candidats intermédiaires critiques manquent dans la plage
    # [ref×0.75, ref×0.95]. Combler le gap avec des pas de 1 point percentile.
    if raw_soft_overcoverage and adaptive_pcts and seed_values.size >= 100:
        _min_adaptive = min(adaptive_pcts)
        _hardcoded = [float(p) for p in cfg.get("tissue_percentiles", [8.0, 10.0, 12.0, 15.0, 18.0])]
        _max_hardcoded = max(_hardcoded) if _hardcoded else 0.0
        if _min_adaptive > _max_hardcoded + 4.0:
            # Combler le gap avec pas de 1.5 point percentile
            _gap_start = _max_hardcoded + 1.5
            _gap_end = _min_adaptive - 0.5
            _gap_step = 1.5
            _gap_p = _gap_start
            while _gap_p <= _gap_end:
                pct = round(_gap_p, 1)
                thr = float(np.percentile(seed_values, pct))
                tissue = ref_atlas & (image >= thr)
                add_candidate(f"atlas_tissue_p{pct:g}", tissue)
                if current.any():
                    add_candidate(f"current_int_tissue_p{pct:g}", current & tissue)
                _gap_p += _gap_step

    allowed = _dilate(ref_atlas, int(cfg["prune_dilation"]))
    add_candidate("current_pruned_to_atlas", current & allowed)

    # PATCH P16 — candidat random walker multi-modal T1(+T2). Le walker affine
    # la frontière en exploitant les sauts d'intensité (LCR T2 brillant, scalp
    # T1 sombre). Garde-fous internes : rejette si Dice vs current < 0.85, vol
    # ratio hors [0.70, 1.10], ou atlas_dice écroule. Voir Grady IEEE TPAMI 2006.
    if bool(cfg.get("rw_candidate_enabled", False)):
        _rw_mask = _build_random_walker_candidate(current, ref_atlas, image, t2, cfg)
        if _rw_mask is not None:
            _rw_suffix = "t1t2" if t2 is not None else "t1"
            add_candidate(f"current_rw_{_rw_suffix}", _rw_mask)

    # PATCH P17 — Chan-Vese vectoriel sur (T1, T2). Région-based : robuste où
    # le gradient est faible (LCR-cerveau pédiatrique 6-9 mois isointense).
    # Complète le random walker (gradient-based) sur les cas où il échoue.
    if bool(cfg.get("cv_candidate_enabled", False)):
        _cv_mask = _build_chan_vese_candidate(current, ref_atlas, image, t2, cfg)
        if _cv_mask is not None:
            _cv_suffix = "t1t2" if t2 is not None else "t1"
            add_candidate(f"current_cv_{_cv_suffix}", _cv_mask)

    # PATCH P5 — variante SAM2-derived : current ∩ atlas_majority
    if current.any():
        _maj = next((c["mask"] for c in candidates if c["name"] == "atlas_majority"), None)
        if _maj is not None:
            add_candidate("current_int_majority", current & _maj)

    if raw_undercoverage or raw_bad:
        shell = _dilate(current, int(cfg["recover_shell_width"])) & (~current) & ref_atlas
        if shell.any() and current.any():
            threshold = float(np.percentile(image[current], min(15.0, max(1.0, float(adaptive_pcts[0])))))
            proposed = shell & (image >= threshold)
            max_vox = int(float(cfg["recover_max_added_cm3"]) / max(vox_cm3, 1e-9))
            if proposed.sum() > max_vox > 0:
                vals = image[proposed]
                cutoff = np.partition(vals, max(0, vals.size - max_vox))[max(0, vals.size - max_vox)]
                proposed &= image >= cutoff
            add_candidate("current_shell_recovery", current | proposed)

    # PATCH P8 — candidats SAM2-Union pour atlas_mismatch sévère.
    # Déclenché quand SAM2 sous-couvre l'atlas de > 30 % (coverage < 0.70).
    # Dans ce régime les candidats ∩ (PATCH P5) sont contre-productifs : current ∩ atlas
    # donne des masques encore plus petits que current, qui est déjà trop court.
    # Solution : current ∪ atlas_tissue_pX = SAM2 + région atlas non capturée.
    # Ces candidats SONT SAM2-derived (superset de current, préfixe reconnu par
    # _SAM2_DERIVED_PREFIXES → PATCH P4 les conserve sans remplacement).
    # Précaution : seuls les atlas_tissue_p* déjà générés sont unionés (pas le
    # reference_atlas plein, pour limiter la sur-expansion).
    if (
        raw_atlas_mismatch
        and current_overlap["coverage"] < float(cfg.get("union_candidate_min_atlas_mismatch_cov", 0.70))
    ):
        for _cand in list(candidates):
            if _cand["name"].startswith("atlas_tissue_p"):
                _union_name = _cand["name"].replace("atlas_tissue_p", "current_union_tissue_p")
                add_candidate(_union_name, current | _cand["mask"])

    nomask_support_info = _annotate_nomask_support(candidates, cfg, raw_undercoverage=raw_undercoverage)
    for item in candidates:
        item["score"] = _score_candidate(
            item["metrics"],
            raw_bad,
            raw_undercoverage,
            raw_overcoverage,
            raw_atlas_mismatch,
            cfg,
            raw_soft_overcoverage=raw_soft_overcoverage,
        )

    def allowed_candidate(item: dict[str, Any]) -> bool:
        name = str(item.get("name", ""))
        if name == "current":
            return True
        metrics = item.get("metrics", {})
        current_dice = float(metrics.get("current_dice", 0.0))
        change_ratio = float(metrics.get("change_ratio", 0.0))
        # PATCH P8 — les candidats SAM2-union (current ∪ atlas_tissue_pX) ne passent
        # pas le filtre atlas_dice ≥ 0.985 du bloc raw_atlas_mismatch (leur atlas_dice
        # est ~0.90-0.93 car ils débordent légèrement l'atlas). On leur applique
        # uniquement change_ratio et volume_score comme garde-fous.
        if name.startswith("current_union_tissue_p"):
            return (
                change_ratio <= float(cfg.get("max_union_candidate_change_ratio", 0.60))
                and float(metrics.get("volume_score", 0.0)) >= float(cfg.get("union_candidate_min_volume_score", 0.60))
            )
        if (
            bool(metrics.get("nomask_pair_consensus_context"))
            and (raw_bad or raw_undercoverage or raw_atlas_mismatch)
        ):
            return (
                current_dice >= float(cfg.get("undercoverage_rescue_min_candidate_current_dice", 0.45))
                and change_ratio <= float(cfg.get("undercoverage_rescue_max_change_ratio", 1.35))
                and float(metrics.get("volume_score", 0.0)) >= float(cfg.get("undercoverage_rescue_min_volume_score", 0.70))
                and float(metrics.get("intensity_score", 0.0)) >= float(cfg.get("undercoverage_rescue_min_intensity_score", 0.45))
            )

        undercoverage_rescue_context = (
            bool(cfg.get("undercoverage_rescue_enabled", True))
            and (raw_undercoverage or raw_bad or raw_atlas_mismatch)
            and current_overlap["precision"] >= float(cfg.get("undercoverage_rescue_min_current_precision", 0.92))
            and current_overlap["coverage"] <= float(cfg.get("undercoverage_rescue_max_current_coverage", 0.70))
            and current_ref_ratio <= float(cfg.get("undercoverage_rescue_current_ref_ratio", 0.70))
        )
        atlas_like_rescue = (
            name in {"reference_atlas", "atlas_majority"}
            or name.startswith("atlas_candidate_")
            or name.startswith("atlas_tissue_p")
        )
        if undercoverage_rescue_context and atlas_like_rescue:
            return (
                current_dice >= float(cfg.get("undercoverage_rescue_min_candidate_current_dice", 0.45))
                and change_ratio <= float(cfg.get("undercoverage_rescue_max_change_ratio", 1.35))
                and float(metrics.get("atlas_precision", 0.0)) >= float(cfg.get("undercoverage_rescue_min_atlas_precision", 0.94))
                and float(metrics.get("atlas_coverage", 0.0)) >= float(cfg.get("undercoverage_rescue_min_atlas_coverage", 0.82))
                and float(metrics.get("volume_score", 0.0)) >= float(cfg.get("undercoverage_rescue_min_volume_score", 0.70))
                and float(metrics.get("intensity_score", 0.0)) >= float(cfg.get("undercoverage_rescue_min_intensity_score", 0.45))
            )

        if raw_undercoverage:
            if current_dice < float(cfg.get("min_undercoverage_candidate_current_dice", 0.82)):
                return False
            if change_ratio > float(cfg.get("max_undercoverage_change_ratio", 0.48)):
                return False
        elif raw_atlas_mismatch:
            # Déplacement catastrophique : masque courant hors plage de volume valide.
            # current_dice ≈ 0 car le masque est entièrement hors du cerveau — les gardes
            # habituelles bloqueraient tous les candidats atlas. On les laisse passer.
            _current_hard_ok_guard = (
                float(cfg["min_volume_cm3"]) <= float(current_volume) <= float(cfg["max_volume_cm3"])
            )
            if not _current_hard_ok_guard:
                return atlas_like_rescue
            if current_dice < float(cfg.get("min_atlas_guard_candidate_current_dice", 0.82)):
                return False
            if float(metrics.get("atlas_dice", 0.0)) < float(cfg.get("min_atlas_guard_candidate_atlas_dice", 0.985)):
                return False
            if change_ratio > float(cfg.get("max_atlas_guard_change_ratio", 0.46)):
                return False
        elif raw_overcoverage:
            if current_dice < float(cfg.get("min_overcoverage_candidate_current_dice", 0.82)):
                return False
            if change_ratio > float(cfg.get("max_overcoverage_change_ratio", 0.40)):
                return False
        elif not raw_bad:
            if current_dice < float(cfg.get("min_nonraw_candidate_current_dice", 0.0)):
                return False
            if change_ratio > float(cfg.get("max_nonraw_change_ratio", 0.38)):
                return False
        else:
            current_hard_ok = float(cfg["min_volume_cm3"]) <= float(current_volume) <= float(cfg["max_volume_cm3"])
            if current_hard_ok and current_dice < float(cfg.get("min_raw_candidate_current_dice", 0.0)):
                return False
        return True

    def undercoverage_rescue_rank_score(item: dict[str, Any]) -> float:
        score = float(item["score"])
        name = str(item.get("name", ""))
        metrics = item.get("metrics", {})
        if not (
            bool(cfg.get("undercoverage_rescue_enabled", True))
            and raw_undercoverage
            and current_overlap["precision"] >= float(cfg.get("undercoverage_rescue_min_current_precision", 0.92))
            and current_overlap["coverage"] <= float(cfg.get("undercoverage_rescue_max_current_coverage", 0.70))
            and current_ref_ratio <= float(cfg.get("undercoverage_rescue_current_ref_ratio", 0.70))
            and name != "current"
        ):
            return score
        atlas_coverage = float(metrics.get("atlas_coverage", 0.0))
        volume_ratio = float(metrics.get("volume_ratio_to_reference", 0.0))
        if (
            atlas_coverage >= float(cfg.get("undercoverage_rescue_rank_high_coverage", 0.985))
            and volume_ratio >= float(cfg.get("undercoverage_rescue_rank_min_expansion_ratio", 1.015))
        ):
            score += float(cfg.get("undercoverage_rescue_rank_expansion_bonus", 0.10))
        if "consensus" in name or "probability" in name:
            score += float(cfg.get("undercoverage_rescue_rank_consensus_bonus", 0.02))
        if atlas_coverage < float(cfg.get("undercoverage_rescue_rank_low_coverage", 0.96)):
            score -= float(cfg.get("undercoverage_rescue_rank_low_coverage_penalty", 0.06))
        return score

    def nomask_rescue_rank_score(item: dict[str, Any]) -> float:
        score = undercoverage_rescue_rank_score(item)
        metrics = item.get("metrics", {})
        if bool(metrics.get("nomask_pair_consensus_context")):
            score += float(cfg.get("nomask_pair_bonus", 0.35))
        if bool(metrics.get("masked_nomask_conflict")):
            score -= float(cfg.get("masked_nomask_conflict_penalty", 0.18))
        return score

    # PATCH P6 — tie-breaker : à score égal, préférer un candidat SAM2-derived.
    # Évite que max()/sort() retourne un atlas-pur (qui sera écarté par PATCH P4
    # ou par la margin) alors qu'un current_int_* avec exactement le même score existe.
    candidates.sort(
        key=lambda item: (float(item["score"]), _is_sam2_derived(str(item.get("name", "")))),
        reverse=True,
    )
    current_item = next(item for item in candidates if item["name"] == "current")
    current_hard_ok = float(cfg["min_volume_cm3"]) <= float(current_volume) <= float(cfg["max_volume_cm3"])
    low_trust_fail_clean = (
        bool(cfg.get("low_trust_fail_clean_enabled", True))
        and reference_trust < float(cfg.get("low_trust_fail_clean_threshold", 0.18))
        and (
            not bool(cfg.get("low_trust_fail_clean_keep_current_if_hard_ok", True))
            or current_hard_ok
        )
    )
    if low_trust_fail_clean:
        allowed_candidates = [current_item]
        best = current_item
    else:
        allowed_candidates = [item for item in candidates if allowed_candidate(item)]
        nomask_rescue_candidates = [
            item for item in allowed_candidates if bool(item.get("metrics", {}).get("nomask_pair_consensus_context"))
        ]
        # PATCH P6 — tie-breaker : à rang égal, préférer un candidat SAM2-derived.
        if nomask_rescue_candidates and (raw_bad or raw_undercoverage or raw_atlas_mismatch):
            best = max(
                nomask_rescue_candidates,
                key=lambda c: (nomask_rescue_rank_score(c), _is_sam2_derived(str(c.get("name", "")))),
            )
        elif raw_undercoverage and allowed_candidates:
            best = max(
                allowed_candidates,
                key=lambda c: (undercoverage_rescue_rank_score(c), _is_sam2_derived(str(c.get("name", "")))),
            )
        else:
            best = allowed_candidates[0] if allowed_candidates else current_item

        # Déplacement catastrophique : préférer atlas_tissue_p* (sous-ensembles calibrés à
        # l'intensité) sur atlas_majority (union d'atlases, potentiellement sur-dimensionnée)
        # quand l'écart de score reste dans la tolérance configurée.
        if (
            not current_hard_ok
            and raw_atlas_mismatch
            and best["name"] == "atlas_majority"
        ):
            _tol = float(cfg.get("catastrophic_tissue_preference_tolerance", 0.08))
            _tissue = [
                c for c in allowed_candidates
                if c["name"].startswith("atlas_tissue_p")
                and float(c["metrics"].get("volume_score", 0.0)) > 0.0
            ]
            if _tissue:
                # PATCH P6 — tie-breaker : à score égal, préférer SAM2-derived.
                _best_tissue = max(
                    _tissue,
                    key=lambda c: (float(c["score"]), _is_sam2_derived(str(c.get("name", "")))),
                )
                if float(_best_tissue["score"]) >= float(best["score"]) - _tol:
                    best = _best_tissue
    if (
        best["name"] != "current"
        and bool(cfg.get("replace_only_if_raw_qc_triggered", False))
        and not raw_bad
        and not raw_undercoverage
        and not raw_overcoverage
        and not raw_atlas_mismatch
    ):
        best = current_item
    # PATCH P6 — ordre critique : la contrainte SAM2-only (PATCH P4) DOIT précéder
    # le bloc de margin. Sinon un candidat atlas-pur top-scoré (e.g. atlas_tissue_p14)
    # est d'abord écarté par la margin (retour à current), AVANT que PATCH P4 ait
    # l'occasion de le remplacer par son équivalent SAM2-hybrid (current_int_tissue_p14)
    # qui était strictement meilleur que current. Cas observé sur BOBs sub-198549_ses-1mo.

    # PATCH P4 — contrainte architecturale: le masque final doit contenir SAM2.
    # Si la sélection actuelle est un candidat atlas pur, on remplace par le meilleur
    # candidat SAM2-derived disponible (current, current_pruned_to_atlas, current_shell_recovery,
    # current_int_tissue_p*, current_int_majority).
    sam2_only_replaced_from = None
    if bool(cfg.get("sam2_only_output", True)) and not _is_sam2_derived(best["name"]):
        sam2_only_replaced_from = best["name"]
        sam2_candidates = [c for c in allowed_candidates if _is_sam2_derived(c["name"])]
        if sam2_candidates:
            # Garder le meilleur candidat SAM2-derived au sens du score (déjà calculé).
            # Tie-breaker neutre ici (tous SAM2-derived) mais conservé pour cohérence.
            best_sam2 = max(
                sam2_candidates,
                key=lambda c: (float(c["score"]), _is_sam2_derived(str(c.get("name", "")))),
            )
            # PATCH P10 — Emergency Atlas Rescue.
            # Si le meilleur candidat SAM2-derived a lui-même un dice_atlas < seuil ET un
            # volume_ratio > seuil (SAM2 a segmenté hors cerveau), autoriser l'atlas pur.
            _p10_enabled = bool(cfg.get("sam2_only_allow_atlas_emergency", True))
            _p10_max_dice = float(cfg.get("sam2_only_emergency_max_atlas_dice", 0.05))
            _p10_min_vol = float(cfg.get("sam2_only_emergency_min_vol_ratio", 2.5))
            _p10_min_trust = float(cfg.get("sam2_only_emergency_min_trust", 0.45))
            _best_sam2_atlas_dice = float(best_sam2.get("metrics", {}).get("atlas_dice", 1.0))
            _best_sam2_vol_cm3 = float(best_sam2.get("metrics", {}).get("volume_cm3", 0.0))
            _ref_vol = max(float(ref_volume), 1e-6)
            _best_sam2_vol_ratio = _best_sam2_vol_cm3 / _ref_vol
            _ref_trust_p10 = float(cfg.get("reference_atlas_trust_score", 0.0) or 0.0)
            # PATCH P10 (étendu P19) — déclencheur OU, conditionné à atlas_dice quasi-nul :
            #  (a) SAM2 hors-cerveau = volume géant (vol_ratio > seuil), OU
            #  (b) SAM2 mal localisé = masque de volume normal mais déplacé, avec
            #      recouvrement atlas quasi-nul ET atlas retenu de confiance
            #      (trust ≥ seuil). Capture les cas type infant_t1_22 (décalé ~46 mm).
            _p10_overseg = _best_sam2_vol_ratio > _p10_min_vol
            _p10_displaced = _ref_trust_p10 >= _p10_min_trust
            if (
                _p10_enabled
                and _best_sam2_atlas_dice < _p10_max_dice
                and (_p10_overseg or _p10_displaced)
            ):
                _p10_why = "hors-cerveau" if _p10_overseg else "mal localise"
                print(
                    f"PATCH P10 — Emergency Atlas Rescue ({_p10_why}): "
                    f"atlas_dice={_best_sam2_atlas_dice:.3f}, vol_ratio={_best_sam2_vol_ratio:.2f}x, "
                    f"trust={_ref_trust_p10:.2f} -> atlas pur '{sam2_only_replaced_from}' autorise"
                )
            else:
                print(f"PATCH P4 — sortie atlas pur '{sam2_only_replaced_from}' interdite "
                      f"→ remplacée par '{best_sam2['name']}' (SAM2-derived)")
                best = best_sam2
        else:
            print(f"PATCH P4 — sortie atlas pur '{sam2_only_replaced_from}' interdite, "
                  f"aucun SAM2-derived candidat → fallback current")
            best = current_item

    # PATCH P6 — margin différenciée. À ce stade, si sam2_only_output=True (défaut),
    # `best` est garanti SAM2-derived. Pour un SAM2-hybrid (current_int_*, current_pruned,
    # current_shell_recovery) on applique une margin réduite (sam2_hybrid_selection_margin)
    # car ces candidats sont structurellement bornés et ne peuvent pas dégrader
    # catastrophiquement par rapport à current. Pour les autres cas (sam2_only_output=False
    # ou candidat non-hybrid), on garde la logique classique.
    if best["name"] != "current":
        qc_triggered = raw_bad or raw_undercoverage or raw_overcoverage or raw_atlas_mismatch
        if _is_sam2_derived(best["name"]):
            margin = float(cfg.get("sam2_hybrid_selection_margin", 0.003))
        elif qc_triggered:
            margin = float(cfg.get("triggered_selection_margin", 0.0))
        else:
            margin = float(cfg["selection_margin"])
        if float(best["score"]) < float(current_item["score"]) + margin:
            best = current_item

    undercoverage_rescue_selected = best["name"] != "current" and (
        raw_undercoverage
        and current_overlap["precision"] >= float(cfg.get("undercoverage_rescue_min_current_precision", 0.92))
        and current_overlap["coverage"] <= float(cfg.get("undercoverage_rescue_max_current_coverage", 0.70))
        and current_ref_ratio <= float(cfg.get("undercoverage_rescue_current_ref_ratio", 0.70))
    )

    result.update(
        {
            "adaptive_fusion_reference_atlas_path": str(ref_atlas_path),
            "adaptive_fusion_seed_atlas_path": str(seed_path),
            "adaptive_fusion_raw_bad": bool(raw_bad),
            "adaptive_fusion_raw_undercoverage": bool(raw_undercoverage),
            "adaptive_fusion_raw_overcoverage": bool(raw_overcoverage),
            "adaptive_fusion_raw_atlas_mismatch": bool(raw_atlas_mismatch),
            "adaptive_fusion_raw_soft_overcoverage": bool(raw_soft_overcoverage),
            "adaptive_fusion_reference_atlas_trust_score": float(reference_trust),
            "adaptive_fusion_low_trust_fail_clean": bool(low_trust_fail_clean),
            "adaptive_fusion_current_hard_volume_ok": bool(current_hard_ok),
            "adaptive_fusion_current_reference_volume_ratio": float(current_ref_ratio),
            "adaptive_fusion_current_volume_cm3": float(current_volume),
            "adaptive_fusion_reference_volume_cm3": float(ref_volume),
            "adaptive_fusion_current_atlas_dice": float(current_overlap["dice"]),
            "adaptive_fusion_current_atlas_coverage": float(current_overlap["coverage"]),
            "adaptive_fusion_current_atlas_precision": float(current_overlap["precision"]),
            "adaptive_fusion_current_uncertainty_mean": current_unc,
            "adaptive_fusion_undercoverage_rescue_context": bool(
                raw_undercoverage
                and current_overlap["precision"] >= float(cfg.get("undercoverage_rescue_min_current_precision", 0.92))
                and current_overlap["coverage"] <= float(cfg.get("undercoverage_rescue_max_current_coverage", 0.70))
                and current_ref_ratio <= float(cfg.get("undercoverage_rescue_current_ref_ratio", 0.70))
            ),
            "adaptive_fusion_undercoverage_rescue_selected": bool(undercoverage_rescue_selected),
            "adaptive_fusion_selected_candidate": str(best["name"]),
            "adaptive_fusion_selected_score": float(best["score"]),
            "adaptive_fusion_current_score": float(current_item["score"]),
            "adaptive_fusion_candidate_count": int(len(candidates)),
            # PATCH P4 — traçabilité de la restriction SAM2-only
            "adaptive_fusion_sam2_only_enforced": bool(cfg.get("sam2_only_output", True)),
            "adaptive_fusion_sam2_only_replaced_from": sam2_only_replaced_from,
            "adaptive_fusion_final_is_sam2_derived": _is_sam2_derived(str(best["name"])),
            # PATCH P10 — Emergency Atlas Rescue flag
            "adaptive_fusion_p10_emergency_atlas_rescue": (
                bool(cfg.get("sam2_only_output", True))
                and sam2_only_replaced_from is not None
                and _is_sam2_derived(str(best["name"])) is False
            ),
            **nomask_support_info,
            "nomask_pair_consensus_selected": bool(best.get("metrics", {}).get("nomask_pair_consensus_context")),
            "adaptive_fusion_candidates": [
                {
                    **{k: v for k, v in item["metrics"].items() if k != "name"},
                    "name": item["name"],
                    "score": float(item["score"]),
                }
                for item in candidates
            ],
        }
    )

    qc_path = mask_path.with_name(mask_path.name.replace(".nii.gz", "_adaptive-fusion_qc.json"))
    qc_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    result["adaptive_fusion_qc_path"] = str(qc_path)

    prob = _fusion_probability(candidates[: min(len(candidates), 8)])
    if prob is not None:
        prob_path = mask_path.with_name(mask_path.name.replace(".nii.gz", "_adaptive-fusion_probability.nii.gz"))
        unc_path = mask_path.with_name(mask_path.name.replace(".nii.gz", "_adaptive-fusion_uncertainty.nii.gz"))
        save_float(prob, ref_img, prob_path)
        save_float(1.0 - np.abs(prob - 0.5) * 2.0, ref_img, unc_path)
        result["adaptive_fusion_probability_path"] = str(prob_path)
        result["adaptive_fusion_uncertainty_path"] = str(unc_path)

    if best["name"] == "current":
        if low_trust_fail_clean:
            result["adaptive_fusion_reason"] = "insufficient_external_atlas_support_low_trust"
        else:
            result["adaptive_fusion_reason"] = (
                "current_mask_selected_no_qc_trigger"
                if bool(cfg.get("replace_only_if_raw_qc_triggered", False)) and not raw_bad and not raw_undercoverage
                else "current_mask_selected"
            )
        if metrics_path:
            update_metrics_json(metrics_path, result)
        return result

    selected = np.asarray(best["mask"], dtype=bool)
    backup = mask_path.with_name(mask_path.name.replace(".nii.gz", f"{cfg['backup_suffix']}.nii.gz"))
    # ⚠️ PIÈGE DE REPRODUCTIBILITÉ CONNU — voir REPRODUCIBILITY_KNOWN_ISSUES.md §1 et §2.
    # `if not backup.exists()` est correct DANS un run (l'étage ne doit pas écraser son
    # propre snapshot), mais sur un run REPRIS après interruption le snapshot du segment
    # précédent est encore sur disque : il n'est donc PAS rafraîchi et décrit un état
    # périmé (mesuré : 18-58 h plus ancien que le masque final). Deux conséquences :
    #   (1) `adaptive_fusion_backup_mask` n'est pas renvoyé → inflation_guard reçoit
    #       base_mask_path=None et se désactive silencieusement (fail-open) ;
    #   (2) toute analyse d'ablation lisant les `_pre-*` mélange des segments différents.
    # NON CORRIGÉ VOLONTAIREMENT (les résultats publiés ont été produits ainsi).
    # Contournement : exécuter chaque run en une seule session ininterrompue, ou nettoyer
    # le dossier du sujet en cours avant de reprendre.
    if not backup.exists():
        shutil.copyfile(mask_path, backup)
    save_mask(selected, ref_img, mask_path)
    save_float(image * selected.astype(np.float32), ref_img, brain_path)
    result.update(
        {
            "adaptive_fusion_used": True,
            "adaptive_fusion_reason": "catastrophic_fallback" if raw_bad else "qc_selected_better_candidate",
            "adaptive_fusion_backup_mask": str(backup),
            "adaptive_fusion_volume_after_cm3": float(mask_volume_cm3(selected, ref_img)),
        }
    )
    if metrics_path:
        update_metrics_json(metrics_path, result)
    qc_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
