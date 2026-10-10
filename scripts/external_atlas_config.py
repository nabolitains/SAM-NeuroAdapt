"""External atlas routing for lightweight M1 evaluation cells."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Mapping


MNIINFANT_COHORT_AGES_MONTHS = {
    1: (0.0, 2.0),
    2: (2.0, 5.0),
    3: (5.0, 8.0),
    4: (8.0, 11.0),
    5: (11.0, 14.0),
    6: (14.0, 17.0),
    7: (17.0, 21.0),
    8: (21.0, 27.0),
    9: (27.0, 33.0),
    10: (33.0, 44.0),
    11: (44.0, 60.0),
}

ADULT_EXTERNAL_ATLAS_FILES = {
    "brain_mask_oasis30ants": ("oasis30ants", "brain_mask.nii.gz"),
    "brain_mask_mni152_nlin2009c": ("mni152_nlin_2009c", "brain_mask.nii.gz"),
    "brain_mask_miitra_older_adult": ("miitra_older_adult", "brain_mask.nii.gz"),
    "brain_mask_sri24": ("sri24", "brain_mask.nii.gz"),
    "brain_mask_colin27_2008": ("colin27_2008", "brain_mask.nii.gz"),
    "brain_mask_icbm452": ("icbm452", "brain_mask.nii.gz"),
    "brain_mask_lpba40_average": ("lpba40_average", "brain_mask.nii.gz"),
}
ADULT_PUBLICATION_BASE_ATLAS_KEYS = (
    "brain_mask_oasis30ants",
    "brain_mask_mni152_nlin2009c",
)
ADULT_DIAGNOSTIC_TEMPLATE_ENV = "SAM_NEUROADAPT_IBSR18_USE_ALL_ADULT_TEMPLATES"
LPBA40_PUBLIC_ATLAS_ENV = "SAM_NEUROADAPT_LPBA40_USE_LPBA_AVERAGE"
PEDIATRIC_DIAGNOSTIC_TEMPLATE_ENV = "SAM_NEUROADAPT_PEDIATRIC_USE_ALL_TEMPLATES"
ALBERTS_BANK_ENV = "SAM_NEUROADAPT_ENABLE_ALBERTS_BANK"
# dHCP neonatal pour MCRIB/MCRIB2 : ON par défaut, mettre à "0" pour désactiver
# (comparaison déterministe A/B dHCP-on vs dHCP-off).
MCRIB_DHCP_ENV = "SAM_NEUROADAPT_MCRIB_DHCP"
ALBERTS_GOUSIAS_IDS = tuple(f"alberts_gousias_{idx:02d}" for idx in range(1, 21))


PROFILE_CONFIGS = {
    "adult_t1": {
        "confidence": 0.70,
        "atlas_weight": 0.72,
        "min_volume_cm3": 650.0,
        "max_volume_cm3": 2100.0,
        "atlas_qc_min_cm3": 650.0,
        "atlas_qc_max_cm3": 2100.0,
        "max_components": 3,
        "refine": {
            "enabled": True,
            "min_volume_cm3": 650.0,
            "max_volume_cm3": 2100.0,
            "prune_if_atlas_precision_below": 0.90,
            "prune_if_volume_ratio_above": 1.12,
            "recover_max_added_cm3": 90.0,
            "recover_max_volume_ratio": 1.06,
            "recover_min_atlas_precision": 0.86,
        },
    },
    "infant_t1": {
        "confidence": 0.64,
        "atlas_weight": 0.78,
        "min_volume_cm3": 250.0,
        "max_volume_cm3": 1400.0,
        "atlas_qc_min_cm3": 220.0,
        "atlas_qc_max_cm3": 1400.0,
        "max_components": 3,
        "refine": {
            "enabled": True,
            "min_volume_cm3": 250.0,
            "max_volume_cm3": 1400.0,
            "prune_if_atlas_precision_below": 0.88,
            "prune_if_volume_ratio_above": 1.16,
            "recover_shell_width": 2,
            "recover_atlas_dilation": 2,
            "recover_max_added_cm3": 55.0,
            "recover_max_volume_ratio": 1.08,
            "recover_min_atlas_precision": 0.84,
        },
    },
    "fetal_fmri": {
        # Fetal functional EPI stress test: tiny brains (~40-570 cm3 mask range
        # observed) guided by neonatal atlases.  Not used by the pediatric M1
        # runner (it carries its own config) but defined here so any caller of
        # config_for_profile("fetal_fmri") gets small-brain priors instead of
        # silently falling back to the adult_t1 profile.
        "confidence": 0.60,
        "atlas_weight": 0.74,
        "min_volume_cm3": 40.0,
        "max_volume_cm3": 700.0,
        "atlas_qc_min_cm3": 30.0,
        "atlas_qc_max_cm3": 750.0,
        "max_components": 3,
        "refine": {
            "enabled": True,
            "min_volume_cm3": 40.0,
            "max_volume_cm3": 700.0,
            "prune_if_atlas_precision_below": 0.86,
            "prune_if_volume_ratio_above": 1.20,
            "recover_shell_width": 2,
            "recover_atlas_dilation": 2,
            "recover_max_added_cm3": 40.0,
            "recover_max_volume_ratio": 1.10,
            "recover_min_atlas_precision": 0.80,
        },
    },
    "infant_t1t2": {
        "confidence": 0.62,
        "atlas_weight": 0.80,
        "min_volume_cm3": 250.0,
        "max_volume_cm3": 1400.0,
        "atlas_qc_min_cm3": 220.0,
        "atlas_qc_max_cm3": 1400.0,
        "max_components": 3,
        "refine": {
            "enabled": True,
            "min_volume_cm3": 250.0,
            "max_volume_cm3": 1400.0,
            "prune_if_atlas_precision_below": 0.88,
            "prune_if_volume_ratio_above": 1.16,
            "recover_shell_width": 2,
            "recover_atlas_dilation": 2,
            "recover_max_added_cm3": 65.0,
            "recover_max_volume_ratio": 1.08,
            "recover_min_atlas_precision": 0.84,
        },
    },
    "pediatric_t1": {
        "confidence": 0.66,
        "atlas_weight": 0.76,
        "min_volume_cm3": 350.0,
        "max_volume_cm3": 1750.0,
        "atlas_qc_min_cm3": 320.0,
        "atlas_qc_max_cm3": 1750.0,
        "max_components": 3,
        "refine": {
            "enabled": True,
            "min_volume_cm3": 350.0,
            "max_volume_cm3": 1750.0,
            "prune_if_atlas_precision_below": 0.88,
            "prune_if_volume_ratio_above": 1.14,
            "recover_shell_width": 2,
            "recover_atlas_dilation": 2,
            "recover_max_added_cm3": 75.0,
            "recover_max_volume_ratio": 1.08,
            "recover_min_atlas_precision": 0.84,
        },
    },
}


def _adult_v4_dataset_override(
    *,
    min_volume_cm3: float,
    soft_min_volume_cm3: float,
    soft_max_volume_cm3: float,
    max_volume_cm3: float,
    atlas_topk: int = 5,
) -> dict:
    """Publication-frozen adult no-LOO settings for extra T1 gold datasets.

    These settings keep the same M1 architecture used by the IBSR18/LPBA40 v4
    cells while adapting only broad adult volume priors.  They do not introduce
    dataset-subject atlases, LOO caches, or SOTA masks as guidance.
    """

    return {
        "min_volume_cm3": float(min_volume_cm3),
        "max_volume_cm3": float(max_volume_cm3),
        "atlas_qc_min_cm3": float(min_volume_cm3),
        "atlas_qc_max_cm3": float(max_volume_cm3),
        "atlas_soft_min_cm3": float(soft_min_volume_cm3),
        "atlas_soft_max_cm3": float(soft_max_volume_cm3),
        "atlas_ensemble_topk": int(atlas_topk),
        "atlas_ensemble_balance_families": True,
        "atlas_consensus_primary_on_disagreement": True,
        "atlas_consensus_primary_disagreement_min": 0.12,
        "atlas_consensus_primary_min_ratio": 0.82,
        "atlas_consensus_primary_max_ratio": 1.18,
        "atlas_consensus_primary_min_ratio_to_selected_median": 0.88,
        "atlas_consensus_primary_max_intensity_score": 0.20,
        "reference_atlas_policy": "primary_unless_consensus_primary",
        "atlas_rescue_tissue_pruned_enabled": True,
        "atlas_rescue_overlarge_max_cm3": float(max_volume_cm3) * 1.35,
        "atlas_rescue_tissue_percentiles": [5.0, 8.0, 10.0, 12.0, 15.0, 20.0],
        "atlas_rescue_candidate_penalty": 0.10,
        "atlas_ensemble_max_components": 6,
        "atlas_ensemble_max_extent_ratio": 0.990,
        "atlas_agreement_weight": 0.35,
        "atlas_disagreement_penalty": 0.20,
        "atlas_consensus_threshold": 0.55,
        "atlas_intensity_score_weight": 0.30,
        "atlas_nomask_penalty": 0.18,
        "atlas_miitra_prior_bonus": 0.035,
        "atlas_icbm_derived_mask_penalty": 0.035,
        "atlas_registration_random_seed": 20260517,
        "atlas_registration_aff_random_sampling_rate": 1.0,
        "adaptive_soft_prior_from_atlas": True,
        "adaptive_soft_prior_min_ratio": 0.76,
        "adaptive_soft_prior_max_ratio": 1.08,
        "adaptive_soft_prior_iqr_guard_ratio": 0.30,
        "adaptive_soft_prior_unstable_max_ratio": 1.02,
        "external_learned_priors": {
            "enabled": False,
            "tools": (),
            "use_as_guidance": False,
            "use_as_reference": False,
        },
        "adaptive_atlas": {
            "enabled": True,
            "min_volume_cm3": float(min_volume_cm3),
            "soft_min_volume_cm3": float(soft_min_volume_cm3),
            "soft_max_volume_cm3": float(soft_max_volume_cm3),
            "max_volume_cm3": float(max_volume_cm3),
            "max_components": 6,
            "envelope_dilation": 2,
            "boundary_width": 2,
            "shell_width": 8,
            "max_added_cm3": 180.0,
            "max_change_ratio": 0.36,
            "external_volume_floor_enabled": True,
            "external_volume_floor_ratio": 0.88,
            "external_volume_floor_min_cm3": float(soft_min_volume_cm3),
            "external_volume_floor_max_score_drop": 0.12,
            "selection_margin": 0.010,
            "tie_break_margin": 0.004,
            "tie_break_prefer_lower_volume": True,
            "trim_percentiles": [1.0, 2.0, 4.0, 6.0],
            "tissue_percentiles": [2.0, 5.0, 8.0, 12.0],
            "probability_thresholds": [0.45, 0.50, 0.55, 0.60],
            "save_top_candidates": 8,
            "intensity_weight": 0.32,
            "volume_weight": 0.24,
            "agreement_weight": 0.21,
            "shape_weight": 0.13,
            "seed_weight": 0.10,
        },
        "refine": {
            "min_volume_cm3": float(min_volume_cm3),
            "max_volume_cm3": float(max_volume_cm3),
            "prune_if_atlas_precision_below": 0.90,
            "prune_if_volume_ratio_above": 1.10,
            "recover_max_added_cm3": 90.0,
            "recover_max_volume_ratio": 1.06,
            "recover_min_atlas_precision": 0.88,
        },
        "boundary_refine": {
            "enabled": True,
            "min_volume_cm3": float(min_volume_cm3),
            "soft_min_volume_cm3": float(soft_min_volume_cm3),
            "soft_max_volume_cm3": float(soft_max_volume_cm3),
            "max_volume_cm3": float(max_volume_cm3),
            "max_components": 4,
            "atlas_dilation": 2,
            "boundary_width": 2,
            "shell_width": 7,
            "trim_percentiles": [1.0, 2.0, 4.0],
            "recover_percentiles": [2.0, 5.0, 8.0],
            "recover_max_added_cm3": 120.0,
            "max_apply_change_ratio": 0.22,
            "max_candidate_change_ratio": 0.36,
            "min_score_gain_to_apply": 0.012,
            "max_atlas_dice_drop": 0.025,
        },
        "adaptive_fusion": {
            "enabled": True,
            "min_volume_cm3": float(min_volume_cm3),
            "soft_min_volume_cm3": float(soft_min_volume_cm3),
            "soft_max_volume_cm3": float(soft_max_volume_cm3),
            "max_volume_cm3": float(max_volume_cm3),
            "tissue_percentiles": [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0],
            "use_t2_gate": False,
            "coverage_target": 0.82,
            "coverage_sigma": 0.17,
            "catastrophic_min_precision": 0.52,
            "catastrophic_min_coverage": 0.45,
            "catastrophic_min_atlas_dice": 0.55,
            "catastrophic_max_uncertainty_mean": 0.30,
            "undercoverage_min_precision": 0.95,
            "undercoverage_max_coverage": 0.82,
            "undercoverage_max_volume_ratio": 0.86,
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
            "nomask_pair_non_nomask_penalty": 0.22,
            "overcoverage_min_coverage": 0.94,
            "overcoverage_max_precision": 0.88,
            "overcoverage_min_volume_ratio": 1.10,
            "reference_atlas_penalty": 0.10,
            "current_mask_bonus": 0.04,
            "tissue_undercoverage_bonus": 0.08,
            "undercoverage_recovery_bonus": 0.12,
            "current_undercoverage_penalty": 0.10,
            "current_overcoverage_penalty": 0.08,
            "atlas_guard_enabled": True,
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
            "intensity_weight": 0.12,
            "include_atlas_candidates": True,
            "atlas_candidate_penalty": 0.025,
            "max_nonraw_change_ratio": 0.32,
            "min_nonraw_candidate_current_dice": 0.98,
            "min_undercoverage_candidate_current_dice": 0.82,
            "max_undercoverage_change_ratio": 0.48,
            "min_overcoverage_candidate_current_dice": 0.82,
            "max_overcoverage_change_ratio": 0.40,
            "min_raw_candidate_current_dice": 0.94,
            "selection_margin": 0.012,
            "triggered_selection_margin": 0.0,
            "recover_max_added_cm3": 220.0,
            "recover_shell_width": 10,
            "replace_only_if_raw_qc_triggered": False,
            "backup_suffix": "_pre-adult-adaptive-fusion",
        },
        "strong_refine": {
            "enabled": True,
            "min_volume_cm3": float(min_volume_cm3),
            "soft_min_volume_cm3": float(soft_min_volume_cm3),
            "soft_max_volume_cm3": float(soft_max_volume_cm3),
            "max_volume_cm3": float(max_volume_cm3),
            "raw_bad_soft_volume": True,
            "roi_dilation": 12,
            "atlas_dilation": 5,
            "atlas_weight": 0.07,
            "current_similarity_weight": 0.18,
            "volume_weight": 0.30,
            "intensity_weight": 0.22,
            "shape_weight": 0.12,
            "probability_weight": 0.10,
            "uncertainty_weight": 0.08,
            "selection_margin": 0.020,
            "max_change_ratio": 0.42,
            "max_triggered_change_ratio": 0.55,
            "tissue_percentiles": [2.0, 5.0, 10.0],
            "probability_thresholds": [0.45, 0.60, 0.75, 0.80],
            "shell_recovery_width": 12,
            "shell_recovery_percentile": 2.0,
            "shell_recovery_max_added_cm3": 180.0,
            "min_reference_trust_for_any_switch": 0.12,
        },
        "candidate_adjudicator": {
            "enabled": True,
            "min_volume_cm3": float(min_volume_cm3),
            "soft_min_volume_cm3": float(soft_min_volume_cm3),
            "soft_max_volume_cm3": float(soft_max_volume_cm3),
            "max_volume_cm3": float(max_volume_cm3),
            "max_components": 6,
            "high_uncertainty_threshold": 0.30,
            "soft_volume_trigger": True,
            "atlas_mismatch_min_trust": 0.50,
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
            "atlas_rescue_enabled": True,
            "atlas_rescue_min_current_dice": 0.62,
            "atlas_rescue_undercoverage_min_current_dice": 0.40,
            "atlas_rescue_min_precision": 0.985,
            "atlas_rescue_min_coverage": 0.82,
            "atlas_rescue_undercoverage_min_precision": 0.84,
            "atlas_rescue_undercoverage_min_coverage": 0.64,
            "atlas_rescue_relaxed_min_precision": 0.74,
            "atlas_rescue_relaxed_min_coverage": 0.90,
            "atlas_rescue_min_volume_score": 0.50,
            "atlas_rescue_min_intensity_score": 0.50,
            "atlas_rescue_min_probability_score": 0.76,
            "atlas_rescue_undercoverage_min_probability_score": 0.45,
            "atlas_rescue_min_uncertainty_score": 0.78,
            "atlas_rescue_undercoverage_min_uncertainty_score": 0.45,
            "atlas_rescue_overcoverage_max_current_volume_ratio": 0.92,
            "atlas_rescue_bonus": 0.14,
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
            "single_nomask_non_nomask_penalty": 0.24,
            "atlas_switch_min_current_dice": 0.86,
            "compact_consensus_large_change_penalty": 0.22,
            "compact_atlas_min_probability_score": 0.78,
            "compact_atlas_low_probability_penalty": 0.16,
            "compact_atlas_large_change_current_dice": 0.90,
            "compact_atlas_large_change_penalty": 0.08,
            "current_prune_rescue_enabled": True,
            "current_prune_rescue_min_current_dice": 0.965,
            "current_prune_rescue_min_probability_score": 0.84,
            "current_prune_rescue_min_uncertainty_score": 0.84,
            "current_prune_rescue_bonus": 0.12,
            "atlas_vote_family_balanced": True,
            "atlas_vote_compact_fraction": 0.58,
            "stability_enabled": True,
            "stability_atlas_disagreement_threshold": 0.32,
            "stability_volume_iqr_ratio_threshold": 0.24,
            "stability_disagreement_safe_ratio_low": 0.87,
            "stability_disagreement_safe_ratio_high": 1.15,
            "stability_overseg_current_to_median_ratio": 1.14,
            "stability_current_atlas_precision_max": 0.935,
            "stability_current_atlas_coverage_min": 0.94,
            "stability_low_trust_reference_threshold": 0.11,
            "stability_surface_volume_ratio_to_softmax": 0.93,
            "stability_current_penalty": 0.14,
            "stability_compact_candidate_bonus": 0.12,
            "stability_prune_candidate_bonus": 0.08,
            "stability_candidate_score_drop_tolerance": 0.04,
            "stability_force_switch_score_drop_tolerance": 0.06,
            "stability_min_noncurrent_score": 0.82,
            "stability_inconclusive_selection_margin": 0.10,
            "min_inconclusive_candidate_current_dice": 0.96,
            "save_generated_candidates": True,
            "generated_candidate_save_limit": 48,
            "atlas_envelope_prune_dilation": 1,
            "current_pruning_enabled": True,
            "current_prune_atlas_dilations": [0, 1, 2],
            "current_prune_probability_quantiles": [10, 20],
            "current_prune_uncertainty_quantiles": [75, 85],
            "current_prune_t1_boundary_quantiles": [5, 10, 20, 30],
            "current_prune_min_volume_ratio": 0.80,
            "current_prune_max_volume_ratio": 0.99,
            "current_prune_bonus": 0.04,
            "current_prune_good_reduction_bonus": 0.035,
            "current_prune_precision_gain_bonus": 0.035,
            "boundary_energy_enabled": True,
            "boundary_energy_candidate_quantiles": [15, 25, 35],
            "boundary_energy_recover_quantiles": [75, 85],
            "boundary_energy_min_volume_ratio": 0.72,
            "boundary_energy_max_volume_ratio": 1.05,
            "surface_current_penalty": 0.04,
            "min_mild_overcoverage_candidate_current_dice": 0.97,
            "min_surface_candidate_current_dice": 0.99,
            "min_current_prune_triggered_candidate_current_dice": 0.93,
            "stable_selection_margin": 0.06,
            "triggered_selection_margin": 0.020,
            "current_trigger_penalty": 0.11,
            "high_uncertainty_current_penalty": 0.18,
            "min_stable_candidate_current_dice": 0.985,
            "min_triggered_candidate_current_dice": 0.78,
            "min_high_uncertainty_candidate_current_dice": 0.52,
            "min_raw_bad_candidate_current_dice": 0.82,
            "save_top_candidates": 24,
        },
    }


DATASET_PROFILE_OVERRIDES = {
    ("NFBS", "adult_t1"): _adult_v4_dataset_override(
        min_volume_cm3=800.0,
        soft_min_volume_cm3=1000.0,
        soft_max_volume_cm3=1750.0,
        max_volume_cm3=1900.0,
    ),
    ("MRBrainS13", "adult_t1"): _adult_v4_dataset_override(
        min_volume_cm3=900.0,
        soft_min_volume_cm3=1100.0,
        soft_max_volume_cm3=1800.0,
        max_volume_cm3=1950.0,
    ),
    ("SynthStripT1", "adult_t1"): _adult_v4_dataset_override(
        min_volume_cm3=900.0,
        soft_min_volume_cm3=1150.0,
        soft_max_volume_cm3=1950.0,
        max_volume_cm3=2200.0,
    ),
    ("IBSR18", "adult_t1"): {
        "max_volume_cm3": 1900.0,
        "atlas_qc_max_cm3": 1900.0,
        "atlas_target_volume_cm3": None,
        "atlas_soft_min_cm3": 850.0,
        "atlas_soft_max_cm3": 1700.0,
        "atlas_ensemble_topk": 5,
        "atlas_ensemble_balance_families": True,
        "atlas_consensus_primary_on_disagreement": True,
        "atlas_consensus_primary_disagreement_min": 0.12,
        "atlas_consensus_primary_min_ratio": 0.82,
        "atlas_consensus_primary_max_ratio": 1.18,
        "atlas_consensus_primary_min_ratio_to_selected_median": 0.88,
        "atlas_consensus_primary_max_intensity_score": 0.20,
        "reference_atlas_policy": "primary_unless_consensus_primary",
        "atlas_rescue_tissue_pruned_enabled": True,
        "atlas_rescue_overlarge_max_cm3": 2600.0,
        "atlas_rescue_tissue_percentiles": [5.0, 8.0, 10.0, 12.0, 15.0, 20.0],
        "atlas_rescue_candidate_penalty": 0.10,
        "atlas_ensemble_max_components": 6,
        "atlas_ensemble_max_extent_ratio": 0.990,
        "atlas_agreement_weight": 0.35,
        "atlas_disagreement_penalty": 0.20,
        "atlas_consensus_threshold": 0.55,
        "atlas_intensity_score_weight": 0.45,
        "atlas_nomask_penalty": 0.18,
        "atlas_low_trust_rescue_enabled": True,
        "atlas_low_trust_intensity_threshold": 0.12,
        "atlas_low_trust_rescue_min_intensity_gain": 0.25,
        "atlas_low_trust_rescue_primary_bonus": 0.30,
        "atlas_low_trust_rescue_max_extent_ratio": 1.0,
        "atlas_registration_backend": "native_ants_syn",
        "atlas_registration_transform": "SyN",
        "atlas_registration_variants": ("syn_masked", "syn_nomask"),
        "atlas_registration_syn_metric": "CC",
        "atlas_registration_syn_sampling": 4,
        "atlas_registration_syn_iterations": (60, 40, 20, 0),
        "atlas_registration_random_seed": 20260517,
        "atlas_registration_aff_random_sampling_rate": 1.0,
        "adaptive_soft_prior_from_atlas": True,
        "adaptive_soft_prior_min_ratio": 0.78,
        "adaptive_soft_prior_max_ratio": 1.08,
        "work_resample_spacing_mm": None,
        "ibsr_annotation_alignment": {
            "enabled": True,
            "max_abs_shift_vox": 8,
        },
        "external_learned_priors": {
            "enabled": False,
            "tools": (),
            "timeout_sec": 600.0,
            "use_as_guidance": False,
            "use_as_reference": False,
            "score": {
                "min_volume_cm3": 650.0,
                "soft_min_volume_cm3": 850.0,
                "soft_max_volume_cm3": 1700.0,
                "max_volume_cm3": 1900.0,
                "max_components": 3,
                "max_extent_ratio": 0.990,
                "min_intensity_score": 0.15,
                "min_score": 0.58,
            },
        },
        "adaptive_atlas": {
            "enabled": True,
            "min_volume_cm3": 650.0,
            "soft_min_volume_cm3": 850.0,
            "soft_max_volume_cm3": 1700.0,
            "max_volume_cm3": 1900.0,
            "max_components": 6,
            "envelope_dilation": 3,
            "boundary_width": 2,
            "shell_width": 12,
            "max_added_cm3": 260.0,
            "max_change_ratio": 0.42,
            "external_volume_floor_enabled": True,
            "external_volume_floor_ratio": 0.88,
            "external_volume_floor_min_cm3": 1250.0,
            "external_volume_floor_max_score_drop": 0.14,
            "selection_margin": 0.0,
            "tie_break_margin": 0.004,
            "tie_break_prefer_lower_volume": False,
            "trim_percentiles": [1.0, 2.0, 4.0, 6.0],
            "tissue_percentiles": [1.0, 2.0, 5.0, 8.0, 12.0, 16.0],
            "probability_thresholds": [0.45, 0.50, 0.55, 0.60],
            "save_top_candidates": 10,
            "intensity_weight": 0.30,
            "volume_weight": 0.20,
            "agreement_weight": 0.24,
            "shape_weight": 0.12,
            "seed_weight": 0.14,
        },
        "refine": {
            "max_volume_cm3": 1900.0,
            "prune_if_atlas_precision_below": 0.88,
            "prune_if_volume_ratio_above": 1.08,
            "recover_max_added_cm3": 80.0,
            "recover_max_volume_ratio": 1.05,
            "recover_min_atlas_precision": 0.88,
        },
        "boundary_refine": {
            "enabled": True,
            "min_volume_cm3": 650.0,
            "soft_min_volume_cm3": 850.0,
            "soft_max_volume_cm3": 1700.0,
            "max_volume_cm3": 1900.0,
            "max_components": 4,
            "atlas_dilation": 2,
            "boundary_width": 2,
            "shell_width": 7,
            "trim_percentiles": [1.0, 2.0, 4.0],
            "recover_percentiles": [2.0, 5.0, 8.0],
            "recover_max_added_cm3": 160.0,
            "max_apply_change_ratio": 0.26,
            "max_candidate_change_ratio": 0.42,
            "min_score_gain_to_apply": 0.006,
            "max_atlas_dice_drop": 0.025,
        },
        "adaptive_fusion": {
            "enabled": True,
            "min_volume_cm3": 650.0,
            "soft_min_volume_cm3": 875.0,
            "soft_max_volume_cm3": 1700.0,
            "max_volume_cm3": 1900.0,
            "tissue_percentiles": [2.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0],
            "use_t2_gate": False,
            "coverage_target": 0.90,
            "coverage_sigma": 0.16,
            "catastrophic_min_precision": 0.52,
            "catastrophic_min_coverage": 0.42,
            "catastrophic_min_atlas_dice": 0.52,
            "catastrophic_max_uncertainty_mean": 1.0,
            "undercoverage_min_precision": 0.92,
            "undercoverage_max_coverage": 0.84,
            "undercoverage_max_volume_ratio": 0.86,
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
            "overcoverage_min_coverage": 0.94,
            "overcoverage_max_precision": 0.88,
            "overcoverage_min_volume_ratio": 1.10,
            "reference_atlas_penalty": 0.025,
            "current_mask_bonus": 0.0,
            "tissue_undercoverage_bonus": 0.10,
            "undercoverage_recovery_bonus": 0.14,
            "current_undercoverage_penalty": 0.12,
            "current_overcoverage_penalty": 0.08,
            "atlas_guard_enabled": True,
            "atlas_guard_min_trust": 0.45,
            "atlas_guard_max_current_atlas_dice": 0.96,
            "atlas_guard_over_volume_ratio": 1.14,
            "atlas_guard_under_volume_ratio": 0.94,
            "min_atlas_guard_candidate_current_dice": 0.82,
            "min_atlas_guard_candidate_atlas_dice": 0.985,
            "max_atlas_guard_change_ratio": 0.46,
            "atlas_guard_reference_bonus": 0.20,
            "atlas_guard_current_penalty": 0.16,
            "intensity_weight": 0.14,
            "include_atlas_candidates": True,
            "atlas_candidate_penalty": 0.015,
            "max_nonraw_change_ratio": 0.30,
            "min_nonraw_candidate_current_dice": 0.98,
            "min_undercoverage_candidate_current_dice": 0.82,
            "max_undercoverage_change_ratio": 0.48,
            "min_overcoverage_candidate_current_dice": 0.82,
            "max_overcoverage_change_ratio": 0.40,
            "min_raw_candidate_current_dice": 0.96,
            "selection_margin": 0.018,
            "triggered_selection_margin": 0.0,
            "recover_max_added_cm3": 260.0,
            "recover_shell_width": 12,
            "replace_only_if_raw_qc_triggered": False,
            "backup_suffix": "_pre-adult-adaptive-fusion",
        },
        "strong_refine": {
            "enabled": True,
            "min_volume_cm3": 650.0,
            "soft_min_volume_cm3": 850.0,
            "soft_max_volume_cm3": 1700.0,
            "max_volume_cm3": 1900.0,
            "raw_bad_soft_volume": True,
            "roi_dilation": 12,
            "atlas_dilation": 5,
            "atlas_weight": 0.08,
            "current_similarity_weight": 0.14,
            "volume_weight": 0.32,
            "intensity_weight": 0.26,
            "shape_weight": 0.12,
            "probability_weight": 0.08,
            "uncertainty_weight": 0.08,
            "selection_margin": 0.012,
            "max_change_ratio": 0.36,
            "max_triggered_change_ratio": 0.55,
            "undercoverage_min_precision": 0.92,
            "undercoverage_max_coverage": 0.84,
            "undercoverage_max_volume_ratio": 0.86,
            "overcoverage_min_coverage": 0.94,
            "overcoverage_max_precision": 0.88,
            "overcoverage_min_volume_ratio": 1.10,
            "current_undercoverage_penalty": 0.12,
            "current_overcoverage_penalty": 0.08,
            "shell_recovery_bonus": 0.12,
            "atlas_vote_undercoverage_bonus": 0.06,
            "prune_overcoverage_bonus": 0.08,
            "undercoverage_requires_low_volume_or_high_trust": True,
            "undercoverage_low_volume_ratio": 0.98,
            "undercoverage_min_reference_trust": 0.85,
            "shell_recovery_soft_ok_min_reference_trust": 0.90,
            "shell_recovery_soft_ok_max_volume_ratio": 1.05,
            "shell_recovery_soft_ok_penalty": 0.22,
            "tissue_percentiles": [2.0, 5.0, 10.0],
            "probability_thresholds": [0.45, 0.60, 0.75, 0.80],
            "shell_recovery_width": 12,
            "shell_recovery_percentile": 2.0,
            "shell_recovery_max_added_cm3": 160.0,
        },
        "candidate_adjudicator": {
            "enabled": True,
            "min_volume_cm3": 650.0,
            "soft_min_volume_cm3": 850.0,
            "soft_max_volume_cm3": 1700.0,
            "max_volume_cm3": 1900.0,
            "max_components": 6,
            "high_uncertainty_threshold": 0.27,
            "soft_volume_trigger": True,
            "atlas_mismatch_min_trust": 0.45,
            "atlas_mismatch_max_dice": 0.92,
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
            "overcoverage_precision_dominant_min_precision": 0.985,
            "overcoverage_precision_dominant_max_coverage": 0.93,
            "overcoverage_precision_dominant_bonus": 0.08,
            "overcoverage_boundary_min_precision": 0.965,
            "overcoverage_boundary_max_current_volume_ratio": 0.98,
            "overcoverage_boundary_penalty": 0.08,
            "high_uncertainty_tissue_recovery_bonus": 0.065,
            "high_uncertainty_tissue_recovery_min_atlas_coverage": 0.70,
            "high_uncertainty_tissue_recovery_min_current_dice": 0.58,
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
            "atlas_rescue_min_probability_score": 0.74,
            "atlas_rescue_undercoverage_min_probability_score": 0.45,
            "atlas_rescue_min_uncertainty_score": 0.74,
            "atlas_rescue_undercoverage_min_uncertainty_score": 0.45,
            "atlas_rescue_overcoverage_max_current_volume_ratio": 0.93,
            "atlas_rescue_bonus": 0.22,
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
            "single_nomask_non_nomask_penalty": 0.24,
            "atlas_switch_min_current_dice": 0.86,
            "compact_consensus_large_change_penalty": 0.22,
            "compact_atlas_min_probability_score": 0.78,
            "compact_atlas_low_probability_penalty": 0.18,
            "compact_atlas_large_change_current_dice": 0.90,
            "compact_atlas_large_change_penalty": 0.10,
            "current_prune_rescue_enabled": True,
            "current_prune_rescue_min_current_dice": 0.965,
            "current_prune_rescue_min_probability_score": 0.84,
            "current_prune_rescue_min_uncertainty_score": 0.84,
            "current_prune_rescue_bonus": 0.14,
            "atlas_vote_family_balanced": True,
            "atlas_vote_compact_fraction": 0.55,
            "stability_enabled": True,
            "stability_atlas_disagreement_threshold": 0.28,
            "stability_volume_iqr_ratio_threshold": 0.20,
            "stability_disagreement_safe_ratio_low": 0.88,
            "stability_disagreement_safe_ratio_high": 1.12,
            "stability_overseg_current_to_median_ratio": 1.10,
            "stability_current_atlas_precision_max": 0.945,
            "stability_current_atlas_coverage_min": 0.94,
            "stability_low_trust_reference_threshold": 0.12,
            "stability_surface_volume_ratio_to_softmax": 0.90,
            "stability_current_penalty": 0.20,
            "stability_compact_candidate_bonus": 0.18,
            "stability_prune_candidate_bonus": 0.12,
            "stability_candidate_score_drop_tolerance": 0.05,
            "stability_force_switch_score_drop_tolerance": 0.10,
            "stability_min_noncurrent_score": 0.80,
            "stability_inconclusive_selection_margin": 0.10,
            "min_inconclusive_candidate_current_dice": 0.95,
            "save_generated_candidates": True,
            "generated_candidate_save_limit": 56,
            "atlas_envelope_prune_dilation": 1,
            "current_pruning_enabled": True,
            "current_prune_atlas_dilations": [0, 1, 2, 3],
            "current_prune_probability_quantiles": [10, 20, 30],
            "current_prune_uncertainty_quantiles": [70, 80, 90],
            "current_prune_t1_boundary_quantiles": [5, 10, 15, 20, 25, 35],
            "current_prune_min_volume_ratio": 0.74,
            "current_prune_max_volume_ratio": 0.995,
            "current_prune_bonus": 0.06,
            "current_prune_good_reduction_bonus": 0.05,
            "current_prune_precision_gain_bonus": 0.05,
            "boundary_energy_enabled": True,
            "boundary_energy_candidate_quantiles": [15, 25, 35],
            "boundary_energy_recover_quantiles": [75, 85],
            "boundary_energy_min_volume_ratio": 0.72,
            "boundary_energy_max_volume_ratio": 1.05,
            "surface_current_penalty": 0.06,
            "min_mild_overcoverage_candidate_current_dice": 0.955,
            "min_surface_candidate_current_dice": 0.985,
            "min_current_prune_triggered_candidate_current_dice": 0.90,
            "stable_selection_margin": 0.05,
            "triggered_selection_margin": 0.015,
            "current_trigger_penalty": 0.14,
            "high_uncertainty_current_penalty": 0.22,
            "min_stable_candidate_current_dice": 0.98,
            "min_triggered_candidate_current_dice": 0.74,
            "min_high_uncertainty_candidate_current_dice": 0.45,
            "min_raw_bad_candidate_current_dice": 0.80,
            "save_top_candidates": 24,
        },
    },
    ("LPBA40", "adult_t1"): {
        "min_volume_cm3": 800.0,
        "max_volume_cm3": 1900.0,
        "atlas_qc_min_cm3": 800.0,
        "atlas_qc_max_cm3": 1900.0,
        "reference_atlas_policy": "primary_unless_consensus_primary",
        "atlas_consensus_primary_min_ratio_to_selected_median": 0.88,
        "atlas_registration_random_seed": 20260517,
        "atlas_registration_aff_random_sampling_rate": 1.0,
        "adaptive_soft_prior_from_atlas": True,
        "adaptive_soft_prior_min_ratio": 0.76,
        "adaptive_soft_prior_max_ratio": 1.08,
        "atlas_intensity_score_weight": 0.30,
        "atlas_nomask_penalty": 0.18,
        "refine": {
            "min_volume_cm3": 800.0,
            "max_volume_cm3": 1900.0,
            "prune_if_atlas_precision_below": 0.90,
            "prune_if_volume_ratio_above": 1.10,
            "recover_max_added_cm3": 90.0,
            "recover_max_volume_ratio": 1.06,
            "recover_min_atlas_precision": 0.88,
        },
        "adaptive_fusion": {
            "enabled": True,
            "min_volume_cm3": 800.0,
            "soft_min_volume_cm3": 900.0,
            "soft_max_volume_cm3": 1700.0,
            "max_volume_cm3": 1900.0,
            "tissue_percentiles": [6.0, 8.0, 10.0, 12.0, 15.0, 18.0],
            "use_t2_gate": False,
            "coverage_target": 0.78,
            "coverage_sigma": 0.17,
            "catastrophic_min_precision": 0.52,
            "catastrophic_min_coverage": 0.45,
            "catastrophic_min_atlas_dice": 0.55,
            "catastrophic_max_uncertainty_mean": 0.25,
            "undercoverage_min_precision": 0.96,
            "undercoverage_max_coverage": 0.80,
            "undercoverage_max_volume_ratio": 0.84,
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
            "nomask_pair_non_nomask_penalty": 0.22,
            "reference_atlas_penalty": 0.12,
            "current_mask_bonus": 0.05,
            "tissue_undercoverage_bonus": 0.08,
            "selection_margin": 0.010,
            "recover_max_added_cm3": 150.0,
            "replace_only_if_raw_qc_triggered": True,
            "backup_suffix": "_pre-adult-adaptive-fusion",
        },
        "strong_refine": {
            "enabled": True,
            "min_volume_cm3": 800.0,
            "soft_min_volume_cm3": 900.0,
            "soft_max_volume_cm3": 1550.0,
            "max_volume_cm3": 1900.0,
            "raw_bad_soft_volume": True,
            "roi_dilation": 12,
            "atlas_dilation": 5,
            "atlas_weight": 0.07,
            "current_similarity_weight": 0.19,
            "volume_weight": 0.30,
            "intensity_weight": 0.22,
            "shape_weight": 0.12,
            "probability_weight": 0.10,
            "uncertainty_weight": 0.08,
            "selection_margin": 0.020,
            "max_change_ratio": 0.42,
            "tissue_percentiles": [2.0, 5.0, 10.0],
            "probability_thresholds": [0.45, 0.60, 0.75, 0.80],
            "shell_recovery_width": 12,
            "shell_recovery_percentile": 2.0,
            "shell_recovery_max_added_cm3": 180.0,
        },
        "candidate_adjudicator": {
            "enabled": True,
            "min_volume_cm3": 800.0,
            "soft_min_volume_cm3": 900.0,
            "soft_max_volume_cm3": 1700.0,
            "max_volume_cm3": 1900.0,
            "max_components": 6,
            "high_uncertainty_threshold": 0.32,
            "soft_volume_trigger": True,
            "atlas_mismatch_min_trust": 0.55,
            "atlas_mismatch_max_dice": 0.88,
            "overcoverage_min_coverage": 0.94,
            "overcoverage_max_precision": 0.88,
            "overcoverage_min_volume_ratio": 1.10,
            "undercoverage_min_precision": 0.95,
            "undercoverage_max_coverage": 0.82,
            "undercoverage_max_volume_ratio": 0.86,
            "mild_overcoverage_min_coverage": 0.99,
            "mild_overcoverage_max_precision": 0.92,
            "mild_overcoverage_min_volume_ratio": 1.06,
            "surface_low_trust_threshold": 0.10,
            "surface_low_trust_min_volume_ratio_to_softmax": 0.95,
            "surface_low_trust_max_atlas_dice": 0.92,
            "surface_boundary_uncertainty_threshold": 0.20,
            "surface_boundary_low_probability_threshold": 0.28,
            "surface_boundary_low_probability_fraction": 0.25,
            "surface_signal_max_reference_trust": 0.30,
            "overcoverage_precision_dominant_min_precision": 0.99,
            "overcoverage_precision_dominant_max_coverage": 0.90,
            "overcoverage_precision_dominant_bonus": 0.05,
            "overcoverage_boundary_min_precision": 0.97,
            "overcoverage_boundary_max_current_volume_ratio": 0.98,
            "overcoverage_boundary_penalty": 0.06,
            "high_uncertainty_tissue_recovery_bonus": 0.035,
            "high_uncertainty_tissue_recovery_min_atlas_coverage": 0.76,
            "high_uncertainty_tissue_recovery_min_current_dice": 0.70,
            "high_uncertainty_tissue_recovery_min_uncertainty_score": 0.92,
            "high_uncertainty_tissue_recovery_max_volume_cm3": 1780.0,
            "atlas_rescue_enabled": True,
            "atlas_rescue_min_current_dice": 0.70,
            "atlas_rescue_undercoverage_min_current_dice": 0.48,
            "atlas_rescue_min_precision": 0.990,
            "atlas_rescue_min_coverage": 0.84,
            "atlas_rescue_undercoverage_min_precision": 0.88,
            "atlas_rescue_undercoverage_min_coverage": 0.70,
            "atlas_rescue_relaxed_min_precision": 0.78,
            "atlas_rescue_relaxed_min_coverage": 0.92,
            "atlas_rescue_min_volume_score": 0.55,
            "atlas_rescue_min_intensity_score": 0.55,
            "atlas_rescue_min_probability_score": 0.78,
            "atlas_rescue_undercoverage_min_probability_score": 0.52,
            "atlas_rescue_min_uncertainty_score": 0.80,
            "atlas_rescue_undercoverage_min_uncertainty_score": 0.52,
            "atlas_rescue_overcoverage_max_current_volume_ratio": 0.90,
            "atlas_rescue_bonus": 0.12,
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
            "single_nomask_non_nomask_penalty": 0.24,
            "atlas_switch_min_current_dice": 0.86,
            "compact_consensus_large_change_penalty": 0.22,
            "compact_atlas_min_probability_score": 0.78,
            "compact_atlas_low_probability_penalty": 0.12,
            "compact_atlas_large_change_current_dice": 0.90,
            "compact_atlas_large_change_penalty": 0.06,
            "current_prune_rescue_enabled": True,
            "current_prune_rescue_min_current_dice": 0.970,
            "current_prune_rescue_min_probability_score": 0.86,
            "current_prune_rescue_min_uncertainty_score": 0.86,
            "current_prune_rescue_bonus": 0.10,
            "atlas_vote_family_balanced": True,
            "atlas_vote_compact_fraction": 0.60,
            "stability_enabled": True,
            "stability_atlas_disagreement_threshold": 0.35,
            "stability_volume_iqr_ratio_threshold": 0.26,
            "stability_disagreement_safe_ratio_low": 0.86,
            "stability_disagreement_safe_ratio_high": 1.16,
            "stability_overseg_current_to_median_ratio": 1.16,
            "stability_current_atlas_precision_max": 0.925,
            "stability_current_atlas_coverage_min": 0.95,
            "stability_low_trust_reference_threshold": 0.10,
            "stability_surface_volume_ratio_to_softmax": 0.95,
            "stability_current_penalty": 0.12,
            "stability_compact_candidate_bonus": 0.10,
            "stability_prune_candidate_bonus": 0.07,
            "stability_candidate_score_drop_tolerance": 0.03,
            "stability_force_switch_score_drop_tolerance": 0.05,
            "stability_min_noncurrent_score": 0.84,
            "stability_inconclusive_selection_margin": 0.12,
            "min_inconclusive_candidate_current_dice": 0.97,
            "save_generated_candidates": True,
            "generated_candidate_save_limit": 48,
            "atlas_envelope_prune_dilation": 1,
            "current_pruning_enabled": True,
            "current_prune_atlas_dilations": [0, 1, 2],
            "current_prune_probability_quantiles": [10, 20],
            "current_prune_uncertainty_quantiles": [75, 85],
            "current_prune_t1_boundary_quantiles": [5, 10, 20, 30],
            "current_prune_min_volume_ratio": 0.80,
            "current_prune_max_volume_ratio": 0.99,
            "current_prune_bonus": 0.04,
            "current_prune_good_reduction_bonus": 0.035,
            "current_prune_precision_gain_bonus": 0.035,
            "boundary_energy_enabled": True,
            "boundary_energy_candidate_quantiles": [15, 25, 35],
            "boundary_energy_recover_quantiles": [75, 85],
            "boundary_energy_min_volume_ratio": 0.76,
            "boundary_energy_max_volume_ratio": 1.04,
            "surface_current_penalty": 0.04,
            "min_mild_overcoverage_candidate_current_dice": 0.97,
            "min_surface_candidate_current_dice": 0.99,
            "min_current_prune_triggered_candidate_current_dice": 0.93,
            "stable_selection_margin": 0.07,
            "triggered_selection_margin": 0.025,
            "current_trigger_penalty": 0.10,
            "high_uncertainty_current_penalty": 0.18,
            "min_stable_candidate_current_dice": 0.985,
            "min_triggered_candidate_current_dice": 0.80,
            "min_high_uncertainty_candidate_current_dice": 0.55,
            "min_raw_bad_candidate_current_dice": 0.84,
            "save_top_candidates": 24,
        },
    },
}

DATASET_PROFILE_OVERRIDES[("NFBS", "adult_t1")]["candidate_adjudicator"].update(
    {
        "external_family_rescue_enabled": True,
        "external_family_rescue_min_current_dice": 0.42,
        "external_family_rescue_min_probability_score": 0.42,
        "external_family_rescue_min_uncertainty_score": 0.72,
        "external_family_rescue_bonus": 0.42,
        "external_family_rescue_current_prune_penalty": 0.24,
        "atlas_rescue_min_current_dice": 0.48,
        "atlas_rescue_bonus": 0.24,
        "save_top_candidates": 32,
    }
)

DATASET_PROFILE_OVERRIDES[("MRBrainS13", "adult_t1")]["candidate_adjudicator"].update(
    {
        "external_family_rescue_enabled": True,
        "external_family_rescue_min_current_dice": 0.40,
        "external_family_rescue_min_probability_score": 0.40,
        "external_family_rescue_min_uncertainty_score": 0.72,
        "external_family_rescue_bonus": 0.36,
        "external_family_rescue_current_prune_penalty": 0.22,
        "atlas_rescue_min_current_dice": 0.45,
        "atlas_rescue_bonus": 0.24,
        "save_top_candidates": 32,
    }
)

DATASET_PROFILE_OVERRIDES[("SynthStripT1", "adult_t1")]["candidate_adjudicator"].update(
    {
        "stable_external_rescue_enabled": True,
        "stable_external_rescue_max_current_atlas_dice": 0.96,
        "stable_external_rescue_min_volume_ratio_to_current": 0.88,
        "stable_external_rescue_max_volume_ratio_to_current": 0.99,
        "stable_external_rescue_min_current_dice": 0.88,
        "stable_external_rescue_min_probability_score": 0.82,
        "stable_external_rescue_min_uncertainty_score": 0.84,
        "stable_external_rescue_score_drop_tolerance": 0.12,
        "stable_external_rescue_bonus": 0.18,
        "save_top_candidates": 32,
    }
)


DATASET_DEFAULT_PROFILE = {
    "BOBs": "infant_t1t2",
    "iSeg2017": "infant_t1t2",
    "MCRIB": "infant_t1t2",
    "MCRIB_WM": "infant_t1t2",
    "MCRIB2": "infant_t1t2",
    "ALBERTs": "infant_t1t2",
    "iSeg2019": "infant_t1t2",
    "NeoBrainS12": "infant_t1t2",
    "dHCPNeonatal": "infant_t1t2",
    "FetalFMRI": "fetal_fmri",
    "CANDIShare": "pediatric_t1",
    "SynthStripT1": "adult_t1",
    "NFBS": "adult_t1",
    "IBSR18": "adult_t1",
    "LPBA40": "adult_t1",
    "MRBrainS13": "adult_t1",
}


def atlas_root() -> Path:
    return Path(os.environ.get("SAM_NEUROADAPT_ATLAS_ROOT", "dataset/atlas_library"))


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def no_atlas_ablation() -> bool:
    """Ablation « SAM2 sans guidance atlas » (SAM_NEUROADAPT_NO_ATLAS=1).

    Coupe UNIQUEMENT l'atlas à l'étage SAM2 (prompts, mask_input, masquage de
    l'entrée, fallback). Les raffinements post-SAM2 restent actifs : ils forment
    une ligne distincte du tableau d'ablation. Défaut OFF → pipeline inchangé.
    """
    return _env_truthy("SAM_NEUROADAPT_NO_ATLAS")


def _row_value(row: Mapping, key: str, default: str = "") -> str:
    value = row.get(key, default)
    if value is None:
        return default
    return str(value)


def month_from_session(session: str | None) -> int | None:
    match = re.search(r"(\d+)\s*mo", str(session or ""), re.IGNORECASE)
    return int(match.group(1)) if match else None


def mniinfant_cohort_for_month(month: int | float | None) -> int:
    if month is None:
        return 3
    value = float(month)
    for cohort, (lo, hi) in MNIINFANT_COHORT_AGES_MONTHS.items():
        if lo <= value < hi:
            return cohort
    return 11 if value >= 44 else 3


def mniinfant_neighbor_cohorts(month: int | float | None, radius: int = 1) -> tuple[int, ...]:
    cohort = mniinfant_cohort_for_month(month)
    lo = max(1, cohort - int(radius))
    hi = min(max(MNIINFANT_COHORT_AGES_MONTHS), cohort + int(radius))
    return tuple(range(lo, hi + 1))


def profile_for_row(dataset: str, row: Mapping) -> str:
    profile = _row_value(row, "profile")
    if profile:
        return profile
    return DATASET_DEFAULT_PROFILE.get(dataset, "adult_t1")


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base


def config_for_profile(profile: str) -> dict:
    return deepcopy(PROFILE_CONFIGS.get(profile, PROFILE_CONFIGS["adult_t1"]))


def config_for_dataset_profile(dataset: str, profile: str | None = None) -> dict:
    profile = profile or DATASET_DEFAULT_PROFILE.get(dataset, "adult_t1")
    cfg = config_for_profile(profile)
    override = DATASET_PROFILE_OVERRIDES.get((dataset, profile), {})
    return _deep_update(cfg, override)


def _existing(paths: dict[str, Path]) -> dict[str, Path]:
    return {name: path for name, path in paths.items() if Path(path).exists()}


def _existing_atlas_mask(root: Path, atlas_id: str) -> Path:
    return root / atlas_id / "brain_mask.nii.gz"


def external_atlas_files_for_row(dataset: str, row: Mapping, profile_filter: str | None = None) -> dict[str, Path]:
    """Return atlas mask paths for a dataset manifest row.

    The paths point to runtime-ready directories created by
    ``scripts/prepare_external_atlases.py``.  Each mask directory also contains
    a ``MNI152_T1_2mm.nii.gz`` alias expected by the existing M1 registration
    helper.
    """

    root = atlas_root()
    profile = profile_filter or profile_for_row(dataset, row)
    session = _row_value(row, "session")

    if profile == "fetal_fmri" or dataset == "FetalFMRI":
        # Atlas FŒTAUX dédiés s'ils ont été déposés/construits (banque GA-binnée
        # fetal_*, cf. dataset/raw_external_atlases/fetal/DEPOT_README.md). On
        # route TOUS les bins GA présents et _select_registered_atlas retient le
        # mieux aligné (l'image fœtale n'a pas de GA dans le manifest).
        fetal_atlases: dict[str, Path] = {}
        for d in sorted(root.glob("fetal_*")):
            mask = d / "brain_mask.nii.gz"
            if d.is_dir() and mask.exists():
                fetal_atlases[f"brain_mask_{d.name}"] = mask
        if fetal_atlases:
            return _existing(fetal_atlases)
        # Repli (aucun atlas fœtal déposé) : proxies néonataux — les plus proches
        # par la taille. Guidage de registration uniquement (pas de sortie) ; une
        # registration ratée sur du BOLD basse-résolution dégrade proprement vers
        # SAM2-seul (stop_if_no_valid_atlas désactivé pour la config fœtale).
        atlases = {
            "brain_mask_fbn125_neonatal": _existing_atlas_mask(root, "fbn125_neonatal"),
            "brain_mask_unc_neo": root / "unc_infant_neo" / "brain_mask.nii.gz",
            "brain_mask_unc_bcp_06m": _existing_atlas_mask(root, "unc_bcp_06m"),
        }
        return _existing(atlases)

    if profile in {"infant_t1", "infant_t1t2"} or dataset in {"BOBs", "iSeg2017"}:
        month = month_from_session(session)
        pediatric_diagnostic = _env_truthy(PEDIATRIC_DIAGNOSTIC_TEMPLATE_ENV)
        neonatal_dataset = dataset in {"MCRIB", "MCRIB_WM", "MCRIB2", "ALBERTs", "NeoBrainS12", "dHCPNeonatal"}
        neonatal_age = month is not None and month <= 2
        atlases: dict[str, Path] = {}
        if dataset == "BOBs":
            # BOBs spans roughly 1-9 months.  A small no-gold neighbourhood of
            # age-matched MNIInfant templates is more stable than a single cohort.
            for cohort in mniinfant_neighbor_cohorts(month, radius=1):
                atlases[f"brain_mask_mniinfant_c{cohort:02d}"] = _existing_atlas_mask(
                    root, f"mniinfant_cohort_{cohort:02d}"
                )
        elif month is None and not neonatal_dataset and dataset != "iSeg2017":
            # Âge inconnu (ex. SynthStripInfant, session vide) : impossible de
            # choisir un cohort fiable. On route un éventail représentatif
            # couvrant 0-33 mois et on laisse _select_registered_atlas retenir la
            # mieux alignée (score image-intensité). Le candidat gagnant fournit
            # une ESTIMATION D'ÂGE qui réactive en aval les priors volumétriques
            # (P12/P13/P9) et la pondération d'âge (P1.2).
            # MNIInfant (2 mm, 0-33 mo via cohortes 1/3/5/7/9, désormais toutes
            # présentes) + UNC-BCP 4D (haute-rés T1+T2, plus appariés). Chemin
            # exclusif aux datasets infant à âge inconnu : BOBs (mois connu) et
            # iSeg2017 (exclu) ne passent jamais ici → aucun risque de régression.
            for cohort in (1, 3, 5, 7, 9):
                atlases[f"brain_mask_mniinfant_c{cohort:02d}"] = _existing_atlas_mask(
                    root, f"mniinfant_cohort_{cohort:02d}"
                )
            for m in (1, 3, 5, 7, 9):
                atlases[f"brain_mask_unc_bcp_{m:02d}m"] = _existing_atlas_mask(
                    root, f"unc_bcp_{m:02d}m"
                )
        else:
            cohort = mniinfant_cohort_for_month(month)
            atlases[f"brain_mask_mniinfant_c{cohort:02d}"] = _existing_atlas_mask(
                root, f"mniinfant_cohort_{cohort:02d}"
            )
        if dataset == "iSeg2017":
            atlases = {
                "brain_mask_unc_bcp_06m": root / "unc_bcp_06m" / "brain_mask.nii.gz",
                **atlases,
            }
        elif neonatal_dataset or neonatal_age or pediatric_diagnostic:
            atlases = {
                "brain_mask_fbn125_neonatal": _existing_atlas_mask(root, "fbn125_neonatal"),
                "brain_mask_unc_bcp_06m": _existing_atlas_mask(root, "unc_bcp_06m"),
                **atlases,
            }
            if _env_truthy(ALBERTS_BANK_ENV) or neonatal_dataset or pediatric_diagnostic:
                for atlas_id in ALBERTS_GOUSIAS_IDS:
                    atlases[f"brain_mask_{atlas_id}"] = _existing_atlas_mask(root, atlas_id)
            # Atlas dHCP neonatal (templates moyennés 0.5 mm, term-né) ajoutés
            # UNIQUEMENT pour MCRIB/MCRIB2 (perdants, term-born). MCRIB_WM (gagnant,
            # +0.002) est EXCLU pour rester strictement inchangé (décision opt-in).
            # ON par défaut ; désactivable (SAM_NEUROADAPT_MCRIB_DHCP=0) pour la
            # comparaison déterministe A/B dHCP-on vs dHCP-off.
            _mcrib_dhcp_on = os.environ.get(MCRIB_DHCP_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}
            if dataset in {"MCRIB", "MCRIB2"} and _mcrib_dhcp_on:
                for ga in (38, 40, 42, 44):
                    atlases[f"brain_mask_dhcp_neonatal_ga{ga:02d}"] = _existing_atlas_mask(
                        root, f"dhcp_neonatal_ga{ga:02d}"
                    )
        if month is None or month < 4:
            atlases["brain_mask_unc_neo"] = root / "unc_infant_neo" / "brain_mask.nii.gz"
        elif month < 18:
            atlases["brain_mask_unc_1yr"] = root / "unc_infant_1yr" / "brain_mask.nii.gz"
        else:
            atlases["brain_mask_unc_2yr"] = root / "unc_infant_2yr" / "brain_mask.nii.gz"
        return _existing(atlases)

    adult_keys = tuple(ADULT_EXTERNAL_ATLAS_FILES)
    if dataset == "IBSR18" and profile == "adult_t1" and not _env_truthy(ADULT_DIAGNOSTIC_TEMPLATE_ENV):
        # SRI24/Colin27/ICBM/LPBA are prepared for the diagnostic benchmark,
        # but they are not publication-frozen M1 guidance until the no-gold
        # benchmark proves they improve IBSR18 globally.
        adult_keys = ADULT_PUBLICATION_BASE_ATLAS_KEYS
    elif dataset == "LPBA40" and profile == "adult_t1" and not _env_truthy(LPBA40_PUBLIC_ATLAS_ENV):
        # LPBA40 average is a useful public-atlas diagnostic ceiling, but for a
        # strict LPBA40 evaluation we keep it out of M1 guidance/selection.
        adult_keys = tuple(name for name in adult_keys if name != "brain_mask_lpba40_average")
    return _existing(
        {
            name: root / ADULT_EXTERNAL_ATLAS_FILES[name][0] / ADULT_EXTERNAL_ATLAS_FILES[name][1]
            for name in adult_keys
        }
    )


def default_output_slug(dataset: str, profile_filter: str | None = None) -> str:
    slug = re.sub(r"[^a-z0-9]+", "", dataset.lower())
    if profile_filter:
        slug += "_" + re.sub(r"[^a-z0-9]+", "", profile_filter.lower())
    return slug
