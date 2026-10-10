#!/usr/bin/env python3
"""Terminal launcher for the unified iSeg-style M1 pipeline.

Loads only the notebook DEFINITIONS required by the iSeg-style pipeline
(preprocessing, atlas registration, SAM2 extractor) then runs
scripts.iseg_style_pipeline.run_iseg_style_pipeline for any dataset:
iSeg2017, BOBs, MCRIB, MCRIB_WM.

Usage:
    python -m scripts.run_m1_isegstyle --dataset iSeg2017 --n-subjects 1
    python -m scripts.run_m1_isegstyle --dataset BOBs --n-subjects 3
    python -m scripts.run_m1_isegstyle --dataset MCRIB
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import types
from pathlib import Path

NOTEBOOK = Path("sam_neuroadapt_pipeline.ipynb")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_notebook_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(cell.get("source", [])) for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]


def _cell_with(cells: list[str], needle: str) -> str:
    for cell in cells:
        if needle in cell:
            return cell
    raise RuntimeError(f"Could not find notebook cell containing {needle!r}")


def _prefix_until(cell: str, stop_patterns: list[str]) -> str:
    lines = cell.splitlines()
    out = []
    for line in lines:
        if any(re.search(pattern, line) for pattern in stop_patterns):
            break
        out.append(line)
    return "\n".join(out) + "\n"


def _prepare_notebook_globals() -> dict:
    cells = _load_notebook_cells()
    module_name = "__m1_isegstyle_notebook_defs__"
    module = types.ModuleType(module_name)
    module.__file__ = str(NOTEBOOK)
    sys.modules[module_name] = module
    g: dict = module.__dict__
    bootstrap = r'''
import os, sys, time, json, warnings, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
warnings.filterwarnings("ignore")
BASE_DIR = Path("./sam_neuroadapt_v2_real_atlases")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ATLASES_DIR = BASE_DIR / "atlases"
RESULTS_DIR = BASE_DIR / "results"
for _p in [DATA_DIR, MODELS_DIR, ATLASES_DIR, RESULTS_DIR]:
    _p.mkdir(parents=True, exist_ok=True)
SUBJECT_RAW = "A00043299"
SUBJECT_ID = f"{SUBJECT_RAW}_NFB3_T1w"
SUBJECT_BIDS_ID = SUBJECT_ID
GS_BASE = Path("datasets/NFBS/NFBS_Dataset/NFBS_Dataset")
T1_RAW_PATH = GS_BASE / SUBJECT_RAW / f"sub-{SUBJECT_RAW}_ses-NFB3_T1w.nii.gz"
GS_MASK_PATH = GS_BASE / SUBJECT_RAW / f"sub-{SUBJECT_RAW}_ses-NFB3_T1w_brainmask.nii.gz"
DERIVATIVES_DIR = Path("outputs/runtime_bootstrap") / SUBJECT_RAW
ANAT_DIR = DERIVATIVES_DIR / "anat"
ANAT_DIR.mkdir(parents=True, exist_ok=True)
'''
    exec(bootstrap, g)

    preproc = _cell_with(cells, "def run_preprocessing_pipeline")
    exec(_prefix_until(preproc, [r"^t1_path\s*="]), g)

    registration = _cell_with(cells, "def advanced_atlas_registration")
    exec(_prefix_until(registration, [r"^atlas_dir\s*=", r"^derivatives_dir\s*="]), g)

    exec(_cell_with(cells, "sam2_checkpoint_path"), g)
    exec(_cell_with(cells, "class SAM2BrainExtractor"), g)

    from scripts.m1_external_adaptive_runner import compute_metrics
    g["compute_metrics"] = compute_metrics
    return g


def _parse_subjects(value: str | None):
    if not value:
        return None
    items = [chunk.strip() for chunk in re.split(r"[,;\s]+", value) if chunk.strip()]
    return items or None


def _spec_for(dataset: str, n_subjects: int | None, subjects: list[str] | None,
              *, use_loo: bool = False):
    from scripts.iseg_style_pipeline import (
        IsegStyleDatasetSpec,
        ISEG_DEFAULT_PRIORS, BOBS_DEFAULT_PRIORS, MCRIB_DEFAULT_PRIORS, MCRIB_WM_DEFAULT_PRIORS,
        iseg_records_resolver, bids_gold_records_resolver,
        fixed_atlas_resolver, external_atlas_by_age_resolver, loo_atlas_resolver,
    )

    if dataset == "iSeg2017":
        atlas_dir = Path("datasets/iSeg2017/atlas/UNC_BCP_06M")
        return IsegStyleDatasetSpec(
            dataset="iSeg2017",
            manifest_csv=Path("datasets/iSeg2017/training_manifest.csv"),
            out_base=Path("outputs/iseg2017_m1_eval"),
            selected_records_fn=lambda: iseg_records_resolver(
                manifest=Path("datasets/iSeg2017/training_manifest.csv"),
                split_filter="train_labeled",
                subjects=subjects, n_subjects=n_subjects,
            ),
            atlas_resolver=fixed_atlas_resolver(
                t1=atlas_dir / "tpl-UNCBCP06M_T1w.nii.gz",
                brain_mask=atlas_dir / "tpl-UNCBCP06M_desc-brain_mask.nii.gz",
            ),
            volume_priors=ISEG_DEFAULT_PRIORS,
        )

    out_slug_map = {"BOBs": "bobs", "MCRIB": "mcrib", "MCRIB_WM": "mcribwm"}
    out_slug = out_slug_map.get(dataset, dataset.lower())
    priors_by_dataset = {
        "BOBs": BOBS_DEFAULT_PRIORS,
        "MCRIB": MCRIB_DEFAULT_PRIORS,
        "MCRIB_WM": MCRIB_WM_DEFAULT_PRIORS,
    }
    if dataset not in priors_by_dataset:
        raise ValueError(f"Dataset non supporte: {dataset}. Choisir parmi: iSeg2017, BOBs, MCRIB, MCRIB_WM.")

    dataset_tags = {"BOBs": "BOBS", "MCRIB": "MCRIB", "MCRIB_WM": "MCRIBWM"}
    if use_loo:
        atlas_resolver_fn = loo_atlas_resolver(
            atlas_root=Path(f"datasets/{dataset}/atlas_loo"),
            dataset_tag=dataset_tags[dataset],
        )
        out_base = Path(f"outputs/{out_slug}_m1_isegstyle_loo_eval")
    else:
        atlas_resolver_fn = external_atlas_by_age_resolver()
        out_base = Path(f"outputs/{out_slug}_m1_isegstyle_eval")

    return IsegStyleDatasetSpec(
        dataset=dataset,
        manifest_csv=Path(f"datasets/manifests/{dataset}.csv"),
        out_base=out_base,
        selected_records_fn=lambda: bids_gold_records_resolver(
            manifest=Path(f"datasets/manifests/{dataset}.csv"),
            subjects=subjects, n_subjects=n_subjects,
        ),
        atlas_resolver=atlas_resolver_fn,
        volume_priors=priors_by_dataset[dataset],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="iSeg2017 | BOBs | MCRIB | MCRIB_WM")
    parser.add_argument("--n-subjects", type=int, default=None)
    parser.add_argument("--subjects", default=None, help="Comma/space separated subject ids")
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--use-loo", action="store_true",
                        help="Use per-subject consolidated LOO atlases under datasets/<dataset>/atlas_loo/.")
    args = parser.parse_args(argv)

    from scripts.iseg_style_pipeline import run_iseg_style_pipeline

    print(f"[run_m1_isegstyle] Initialisation environnement notebook...")
    g = _prepare_notebook_globals()
    print(f"[run_m1_isegstyle] OK: symboles charges = {sorted(k for k in g if not k.startswith('_'))[:20]}...")

    spec = _spec_for(args.dataset, args.n_subjects, _parse_subjects(args.subjects),
                     use_loo=args.use_loo)
    mode = "LOO consolidated" if args.use_loo else "external age-mapped"
    print(f"[run_m1_isegstyle] Dataset={spec.dataset} | mode atlas={mode} | manifest={spec.manifest_csv} | out_base={spec.out_base}")

    result = run_iseg_style_pipeline(
        g, spec,
        reproducibility_seed=args.seed,
    )

    print(f"\n[run_m1_isegstyle] Termine. mean_dice={result['mean_dice']:.4f} sur {len(result['df_ok'])} sujets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
