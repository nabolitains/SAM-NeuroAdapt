#!/usr/bin/env python3
"""Terminal launcher for the scalable pediatric M1 iSeg-style runner.

It executes only the notebook definition blocks required by the original
M1-iSeg2017 cell, then calls scripts.m1_pediatric_iseg_style_runner.  This
allows one terminal tab per dataset without running the whole notebook.
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


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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
    module_name = "__m1_pediatric_notebook_defs__"
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

    # Preprocessing definitions only; skip the example execution at the end.
    preproc = _cell_with(cells, "def run_preprocessing_pipeline")
    exec(_prefix_until(preproc, [r"^t1_path\s*="]), g)

    # Atlas registration definitions only; skip the example execution at the end.
    registration = _cell_with(cells, "def advanced_atlas_registration")
    exec(_prefix_until(registration, [r"^atlas_dir\s*=", r"^derivatives_dir\s*="]), g)

    # SAM2 environment and extractor definitions.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset name: BOBs, MCRIB, MCRIB_WM, MCRIB2, iSeg2017, etc.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out-base", default=None)
    parser.add_argument("--profile-filter", default=None)
    parser.add_argument("--n-subjects", type=int, default=None)
    parser.add_argument("--subjects", default=None, help="Comma/space separated subject ids.")
    parser.add_argument("--target-mean-dice", type=float, default=None)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--allow-missing-atlas", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Load notebook definitions and records, but do not run SAM2.")
    parser.add_argument("--no-force-extraction", action="store_true",
                        help="Use cached SAM2 masks when available (skip re-extraction).")
    parser.add_argument("--no-idempotent-refinement", action="store_true",
                        help="Disable the backup/restore idempotence of upstream refinements "
                             "(restores the legacy accumulating behaviour).")
    args = parser.parse_args(argv)

    g = _prepare_notebook_globals()
    from scripts.m1_pediatric_iseg_style_runner import _load_records, run_m1_pediatric_iseg_style_gold

    if args.dry_run:
        records = _load_records(
            args.dataset,
            manifest=args.manifest,
            profile_filter=args.profile_filter,
            subjects=_parse_subjects(args.subjects),
            n_subjects=args.n_subjects,
            require_atlas=not args.allow_missing_atlas,
        )
        print(f"dry-run ok: dataset={args.dataset} records={len(records)}")
        for rec in records[:5]:
            print(f"  {rec['subject']} session={rec['session']} atlases={','.join(rec['atlas_paths'].keys())}")
        return 0

    run_m1_pediatric_iseg_style_gold(
        g,
        dataset=args.dataset,
        manifest=args.manifest,
        out_base=args.out_base,
        profile_filter=args.profile_filter,
        n_subjects=args.n_subjects,
        subjects=_parse_subjects(args.subjects),
        require_atlas=not args.allow_missing_atlas,
        force_clean_outputs=not args.no_clean,
        force_extraction=not args.no_force_extraction,
        refinement_idempotent_mode=not args.no_idempotent_refinement,
        display_tables=False,
        target_mean_dice=args.target_mean_dice,
        reproducibility_seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
