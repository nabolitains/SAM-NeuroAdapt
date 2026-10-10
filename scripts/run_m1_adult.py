#!/usr/bin/env python3
"""Launcher CLI pour le runner ADULTE externe-adaptatif (IBSR18/LPBA40/NFBS…).

Équivalent de run_m1_pediatric_isegstyle pour les datasets adultes : construit les
symboles notebook requis (réutilise _prepare_notebook_globals) puis appelle
run_m1_external_adaptive_gold. Permet de lancer un dataset adulte hors notebook,
et d'expérimenter la fusion STAPLE portée du pédiatrique (#4.1) via
SAM_NEUROADAPT_ATLAS_CONSENSUS_METHOD=staple.

Exemple :
    SAM_NEUROADAPT_ATLAS_CONSENSUS_METHOD=staple \
    python -m scripts.run_m1_adult --dataset LPBA40 \
        --out-base outputs/validation_runs/lpba40_staple --seed 20260517
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="LPBA40 | IBSR18 | NFBS | MRBrainS13 | SynthStripT1")
    p.add_argument("--manifest", default=None)
    p.add_argument("--out-base", default=None)
    p.add_argument("--n-subjects", type=int, default=None)
    p.add_argument("--subjects", default=None)
    p.add_argument("--seed", type=int, default=20260517)
    p.add_argument("--profile-filter", default="adult_t1")
    p.add_argument("--deterministic-threads", action="store_true",
                   help="Mono-thread ITK/OMP (déterminisme total mais lent). Défaut: off "
                        "(rapide, déterminisme via ants.config._random_seed).")
    p.add_argument("--no-force-extraction", action="store_true",
                   help="Réutilise les masques SAM2 cachés (reprise après interruption).")
    args = p.parse_args(argv)

    # Réutilise la construction des symboles notebook du launcher pédiatrique
    # (run_preprocessing_pipeline, advanced_atlas_registration, SegmentationParams,
    # SAM2BrainExtractor) — strictement en lecture, n'altère pas le runner pédiatrique.
    from scripts.run_m1_pediatric_isegstyle import _prepare_notebook_globals
    from scripts.m1_external_adaptive_runner import run_m1_external_adaptive_gold

    g = _prepare_notebook_globals()
    print(f"[run_m1_adult] dataset={args.dataset} profile={args.profile_filter} "
          f"out_base={args.out_base} seed={args.seed} det_threads={args.deterministic_threads}")

    run_m1_external_adaptive_gold(
        g,
        dataset=args.dataset,
        profile_filter=args.profile_filter,
        manifest=args.manifest,
        out_base=args.out_base,
        n_subjects=args.n_subjects,
        subjects=[s.strip() for s in args.subjects.replace(",", " ").split()] if args.subjects else None,
        force_clean_outputs=False,
        force_atlas_registration=True,
        force_extraction=not args.no_force_extraction,
        reproducibility_seed=args.seed,
        deterministic_threads=args.deterministic_threads,
        display_tables=False,
        target_mean_dice=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
