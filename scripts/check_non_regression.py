"""Filet de non-régression — datasets déjà au-dessus du SOTA.

Garde-fou OBLIGATOIRE avant tout merge d'un changement de cœur partagé (fusion
Spatial STAPLE/JLF, registration MIND/LNCC, contrôle volumétrique unifié…). Les
runs idempotents du pipeline sont déterministes (écart 0.00e+00 documenté dans
``ROBUSTNESS_REPORT.md``) : un changement de cœur ne DOIT pas bouger le Dice
par-sujet des datasets gagnants.

Usage :

    # Comparaison pure de deux CSV de résultats (testable sans GPU) :
    python -m scripts.check_non_regression --compare BASELINE.csv CURRENT.csv

    # Vérifier un dataset gagnant vs sa baseline figée :
    python -m scripts.check_non_regression --check iSeg2017 CURRENT.csv

    # Auto-test du cœur de comparaison :
    python -m scripts.check_non_regression --selftest

Pour produire CURRENT.csv, rejouer le runner en mode idempotent avec le MÊME
``out_base`` que la baseline, p.ex. ::

    run_m1_pediatric_iseg_style_gold(
        g, dataset="iSeg2017", ...,
        refinement_idempotent_mode=True, force_extraction=True,
    )

puis pointer ``--check`` sur le ``sam_neuroadapt_<slug>_results.csv`` produit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Tolérance stricte : absorbe le formatage CSV (dice à précision finie) tout en
# attrapant toute vraie régression (ex. 0.964 → 0.801).
DEFAULT_TOL = 1e-4

# Datasets gagnants → CSV baseline figé (par-sujet, reproductible).
WINNING_BASELINES: dict[str, Path] = {
    "iSeg2017": Path("outputs/iseg2017_m1_pediatric_isegstyle_eval/sam_neuroadapt_iseg2017_results_idempotent_reproducible.csv"),
    "BOBs":     Path("outputs/bobs_m1_pediatric_isegstyle_eval/sam_neuroadapt_bobs_results_idempotent_reproducible.csv"),
    "MCRIB_WM": Path("outputs/mcribwm_m1_pediatric_isegstyle_eval/sam_neuroadapt_mcribwm_results_idempotent_reproducible.csv"),
    # IBSR18 (adulte) : harnais d'éval distinct ; baseline sous
    # outputs/ibsr18_sota_eval_v4_external_templates_frozen/ — à brancher quand le
    # CSV SAM par-sujet figé y est disponible.
}


class RegressionError(RuntimeError):
    """Levée quand une régression (ou un changement de couverture) est détectée."""


def _load_dice(csv_path: Path) -> pd.Series:
    """Charge une série dice indexée par subject depuis un CSV de résultats."""
    df = pd.read_csv(csv_path)
    for col in ("subject", "dice"):
        if col not in df.columns:
            raise ValueError(f"{csv_path} : colonne '{col}' manquante")
    s = df.set_index("subject")["dice"].astype(float)
    if s.index.duplicated().any():
        dups = s.index[s.index.duplicated()].unique().tolist()
        raise ValueError(f"{csv_path} : sujets dupliqués {dups}")
    return s


def compare_dice(
    baseline: pd.Series,
    current: pd.Series,
    tol: float = DEFAULT_TOL,
) -> dict:
    """Compare deux séries dice (indexées subject). Retourne un rapport.

    Échec si : un sujet baseline manque dans current (couverture réduite), un
    sujet supplémentaire apparaît (couverture modifiée), ou |Δdice| > tol.
    """
    missing = sorted(set(baseline.index) - set(current.index))
    extra = sorted(set(current.index) - set(baseline.index))
    common = sorted(set(baseline.index) & set(current.index))

    deltas = {sid: float(current[sid] - baseline[sid]) for sid in common}
    regressions = {sid: d for sid, d in deltas.items() if d < -tol}
    improvements = {sid: d for sid, d in deltas.items() if d > tol}
    max_abs = max((abs(d) for d in deltas.values()), default=0.0)

    ok = not missing and not extra and not regressions and max_abs <= tol
    return {
        "ok": ok,
        "n_common": len(common),
        "missing": missing,
        "extra": extra,
        "max_abs_delta": max_abs,
        "regressions": dict(sorted(regressions.items(), key=lambda kv: kv[1])),
        "improvements": improvements,
        "mean_delta": (sum(deltas.values()) / len(deltas)) if deltas else 0.0,
    }


def _print_report(name: str, rep: dict, tol: float) -> None:
    status = "OK ✓" if rep["ok"] else "ÉCHEC ✗"
    print(f"[{name}] {status} | n={rep['n_common']} | "
          f"max|Δ|={rep['max_abs_delta']:.2e} | moy(Δ)={rep['mean_delta']:+.2e} (tol={tol:.0e})")
    if rep["missing"]:
        print(f"  ✗ sujets MANQUANTS dans current ({len(rep['missing'])}): {rep['missing'][:8]}")
    if rep["extra"]:
        print(f"  ✗ sujets EN TROP dans current ({len(rep['extra'])}): {rep['extra'][:8]}")
    for sid, d in rep["regressions"].items():
        print(f"  ✗ RÉGRESSION {sid}: Δdice={d:+.4f}")
    if rep["improvements"] and not rep["regressions"]:
        # Une amélioration sur un dataset gagnant est suspecte (non-déterminisme).
        print(f"  ⚠ {len(rep['improvements'])} sujet(s) AMÉLIORÉ(s) — inattendu en mode idempotent")


def check_dataset(dataset: str, current_csv: Path, tol: float = DEFAULT_TOL) -> dict:
    """Vérifie un dataset gagnant vs sa baseline figée."""
    if dataset not in WINNING_BASELINES:
        raise KeyError(f"Dataset '{dataset}' non enregistré. Connus: {sorted(WINNING_BASELINES)}")
    baseline_csv = WINNING_BASELINES[dataset]
    if not baseline_csv.exists():
        raise FileNotFoundError(f"Baseline introuvable: {baseline_csv}")
    rep = compare_dice(_load_dice(baseline_csv), _load_dice(Path(current_csv)), tol)
    _print_report(dataset, rep, tol)
    return rep


def _selftest() -> None:
    """Auto-test du cœur de comparaison (sans GPU, synthétique)."""
    base = pd.Series({"s1": 0.95, "s2": 0.90, "s3": 0.88})

    # Identique → OK.
    rep = compare_dice(base, base.copy())
    assert rep["ok"], rep

    # Bruit de formatage sous tolérance → OK.
    rep = compare_dice(base, base + 5e-5)
    assert rep["ok"], rep

    # Régression franche → ÉCHEC.
    reg = base.copy(); reg["s1"] = 0.80
    rep = compare_dice(base, reg)
    assert not rep["ok"] and "s1" in rep["regressions"], rep

    # Couverture réduite → ÉCHEC.
    rep = compare_dice(base, base.drop("s3"))
    assert not rep["ok"] and "s3" in rep["missing"], rep

    # Sujet en trop → ÉCHEC.
    extra = pd.concat([base, pd.Series({"s4": 0.9})])
    rep = compare_dice(base, extra)
    assert not rep["ok"] and "s4" in rep["extra"], rep

    # Amélioration au-dessus de tol → BLOQUANTE en idempotent strict : sur un
    # dataset gagnant, toute variation (même positive) trahit un effet non voulu
    # ou du non-déterminisme. On la signale ET on échoue.
    imp = base.copy(); imp["s2"] = 0.93
    rep = compare_dice(base, imp)
    assert not rep["ok"] and "s2" in rep["improvements"] and not rep["regressions"], rep

    print("check_non_regression: self-tests OK")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--compare", nargs=2, metavar=("BASELINE", "CURRENT"), help="Comparer deux CSV")
    g.add_argument("--check", nargs=2, metavar=("DATASET", "CURRENT"), help="Vérifier un dataset gagnant")
    g.add_argument("--selftest", action="store_true", help="Auto-test du cœur de comparaison")
    p.add_argument("--tol", type=float, default=DEFAULT_TOL)
    args = p.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0
    if args.compare:
        baseline_csv, current_csv = map(Path, args.compare)
        rep = compare_dice(_load_dice(baseline_csv), _load_dice(current_csv), args.tol)
        _print_report(f"{baseline_csv.name} vs {current_csv.name}", rep, args.tol)
        return 0 if rep["ok"] else 1
    # --check
    dataset, current_csv = args.check
    rep = check_dataset(dataset, Path(current_csv), args.tol)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
