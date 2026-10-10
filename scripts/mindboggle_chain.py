"""Chaîne Mindboggle-101 : notre pipeline (déterministe + garde) vs HD-BET sur 25 sujets.

Sous-ensemble STRATIFIÉ = 5 sujets par groupe (Extra-18, MMRR-21, NKI-RS-22, NKI-TRT-20,
OASIS-TRT-20). Benchmark adulte ÉQUITABLE (hors benchmarks HD-BET). Si on gagne → étendre
aux 101. Résumable (saute si CSV complet). Garde anti-inflation activée.

Sortie : outputs/validation_runs/mindboggle/SUMMARY.md + summary.json.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from scripts.repro_chain import run_ours, paired_vs_hdbet, _key

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/validation_runs/mindboggle"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "mindboggle_chain.log"
MAN = REPO / "datasets/manifests/Mindboggle101.csv"
HDBET_CSV = OUT / "mindboggle_hdbet_baseline.csv"
SEED = 20260517
N_PER_GROUP = 5


def log(m: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def stratified_subset() -> list[str]:
    d = pd.read_csv(MAN)
    subs = []
    for _, grp in d.groupby("group"):
        subs += sorted(grp["subject"].astype(str))[:N_PER_GROUP]
    return subs


def ensure_hdbet(subset: list[str]) -> pd.DataFrame:
    if HDBET_CSV.exists():
        log(f"  SKIP baseline HD-BET (déjà: {HDBET_CSV})")
        return pd.read_csv(HDBET_CSV)
    log("  GÉNÈRE baseline HD-BET Mindboggle (m2, CPU)…")
    man = pd.read_csv(MAN)
    man = man[man["subject"].astype(str).isin(subset)].copy()
    mini = pd.DataFrame({
        "subject": man["subject"].astype(str),
        "source_t1_path": man["input_path"].fillna(man["t1_path"]),
        "gold_mask_path": man["brain_mask_path"],
        "error": "",
    })
    mini_csv = OUT / "_mini_m1.csv"
    mini.to_csv(mini_csv, index=False)
    os.environ["HDBET_DEVICE"] = "cpu"
    os.environ.setdefault("HDBET_DISABLE_TTA", "1")
    from scripts.m2_external_sota_runner import run_m2_external_sota
    run_m2_external_sota(
        dataset="Mindboggle101", m1_csv=str(mini_csv), out_base=str(OUT / "sota_eval"),
        subjects=subset, force_rerun=False,
        require_all_tools=False, require_all_complete=False, require_clean_m1=False,
    )
    longp = OUT / "sota_eval" / "all_methods_long.csv"
    lon = pd.read_csv(longp)
    log("  SOTA Mindboggle (par méthode): " + lon.dropna(subset=["dice"]).groupby("method")["dice"].mean().round(4).to_dict().__str__())
    hd = lon[lon["method"].astype(str).str.upper().str.replace("_", "-") == "HD-BET"][["subject", "dice"]].dropna()
    hd.to_csv(HDBET_CSV, index=False)
    log(f"  baseline HD-BET: N={len(hd)} mean={hd['dice'].mean():.4f}")
    return hd


def main() -> int:
    log("=" * 70)
    log("CHAÎNE MINDBOGGLE-101 (sous-ensemble stratifié 25) vs HD-BET")
    subset = stratified_subset()
    log(f"sous-ensemble N={len(subset)}: {subset}")

    out_ours = OUT / f"ours_s{SEED}"
    ours = run_ours("Mindboggle101", SEED, out_ours, subset, len(subset))

    hd = ensure_hdbet(subset)
    r = paired_vs_hdbet(ours, hd)
    log(f"  RÉSULTAT: nous={r['ours_mean']:.4f} vs HD-BET={r['hdbet_mean']:.4f} "
        f"Δ={r['delta']:+.4f} p={r['wilcoxon_p']} ≥{r['n_ge']}/{r['n']} win={r['win']}")

    # détail par groupe
    ours2 = ours.copy()
    man = pd.read_csv(MAN)[["subject", "group"]]
    man["k"] = man["subject"].map(_key)
    ours2 = ours2.merge(man[["k", "group"]], on="k", how="left")
    hd2 = hd.copy(); hd2["k"] = hd2["subject"].map(_key)
    merged = ours2.merge(hd2[["k", "dice"]], on="k", suffixes=("", "_hd"))
    per_group = {}
    for g, gg in merged.groupby("group"):
        per_group[g] = {"n": int(len(gg)), "ours": round(float(gg["dice"].mean()), 4),
                        "hdbet": round(float(gg["dice_hd"].mean()), 4)}
        log(f"    [{g}] nous={gg['dice'].mean():.4f} HD-BET={gg['dice_hd'].mean():.4f} (N={len(gg)})")

    summary = {"subset": subset, "result": r, "per_group": per_group}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Mindboggle-101 (stratifié 25) — nous (déterministe+garde) vs HD-BET", "",
             f"N={r['n']} | nous **{r['ours_mean']:.4f}** vs HD-BET **{r['hdbet_mean']:.4f}** | "
             f"Δ={r['delta']:+.4f} | Wilcoxon p={r['wilcoxon_p']} | ≥SOTA {r['n_ge']}/{r['n']} | "
             f"{'✅ WIN' if r['win'] else '❌'}", "", "## Par groupe", "| groupe | N | nous | HD-BET |",
             "|---|---|---|---|"]
    for g, v in per_group.items():
        lines.append(f"| {g} | {v['n']} | {v['ours']} | {v['hdbet']} |")
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    log("TERMINÉ. Voir SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
