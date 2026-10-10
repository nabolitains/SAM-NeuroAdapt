"""Chaîne automatique IBSR18 → NFBS → multi-seeds (reproductibilité vs HD-BET).

Demande utilisateur (2026-06-24) : relancer IBSR18 frais+garde, enchaîner NFBS en
déterministe, et SI les deux gagnent → relancer chacun sur plusieurs seeds pour
prouver la reproductibilité (mean±std + Wilcoxon apparié vs HD-BET).

Garanties :
- garde anti-inflation activée (SAM_NEUROADAPT_INFLATION_GUARD=1) sur tous les runs ;
- RÉSUMABLE : chaque run dont le CSV final est complet est SAUTÉ (survit à une
  extinction PC — il suffit de relancer ce script) ;
- NFBS = sous-ensemble FIXE de 20 sujets (déterministe, pré-enregistré) car le
  dataset complet (125) rendrait le multi-seed irréaliste ;
- baseline SOTA : IBSR18 réutilise la baseline HD-BET figée existante ; NFBS la
  génère une fois (mêmes sujets à chaque seed) via le runner SOTA dédié ;
- porte « si gagne » : on ne lance les seeds 2/3 QUE si seed1 bat HD-BET sur LES DEUX.

Sortie : outputs/validation_runs/repro/SUMMARY.md + summary.json (+ log master).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
REPRO = REPO / "outputs/validation_runs/repro"
REPRO.mkdir(parents=True, exist_ok=True)
MASTER_LOG = REPRO / "repro_chain.log"
SEEDS = [20260517, 20260518, 20260519]
NFBS_N = 20

IBSR_HDBET = REPO / "outputs/ibsr18_sota_eval_v4_external_templates_frozen/HD-BET_results.csv"
NFBS_HDBET = REPRO / "nfbs_hdbet_baseline.csv"  # généré une fois


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(MASTER_LOG, "a") as f:
        f.write(line + "\n")


def _key(s: str) -> str:
    return str(s).replace("_T1w", "").replace("-", "_").strip()


def nfbs_subset() -> list[str]:
    df = pd.read_csv(REPO / "datasets/manifests/NFBS.csv")
    df = df[df["has_gold"] == True]  # noqa: E712
    subs = sorted(df["subject"].astype(str).unique())[:NFBS_N]
    return subs


def results_csv(dataset: str, out_base: Path) -> Path:
    return out_base / f"sam_neuroadapt_{dataset.lower()}_adultt1_results.csv"


def run_complete(dataset: str, out_base: Path, expected_n: int) -> bool:
    csv = results_csv(dataset, out_base)
    if not csv.exists():
        return False
    try:
        d = pd.read_csv(csv).dropna(subset=["dice"])
    except Exception:
        return False
    return len(d) >= expected_n


def run_ours(dataset: str, seed: int, out_base: Path, subjects: list[str] | None, expected_n: int) -> pd.DataFrame:
    """Lance run_m1_adult (garde ON) si pas déjà complet. Retourne le DataFrame résultat."""
    if run_complete(dataset, out_base, expected_n):
        log(f"  SKIP {dataset} seed={seed} (déjà complet: {results_csv(dataset, out_base)})")
    else:
        cmd = [
            sys.executable, "-m", "scripts.run_m1_adult",
            "--dataset", dataset,
            "--out-base", str(out_base),
            "--seed", str(seed),
        ]
        if subjects:
            cmd += ["--subjects", ",".join(subjects)]
        env = dict(os.environ)
        env["SAM_NEUROADAPT_INFLATION_GUARD"] = "1"
        log(f"  RUN  {dataset} seed={seed} -> {out_base}  (N attendu={expected_n})")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(REPO), env=env)
        log(f"  done {dataset} seed={seed} rc={r.returncode} en {(time.time()-t0)/60:.1f} min")
        if r.returncode != 0:
            raise RuntimeError(f"run_m1_adult échec {dataset} seed={seed}")
    d = pd.read_csv(results_csv(dataset, out_base)).dropna(subset=["dice"])
    d["k"] = d["subject"].map(_key)
    return d


def ensure_nfbs_hdbet(seed1_csv: Path, subjects: list[str]) -> pd.DataFrame:
    """Génère (une fois) la baseline HD-BET NFBS sur le sous-ensemble, via le runner SOTA."""
    if NFBS_HDBET.exists():
        log(f"  SKIP baseline HD-BET NFBS (déjà: {NFBS_HDBET})")
        return pd.read_csv(NFBS_HDBET)
    log("  GÉNÈRE baseline HD-BET NFBS (runner SOTA, sous-ensemble 20)…")
    from scripts.m2_external_sota_runner import run_m2_external_sota
    out = REPRO / "nfbs_sota_eval"
    run_m2_external_sota(
        dataset="NFBS",
        m1_csv=str(seed1_csv),
        out_base=str(out),
        subjects=subjects,
        force_rerun=False,
        require_all_tools=False,
        require_all_complete=False,
        require_clean_m1=False,
    )
    longp = out / "all_methods_long.csv"
    if not longp.exists():
        raise RuntimeError("baseline SOTA NFBS: all_methods_long.csv introuvable")
    lon = pd.read_csv(longp)
    hd = lon[lon["method"].astype(str).str.upper().str.replace("_", "-") == "HD-BET"][["subject", "dice"]].dropna()
    if hd.empty:
        raise RuntimeError("baseline SOTA NFBS: aucune ligne HD-BET")
    hd = hd.rename(columns={"dice": "dice"})
    hd.to_csv(NFBS_HDBET, index=False)
    log(f"  baseline HD-BET NFBS sauvée: {NFBS_HDBET} (N={len(hd)}, mean={hd['dice'].mean():.4f})")
    return hd


def paired_vs_hdbet(ours: pd.DataFrame, hd: pd.DataFrame) -> dict:
    hd = hd.copy()
    hd["k"] = hd["subject"].map(_key)
    m = ours.merge(hd[["k", "dice"]], on="k", how="inner", suffixes=("", "_hd"))
    a = m["dice"].values
    b = m["dice_hd"].values
    res = {
        "n": int(len(m)),
        "ours_mean": float(np.mean(a)) if len(a) else float("nan"),
        "hdbet_mean": float(np.mean(b)) if len(b) else float("nan"),
        "delta": float(np.mean(a) - np.mean(b)) if len(a) else float("nan"),
        "n_ge": int(np.sum(a >= b)),
        "win": bool(len(a) and np.mean(a) > np.mean(b)),
    }
    if len(a) > 5 and np.any(a - b != 0):
        try:
            res["wilcoxon_p"] = float(wilcoxon(a, b).pvalue)
        except Exception:
            res["wilcoxon_p"] = None
    else:
        res["wilcoxon_p"] = None
    return res


def main() -> int:
    log("=" * 70)
    log("CHAÎNE REPRO IBSR18 → NFBS → multi-seeds (garde anti-inflation ON)")
    subset = nfbs_subset()
    log(f"NFBS sous-ensemble fixe (N={len(subset)}): {subset[0]}..{subset[-1]}")

    datasets = {
        "IBSR18": {"subjects": None, "n": 20, "hdbet": "ibsr"},
        "NFBS": {"subjects": subset, "n": len(subset), "hdbet": "nfbs"},
    }

    summary: dict = {"seeds": SEEDS, "nfbs_subset": subset, "per_seed": {}, "reproducibility": {}}

    # ---- SEED 1 (base) : les deux datasets + baseline + porte "si gagne" ----
    seed1 = SEEDS[0]
    seed1_results = {}
    for ds, cfg in datasets.items():
        out = REPRO / f"{ds.lower()}_s{seed1}"
        seed1_results[ds] = run_ours(ds, seed1, out, cfg["subjects"], cfg["n"])

    # baselines HD-BET
    ibsr_hd = pd.read_csv(IBSR_HDBET)[["subject", "dice"]].dropna()
    nfbs_hd = ensure_nfbs_hdbet(results_csv("NFBS", REPRO / f"nfbs_s{seed1}"), subset)
    baselines = {"IBSR18": ibsr_hd, "NFBS": nfbs_hd}

    summary["per_seed"][str(seed1)] = {}
    all_win = True
    for ds in datasets:
        r = paired_vs_hdbet(seed1_results[ds], baselines[ds])
        summary["per_seed"][str(seed1)][ds] = r
        log(f"  [seed {seed1}] {ds}: ours={r['ours_mean']:.4f} vs HD-BET={r['hdbet_mean']:.4f} "
            f"Δ={r['delta']:+.4f} p={r['wilcoxon_p']} ≥{r['n_ge']}/{r['n']} win={r['win']}")
        all_win = all_win and r["win"]

    _write_summary(summary)
    log(f"PORTE 'si gagne' (seed {seed1}) : IBSR18 & NFBS battent HD-BET ? -> {all_win}")

    # ---- SEEDS 2/3 seulement si les DEUX gagnent en seed1 ----
    if all_win:
        for seed in SEEDS[1:]:
            for ds, cfg in datasets.items():
                out = REPRO / f"{ds.lower()}_s{seed}"
                d = run_ours(ds, seed, out, cfg["subjects"], cfg["n"])
                r = paired_vs_hdbet(d, baselines[ds])
                summary["per_seed"].setdefault(str(seed), {})[ds] = r
                log(f"  [seed {seed}] {ds}: ours={r['ours_mean']:.4f} Δ={r['delta']:+.4f} "
                    f"p={r['wilcoxon_p']} win={r['win']}")
                _write_summary(summary)

        # ---- reproductibilité : mean±std across seeds + pooled Wilcoxon ----
        for ds, cfg in datasets.items():
            per_subj = None
            seed_means = []
            for seed in SEEDS:
                d = pd.read_csv(results_csv(ds, REPRO / f"{ds.lower()}_s{seed}")).dropna(subset=["dice"])
                d["k"] = d["subject"].map(_key)
                seed_means.append(float(d["dice"].mean()))
                s = d.set_index("k")["dice"].rename(f"s{seed}")
                per_subj = s.to_frame() if per_subj is None else per_subj.join(s, how="inner")
            pooled = per_subj.mean(axis=1)  # dice moyen par sujet sur les seeds
            hd = baselines[ds].copy()
            hd["k"] = hd["subject"].map(_key)
            hd = hd.set_index("k")["dice"]
            common = [k for k in pooled.index if k in hd.index]
            a = pooled.loc[common].values
            b = hd.loc[common].values
            rep = {
                "seed_means": seed_means,
                "mean_of_seed_means": float(np.mean(seed_means)),
                "std_of_seed_means": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
                "max_swing": float(np.max(seed_means) - np.min(seed_means)),
                "pooled_mean": float(np.mean(a)),
                "hdbet_mean": float(np.mean(b)),
                "pooled_delta": float(np.mean(a) - np.mean(b)),
                "pooled_n_ge": int(np.sum(a >= b)),
                "pooled_wilcoxon_p": (float(wilcoxon(a, b).pvalue) if np.any(a - b != 0) else None),
            }
            summary["reproducibility"][ds] = rep
            log(f"  [REPRO {ds}] seeds={['%.4f'%x for x in seed_means]} "
                f"mean±std={rep['mean_of_seed_means']:.4f}±{rep['std_of_seed_means']:.4f} "
                f"swing={rep['max_swing']:.4f} | pooled vs HD-BET Δ={rep['pooled_delta']:+.4f} "
                f"p={rep['pooled_wilcoxon_p']}")
            _write_summary(summary)
    else:
        log("Porte fermée : au moins un dataset ne bat pas HD-BET en seed1 → "
            "on N'ENCHAÎNE PAS les seeds 2/3 (économie de calcul). Verdict = seed1 ci-dessus.")

    _write_summary(summary)
    log("CHAÎNE TERMINÉE. Voir SUMMARY.md")
    return 0


def _write_summary(summary: dict) -> None:
    (REPRO / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Chaîne repro IBSR18 + NFBS vs HD-BET (garde anti-inflation)", ""]
    lines.append(f"Seeds: {summary['seeds']} | NFBS sous-ensemble N={len(summary['nfbs_subset'])} (fixe)")
    lines.append("")
    lines.append("## Par seed")
    lines.append("| seed | dataset | nous | HD-BET | Δ | Wilcoxon p | ≥SOTA | gagne |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for seed, dss in summary["per_seed"].items():
        for ds, r in dss.items():
            p = f"{r['wilcoxon_p']:.4f}" if r.get("wilcoxon_p") is not None else "—"
            lines.append(f"| {seed} | {ds} | {r['ours_mean']:.4f} | {r['hdbet_mean']:.4f} | "
                         f"{r['delta']:+.4f} | {p} | {r['n_ge']}/{r['n']} | {'✅' if r['win'] else '❌'} |")
    if summary.get("reproducibility"):
        lines += ["", "## Reproductibilité (across seeds)"]
        lines.append("| dataset | means par seed | mean±std | swing | pooled Δ vs HD-BET | pooled p |")
        lines.append("|---|---|---|---|---|---|")
        for ds, r in summary["reproducibility"].items():
            sm = ", ".join(f"{x:.4f}" for x in r["seed_means"])
            pp = f"{r['pooled_wilcoxon_p']:.4f}" if r.get("pooled_wilcoxon_p") is not None else "—"
            lines.append(f"| {ds} | {sm} | {r['mean_of_seed_means']:.4f}±{r['std_of_seed_means']:.4f} | "
                         f"{r['max_swing']:.4f} | {r['pooled_delta']:+.4f} | {pp} |")
    (REPRO / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
