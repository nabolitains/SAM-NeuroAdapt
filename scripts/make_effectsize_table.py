"""Quatre compléments statistiques empruntés au protocole de LifespanStrip.

LifespanStrip (Nature Biomed. Eng. 2025) rapporte, pour chaque comparaison : moyenne ± s.d.,
IC 95 %, et le **d de Cohen** comme mesure de taille d'effet. Il insiste par ailleurs sur la
*borne inférieure* de performance plutôt que sur la seule moyenne, et stratifie ses résultats
par constructeur d'IRM. Aucun de ces quatre éléments n'est coûteux à calculer chez nous, et
tous rendent nos chiffres directement comparables aux siens, qui sont la référence du domaine.

Ce script produit :
  1. le **d de Cohen apparié** (d_z = moyenne des différences / écart-type des différences),
     à côté de notre r = |Z|/sqrt(N) issu du Wilcoxon ;
  2. l'**IC 95 % bootstrap** de l'écart moyen (10 000 rééchantillonnages) ;
  3. le **pire cas** par méthode — minimum pour le Dice, maximum pour les distances ;
  4. la **stratification par cohorte d'acquisition** de Mindboggle-101 (7 cohortes),
     notre équivalent de leur robustesse inter-constructeur.

Sorties :
  outputs/ablation_2026-07/tables/effect_sizes.csv
  outputs/ablation_2026-07/tables/mindboggle_cohorts.csv

    .venv/bin/python scripts/make_effectsize_table.py
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

TAB = "outputs/ablation_2026-07/tables"
N_BOOT = 10_000
SEED = 20260517

# Cohortes d'acquisition de Mindboggle-101, lues depuis l'identifiant de sujet.
COHORT_RE = [
    ("OASIS-TRT", r"^OASIS_TRT"), ("NKI-RS", r"^NKI_RS"), ("NKI-TRT", r"^NKI_TRT"),
    ("MMRR-21", r"^MMRR_21"), ("MMRR-3T7T", r"^MMRR_3T7T"), ("HLN-12", r"^HLN"),
    ("Twins-2", r"^Twins"), ("Colin27", r"^Colin27"), ("Afterthought", r"^Afterthought"),
]


def cohort(subject: str) -> str:
    for name, pat in COHORT_RE:
        if re.match(pat, subject):
            return name
    return "other"


def load():
    sm = pd.read_csv(f"{TAB}/surface_metrics_corrected.csv")
    ms = pd.read_csv(f"{TAB}/ibsr18_multiseed_surface.csv")
    ped = pd.read_csv("outputs/pediatric_sota_eval/all_pediatric_sota_results_long.csv")
    ours_b = pd.read_csv(
        "outputs/validation_runs/bobs_determ/sam_neuroadapt_bobs_results.csv"
    ).set_index("subject")["dice"]
    return sm, ms, ped, ours_b


def paired(sm, ms, ped, ours_b, bench: str, metric: str) -> pd.DataFrame:
    """Séries appariées par sujet (Ours, HD-BET, SynthStrip)."""
    if bench == "BOBs" and metric == "dice":
        o = ours_b
        b = ped[ped.dataset.astype(str) == "BOBs"]
        h = b[b.method == "HD-BET"].set_index("subject")["dice"]
        s = b[b.method == "SynthStrip"].set_index("subject")["dice"]
    else:
        o = (ms.groupby("subject")[metric].mean() if bench == "IBSR18"
             else sm[(sm.benchmark == bench) & (sm.method == "Ours")].set_index("subject")[metric])
        h = sm[(sm.benchmark == bench) & (sm.method == "HD-BET")].set_index("subject")[metric]
        s = sm[(sm.benchmark == bench) & (sm.method == "SynthStrip")].set_index("subject")[metric]
    return pd.DataFrame({"Ours": o, "HD-BET": h, "SynthStrip": s}).replace(
        [np.inf, -np.inf], np.nan).dropna()


def worst(v: pd.Series, metric: str) -> float:
    """Pire valeur : minimum pour un recouvrement, maximum pour une distance."""
    return float(v.min() if metric == "dice" else v.max())


def main() -> int:
    sm, ms, ped, ours_b = load()
    rng = np.random.default_rng(SEED)
    rows = []
    for bench in ("BOBs", "IBSR18", "Mindboggle-101"):
        for metric in ("dice", "assd", "hd95"):
            d = paired(sm, ms, ped, ours_b, bench, metric)
            diff = d["Ours"] - d["HD-BET"]
            # d de Cohen apparié : la version d_z, adaptée à un test apparié
            dz = float(diff.mean() / diff.std(ddof=1))
            boot = np.array([rng.choice(diff.values, len(diff), replace=True).mean()
                             for _ in range(N_BOOT)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(
                benchmark=bench, metric=metric.upper(), n=len(d),
                mean_ours=d["Ours"].mean(), mean_hdbet=d["HD-BET"].mean(),
                delta=float(diff.mean()), ci_low=float(lo), ci_high=float(hi),
                cohens_dz=dz,
                worst_ours=worst(d["Ours"], metric),
                worst_hdbet=worst(d["HD-BET"], metric),
                worst_synthstrip=worst(d["SynthStrip"], metric),
            ))
    eff = pd.DataFrame(rows)
    eff.to_csv(f"{TAB}/effect_sizes.csv", index=False)

    pd.set_option("display.width", 220)
    print("=== d de Cohen, IC 95 % bootstrap et pire cas ===")
    print(eff.round(4).to_string(index=False))

    # ---------------------------------------- stratification Mindboggle-101 ----
    mb = sm[sm.benchmark == "Mindboggle-101"].copy()
    mb["cohort"] = mb.subject.map(cohort)
    coh = (mb.pivot_table(index=["cohort", "subject"], columns="method", values="dice")
             .reset_index())
    agg = (coh.groupby("cohort")
              .agg(n=("subject", "size"), ours=("Ours", "mean"), hdbet=("HD-BET", "mean"),
                   synthstrip=("SynthStrip", "mean"), ours_worst=("Ours", "min"))
              .round(4))
    agg["delta"] = (agg.ours - agg.hdbet).round(4)
    agg = agg.sort_values("delta", ascending=False)
    agg.to_csv(f"{TAB}/mindboggle_cohorts.csv")
    print("\n=== Mindboggle-101 par cohorte d'acquisition (Dice) ===")
    print(agg.to_string())
    print(f"\nécrit : {TAB}/effect_sizes.csv et {TAB}/mindboggle_cohorts.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
