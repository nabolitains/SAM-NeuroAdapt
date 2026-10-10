"""Correction de multiplicité sur les neuf tests appariés du Tableau 1.

Le Tableau 1 rapporte neuf comparaisons appariées Ours vs HD-BET (3 benchmarks x 3
metriques). Sous l'hypothese nulle globale, la probabilite qu'au moins un p brut tombe
sous 0.05 par hasard vaut 1 - 0.95^9 ~ 37 % : sans correction, aucune de ces marges ne
soutient une affirmation individuelle.

Deux corrections sont calculees et TOUTES DEUX rapportees dans l'article :

  * Benjamini-Hochberg (FDR)  -- methode PRIMAIRE. Controle le taux de fausses
    decouvertes, ce qui est le cadre approprie quand on rapporte un panel de metriques
    liees plutot qu'une hypothese unique, et c'est la convention du domaine.
  * Holm-Bonferroni (FWER)    -- rapporte en parallele, plus conservateur.

Le choix ne change qu'UNE cellule sur neuf (IBSR18 ASSD). L'article le declare
explicitement plutot que de choisir en silence -- choisir la correction apres avoir vu
laquelle donne le plus de resultats significatifs serait exactement le comportement que
le manuscrit denonce ailleurs.

Contexte : LifespanStrip (Nature Biomed. Eng. 2025), le concurrent direct, corrige lui
aussi la multiplicite (ANOVA a mesures repetees + test de Dunnett). Ne pas corriger nous
placerait sous le standard de l'article contre lequel nous nous positionnons.

    .venv/bin/python scripts/make_multiplicity_correction.py

Sortie : outputs/ablation_2026-07/tables/t1_multiplicity_correction.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, ".")

TAB = "outputs/ablation_2026-07/tables"
OUT = f"{TAB}/t1_multiplicity_correction.csv"


def main() -> int:
    sm = pd.read_csv(f"{TAB}/surface_metrics_corrected.csv")
    ms = pd.read_csv(f"{TAB}/ibsr18_multiseed_surface.csv")
    ped = pd.read_csv("outputs/pediatric_sota_eval/all_pediatric_sota_results_long.csv")
    ours_b = pd.read_csv(
        "outputs/validation_runs/bobs_determ/sam_neuroadapt_bobs_results.csv"
    ).set_index("subject")["dice"]

    rows = []

    def add(bench: str, metric: str, a: pd.Series, b: pd.Series) -> None:
        d = pd.DataFrame({"o": a, "h": b}).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append((bench, metric.upper(), float(wilcoxon(d.o, d.h).pvalue), len(d)))

    hd_b = ped[(ped.dataset.astype(str) == "BOBs") & (ped.method == "HD-BET")].set_index("subject")["dice"]
    add("BOBs", "dice", ours_b, hd_b)
    for m in ("assd", "hd95"):
        add("BOBs", m,
            sm[(sm.benchmark == "BOBs") & (sm.method == "Ours")].set_index("subject")[m],
            sm[(sm.benchmark == "BOBs") & (sm.method == "HD-BET")].set_index("subject")[m])
    for m in ("dice", "assd", "hd95"):
        add("IBSR18", m, ms.groupby("subject")[m].mean(),
            sm[(sm.benchmark == "IBSR18") & (sm.method == "HD-BET")].set_index("subject")[m])
    for m in ("dice", "assd", "hd95"):
        add("Mindboggle-101", m,
            sm[(sm.benchmark == "Mindboggle-101") & (sm.method == "Ours")].set_index("subject")[m],
            sm[(sm.benchmark == "Mindboggle-101") & (sm.method == "HD-BET")].set_index("subject")[m])

    df = pd.DataFrame(rows, columns=["benchmark", "metric", "p", "n"]).sort_values("p").reset_index(drop=True)
    k = len(df)

    # Holm-Bonferroni : max cumulatif de (k-i)*p, monotone croissant, borne a 1
    holm, run = [], 0.0
    for i, p in enumerate(df.p):
        run = max(run, (k - i) * p)
        holm.append(min(run, 1.0))
    df["p_holm"] = holm

    # Benjamini-Hochberg : min cumulatif de p*k/(i+1) parcouru a rebours, borne a 1
    bh, run = [None] * k, 1.0
    for i in range(k - 1, -1, -1):
        run = min(run, df.p[i] * k / (i + 1))
        bh[i] = min(run, 1.0)
    df["p_BH"] = bh

    df["signif_BH"] = df.p_BH < 0.05
    df["signif_holm"] = df.p_holm < 0.05
    df.to_csv(OUT, index=False)

    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nBH-FDR (primaire) : {int(df.signif_BH.sum())}/9 significatifs")
    print(f"Holm    (rapporte) : {int(df.signif_holm.sum())}/9 significatifs")
    diff = df[df.signif_BH != df.signif_holm]
    print("Cellules ou les deux corrections different :",
          ", ".join(f"{r.benchmark} {r.metric}" for r in diff.itertuples()) or "aucune")
    print(f"\necrit : {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
