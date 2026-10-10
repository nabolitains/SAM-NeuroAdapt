"""T2 — tableau des ablations (déterminisme / raffinement intensité / collapse-rescue).
Réutilise les mêmes données mesurées/citées que make_fig6_ablations.py (voir ce script
pour la provenance détaillée de chaque valeur). Sortie : assets/tables/table2_ablations.md + .tex
"""
import os
import numpy as np

OUT = "assets/tables"; os.makedirs(OUT, exist_ok=True)

rows = []

# --- a) déterminisme (IBSR18) ---
runs_after = [0.9194, 0.9258, 0.9363, 0.9211, 0.9428, 0.9332]
std_after = float(np.std(runs_after, ddof=1))
rows.append(dict(
    levier="Déterminisme (seed ants/cuDNN)", benchmark="IBSR18 (N=20)",
    avant="bruit inter-runs ≈0.050 (rapporté, runner pré-fix non rejouable)",
    apres=f"bruit inter-runs {std_after:.3f} (mesuré, {len(runs_after)} runs, protocole figé)",
    metrique="écart-type Dice inter-runs", n="6 runs × N=20",
    source="RESULTS_SYNTHESIS.md (avant) ; 6 CSV figés sur disque (après)",
))

# --- b) chaîne de raffinement (Mindboggle-101) ---
rows.append(dict(
    levier="Raffinement intensité (Chan-Vese)", benchmark="Mindboggle-101 (N=101)",
    avant="0.8730 (SAM2 brut, cité RESULTS_SYNTHESIS.md)",
    apres="0.9330 (mesuré, comparison_101.csv)",
    metrique="Dice moyen", n="N=101",
    source="RESULTS_SYNTHESIS.md (avant) ; comparison_101.csv (après)",
))
rows.append(dict(
    levier="Collapse-rescue (ré-ancrage atlas robuste)", benchmark="Mindboggle-101 (N=101)",
    avant="0.9330 (mesuré, comparison_101.csv)",
    apres="0.9423 (mesuré, comparison_101_rescued.csv)",
    metrique="Dice moyen", n="N=101 (dont 7 sujets réellement modifiés)",
    source="comparison_101.csv → comparison_101_rescued.csv",
))

# --- c) collapse-rescue isolé sur les 7 sujets concernés ---
rescue_log = [
    ("MMRR_21_14", 0.696, 0.908), ("MMRR_21_17", 0.588, 0.911), ("HLN_12_12", 0.915, 0.949),
    ("HLN_12_3", 0.918, 0.945), ("HLN_12_4", 0.935, 0.955), ("Twins_2_2", 0.720, 0.957),
    ("MMRR_21_21", 0.857, 0.943),
]
before = np.array([r[1] for r in rescue_log]); after = np.array([r[2] for r in rescue_log])
rows.append(dict(
    levier="Collapse-rescue — sous-groupe désaccord sévère", benchmark="Mindboggle-101 (7 sujets, dis>0.15)",
    avant=f"{before.mean():.3f} ± {before.std(ddof=1):.3f} (min {before.min():.3f})",
    apres=f"{after.mean():.3f} ± {after.std(ddof=1):.3f} (min {after.min():.3f})",
    metrique="Dice moyen (sous-groupe)", n="N=7",
    source="_collapse_rescue2.log",
))

if __name__ == "__main__":
    md = ["# T2 — Étude d'ablation (3 leviers méthodologiques)\n",
          "| Levier | Benchmark | Métrique | Avant | Après | N | Source |",
          "|---|---|---|---|---|---|---|"]
    for row in rows:
        md.append(f"| {row['levier']} | {row['benchmark']} | {row['metrique']} | {row['avant']} | "
                   f"{row['apres']} | {row['n']} | {row['source']} |")
    md_text = "\n".join(md)
    with open(f"{OUT}/table2_ablations.md", "w") as f:
        f.write(md_text)
    print(md_text)

    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Étude d'ablation : importance de 3 leviers méthodologiques du pipeline zero-training.}",
        r"\label{tab:ablations}",
        r"\begin{tabular}{p{3.2cm}p{2.5cm}p{3.2cm}p{3.2cm}l}",
        r"\toprule",
        r"Levier & Benchmark & Avant & Après & N \\",
        r"\midrule",
    ]
    for row in rows:
        def esc(s):
            return s.replace("±", "$\\pm$").replace(">", "$>$")
        tex.append(f"{esc(row['levier'])} & {esc(row['benchmark'])} & {esc(row['avant'])} & "
                    f"{esc(row['apres'])} & {esc(row['n'])} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_text = "\n".join(tex)
    with open(f"{OUT}/table2_ablations.tex", "w") as f:
        f.write(tex_text)
    print("\nécrit:", f"{OUT}/table2_ablations.md", "et", f"{OUT}/table2_ablations.tex")
