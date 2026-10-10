"""Figure 4 — Dice, ASSD et HD95 par benchmark (remplace make_fig4_boxplots.py).

Différence essentielle avec la version précédente : les distances de surface sont lues
depuis `surface_metrics_corrected.csv`, c'est-à-dire recalculées **surface-à-surface** et
**identiquement pour les trois méthodes**. L'ancienne version mélangeait un recalcul correct
pour notre méthode avec les colonnes du harnais (biaisées en faveur de la sur-segmentation)
pour les baselines, ce qui inversait la comparaison sur Mindboggle-101
(cf. REPRODUCIBILITY_KNOWN_ISSUES.md §5).

IBSR18 « Ours » = moyenne par sujet des 4 graines ininterrompues, cohérent avec T1.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, os

OUT = "assets/figures"; os.makedirs(OUT, exist_ok=True)
TAB = "outputs/ablation_2026-07/tables"
METHODS = ["SynthStrip", "HD-BET", "Ours"]
COLORS = {"SynthStrip": "#7fb3ff", "HD-BET": "#ff9f4d", "Ours": "#4dbd6f"}
BENCHES = [("BOBs", "BOBs\n(infant, 71 visits)"),
           ("IBSR18", "IBSR-20\n(adult, N=20)"),
           ("Mindboggle-101", "Mindboggle-101\n(adult, N=101)")]

surf = pd.read_csv(f"{TAB}/surface_metrics_corrected.csv")
ms = pd.read_csv(f"{TAB}/ibsr18_multiseed_surface.csv")   # 4 graines ininterrompues


def series(bench: str, method: str, metric: str) -> np.ndarray:
    """Valeurs par sujet ; pour IBSR18/Ours, moyenne des 4 graines (cohérent avec T1)."""
    if bench == "IBSR18" and method == "Ours":
        v = ms.groupby("subject")[metric].mean().values
    else:
        s = surf[(surf.benchmark == bench) & (surf.method == method)]
        v = s[metric].values
    v = np.asarray(v, dtype=float)
    return v[np.isfinite(v)]


# Les tests ne sont PAS recalculés ici : ils sont lus dans la table figée produite par
# scripts/make_multiplicity_correction.py, celle-là même qui alimente le Tableau 1. Deux raisons.
# 1) Une version antérieure recalculait le Wilcoxon en appariant les deux séries par
#    troncature (`a[:n], b[:n]`) au lieu d'une jointure par sujet — appariement fragile.
# 2) Surtout, elle affichait le p BRUT et en dérivait les étoiles, si bien que la figure
#    marquait IBSR18 Dice d'une étoile alors que le Tableau 1 le déclare n.s. après
#    correction. Un relecteur regardant la figure sans le tableau y lisait un résultat
#    significatif de plus. Toute significativité vient désormais de p_adj (BH).
holm = pd.read_csv(f"{TAB}/t1_multiplicity_correction.csv")
holm["metric"] = holm["metric"].str.upper()
# p_BH = correction primaire (Benjamini-Hochberg), cohérente avec le Tableau 1.
HOLM = {(r.benchmark, r.metric): (float(r.p), float(r.p_BH)) for r in holm.itertuples()}


def stars(p_adj):
    """Étoiles fondées sur le p AJUSTÉ, jamais sur le brut."""
    if p_adj is None: return "n.s."
    return "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "n.s."


def fmt(label, p):
    """« p<1e-12 » plutôt que « p=<1e-12 » : le comparateur remplace le signe égal."""
    if p < 1e-12:
        return f"${label}<$1e-12"
    return f"${label}=${p:.1e}" if p < 0.01 else f"${label}=${p:.3f}"


fig, axs = plt.subplots(3, 3, figsize=(13, 11), dpi=190)
for col, (bench, label) in enumerate(BENCHES):
    for row, (metric, ylab) in enumerate([("dice", "Dice"), ("assd", "ASSD (mm)"), ("hd95", "HD95 (mm)")]):
        ax = axs[row, col]
        data = [series(bench, m, metric) for m in METHODS]
        bp = ax.boxplot(data, tick_labels=METHODS, patch_artist=True, widths=0.55,
                        showfliers=False, whis=(5, 95), medianprops=dict(color="black", lw=1.4))
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(COLORS[m]); patch.set_alpha(0.75)
        rng = np.random.default_rng(0)
        for k, d in enumerate(data):
            ax.scatter(1 + k + rng.uniform(-0.12, 0.12, len(d)), d, s=7,
                       color="black", alpha=0.35, zorder=3, linewidths=0)
        # annotation Ours vs HD-BET (indices 2 et 1), valeurs figées du Tableau 1
        p, p_adj = HOLM[(bench, metric.upper())]
        hi = max(d.max() for d in data if len(d)); lo = min(d.min() for d in data if len(d))
        span = (hi - lo) or 1.0
        top = hi + 0.08 * span
        ax.plot([2, 2, 3, 3], [top, top + 0.03 * span, top + 0.03 * span, top], color="black", lw=1.0)
        ax.text(2.5, top + 0.05 * span,
                f"{stars(p_adj)}\n{fmt('p', p)}  {fmt('p_{adj}', p_adj)}",
                ha="center", va="bottom", fontsize=7.2)
        ax.set_ylim(lo - 0.05 * span, top + 0.22 * span)
        if row == 0:
            ax.set_title(label, fontsize=10.5, fontweight="bold")
        if col == 0:
            ax.set_ylabel(ylab, fontsize=10.5, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8.5)
        ax.grid(axis="y", alpha=0.25)

# Le texte explicatif vit dans la légende LaTeX, pas dans l'image (consignes de la revue).
plt.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig4_boxplots.{ext}", bbox_inches="tight", pad_inches=0.08, facecolor="white")
print("écrit:", f"{OUT}/fig4_boxplots.png")
for bench, _ in BENCHES:
    for metric in ("dice", "assd", "hd95"):
        o, h = series(bench, "Ours", metric), series(bench, "HD-BET", metric)
        p, p_adj = HOLM[(bench, metric.upper())]
        print(f"  {bench:16s} {metric:5s} Ours={o.mean():7.3f}  HD-BET={h.mean():7.3f}  "
              f"p={p:.3g}  p_adj={p_adj:.3g}  {stars(p_adj)}")
