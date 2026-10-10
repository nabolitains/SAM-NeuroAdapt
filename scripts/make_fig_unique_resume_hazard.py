"""Figure « unique » — le piège de la reprise après interruption (IBSR18).

Remplace l'ancienne figure « non-déterminisme run-to-run », dont le récit était faux :
la variabilité observée n'était pas du bruit de la méthode mais un défaut de notre
harnais d'évaluation (garde-fou qui échoue en mode ouvert sur un run repris).

Panel (a) : Dice par run des 5 graines du protocole figé. Les 4 runs ININTERROMPUS
            s'accordent à ±0.0006 ; le seul run INTERROMPU (3 segments) tombe à 0.889.
Panel (b) : le mécanisme, par sujet — les sujets dégradés du run interrompu sont
            exactement ceux dont le masque est SUR-GONFLÉ (volume/GT > 1.15), c.-à-d.
            ceux pour qui la garde anti-inflation aurait dû agir mais s'est désactivée
            faute de masque de référence (`no_base_mask`).
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, os

OUT = "assets/figures"; os.makedirs(OUT, exist_ok=True)
ROOT = "outputs/ablation_2026-07/multiseed"
INTERRUPTED = "20260517"          # 3 segments d'exécution (2 redémarrages machine)
UNINTERRUPTED = ["20260518", "20260519", "20260520", "20260521"]
HDBET = 0.9217

def load(seed):
    return pd.read_csv(f"{ROOT}/ibsr18_s{seed}/sam_neuroadapt_ibsr18_adultt1_results.csv").set_index("subject")

runs = {s: load(s) for s in [INTERRUPTED] + UNINTERRUPTED}
means = {s: d["dice"].mean() for s, d in runs.items()}

fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), dpi=190,
                        gridspec_kw={"width_ratios": [1, 1.35]})

# ---------------------------------------------------------------- (a) par run ----
ax = axs[0]
order = [INTERRUPTED] + UNINTERRUPTED
vals = [means[s] for s in order]
colors = ["#c9463f"] + ["#4dbd6f"] * len(UNINTERRUPTED)
bars = ax.bar(range(len(order)), vals, color=colors, alpha=0.9)
ax.axhline(HDBET, color="black", ls="--", lw=1.3, label=f"HD-BET ({HDBET:.4f})")
for i, v in enumerate(vals):
    ax.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8.5)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"seed {s}\n{'INTERRUPTED' if s == INTERRUPTED else 'uninterrupted'}" for s in order],
                   rotation=30, ha="right", fontsize=8)
ax.set_ylabel("mean Dice (N=20)", fontsize=10.5)
ax.set_ylim(0.85, 0.96)
u = np.array([means[s] for s in UNINTERRUPTED])
ax.set_title("(a) Mean Dice per run (same configuration, five seeds)",
             fontsize=9.8, fontweight="bold")
ax.legend(fontsize=8, loc="lower right")
ax.grid(axis="y", alpha=0.25)

# ---------------------------------------------------------------- (b) mécanisme ----
ax = axs[1]
di = runs[INTERRUPTED]
ref = pd.concat([runs[s]["dice"] for s in UNINTERRUPTED], axis=1).mean(axis=1)
ratio = di["vol_pred_cm3"] / di["vol_gs_cm3"]
# « dégradé » = perte RÉELLE due à l'interruption, donc écart au run continu du même
# sujet — pas un Dice bas dans l'absolu (un sujet intrinsèquement difficile est bas
# dans les deux cas et n'est pas une victime de l'interruption).
degraded = (ref - di["dice"]) > 0.05
ax.scatter(ratio[~degraded], di["dice"][~degraded], s=45, color="#4dbd6f",
           edgecolor="black", linewidth=0.4, label="other subjects", zorder=3)
ax.scatter(ratio[degraded], di["dice"][degraded], s=70, color="#c9463f",
           edgecolor="black", linewidth=0.5, label="Dice more than 0.05 below uninterrupted runs", zorder=4)
loss = (ref - di["dice"])[degraded].sort_values(ascending=False)
for s in loss.index[:5]:                      # les 5 pertes les plus fortes, pour rester lisible
    ax.annotate(f"{s}: {di.loc[s,'dice']:.3f} → {ref[s]:.3f}",
                (ratio[s], di.loc[s, "dice"]), textcoords="offset points",
                xytext=(-10, 0), fontsize=7, va="center", ha="right")
ax.axvline(1.15, color="black", ls=":", lw=1.4)
ax.text(1.16, 0.745, "volume ratio = 1.15\n(retrospective diagnostic)", fontsize=7.5, va="top")
ax.set_xlim(0.78, 1.60)
ax.set_xlabel("predicted / reference volume", fontsize=10.5)
ax.set_ylabel("Dice (interrupted run)", fontsize=10.5)
ax.set_title("(b) Interrupted run, per subject", fontsize=9.8, fontweight="bold")
ax.legend(fontsize=8.5, loc="lower left")
ax.grid(alpha=0.25)

plt.subplots_adjust(wspace=0.28, bottom=0.2, top=0.9)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig_unique_resume_hazard.{ext}", bbox_inches="tight",
                pad_inches=0.1, facecolor="white")
print("écrit:", f"{OUT}/fig_unique_resume_hazard.png")
print(f"  runs continus : {u.mean():.4f} ± {u.std(ddof=1):.4f}")
print(f"  run interrompu: {means[INTERRUPTED]:.4f}  ({int(degraded.sum())} sujets dégradés)")
