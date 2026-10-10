"""Figure 6 (remplace l'ancienne figure d'ablation) — qui fait le travail dans le pipeline.

Panel (a) : Dice par configuration et par benchmark — atlas seul (sans SAM2),
            SAM2 seul (sans atlas), chaîne complète.
Panel (b) : taux d'échec (Dice<0.80) pour les mêmes configurations. C'est LE panel :
            l'atlas ne déplace pas la moyenne, il supprime les échecs.
Panel (c) : sur les sujets où SAM2 ne s'effondre PAS sans atlas, il égale la chaîne
            complète → l'atlas est du confinement d'échec, pas de la précision.

Provenance des valeurs — deux régimes, à ne pas confondre :
  * les lignes « SAM2 sans atlas » sont recalculées à chaque exécution depuis les CSV
    par sujet de la campagne d'ablation (`noatlas_stats`) ;
  * les lignes « atlas seul » et « chaîne complète » sont recopiées en dur ci-dessous
    depuis `outputs/ablation_2026-07/tables/ablation_cascade.csv`, le fichier figé qui
    alimente aussi le Tableau 2 de l'article. Elles y sont identiques (BOBs
    0.9230/0.9201, IBSR18 0.9007/0.9388, Mindboggle 0.9145/0.9418).
Conséquence : si `ablation_cascade.csv` était régénéré avec des valeurs différentes, cette
figure ne suivrait pas. `scripts/verify_manuscript_numbers.py` compare le Tableau 2 au CSV
et détecterait la divergence, mais pas le décalage de cette figure — le contrôle reste
manuel tant que ces constantes ne sont pas lues depuis le CSV.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, glob, os

OUT = "assets/figures"; os.makedirs(OUT, exist_ok=True)
ABL = "outputs/ablation_2026-07"

TAB = f"{ABL}/tables"
SEEDS = ["20260518", "20260519", "20260520", "20260521"]          # runs ininterrompus

def full_per_subject(bench):
    """Dice de la chaîne complète, par sujet — même source que le Tableau 1."""
    if bench == "IBSR18":
        ms = pd.read_csv(f"{TAB}/ibsr18_multiseed_surface.csv")
        return ms[ms.seed.astype(str).isin(SEEDS)].groupby("subject")["dice"].mean()
    sm = pd.read_csv(f"{TAB}/surface_metrics_corrected.csv")
    return sm[(sm.benchmark == bench) & (sm.method == "Ours")].set_index("subject")["dice"]

def noatlas_stats(paths, bench=None):
    d = pd.concat([pd.read_csv(f) for f in paths], ignore_index=True)
    filled = d["dice"].fillna(0.0)                    # masque non produit = échec
    ok = d["dice"].notna() & (d["dice"] >= 0.80)
    surv = d.loc[ok, "dice"]
    out = dict(n=len(d), dice_all=filled.mean(), fail=(filled < 0.80).mean(),
               dice_surv=surv.mean() if len(surv) else np.nan, n_surv=int(ok.sum()),
               full_same=np.nan)
    # Panel (c) : la chaîne complète doit être lue SUR CES MÊMES SUJETS. La comparer à la
    # moyenne de la cohorte entière oppose deux populations différentes (relecture du
    # 2026-10-01) ; l'appariement réel donne IBSR 0.9481 (n=11), Mindboggle 0.9388 (n=26).
    if bench is not None and len(surv):
        same = full_per_subject(bench).reindex(d.loc[ok, "subject"])
        assert same.notna().all(), f"{bench} : sujets non appariés"
        out["full_same"] = float(same.mean())
    return out

# --- données mesurées ---
B = {}
B["BOBs\n(infant)"] = dict(
    atlas=0.9230, atlas_fail=0.028, full=0.9201, full_fail=0.0,
    **noatlas_stats([f"{ABL}/noatlas/bobs/sam_neuroadapt_bobs_results.csv"]))
B["IBSR-20\n(adult)"] = dict(
    atlas=0.9007, atlas_fail=0.10, full=0.9388, full_fail=0.0,
    **noatlas_stats([f"{ABL}/noatlas/ibsr18/sam_neuroadapt_ibsr18_adultt1_results.csv"], "IBSR18"))
B["Mindboggle-101\n(adult)"] = dict(
    atlas=0.9145, atlas_fail=0.053, full=0.9418, full_fail=0.0,
    **noatlas_stats(sorted(glob.glob(f"{ABL}/noatlas/mbg101/b*/sam_neuroadapt_*results.csv")),
                    "Mindboggle-101"))

names = list(B)
x = np.arange(len(names)); w = 0.26
C = {"atlas": "#1f77b4", "sam2": "#c9463f", "full": "#4dbd6f"}

fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), dpi=190)

# ---------------------------------------------------------------- (a) Dice ----
ax = axs[0]
ax.bar(x - w, [B[n]["atlas"] for n in names], w, color=C["atlas"], label="atlas alone (no SAM2)")
ax.bar(x,     [B[n]["dice_all"] for n in names], w, color=C["sam2"], label="SAM2 without atlas prompts")
ax.bar(x + w, [B[n]["full"] for n in names], w, color=C["full"], label="complete pipeline")
for i, n in enumerate(names):
    for dx, k in ((-w, "atlas"), (0, "dice_all"), (w, "full")):
        ax.text(i + dx, B[n][k] + 0.012, f"{B[n][k]:.3f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel("mean Dice", fontsize=10.5); ax.set_ylim(0, 1.08)
ax.set_title("(a) Mean Dice, all cases", fontsize=10.5, fontweight="bold")
ax.legend(fontsize=7.6, loc="lower left"); ax.grid(axis="y", alpha=0.25)

# ---------------------------------------------------------------- (b) échecs ----
ax = axs[1]
ax.bar(x - w, [100*B[n]["atlas_fail"] for n in names], w, color=C["atlas"])
ax.bar(x,     [100*B[n]["fail"] for n in names], w, color=C["sam2"])
ax.bar(x + w, [100*B[n]["full_fail"] for n in names], w, color=C["full"])
for i, n in enumerate(names):
    for dx, k in ((-w, "atlas_fail"), (0, "fail"), (w, "full_fail")):
        ax.text(i + dx, 100*B[n][k] + 2, f"{100*B[n][k]:.0f}%", ha="center", fontsize=7.5,
                fontweight="bold" if k == "fail" else "normal")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel("failure rate (Dice < 0.80), %", fontsize=10.5); ax.set_ylim(0, 118)
ax.set_title("(b) Failure rate", fontsize=10.5, fontweight="bold")
ax.grid(axis="y", alpha=0.25)

# ---------------------------------------------------------------- (c) survivants ----
ax = axs[2]
adults = [n for n in names if not np.isnan(B[n]["dice_surv"])]
xa = np.arange(len(adults))
ax.bar(xa - 0.17, [B[n]["dice_surv"] for n in adults], 0.34, color=C["sam2"],
       label="SAM2 without atlas prompts")
ax.bar(xa + 0.17, [B[n]["full_same"] for n in adults], 0.34, color=C["full"],
       label="complete pipeline, same cases")
for i, n in enumerate(adults):
    ax.text(i - 0.17, B[n]["dice_surv"] + 0.002, f"{B[n]['dice_surv']:.4f}", ha="center", fontsize=8)
    ax.text(i + 0.17, B[n]["full_same"] + 0.002, f"{B[n]['full_same']:.4f}", ha="center", fontsize=8)
ax.set_xticks(xa)
ax.set_xticklabels([f"{n.split(chr(10))[0]}\n(n={B[n]['n_surv']})" for n in adults], fontsize=8.5)
ax.set_ylim(0.90, 0.96); ax.set_ylabel("mean Dice", fontsize=10.5)
ax.set_title("(c) Non-failed cases, paired", fontsize=10.5, fontweight="bold")
ax.legend(fontsize=7.6, loc="lower right"); ax.grid(axis="y", alpha=0.25)

plt.subplots_adjust(wspace=0.28, bottom=0.14, top=0.9)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig6_ablation_cascade.{ext}", bbox_inches="tight", pad_inches=0.1, facecolor="white")
print("écrit:", f"{OUT}/fig6_ablation_cascade.png")
for n in names:
    b = B[n]
    print(f"  {n.split(chr(10))[0]:16s} atlas={b['atlas']:.4f}  sans-atlas={b['dice_all']:.4f} "
          f"(échec {100*b['fail']:.0f}%, non-échecs {b['dice_surv']:.4f}, n={b['n_surv']}, "
          f"complète mêmes sujets {b['full_same']:.4f})  complète={b['full']:.4f}")
