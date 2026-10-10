"""Rend la Figure 3 à partir du cache produit par make_fig3_error_maps.py.
Colonnes = méthodes (SynthStrip, HD-BET, Ours) ; 3 rangées du haut = sur-segmentation
(vues sagittale/coronale/axiale) ; 3 rangées du bas = sous-segmentation. Colormap 'hot',
échelle 0→vmax partagée par bloc (sur-seg / sous-seg) pour comparabilité inter-méthodes.
Contour gris clair = silhouette moyenne du cerveau (GT), pour l'orientation anatomique.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

OUT = "assets/figures"
METHODS = ["SynthStrip", "HD-BET", "Ours"]
VIEW_LABELS = ["sagittal", "coronal", "axial"]

d = np.load("outputs/_fig3_cache.npz")
gt = d["gt"]; n_total = int(d["n_total"])
fp = {m: d[f"fp_{m}"] for m in METHODS}
fn = {m: d[f"fn_{m}"] for m in METHODS}

def proj(vol, axis):
    return np.rot90(vol.mean(axis=axis))

def gt_contour(axis):
    return np.rot90(gt.mean(axis=axis))

vmax_fp = max(proj(fp[m], ax).max() for m in METHODS for ax in range(3))
vmax_fn = max(proj(fn[m], ax).max() for m in METHODS for ax in range(3))

nrows, ncols = 6, len(METHODS)
fig, axs = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows), dpi=190)

row_meta = []
for i in range(3):
    row_meta.append(("over", i))
for i in range(3):
    row_meta.append(("under", i))

for j, m in enumerate(METHODS):
    axs[0, j].set_title(m, fontsize=13, fontweight="bold")

for r, (kind, ax_idx) in enumerate(row_meta):
    data = fp if kind == "over" else fn
    vmax = vmax_fp if kind == "over" else vmax_fn
    row_label = f"{'over-seg.' if kind=='over' else 'under-seg.'}\n({VIEW_LABELS[ax_idx]})"
    axs[r, 0].set_ylabel(row_label, fontsize=9.5, fontweight="bold")
    for j, m in enumerate(METHODS):
        a = axs[r, j]
        img = proj(data[m], ax_idx)
        a.imshow(img, cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
        a.contour(gt_contour(ax_idx), levels=[gt_contour(ax_idx).max() * 0.5],
                  colors="#5599ff", linewidths=0.6, alpha=0.55)
        a.set_xticks([]); a.set_yticks([])
        a.set_facecolor("black")

# barres de couleur
cax1 = fig.add_axes([0.92, 0.53, 0.015, 0.35])
sm1 = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, vmax_fp)); sm1.set_array([])
cb1 = fig.colorbar(sm1, cax=cax1); cb1.set_label("over-seg. rate", fontsize=8)
cax2 = fig.add_axes([0.92, 0.11, 0.015, 0.35])
sm2 = plt.cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, vmax_fn)); sm2.set_array([])
cb2 = fig.colorbar(sm2, cax=cax2); cb2.set_label("under-seg. rate", fontsize=8)

fig.suptitle(f"Aggregated error maps — mean projection over {n_total} subjects "
             f"(BOBs N=71, IBSR18 N=20, Mindboggle-101 N=101; three fair benchmarks)",
             fontsize=10.5, y=0.995)
fig.text(0.46, 0.005, "light-blue contour = mean ground-truth brain silhouette",
         ha="center", fontsize=8.5)
plt.subplots_adjust(left=0.09, right=0.90, top=0.94, bottom=0.03, wspace=0.05, hspace=0.08)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig3_error_maps.{ext}", bbox_inches="tight", pad_inches=0.08, facecolor="white")
print("écrit:", f"{OUT}/fig3_error_maps.png")
