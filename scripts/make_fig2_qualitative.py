"""Figure 2 — comparaison qualitative (style LifespanStrip Fig. 2).
Grille lignes = tranches d'âge (infant BOBs, adulte Mindboggle) × colonnes = méthodes
(T1w · SynthStrip · HD-BET · Ours · Ground truth). Overlay masque ROUGE + cercles
JAUNES (sous-seg / FN) et VERTS (sur-seg / FP) vs GT. Zero-training.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np, nibabel as nib, glob, os, pandas as pd
from scipy import ndimage as ndi

OUT = "assets/figures"; os.makedirs(OUT, exist_ok=True)
def L(p): return np.asarray(nib.load(p).dataobj)
def Lb(p): return L(p) > 0
def dice(a, b): s = a.sum()+b.sum(); return 2*(a & b).sum()/s if s else 0.0

# ---- config des lignes (label, T1, GT, {méthode: mask}) ----
def bobs_row():
    sub = "sub-116056_ses-3mo"
    ped = pd.read_csv("outputs/pediatric_sota_eval/all_pediatric_sota_results_long.csv")
    b = ped[(ped.dataset.astype(str).str.contains("BOB", na=False)) & (ped.subject == sub)]
    mp = {r.method: r.mask_path for _, r in b.iterrows()}
    return ("infant\n(BOBs, 3 mo)",
            f"datasets/standardized/BOBs/{sub}/anat/{sub}_T1w.nii.gz",
            f"datasets/standardized/BOBs/{sub}/anat/{sub}_desc-brain_mask.nii.gz",
            {"SynthStrip": mp["SynthStrip"], "HD-BET": mp["HD-BET"],
             "Ours": f"outputs/bobs_m1_pediatric_isegstyle_eval/{sub}/anat/{sub}_T1w_desc-brain_mask.nii.gz"})

def mbg_row(sub):
    lon = pd.read_csv("outputs/validation_runs/mindboggle101_full/sota_eval/all_methods_long.csv")
    r = lon[lon.subject == sub]
    def mp(m):
        x = r[r.method == m]; return x.iloc[0].mask_path if len(x) else None
    d = None
    for bb in glob.glob("outputs/validation_runs/mindboggle101_full/b*/"):
        if os.path.isdir(f"{bb}{sub}/anat"): d = f"{bb}{sub}/anat"; break
    return ("adult\n(Mindboggle)",
            f"datasets/standardized/Mindboggle101/{sub}/anat/sub-{sub}_T1w.nii.gz",
            f"datasets/standardized/Mindboggle101/{sub}/anat/sub-{sub}_desc-brain_mask.nii.gz",
            {"SynthStrip": mp("SynthStrip"), "HD-BET": mp("HD-BET"),
             "Ours": f"{d}/sub-{sub}_desc-brain_mask.nii.gz"})

ROWS = [bobs_row(), mbg_row("NKI_RS_22_2")]
COLS = ["T1w MRI", "SynthStrip", "HD-BET", "Ours", "Ground truth"]

def sag(vol, idx, ax):
    sl = [slice(None)]*3; sl[ax] = idx; return np.rot90(vol[tuple(sl)])

def biggest_cluster(mask2d, min_area=25):
    lbl, n = ndi.label(mask2d)
    if n == 0: return None
    sizes = ndi.sum(np.ones_like(lbl), lbl, range(1, n+1))
    k = int(np.argmax(sizes));
    if sizes[k] < min_area: return None
    cy, cx = ndi.center_of_mass(lbl == k+1)
    r = float(np.sqrt(sizes[k]/np.pi)) + 4
    return cx, cy, r

nr, nc = len(ROWS), len(COLS)
fig, axs = plt.subplots(nr, nc, figsize=(2.15*nc, 2.35*nr), dpi=190)
for j, c in enumerate(COLS):
    axs[0, j].set_title(c, fontsize=11, fontweight="bold")

for i, (lab, t1p, gtp, masks) in enumerate(ROWS):
    t1 = np.asarray(L(t1p), dtype=np.float32); gt = Lb(gtp)
    ax_sag = int(np.argmin(t1.shape))
    # coupe = celle où le cerveau GT est le PLUS CONNECTÉ (vraie médiane, pas para-sagittal
    # qui coupe le cortex plié → apparence fragmentée). On maximise l'aire de la + grande
    # composante connexe 2D de GT, dans le tiers central.
    other = tuple(k for k in range(3) if k != ax_sag)
    n = t1.shape[ax_sag]; lo, hi = int(n*0.33), int(n*0.67)
    best_i, best_a = (lo+hi)//2, -1
    for s in range(lo, hi):  # NB: variable dédiée — ne pas écraser l'index de ligne « i »
        g2 = np.take(gt, s, axis=ax_sag)
        lbl, nc2 = ndi.label(g2)
        if nc2 == 0: continue
        a2 = ndi.sum(np.ones_like(lbl), lbl, range(1, nc2+1)).max()
        if a2 > best_a: best_a, best_i = a2, s
    idx = best_i
    T = sag(t1, idx, ax_sag); G = sag(gt, idx, ax_sag)
    v = np.clip(T/(np.percentile(T[T > 0], 99)+1e-6), 0, 1)
    axs[i, 0].set_ylabel(lab, fontsize=10.5, fontweight="bold")
    for j, col in enumerate(COLS):
        a = axs[i, j]; a.imshow(v, cmap="gray"); a.set_xticks([]); a.set_yticks([])
        if col == "T1w MRI":
            continue
        M = G if col == "Ground truth" else sag(Lb(masks[col]) if (masks.get(col) and os.path.exists(masks[col])) else np.zeros_like(gt), idx, ax_sag)
        M = ndi.binary_fill_holes(M)  # rendu net (pas de speckle sur la coupe)
        ov = np.zeros((*v.shape, 4)); ov[M > 0] = [0.90, 0.16, 0.16, 0.48]
        a.imshow(ov, interpolation="none")
        a.contour(M, levels=[0.5], colors="#d01010", linewidths=0.8)
        if col not in ("Ground truth",):
            d = dice(sag(Lb(masks[col]), idx, ax_sag) if masks.get(col) and os.path.exists(masks[col]) else np.zeros_like(G), G)
            # cercles d'erreur vs GT (jaune = FN/sous-seg, vert = FP/sur-seg)
            fn = biggest_cluster((G & ~M)); fp = biggest_cluster((M & ~G))
            if fn: a.add_patch(Circle((fn[0], fn[1]), fn[2], fill=False, ec="#ffd21e", lw=1.8, ls=(0,(3,2))))
            if fp: a.add_patch(Circle((fp[0], fp[1]), fp[2], fill=False, ec="#38d957", lw=1.8, ls=(0,(3,2))))
for a in axs.ravel(): a.set_facecolor("black")
# légende
fig.text(0.5, 0.005, "red = predicted brain mask   ·   yellow circle = under-segmentation (FN)   ·   green circle = over-segmentation (FP)",
         ha="center", fontsize=9)
plt.subplots_adjust(0.05, 0.05, 0.995, 0.94, wspace=0.04, hspace=0.06)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig2_qualitative.{ext}", bbox_inches="tight", pad_inches=0.05, facecolor="white")
print("écrit:", f"{OUT}/fig2_qualitative.png")
