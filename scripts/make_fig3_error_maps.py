"""Figure 3 — cartes d'erreur agrégées (style LifespanStrip Fig. 3).
Colonnes = méthodes (SynthStrip, HD-BET, Ours) ; rangées = sur-segmentation (3 vues
orthogonales) puis sous-segmentation (3 vues orthogonales), moyennées sur les 192 sujets
des 3 benchmarks équitables (BOBs N=71, IBSR18 N=20, Mindboggle-101 N=101). Chaque sujet
est recentré (translation seule, PAS de mise à l'échelle) sur le centroïde de son masque
GT dans un repère canonique RAS commun avant projection — pas de recalage déformable,
juste un alignement rigide en espace physique (mm) cohérent avec l'esprit "zero-training".
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, nibabel as nib, pandas as pd, os, glob
from scipy import ndimage as ndi

OUT = "assets/figures"; os.makedirs(OUT, exist_ok=True)
GRID = (96, 96, 96)   # axes canoniques R, A, S
SPACING = 2.2         # mm/voxel -> FOV 211 mm de côté (tête+cou adulte, marge nourrisson)
METHODS = ["SynthStrip", "HD-BET", "Ours"]

# ---------------------------------------------------------------- manifest ----
def bobs_manifest():
    # "Ours" = run figé outputs/validation_runs/bobs_determ/ (0.9201 = tête de RESULTS_SYNTHESIS.md)
    ped = pd.read_csv("outputs/pediatric_sota_eval/all_pediatric_sota_results_long.csv")
    b = ped[ped.dataset.astype(str) == "BOBs"]
    ours_df = pd.read_csv("outputs/validation_runs/bobs_determ/sam_neuroadapt_bobs_results.csv").set_index("subject")
    rows = []
    for sub in sorted(b["subject"].unique()):
        gt = f"datasets/standardized/BOBs/{sub}/anat/{sub}_desc-brain_mask.nii.gz"
        if not os.path.exists(gt):
            continue
        masks = {}
        for m in ("SynthStrip", "HD-BET"):
            r = b[(b.subject == sub) & (b.method == m)]
            if len(r) and os.path.exists(r.iloc[0].mask_path):
                masks[m] = r.iloc[0].mask_path
        if sub in ours_df.index and os.path.exists(ours_df.loc[sub, "mask_path"]):
            masks["Ours"] = ours_df.loc[sub, "mask_path"]
        if len(masks) == 3:
            rows.append(("BOBs", sub, gt, masks))
    return rows

def ibsr18_manifest():
    # "Ours" = run figé outputs/validation_runs/repro/ibsr18_s20260517/ (0.9428 = tête de RESULTS_SYNTHESIS.md)
    ib = pd.read_csv("outputs/ibsr18_sota_eval_v4_external_templates_frozen/all_methods_long.csv")
    ours_df = pd.read_csv("outputs/validation_runs/repro/ibsr18_s20260517/sam_neuroadapt_ibsr18_adultt1_results.csv").set_index("subject")
    rows = []
    for sub in sorted(ib["subject"].unique()):
        r0 = ib[ib.subject == sub].iloc[0]
        gt = r0.gold_mask_path
        if not os.path.exists(gt):
            continue
        masks = {}
        for m in ("SynthStrip", "HD-BET"):
            r = ib[(ib.subject == sub) & (ib.method == m)]
            if len(r) and os.path.exists(r.iloc[0].mask_path):
                masks[m] = r.iloc[0].mask_path
        if sub in ours_df.index and os.path.exists(ours_df.loc[sub, "output_mask_path"]):
            masks["Ours"] = ours_df.loc[sub, "output_mask_path"]
        if len(masks) == 3:
            rows.append(("IBSR18", sub, gt, masks))
    return rows

def mbg_manifest():
    # Priorité de résolution du masque final "Ours" (validée par recoupement exact avec
    # comparison_101_rescued.csv, moyenne 0.94225... reproduite au bit près) :
    # 1) _rescue_tmp/  -> les 7 sujets collapse-rescue (MMRR_21_17 etc.), résultat final
    # 2) _corr25/      -> les 25 sujets corrigés par le raffinement intensité Chan-Vese
    # 3) b*/{sub}/anat -> les 69 sujets restants (masque de base, jamais retouché)
    # NB : b*/ contient des masques PRE-rescue/PRE-correction pour les 32 sujets ci-dessus
    # (jamais réécrits en place) -> les ignorer en priorité sous peine de recréer un faux
    # signal de sous-segmentation massive (bug détecté et corrigé ici).
    FULL = "outputs/validation_runs/mindboggle101_full"
    mbg = pd.read_csv(f"{FULL}/sota_eval/all_methods_long.csv")
    bdirs = sorted(glob.glob(f"{FULL}/b*/"))
    rescue = f"{FULL}/_rescue_tmp"
    corr25 = f"{FULL}/_corr25"
    rows = []
    for sub in sorted(mbg["subject"].unique()):
        r0 = mbg[mbg.subject == sub].iloc[0]
        gt = r0.gold_mask_path
        if not os.path.exists(gt):
            continue
        masks = {}
        for m in ("SynthStrip", "HD-BET"):
            r = mbg[(mbg.subject == sub) & (mbg.method == m)]
            if len(r) and os.path.exists(r.iloc[0].mask_path):
                masks[m] = r.iloc[0].mask_path
        ours = None
        for d in (rescue, corr25):
            p = f"{d}/{sub}.nii.gz"
            if os.path.exists(p):
                ours = p; break
        if ours is None:
            for bb in bdirs:
                p = f"{bb}{sub}/anat/sub-{sub}_desc-brain_mask.nii.gz"
                if os.path.exists(p):
                    ours = p; break
        if ours:
            masks["Ours"] = ours
        if len(masks) == 3:
            rows.append(("Mindboggle-101", sub, gt, masks))
    return rows

# ---------------------------------------------------------------- resampling ----
def L(p):
    return nib.load(p)

def centroid_world(mask, affine):
    idx = np.argwhere(mask)
    c_ijk = idx.mean(axis=0)
    return nib.affines.apply_affine(affine, c_ijk[None, :])[0]

_GRID_IJK = np.indices(GRID).reshape(3, -1).T.astype(np.float64)
_OFFSETS = (_GRID_IJK - (np.array(GRID) - 1) / 2.0) * SPACING  # (N,3) mm along R,A,S

def to_canonical(data, affine, center_world):
    world = center_world[None, :] + _OFFSETS
    voxel = nib.affines.apply_affine(np.linalg.inv(affine), world)
    out = ndi.map_coordinates(data.astype(np.float32), voxel.T, order=0, mode="constant", cval=0.0)
    return out.reshape(GRID) > 0.5

def process(rows):
    sum_fp = {m: np.zeros(GRID, dtype=np.float32) for m in METHODS}
    sum_fn = {m: np.zeros(GRID, dtype=np.float32) for m in METHODS}
    sum_gt = np.zeros(GRID, dtype=np.float32)
    n = {m: 0 for m in METHODS}
    for k, (bench, sub, gtp, masks) in enumerate(rows):
        gt_img = L(gtp)
        gt_data = np.asarray(gt_img.dataobj) > 0
        center = centroid_world(gt_data, gt_img.affine)
        gt_c = to_canonical(gt_data, gt_img.affine, center)
        sum_gt += gt_c
        for m in METHODS:
            pimg = L(masks[m])
            pdata = np.asarray(pimg.dataobj) > 0
            p_c = to_canonical(pdata, pimg.affine, center)
            sum_fp[m] += (p_c & ~gt_c)
            sum_fn[m] += (gt_c & ~p_c)
            n[m] += 1
        if (k + 1) % 20 == 0:
            print(f"  ... {k+1}/{len(rows)} sujets ({bench})")
    return sum_fp, sum_fn, sum_gt, n

if __name__ == "__main__":
    rows = bobs_manifest() + ibsr18_manifest() + mbg_manifest()
    print(f"manifest: {len(rows)} sujets (BOBs/IBSR18/Mindboggle-101, 3 méthodes complètes)")
    sum_fp, sum_fn, sum_gt, n = process(rows)
    print("n par méthode:", n)

    avg_fp = {m: sum_fp[m] / n[m] for m in METHODS}
    avg_fn = {m: sum_fn[m] / n[m] for m in METHODS}
    avg_gt_outline = sum_gt / len(rows)

    CACHE = "outputs/_fig3_cache.npz"
    np.savez(CACHE,
             **{f"fp_{m}": avg_fp[m] for m in METHODS},
             **{f"fn_{m}": avg_fn[m] for m in METHODS},
             gt=avg_gt_outline, n_total=len(rows))
    print("cache écrit:", CACHE)
