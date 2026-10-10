"""Figure 5 — galerie d'atlas lifespan + histogramme d'âge + mécanisme âge→volume
(style LifespanStrip Fig. 8). Contrairement à LifespanStrip (atlas UNIQUE appris par
un réseau), notre pipeline zero-training pioche dans une bibliothèque de **21 gabarits
externes classiques** (dHCP, UNC-BCP, UNC-Infant, MNI-Infant, SRI24, MIITRA...) choisis
par âge/volume estimé du sujet — ceci illustre la MÉCANIQUE réelle du module de prior.

Panel a : galerie de 9 gabarits réels (coupe axiale) du fœtal/néonatal à l'âgé.
Panel b : histogramme d'âge des sujets BOBs (réel, N=71 ; seul benchmark à âge connu) +
          rappel des effectifs adultes sans âge publié par sujet (IBSR18, Mindboggle-101).
Panel c : mécanisme âge→volume — bande normative (normative_brain_growth.py) avec les
          71 sujets BOBs réels (âge, volume GT) superposés + exemple d'appariement au
          gabarit le plus proche.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np, nibabel as nib, pandas as pd, re, os, sys

sys.path.insert(0, ".")
from scripts.normative_brain_growth import (
    FETAL_BRAIN_VOLUME_CM3, POSTNATAL_BRAIN_VOLUME_CM3, ga_weeks_to_months,
    expected_volume_band, estimate_age_months_from_volume,
)

OUT = "assets/figures"; os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- panel a : galerie ----
ATLASES = [
    ("dhcp_neonatal_ga40", "term\n(~40 wGA)"),
    ("unc_bcp_01m", "1 month"),
    ("unc_bcp_03m", "3 months"),
    ("unc_bcp_06m", "6 months"),
    ("unc_bcp_09m", "9 months"),
    ("unc_infant_1yr", "1 year"),
    ("unc_infant_2yr", "2 years"),
    ("sri24", "adult\n(19-64 y)"),
    ("miitra_older_adult", "older adult\n(65+ y)"),
]

def axial_slice(path, frac=0.46, margin=6):
    img = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    z = int(data.shape[2] * frac)
    sl = np.rot90(data[:, :, z])
    nz = np.argwhere(sl > np.percentile(sl, 60) * 0.05)
    if len(nz):
        r0, c0 = nz.min(axis=0); r1, c1 = nz.max(axis=0)
        r0, c0 = max(r0 - margin, 0), max(c0 - margin, 0)
        r1, c1 = min(r1 + margin, sl.shape[0]), min(c1 + margin, sl.shape[1])
        sl = sl[r0:r1, c0:c1]
    lo, hi = np.percentile(sl[sl > 0], [2, 99]) if (sl > 0).any() else (0, 1)
    return np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1)

# ---------------------------------------------------------------- panel b/c : BOBs ----
def parse_age(session):
    m = re.search(r"ses-(\d+)mo", str(session))
    return int(m.group(1)) if m else None

bobs = pd.read_csv("dataset/manifests/BOBs.csv")
bobs["age_months"] = bobs["session"].apply(parse_age)

# ---------------------------------------------------------------- figure ----
fig = plt.figure(figsize=(13.5, 11), dpi=190)
gs = gridspec.GridSpec(3, 9, figure=fig, height_ratios=[1.05, 1.0, 1.15], hspace=0.55, wspace=0.15)

for j, (name, label) in enumerate(ATLASES):
    ax = fig.add_subplot(gs[0, j])
    img = axial_slice(f"dataset/atlas_library/{name}/MNI152_T1_2mm.nii.gz")
    ax.imshow(img, cmap="gray")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(label, fontsize=8.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
fig.text(0.01, 0.965, "a", fontsize=15, fontweight="bold")
fig.text(0.5, 0.975, "External T1w templates of the atlas library (fetal/neonatal to older adult) — 21 templates in total, 9 shown",
          ha="center", fontsize=10)

# --- panel b : histogramme d'âge ---
axb = fig.add_subplot(gs[1, 0:5])
counts = bobs["age_months"].value_counts().sort_index()
axb.bar(counts.index, counts.values, color="#4dbd6f", edgecolor="black", width=0.7)
axb.set_xlabel("recorded session age (months)", fontsize=10); axb.set_ylabel("imaging visits (n)", fontsize=10)
axb.set_title("BOBs (71 visits, 51 infants) — the only benchmark with recorded per-session age", fontsize=9.5)
axb.set_xticks(range(int(bobs.age_months.min()), int(bobs.age_months.max()) + 1))

axb2 = fig.add_subplot(gs[1, 5:9])
axb2.axis("off")
axb2.text(0.02, 0.85, "Adult benchmarks (no per-subject age published):", fontsize=9.5, fontweight="bold", transform=axb2.transAxes)
axb2.text(0.02, 0.60, "$\\bullet$ IBSR-20: N=20 (20Normals — no age table in the public metadata)",
          fontsize=8.8, transform=axb2.transAxes)
axb2.text(0.02, 0.40, "$\\bullet$ Mindboggle-101: N=101 (HLN/MMRR/NKI-RS/NKI-TRT/OASIS-TRT/Twins cohorts — no per-subject age)",
          fontsize=8.8, transform=axb2.transAxes)
axb2.text(0.02, 0.12, "$\\rightarrow$ age-based matching (panel c) is therefore validated end-to-end on BOBs only;\n"
          "for adults, templates are selected per subject by image-based criteria, with no age weighting.",
          fontsize=8.2, style="italic", transform=axb2.transAxes, wrap=True)
fig.text(0.01, 0.635, "b", fontsize=15, fontweight="bold")

# --- panel c : mécanisme âge→volume ---
axc = fig.add_subplot(gs[2, :])
xs_fetal = sorted(FETAL_BRAIN_VOLUME_CM3)
xs_fetal_m = [ga_weeks_to_months(w) for w in xs_fetal]
xs_post = sorted(POSTNATAL_BRAIN_VOLUME_CM3)
xs = xs_fetal_m + xs_post
xs_dense = np.linspace(min(xs), 24, 300)
lo = [expected_volume_band(x)[0] for x in xs_dense]
hi = [expected_volume_band(x)[1] for x in xs_dense]
axc.fill_between(xs_dense, lo, hi, color="#4dbd6f", alpha=0.18, label="normative band, 70-135% (normative_brain_growth.py)")
mid = [0.5 * (a + b) for a, b in zip(lo, hi)]
axc.plot(xs_dense, mid, color="#2f8f4e", lw=1.6, label="expected volume (midpoint)")

# gabarits disponibles (âge nominal -> marqueurs sur la courbe)
atlas_ages_months = {"term (dHCP GA40)": 0, "1 mo (UNC-BCP)": 1, "3 mo (UNC-BCP)": 3,
                      "6 mo (UNC-BCP)": 6, "9 mo (UNC-BCP)": 9, "1 y (UNC-Infant)": 12,
                      "2 y (UNC-Infant)": 24}
for lab, m in atlas_ages_months.items():
    v = 0.5 * sum(expected_volume_band(m))
    axc.scatter([m], [v], marker="^", s=70, color="#1f4fd6", zorder=5)
axc.scatter([], [], marker="^", s=70, color="#1f4fd6", label="template available in the library")

# sujets BOBs réels
axc.scatter(bobs["age_months"], bobs["mask_volume_cm3"], s=22, color="#d1481f", alpha=0.75,
            zorder=6, label="BOBs visits (recorded age, reference-mask volume)")

# exemple d'appariement (un sujet illustratif)
ex = bobs.sort_values("age_months").iloc[len(bobs)//2]
nearest_m = min(atlas_ages_months.values(), key=lambda m: abs(m - ex.age_months))
axc.annotate("", xy=(nearest_m, 0.5 * sum(expected_volume_band(nearest_m))),
             xytext=(ex.age_months, ex.mask_volume_cm3),
             arrowprops=dict(arrowstyle="->", color="black", lw=1.3, ls="--"))
axc.text(ex.age_months + 0.6, ex.mask_volume_cm3 - 160,
          f"{ex['subject']}\n{ex.age_months} months, {ex.mask_volume_cm3:.0f} cm$^3$\n$\\rightarrow$ nearest-age template ({nearest_m} mo), illustration",
          fontsize=7.3, va="top",
          bbox=dict(boxstyle="round", fc="white", ec="#999999", alpha=0.9))

axc.set_xlim(-1, 25); axc.set_xlabel("age (postnatal months)", fontsize=10)
axc.set_ylabel("brain volume (cm$^3$)", fontsize=10)
axc.set_title("Normative volume band, available templates and BOBs visits", fontsize=9.5)
axc.legend(fontsize=7.6, loc="upper left", framealpha=0.9)
axc.grid(alpha=0.25)
fig.text(0.01, 0.335, "c", fontsize=15, fontweight="bold")

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig5_atlas_gallery.{ext}", bbox_inches="tight", pad_inches=0.1, facecolor="white")
print("écrit:", f"{OUT}/fig5_atlas_gallery.png")
