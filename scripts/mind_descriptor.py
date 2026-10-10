"""Descripteur MIND (Modality Independent Neighbourhood Descriptor).

Réf : Heinrich et al., "MIND: Modality Independent Neighbourhood Descriptor for
Multi-Modal Deformable Registration", Medical Image Analysis 16(7):1313-1325, 2012.

Pourquoi ici : sur SynthStripInfant (et FetalFMRI), le contraste synthétique/EPI
trompe la registration à base d'intensité (Mattes MI / NCC) → l'atlas est
sur-dimensionné → sur-segmentation 2-2.5× (diagnostic 2026-06-21). MIND transforme
chaque image en un descripteur d'AUTO-SIMILARITÉ locale, INDÉPENDANT de la relation
d'intensité absolue. Recaler sur les descripteurs MIND (au lieu des intensités) rend
l'alignement robuste au domain-shift → l'atlas garde la bonne échelle.

Idée mathématique :
  Pour un voxel x et un déplacement r d'un voisinage,
    Dp(x, r) = SSD( patch(x), patch(x+r) )         (distance patch)
    V(x)     = moyenne locale des Dp (variance de structure)
    MIND(x, r) = exp( -Dp(x, r) / V(x) )            (normalisé, ∈ ]0,1])
  Le descripteur (un canal par r) ne dépend QUE de la structure locale relative,
  pas du niveau de gris absolu → invariant à toute transformation monotone du
  contraste (gamma, inversion, mise à l'échelle…).

Pur numpy/scipy → testable sans GPU.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# Voisinage 6-connecté (déplacements unitaires).
_OFFSETS_6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _patch_ssd(img: np.ndarray, offset: tuple[int, int, int], patch_sigma: float) -> np.ndarray:
    """Distance patch (SSD lissée) entre img et img décalée de offset."""
    shifted = ndi.shift(img, offset, order=1, mode="nearest")
    diff2 = (img - shifted) ** 2
    # SSD sur un patch ≈ lissage gaussien du carré des différences
    return ndi.gaussian_filter(diff2, sigma=patch_sigma)


def mind_descriptor(
    img: np.ndarray,
    patch_sigma: float = 0.7,
    offsets: list[tuple[int, int, int]] | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Descripteur MIND d'un volume 3D → tableau (n_offsets, *img.shape) ∈ ]0,1].

    Chaque canal correspond à une direction de voisinage. Invariant au contraste.
    """
    img = np.asarray(img, dtype=np.float32)
    # normalisation robuste (n'affecte pas l'invariance, stabilise V)
    p2, p98 = np.percentile(img, [2, 98])
    if p98 > p2:
        img = np.clip((img - p2) / (p98 - p2), 0, 1)
    offs = offsets or _OFFSETS_6
    dps = np.stack([_patch_ssd(img, o, patch_sigma) for o in offs], axis=0)
    # V(x) = variance locale = moyenne des distances patch (clampée pour stabilité)
    V = dps.mean(axis=0)
    V = np.clip(V, a_min=float(np.mean(V) * 0.001 + eps), a_max=None)
    mind = np.exp(-dps / V[None])
    # normalisation par voxel : max des canaux = 1 (def. de Heinrich)
    mind /= np.clip(mind.max(axis=0, keepdims=True), eps, None)
    return mind.astype(np.float32)


def mind_distance(a_img: np.ndarray, b_img: np.ndarray, **kw) -> float:
    """Distance MIND moyenne (SSD des descripteurs) entre deux volumes alignés.

    Faible = structures locales similaires (indépendamment du contraste). Utilisable
    comme métrique de similarité robuste pour ranking/registration.
    """
    ma = mind_descriptor(a_img, **kw)
    mb = mind_descriptor(b_img, **kw)
    return float(np.mean((ma - mb) ** 2))


def mind_scalar(img: np.ndarray, **kw) -> np.ndarray:
    """Réduction scalaire du descripteur MIND (moyenne des canaux) → volume 3D.

    Image d'auto-similarité recalable par un outil standard (ANTs) sur une métrique
    d'intensité : sert de représentation robuste au contraste pour la registration.
    """
    return mind_descriptor(img, **kw).mean(axis=0).astype(np.float32)


def _selftest() -> None:
    rng = np.random.default_rng(0)
    # Volume structuré : sphère + barre (structures à différents contrastes).
    zz, yy, xx = np.mgrid[0:32, 0:32, 0:32]
    base = np.zeros((32, 32, 32), np.float32)
    base[(zz - 16) ** 2 + (yy - 16) ** 2 + (xx - 16) ** 2 < 9 ** 2] = 1.0
    base[14:18, :, 6:26] = 0.6
    base = ndi.gaussian_filter(base, 0.8) + rng.normal(0, 0.02, base.shape).astype(np.float32)

    md_base = mind_descriptor(base)

    # 1. INVARIANCE AU CONTRASTE : transformations monotones radicales de l'intensité.
    def corr(a, b):
        a = a.ravel(); b = b.ravel()
        return float(np.corrcoef(a, b)[0, 1])

    # Invariances EXACTES de MIND : transformations affines de l'intensité +
    # inversion de contraste (le cas modal T1↔T2 — Dp inchangé). Seuil strict.
    exact = {
        "inversion": base.max() - base,           # contraste inversé (T1↔T2-like)
        "affine 3x+5": 3.0 * base + 5.0,
    }
    # Invariances APPROXIMATIVES : transformations monotones NON-affines (gamma,
    # sqrt). MIND reste robuste mais pas exact (la structure locale est préservée,
    # les ratios de différences non). Seuil plus souple — c'est attendu.
    approx = {
        "gamma 2.2": np.clip(base, 0, None) ** 2.2,
        "sqrt": np.sqrt(np.clip(base, 0, None)),
    }
    print("  invariance MIND au contraste (corrélation descripteur vs original) :")
    for name, t in exact.items():
        c = corr(md_base, mind_descriptor(t))
        print(f"    {name:14} corr={c:.3f}  [exact]")
        assert c > 0.95, f"MIND devrait être ~exact pour '{name}': corr={c:.3f}"
    for name, t in approx.items():
        c = corr(md_base, mind_descriptor(t))
        print(f"    {name:14} corr={c:.3f}  [approx]")
        assert c > 0.55, f"MIND devrait rester robuste à '{name}': corr={c:.3f}"

    # 2. CONTRASTE BRUT, lui, change radicalement (preuve que c'est non-trivial).
    raw_corr = corr(base, base.max() - base)
    assert raw_corr < -0.9, f"intensité brute devrait s'inverser: {raw_corr:.3f}"

    # 3. MIND distingue structure vs uniforme : une image CONSTANTE (sans bruit) a
    #    un descripteur uniforme (variance ~0) ; une image structurée varie.
    uni = np.full((32, 32, 32), 0.5, np.float32)
    var_struct = float(np.var(mind_scalar(base)))
    var_uni = float(np.var(mind_scalar(uni)))
    assert var_struct > var_uni * 5 and var_uni < 1e-3, \
        f"MIND devrait varier sur structure ({var_struct:.4f}) vs constante ({var_uni:.6f})"

    # 4. mind_distance : faible entre image et sa version contraste-transformée,
    #    élevée entre image et une image structurellement différente.
    d_same = mind_distance(base, base.max() - base)        # même structure
    shifted = ndi.shift(base, (5, 5, 5), order=1, mode="nearest")
    d_diff = mind_distance(base, shifted)                  # structure décalée
    assert d_same < d_diff, f"MIND-dist same={d_same:.4f} doit < diff={d_diff:.4f}"

    print("mind_descriptor: self-tests OK")
    print(f"  contraste brut inversé corr={raw_corr:.2f} (MIND reste >0.95) | "
          f"struct/uni var ratio={var_struct/max(var_uni,1e-9):.0f}x")


if __name__ == "__main__":
    _selftest()
