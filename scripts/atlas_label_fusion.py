"""Fusion d'étiquettes multi-atlas robuste (levier #4.1).

Remplace la sélection "meilleur atlas seul" (winner-take-all) par un CONSENSUS de
plusieurs atlas recalés. Double bénéfice (cf. RESEARCH_SOTA_STRATEGY.md §3.1) :

1. **Déterminisme** : le winner-take-all bascule entre deux atlas de scores proches
   au moindre bruit de registration → gros swing Dice (~0.05-0.14 observé). Un
   consensus pondéré ne bascule pas (une permutation du top-K ne change presque
   rien à la moyenne).
2. **Précision** : les erreurs de warping sont spatialement corrélées ; un
   consensus pondéré (et a fortiori STAPLE) corrige les erreurs d'un atlas isolé.

Méthodes :
- ``weighted_vote_fusion`` : vote pondéré par voxel (poids = score de sélection).
  Robuste, déterministe, sans dépendance lourde.
- ``staple_fusion`` : Simultaneous Truth And Performance Level Estimation
  (Warfield et al., IEEE TMI 2004) — EM estimant sensibilité/spécificité par atlas.

Module sans dépendance lourde (numpy + scipy.ndimage), testable sans GPU.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _largest_component(mask: np.ndarray) -> np.ndarray:
    lbl, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def _clean(mask: np.ndarray, fill: bool = True, largest: bool = True) -> np.ndarray:
    m = mask.astype(bool)
    if largest:
        m = _largest_component(m)
    if fill:
        m = ndi.binary_fill_holes(m)
    return m


def weighted_vote_fusion(
    masks: list[np.ndarray],
    weights: list[float] | None = None,
    threshold: float = 0.5,
    clean: bool = True,
) -> np.ndarray:
    """Vote pondéré par voxel sur des masques binaires de même forme.

    Args:
        masks: liste de masques binaires (même shape), déjà recalés sur le sujet.
        weights: poids par atlas (ex. score de sélection). Défaut = uniformes.
        threshold: seuil sur la probabilité de consensus [0,1].
        clean: nettoyage morphologique (plus grande composante + remplissage).

    Returns:
        masque binaire fusionné (uint8).
    """
    if not masks:
        raise ValueError("aucun masque à fusionner")
    shp = masks[0].shape
    for m in masks:
        if m.shape != shp:
            raise ValueError(f"formes incohérentes: {m.shape} != {shp}")
    n = len(masks)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError("len(weights) != len(masks)")
    w = np.asarray(weights, dtype=np.float64)
    wsum = float(w.sum())
    if wsum <= 0:
        w = np.ones(n); wsum = float(n)
    prob = np.zeros(shp, dtype=np.float64)
    for wi, m in zip(w, masks):
        prob += wi * (np.asarray(m) > 0)
    prob /= wsum
    fused = prob >= float(threshold)
    if clean:
        fused = _clean(fused)
    return fused.astype(np.uint8)


def consensus_probability(
    masks: list[np.ndarray], weights: list[float] | None = None
) -> np.ndarray:
    """Carte de probabilité de consensus pondéré [0,1] (sans seuillage)."""
    n = len(masks)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    wsum = float(w.sum()) or float(n)
    prob = np.zeros(masks[0].shape, dtype=np.float64)
    for wi, m in zip(w, masks):
        prob += wi * (np.asarray(m) > 0)
    return prob / wsum


def staple_fusion(
    masks: list[np.ndarray],
    max_iter: int = 30,
    tol: float = 1e-4,
    threshold: float = 0.5,
    clean: bool = True,
) -> np.ndarray:
    """STAPLE (Warfield et al. 2004) — EM sensibilité/spécificité par atlas.

    Estime itérativement la "vérité" latente W et, pour chaque atlas, sa
    sensibilité p (vrai positif) et spécificité q (vrai négatif). Déterministe
    (init par vote majoritaire). Implémentation globale (non spatiale) suffisante
    pour un masque cerveau ; extension Spatial STAPLE possible par fenêtres.
    """
    D = np.stack([(np.asarray(m) > 0).astype(np.float64).ravel() for m in masks], axis=0)
    R, V = D.shape  # R atlas, V voxels
    gamma = float(D.mean())  # prior P(W=1)
    W = D.mean(axis=0)       # init = vote moyen
    p = np.full(R, 0.99); q = np.full(R, 0.99)
    for _ in range(max_iter):
        # E-step : a = P(données|W=1)*gamma ; b = P(données|W=0)*(1-gamma)
        # log pour stabilité
        log_a = np.log(gamma + 1e-12) + (D * np.log(p[:, None] + 1e-12) +
                                          (1 - D) * np.log(1 - p[:, None] + 1e-12)).sum(0)
        log_b = np.log(1 - gamma + 1e-12) + (D * np.log(1 - q[:, None] + 1e-12) +
                                              (1 - D) * np.log(q[:, None] + 1e-12)).sum(0)
        mx = np.maximum(log_a, log_b)
        Wn = np.exp(log_a - mx) / (np.exp(log_a - mx) + np.exp(log_b - mx) + 1e-12)
        # M-step
        sw = Wn.sum(); siw = (1 - Wn).sum()
        p_new = (D * Wn).sum(1) / (sw + 1e-12)
        q_new = ((1 - D) * (1 - Wn)).sum(1) / (siw + 1e-12)
        if (np.abs(Wn - W).max() < tol and
                np.abs(p_new - p).max() < tol and np.abs(q_new - q).max() < tol):
            W, p, q = Wn, p_new, q_new
            break
        W, p, q = Wn, p_new, q_new
    fused = (W.reshape(masks[0].shape) >= float(threshold))
    if clean:
        fused = _clean(fused)
    return fused.astype(np.uint8)


def _selftest() -> None:
    rng = np.random.default_rng(0)
    # Vérité : sphère centrée dans un volume 40³.
    zz, yy, xx = np.mgrid[0:40, 0:40, 0:40]
    truth = ((zz - 20) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2) < 12 ** 2

    def dice(a, b):
        a = a > 0; b = b > 0
        return 2 * (a & b).sum() / (a.sum() + b.sum())

    # 5 atlas bruités + 1 atlas franchement mauvais (décalé) = outlier corrélé.
    masks = []
    for _ in range(5):
        m = truth.copy()
        flip = rng.random(truth.shape) < 0.03
        m = m ^ flip
        masks.append(m)
    bad = ((zz - 30) ** 2 + (yy - 30) ** 2 + (xx - 20) ** 2) < 12 ** 2  # décalé
    masks.append(bad)

    # 1. Vote pondéré : poids haut aux bons, bas au mauvais → proche de truth.
    weights = [1.0] * 5 + [0.2]
    fused_wv = weighted_vote_fusion(masks, weights)
    d_wv = dice(fused_wv, truth)
    d_bad = dice(bad, truth)
    assert d_wv > 0.9, f"vote pondéré Dice={d_wv:.3f}"
    assert d_wv > d_bad + 0.2, "le consensus doit battre l'atlas outlier"

    # 2. STAPLE : doit down-weighter l'outlier automatiquement et battre le mauvais.
    fused_st = staple_fusion(masks)
    d_st = dice(fused_st, truth)
    assert d_st > 0.9, f"STAPLE Dice={d_st:.3f}"

    # 3. DÉTERMINISME / STABILITÉ : permuter l'ordre + retirer/ajouter l'outlier
    #    ne doit presque pas changer le consensus (anti-bascule winner-take-all).
    import itertools
    perm = list(reversed(masks))
    fused_perm = weighted_vote_fusion(perm, list(reversed(weights)))
    assert dice(fused_wv, fused_perm) > 0.999, "consensus non invariant à la permutation"

    # 4. Idempotence sur masques identiques.
    same = [truth.copy() for _ in range(3)]
    assert dice(weighted_vote_fusion(same), truth) > 0.999

    print("atlas_label_fusion: self-tests OK")
    print(f"  vote pondéré Dice={d_wv:.3f} | STAPLE Dice={d_st:.3f} | atlas outlier seul Dice={d_bad:.3f}")


if __name__ == "__main__":
    _selftest()
