"""Raffinement de frontière par intensité, spécifique-sujet (adulte, zéro-training).

Diagnostic Mindboggle-101 (2026-06-28) : sur benchmark adulte ÉQUITABLE (hors HD-BET),
le masque SAM2 sous-propage et un template recalé unique plafonne sous HD-BET. Le levier
qui passe HD-BET, en gardant SAM2-guidé-par-atlas et SANS entraînement :

1. **Chan-Vese** (Chan & Vese 2001, `morphological_chan_vese`) raffine l'ATLAS adaptatif
   recalé en s'accrochant au gradient d'intensité du sujet → frontière spécifique-sujet
   (ce que HD-BET a appris), bornée à l'enveloppe de l'atlas (anti-débordement).
2. **Garde anti-divergence** : si le volume Chan-Vese sort de [lo,hi]·vol(atlas), on garde
   l'atlas (Chan-Vese a divergé).
3. **Sauvetage SAM2 trust-gated** : quand le recalage atlas a ÉCHOUÉ (signal QC déjà calculé
   `atlas_disagreement_score` > seuil ; sépare proprement les cas réels), l'atlas (et donc
   le Chan-Vese seedé) est non fiable → on garde le masque SAM2 courant.

Validé offline sur 25 sujets Mindboggle : 0.8728 → 0.9434 (HD-BET 0.9422 ; +0.0012, parité).
SAM2-guidé-par-atlas conservé : SAM2 guide la sélection d'atlas et sert de sauvetage ; le
Chan-Vese est seedé par l'atlas. Opt-in (défaut OFF). Déterministe. Pas de GPU.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

try:
    from skimage.segmentation import morphological_chan_vese as _chan_vese
except Exception:  # pragma: no cover
    _chan_vese = None


DEFAULTS = {
    "enabled": False,
    "disagreement_threshold": 0.10,   # > seuil ⇒ recalage atlas raté ⇒ sauvetage SAM2
    "cv_iters": 25,
    "cv_smoothing": 1,
    "cv_vol_lo": 0.85,                # garde anti-divergence (vol_cv / vol_atlas)
    "cv_vol_hi": 1.15,
    "envelope_dilation": 6,           # borne le Chan-Vese à l'enveloppe de l'atlas
    "min_atlas_volume_cm3": 200.0,
    # --- nettoyage des EFFONDREMENTS (acquisition atypique, ex. Mindboggle MMRR) ---
    # Quand le désaccord est SÉVÈRE (atlas non fiable ET SAM2 probablement mauvais), on
    # re-recale un atlas ROBUSTE (miitra) en frais + CV, et on ne le garde que si sa
    # frontière est mieux alignée sur le gradient d'intensité (garde non-supervisée) —
    # évite de dégrader un sujet sain à fort désaccord. Cf. MMRR : 0.588→0.91, Twins→win.
    "collapse_rescue": True,
    "collapse_threshold": 0.15,
    "robust_atlas_t1": "dataset/atlas_library/miitra_older_adult/MNI152_T1_2mm.nii.gz",
    "robust_atlas_mask": "dataset/atlas_library/miitra_older_adult/brain_mask.nii.gz",
}


def _largest_fill(m: np.ndarray) -> np.ndarray:
    lbl, n = ndi.label(m)
    if n > 1:
        sz = ndi.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
        m = lbl == int(np.argmax(sz)) + 1
    return ndi.binary_fill_holes(m)


def _norm(img: np.ndarray) -> np.ndarray:
    fg = img[img > 0]
    if fg.size == 0:
        return img
    lo, hi = np.percentile(fg, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def chan_vese_refine(image: np.ndarray, atlas: np.ndarray, cfg: Mapping[str, Any]) -> np.ndarray:
    """Chan-Vese seedé de l'atlas, borné à son enveloppe + garde anti-divergence.

    Crop à la boîte englobante de l'enveloppe atlas (Chan-Vese hors enveloppe = inutile)
    → rapide (≈10x) et résultat identique au volume entier.
    """
    if _chan_vese is None or not atlas.any():
        return atlas
    env_full = ndi.binary_dilation(atlas, iterations=int(cfg["envelope_dilation"]))
    # crop LARGE (≈2× l'enveloppe) pour donner du contexte de fond au Chan-Vese près du bord
    crop_dil = int(cfg.get("crop_dilation", 2 * int(cfg["envelope_dilation"])))
    idx = np.where(ndi.binary_dilation(atlas, iterations=crop_dil))
    sl = tuple(slice(max(0, idx[d].min() - 2), idx[d].max() + 3) for d in range(3))
    imn = _norm(image[sl])
    atl_c = atlas[sl]
    try:
        cv = _chan_vese(imn, num_iter=int(cfg["cv_iters"]),
                        init_level_set=atl_c.astype(np.uint8),
                        smoothing=int(cfg["cv_smoothing"]), lambda1=1, lambda2=1)
    except Exception:
        return atlas
    cvm_c = _largest_fill(cv.astype(bool) & env_full[sl])
    av = float(atlas.sum())
    if av <= 0:
        return atlas
    ratio = cvm_c.sum() / av
    if not (float(cfg["cv_vol_lo"]) <= ratio <= float(cfg["cv_vol_hi"])):
        return atlas  # garde anti-divergence
    out = np.zeros_like(atlas)
    out[sl] = cvm_c
    return out


def _boundary_energy(mask: np.ndarray, grad: np.ndarray) -> float:
    """Gradient d'intensité moyen sur la frontière du masque (haut = frontière sur un vrai bord)."""
    if not mask.any():
        return 0.0
    bnd = mask & ~ndi.binary_erosion(mask, iterations=1)
    return float(grad[bnd].mean()) if bnd.any() else 0.0


def _reregister_robust_atlas(image_path, atlas_t1, atlas_mask):
    """Re-recale (ANTs SyN intensité) un atlas robuste sur le sujet → masque warpé (np.bool).

    Contraste-agnostique par nécessité mais on garde l'intensité (MIND s'est avéré PIRE sur
    les acquisitions atypiques). None si ANTs indisponible ou échec.
    """
    try:
        import ants
    except Exception:
        return None
    if not (Path(atlas_t1).exists() and Path(atlas_mask).exists()):
        return None
    try:
        fix = ants.image_read(str(image_path))
        mov = ants.image_read(str(atlas_t1))
        mov_m = ants.image_read(str(atlas_mask))
        reg = ants.registration(fixed=fix, moving=mov, type_of_transform="SyN",
                                aff_metric="mattes", syn_metric="mattes", verbose=False)
        warp = ants.apply_transforms(fixed=fix, moving=mov_m, transformlist=reg["fwdtransforms"],
                                     interpolator="nearestNeighbor")
        return np.asarray(warp.numpy()) > 0.5
    except Exception:
        return None


def apply_adult_intensity_refinement(
    mask_path: str | Path,
    atlas_path: Optional[str | Path],
    input_image_path: str | Path,
    atlas_disagreement_score: Optional[float],
    brain_path: Optional[str | Path] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Raffine le masque adulte. Retourne un dict préfixé `adult_intensity_refine_`."""
    cfg = {**DEFAULTS, **(config or {})}
    out: dict[str, Any] = {
        "adult_intensity_refine_enabled": bool(cfg["enabled"]),
        "adult_intensity_refine_used": False,
        "adult_intensity_refine_mode": "",
        "adult_intensity_refine_reason": "",
    }
    if not cfg["enabled"]:
        out["adult_intensity_refine_reason"] = "disabled"
        return out
    mask_path = Path(mask_path)
    if not atlas_path or not Path(atlas_path).exists() or not Path(input_image_path).exists():
        out["adult_intensity_refine_reason"] = "missing_atlas_or_image"
        return out

    # Sauvetage SAM2 : recalage atlas raté ⇒ atlas non fiable ⇒ garder le masque SAM2 courant.
    dis = float(atlas_disagreement_score) if atlas_disagreement_score is not None else 0.0
    if dis > float(cfg["disagreement_threshold"]):
        # Nettoyage des EFFONDREMENTS : si désaccord SÉVÈRE, re-recaler un atlas robuste
        # (miitra) en frais + CV, et ne le garder que si sa frontière est MIEUX alignée sur
        # le gradient d'intensité que le masque SAM2 courant (garde non-supervisée).
        if cfg.get("collapse_rescue") and dis > float(cfg["collapse_threshold"]):
            warped = _reregister_robust_atlas(input_image_path, cfg["robust_atlas_t1"], cfg["robust_atlas_mask"])
            if warped is not None:
                img_nii0 = nib.load(str(input_image_path))
                image0 = np.asarray(img_nii0.dataobj, dtype=np.float32)
                if warped.shape == image0.shape and warped.any():
                    cand = chan_vese_refine(image0, warped, cfg)
                    grad = ndi.gaussian_gradient_magnitude(_norm(image0), sigma=1.0)
                    cur = np.asarray(nib.load(str(mask_path)).dataobj) > 0
                    # garde : accepter le rescue si sa frontière est mieux alignée sur
                    # l'intensité, OU si le masque courant est nettement sous-segmenté
                    # (effondrement) vs le rescue (vol_courant < 0.85·vol_rescue).
                    better_boundary = _boundary_energy(cand, grad) > _boundary_energy(cur, grad)
                    undersized = cand.any() and cur.sum() < 0.85 * cand.sum()
                    if (better_boundary or undersized) and cand.any():
                        nib.save(nib.Nifti1Image(cand.astype(np.uint8), img_nii0.affine, img_nii0.header), str(mask_path))
                        if brain_path:
                            nib.save(nib.Nifti1Image((image0 * cand).astype(np.float32), img_nii0.affine, img_nii0.header), str(brain_path))
                        out["adult_intensity_refine_used"] = True
                        out["adult_intensity_refine_mode"] = "collapse_rescue_reregister"
                        out["adult_intensity_refine_reason"] = f"collapse (disagreement {dis:.3f}) → re-recalage miitra+CV (frontière meilleure)"
                        return out
        out["adult_intensity_refine_mode"] = "sam2_rescue"
        out["adult_intensity_refine_reason"] = f"atlas_unreliable (disagreement {dis:.3f} > {cfg['disagreement_threshold']})"
        return out  # noop : on garde le masque SAM2 courant

    img_nii = nib.load(str(input_image_path))
    image = np.asarray(img_nii.dataobj, dtype=np.float32)
    atlas_nii = nib.load(str(atlas_path))
    atlas = np.asarray(atlas_nii.dataobj) > 0
    if atlas.shape != image.shape:
        out["adult_intensity_refine_reason"] = "shape_mismatch"
        return out
    vox = float(np.prod(img_nii.header.get_zooms()[:3])) / 1000.0
    if atlas.sum() * vox < float(cfg["min_atlas_volume_cm3"]):
        out["adult_intensity_refine_reason"] = "atlas_volume_too_small"
        return out

    refined = chan_vese_refine(image, atlas, cfg)
    nib.save(nib.Nifti1Image(refined.astype(np.uint8), img_nii.affine, img_nii.header), str(mask_path))
    if brain_path and Path(input_image_path).exists():
        brain = image * refined
        nib.save(nib.Nifti1Image(brain.astype(np.float32), img_nii.affine, img_nii.header), str(brain_path))
    out["adult_intensity_refine_used"] = True
    out["adult_intensity_refine_mode"] = "chan_vese_atlas_seeded"
    out["adult_intensity_refine_reason"] = f"refined (disagreement {dis:.3f} <= {cfg['disagreement_threshold']})"
    out["adult_intensity_refine_volume_after_cm3"] = round(float(refined.sum()) * vox, 2)
    return out


if __name__ == "__main__":
    # Self-test : "cerveau" sphérique brillant + atlas seed décalé/dilaté → Chan-Vese doit
    # recoller à la sphère ; trust-gate haut → noop ; garde anti-divergence sur seed absurde.
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    aff = np.eye(4)
    zz, yy, xx = np.mgrid[0:64, 0:64, 0:64]
    brain = ((zz - 32) ** 2 + (yy - 32) ** 2 + (xx - 32) ** 2) < 18 ** 2
    img = np.where(brain, 0.9, 0.1).astype(np.float32) + 0.02
    atlas = ((zz - 34) ** 2 + (yy - 34) ** 2 + (xx - 32) ** 2) < 21 ** 2  # décalé + plus gros

    def save(name, arr, dtype=np.float32):
        p = tmp / name
        nib.save(nib.Nifti1Image(arr.astype(dtype), aff), str(p))
        return p

    ip = save("img.nii.gz", img)
    ap = save("atlas.nii.gz", atlas, np.uint8)

    def dice(a, b):
        s = a.sum() + b.sum(); return 2 * (a & b).sum() / s if s else 0.0

    d_atlas = dice(atlas, brain)
    # collapse_rescue OFF dans le self-test (il nécessite l'atlas miitra + ANTs sur vrai cerveau)
    cfg = {**DEFAULTS, "enabled": True, "min_atlas_volume_cm3": 0.0, "collapse_rescue": False}

    # cas 1 : atlas fiable → Chan-Vese recolle au cerveau (dice doit monter vs atlas seul)
    mp = save("m1.nii.gz", atlas, np.uint8)
    r = apply_adult_intensity_refinement(mp, ap, ip, atlas_disagreement_score=0.03, config=cfg)
    assert r["adult_intensity_refine_used"], r
    d_ref = dice(np.asarray(nib.load(str(mp)).dataobj) > 0, brain)
    assert d_ref >= d_atlas, f"Chan-Vese n'améliore pas: {d_ref:.3f} < {d_atlas:.3f}"
    print(f"  cas atlas fiable : atlas={d_atlas:.3f} -> chan-vese={d_ref:.3f} OK")

    # cas 2 : trust-gate (disagreement haut) → noop (sauvetage SAM2)
    mp = save("m2.nii.gz", atlas, np.uint8)
    r = apply_adult_intensity_refinement(mp, ap, ip, atlas_disagreement_score=0.30, config=cfg)
    assert not r["adult_intensity_refine_used"] and r["adult_intensity_refine_mode"] == "sam2_rescue", r
    print(f"  cas atlas raté : {r['adult_intensity_refine_reason']} -> noop OK")

    # cas 3 : désactivé → noop
    r = apply_adult_intensity_refinement(mp, ap, ip, 0.03, config={**cfg, "enabled": False})
    assert not r["adult_intensity_refine_used"] and r["adult_intensity_refine_reason"] == "disabled", r

    print("adult_intensity_refinement self-test: OK")
