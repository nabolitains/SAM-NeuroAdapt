"""Estimation de la TAILLE du cerveau d'un sujet (sans atlas, sans GPU).

Cœur de la vision "appariement atlas↔sujet par volume/âge" : pour les datasets sans
âge connu (SynthStripInfant, FetalFMRI), on estime grossièrement le volume cérébral
depuis l'image N4 (skull-strip morphologique rapide), puis on choisit l'atlas dont
le volume/âge est le PLUS PROCHE — au lieu de recaler un atlas mal dimensionné.

L'estimation n'a pas besoin d'être parfaite : elle doit distinguer un cerveau de
~326 cm³ d'un de ~1000 cm³ pour orienter la sélection d'atlas. Un facteur de
calibration corrige le biais systématique (scalp résiduel).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

# Facteur de calibration : le skull-strip morphologique (érosion adaptative) sur-
# estime (scalp/dure-mère résiduels). Facteur médian GT/raw calibré sur 16 sujets
# SynthStripInfant réels (corr 0.73, erreur médiane ~16 %). Robuste (sans plancher).
DEFAULT_CALIBRATION = 0.573


def _largest_cc(m: np.ndarray) -> np.ndarray:
    lbl, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def _rough_brain_mask(data: np.ndarray, erode_iter: int = 4) -> np.ndarray:
    """Skull-strip morphologique grossier et ROBUSTE.

    L'érosion à pas fixe détruit les petits cerveaux (masque vidé). On érode donc
    de façon ADAPTATIVE : on réduit le nombre d'itérations tant que l'érosion ne
    laisse pas une composante d'au moins 25 % du foreground initial. Garantit un
    masque non dégénéré sur toute la gamme de tailles.
    """
    fg = data[data > 0]
    if fg.size < 1000:
        return np.zeros_like(data, dtype=bool)
    try:
        from skimage.filters import threshold_otsu
        thr = float(threshold_otsu(fg))
    except Exception:
        thr = float(np.percentile(fg, 55))
    base = ndi.binary_fill_holes(_largest_cc(data > thr))     # foreground principal
    base_vol = float(base.sum())
    if base_vol < 1.0:
        return np.zeros_like(data, dtype=bool)
    # érosion adaptative : la plus forte qui garde ≥25 % du foreground
    for it in range(int(erode_iter), 0, -1):
        er = _largest_cc(ndi.binary_erosion(base, iterations=it))
        if er.sum() >= 0.25 * base_vol:
            m = ndi.binary_dilation(er, iterations=it)
            return ndi.binary_fill_holes(_largest_cc(m & base if base.any() else m))
    # aucune érosion ne tient → foreground avec ouverture légère (anti-scalp fin)
    return ndi.binary_fill_holes(_largest_cc(ndi.binary_opening(base, iterations=1)))


def estimate_brain_volume_cm3(
    image_path: str | Path,
    calibration: float = DEFAULT_CALIBRATION,
) -> float:
    """Volume cérébral estimé (cm³) depuis une image N4. None-safe → 0.0 si échec."""
    img = nib.load(str(image_path))
    data = np.asarray(img.dataobj, dtype=np.float32).squeeze()
    mask = _rough_brain_mask(data)
    if not mask.any():
        return 0.0
    vox_cm3 = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    raw = float(mask.sum()) * vox_cm3
    return raw * float(calibration)


def estimate_age_months_from_image(image_path: str | Path, calibration: float = DEFAULT_CALIBRATION) -> float | None:
    """Âge estimé (mois postnatals ; négatif = fœtal) depuis la taille du cerveau.

    Brise la circularité de l'inférence d'âge par atlas (P19) : l'âge vient de la
    TAILLE réelle du sujet, indépendamment de tout atlas.
    """
    vol = estimate_brain_volume_cm3(image_path, calibration)
    if vol <= 0:
        return None
    try:
        from scripts.normative_brain_growth import estimate_age_months_from_volume
    except Exception:
        from normative_brain_growth import estimate_age_months_from_volume
    return float(estimate_age_months_from_volume(vol))


if __name__ == "__main__":
    # Validation sur sujets réels variés (si présents) : l'estimation doit corréler
    # avec le GT et distinguer petit/gros.
    import glob, sys

    pairs = []
    # SynthStripInfant : petit→gros
    for sid, gtv in [("infant_t1_21", 326), ("infant_t1_15", 527), ("infant_t1_20", 1015)]:
        n4 = glob.glob(f"outputs/*synthstripinfant*/{sid}/anat/*desc-n4_T1w.nii.gz")
        gt = glob.glob(f"outputs/*synthstripinfant*/{sid}/anat/source/*gold-brain_mask.nii.gz")
        if n4 and gt:
            pairs.append((sid, n4[0], gt[0]))

    if not pairs:
        print("aucun sujet de validation trouvé"); sys.exit(0)

    print(f"{'sujet':16}{'GT':>7}{'estimé':>9}{'est/GT':>8}")
    ratios = []
    for sid, n4, gt in pairs:
        gimg = nib.load(gt); gtv = float((gimg.get_fdata() > 0).sum() * np.prod(gimg.header.get_zooms()[:3]) / 1000)
        est = estimate_brain_volume_cm3(n4)
        ratios.append(est / gtv)
        age = estimate_age_months_from_image(n4)
        print(f"{sid:16}{gtv:>7.0f}{est:>9.0f}{est/gtv:>8.2f}   âge≈{age:.1f} mo" if age is not None else "")
    print(f"\nmédiane est/GT = {np.median(ratios):.2f} (cible ~1.0 ; ajuster DEFAULT_CALIBRATION si besoin)")
