"""SAM2 multi-vue (axial + coronal + sagittal) + fusion STAPLE.

SAM2BrainExtractor propage TOUJOURS le long de l'axe Z (slices axiales). Le contour
est donc moins précis sur les bords perpendiculaires à cette vue → erreur de bord
~2× HD-BET sur LPBA40 (diagnostic 2026-06-21 : FP+FN ≈ 104 cm³ vs 52). Idée : faire
voir à SAM2 les 3 orientations en présentant le volume réorienté (chaque axe placé
en position Z), puis fusionner les 3 masques (consensus STAPLE) → contour 3D
cohérent, les erreurs propres à chaque vue s'annulent.

Conception NON-INVASIVE : on n'altère ni le notebook ni la classe SAM2BrainExtractor.
On orchestre 3 appels à ``extract_brain`` sur des copies réorientées de l'image ET
des atlas de guidage, puis on dé-réoriente et on fusionne. ``extract_fn`` est
injectable → testable sans GPU. Opt-in : le runner n'appelle ce wrapper que si
demandé (flag), sinon comportement mono-vue inchangé.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import nibabel as nib


def _reorient_affine(affine: np.ndarray, axis_to_z: int) -> np.ndarray:
    """Affine cohérent après np.moveaxis(data, axis_to_z, 2).

    moveaxis place ``axis_to_z`` en dernier ; le nouvel ordre des axes spatiaux est
    ``[a for a in 0,1,2 if a != axis_to_z] + [axis_to_z]``. On réordonne les
    colonnes spatiales de l'affine en conséquence (translation inchangée).
    """
    order = [a for a in range(3) if a != axis_to_z] + [axis_to_z]
    new = affine.copy()
    new[:, :3] = affine[:, order]
    return new


def reorient_img(img: nib.Nifti1Image, axis_to_z: int) -> nib.Nifti1Image:
    """Réoriente un NIfTI pour mettre ``axis_to_z`` en position Z (axe 2)."""
    data = np.moveaxis(np.asarray(img.dataobj), axis_to_z, 2)
    out = nib.Nifti1Image(data, _reorient_affine(img.affine, axis_to_z), img.header)
    out.set_data_dtype(img.get_data_dtype())
    return out


def deorient_array(arr: np.ndarray, axis_to_z: int) -> np.ndarray:
    """Inverse de reorient : ramène l'axe Z (2) à sa position native ``axis_to_z``."""
    return np.moveaxis(arr, 2, axis_to_z)


def extract_brain_multiview(
    input_path: str | Path,
    output_path: str | Path,
    atlas_paths: Mapping[str, str | Path] | None,
    *,
    extract_fn: Callable,
    axes: Sequence[int] = (0, 1, 2),
    fusion: str = "staple",
    threshold: float = 0.5,
    workdir: str | Path | None = None,
    **extract_kwargs,
) -> dict:
    """Extraction SAM2 multi-vue + fusion.

    Pour chaque axe de ``axes`` : réoriente image + atlas (axe → Z), appelle
    ``extract_fn(input, output, atlas_paths=...)``, dé-réoriente le masque vers le
    natif. Fusionne les masques (STAPLE par défaut) et écrit ``output_path``.

    ``extract_fn`` doit avoir la signature de ``extract_brain`` du notebook et
    renvoyer ``(brain_path, mask_path)``. Injectable pour test sans GPU.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    work = Path(workdir) if workdir else output_path.parent / "_multiview_tmp"
    work.mkdir(parents=True, exist_ok=True)

    ref_img = nib.load(str(input_path))
    ref_shape = ref_img.shape[:3]
    masks_native: list[np.ndarray] = []
    per_axis = {}
    # Chemins RÉELS produits par la vue native (axe 2) : c'est là que le pipeline
    # appelant attend le masque/brain. On y écrira la fusion (pas un chemin deviné).
    native_brain_path = None
    native_mask_path = None

    for axis in axes:
        if axis == 2:
            # vue native : appel avec le VRAI output_path → chemins corrects pour
            # le pipeline appelant (SAM2 dérive son propre nom de masque).
            img_path_ax, atlas_ax, out_ax = input_path, dict(atlas_paths or {}), output_path
        else:
            img_path_ax = work / f"img_axis{axis}.nii.gz"
            nib.save(reorient_img(ref_img, axis), str(img_path_ax))
            atlas_ax = {}
            for key, ap in (atlas_paths or {}).items():
                ap_out = work / f"atlas_{key}_axis{axis}.nii.gz"
                nib.save(reorient_img(nib.load(str(ap)), axis), str(ap_out))
                atlas_ax[key] = str(ap_out)
            out_ax = work / f"mask_axis{axis}.nii.gz"

        brain_p, mask_p = extract_fn(
            str(img_path_ax), str(out_ax),
            atlas_paths=atlas_ax or None, **extract_kwargs,
        )
        if axis == 2:
            native_brain_path, native_mask_path = brain_p, mask_p
        if mask_p is None or not Path(mask_p).exists():
            per_axis[axis] = "failed"
            continue
        m = np.asarray(nib.load(str(mask_p)).dataobj) > 0
        if axis != 2:
            m = deorient_array(m, axis)
        if m.shape != ref_shape:
            per_axis[axis] = f"shape_mismatch {m.shape}"
            continue
        masks_native.append(m)
        per_axis[axis] = f"ok vol={int(m.sum())}"

    if not masks_native:
        return {"multiview_used": False, "reason": "no_view_succeeded", "per_axis": per_axis}

    if len(masks_native) == 1:
        fused = masks_native[0]
        method = "single_view_fallback"
    else:
        from scripts.atlas_label_fusion import staple_fusion, weighted_vote_fusion
        if fusion == "staple":
            fused = staple_fusion(masks_native, threshold=threshold, clean=True)
        else:
            fused = weighted_vote_fusion(masks_native, threshold=threshold, clean=True)
        method = fusion

    # Écrit la fusion AUX chemins natifs (que le pipeline attend), sinon fallback
    # sur output_path. Écrase le masque mono-vue natif par la fusion multi-vue.
    mask_out = native_mask_path or str(output_path)
    nib.save(nib.Nifti1Image(fused.astype(np.uint8), ref_img.affine, ref_img.header), str(mask_out))
    brain_out = native_brain_path
    if brain_out:
        data = np.asarray(ref_img.dataobj).astype(np.float32) * (fused > 0)
        nib.save(nib.Nifti1Image(data, ref_img.affine, ref_img.header), str(brain_out))

    return {
        "multiview_used": True,
        "n_views": len(masks_native),
        "fusion_method": method,
        "per_axis": per_axis,
        "mask_path": str(mask_out),
        "brain_path": str(brain_out) if brain_out else None,
        "volume_voxels": int(fused.sum()),
    }


def _selftest() -> None:
    """Self-test sans GPU : extract_fn factice (seuil) sur sphère bruitée par vue."""
    import tempfile

    zz, yy, xx = np.mgrid[0:48, 0:44, 0:40]   # shape anisotrope → vérifie les axes
    r = np.sqrt((zz - 24) ** 2 + (yy - 22) ** 2 + (xx - 20) ** 2)
    truth = r < 14
    aff = np.diag([1.2, 1.0, 0.9, 1.0])

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        img_p = d / "img.nii.gz"
        # image = sphère intense + bruit
        rng = np.random.default_rng(0)
        img = truth.astype(np.float32) * 100 + rng.normal(0, 5, truth.shape)
        nib.save(nib.Nifti1Image(img, aff), str(img_p))

        # 1. round-trip réorientation = identité (correction des axes)
        ref = nib.load(str(img_p))
        for axis in (0, 1, 2):
            back = deorient_array(np.moveaxis(np.asarray(ref.dataobj), axis, 2), axis)
            assert back.shape == ref.shape and np.allclose(back, ref.get_fdata()), f"round-trip axe {axis}"

        # extract_fn factice DISCRIMINANT : chaque vue rate une tranche centrale
        # perpendiculaire à SON axe Z reçu (= imprécision de bord orientée). Après
        # dé-réorientation, ces tranches tombent sur 3 plans natifs orthogonaux ;
        # un voxel cerveau est raté par ≤1 vue → majority/STAPLE le récupère.
        def fake_drop_central_z(inp, outp, atlas_paths=None, **kw):
            a = np.asarray(nib.load(inp).dataobj)
            m = a > 50
            zc = m.shape[2] // 2
            m[:, :, zc - 1:zc + 2] = False   # supprime 3 slices Z centrales
            nib.save(nib.Nifti1Image(m.astype(np.uint8), nib.load(inp).affine), outp)
            return None, outp

        def dice(a, b): return 2 * (a & b).sum() / (a.sum() + b.sum())

        # mono-vue (axe Z natif seul) : rate la tranche Z centrale du cerveau
        single = truth.copy(); zc = truth.shape[2] // 2
        single[:, :, zc - 1:zc + 2] = False
        d_single = dice(single, truth)

        # multi-vue STAPLE : 3 tranches orthogonales → fusion récupère
        rep = extract_brain_multiview(img_p, d / "mv_mask.nii.gz", None,
                                      extract_fn=fake_drop_central_z, fusion="staple", workdir=d / "wtmp")
        assert rep["multiview_used"] and rep["n_views"] == 3, rep
        fused = nib.load(str(d / "mv_mask.nii.gz")).get_fdata() > 0
        d_mv = dice(fused, truth)
        assert d_mv > d_single + 0.02, f"multi-vue {d_mv:.4f} doit nettement battre mono-vue {d_single:.4f}"
        assert d_mv > 0.97, f"multi-vue Dice trop bas {d_mv:.4f}"

    print("sam2_multiview: self-tests OK")
    print(f"  round-trip réorientation exact | multi-vue Dice={d_mv:.4f} ≥ mono-vue {d_single:.4f}")


if __name__ == "__main__":
    _selftest()
