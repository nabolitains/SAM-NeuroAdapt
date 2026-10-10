"""Standardise Mindboggle-101 (dataverse_files.zip) en benchmark skull-stripping adulte.

Source : téléchargement Harvard Dataverse de Mindboggle-101 (5 groupes = 101 sujets).
Chaque sujet a, en espace NATIF : `t1weighted.nii.gz` (whole-head, crâne présent) et
`labels.DKT31.manual+aseg.nii.gz` (cortex manuel DKT + aseg sous-cortical/cervelet/tronc).
Masque cerveau GT = union(labels > 0) → plus grosse composante → remplissage de trous.

Benchmark ÉQUITABLE vs HD-BET (Mindboggle n'est PAS dans ses benchmarks NFBS/LPBA40/CC-359).
Caveat : masque = cortex MANUEL + sous-cortical aseg (auto) ; source OASIS/NKI/MMRR
(NKI ≠ sujets NFBS mais même cohorte source ; OASIS potentiellement dans SynthStrip).

Sortie : datasets/standardized/Mindboggle101/<sub>/anat/{T1w,desc-brain_mask}.nii.gz
+ manifest datasets/manifests/Mindboggle101.csv (format runner adulte).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
ZIP = Path("/mnt/c/Users/ThinkPad/Downloads/dataverse_files.zip")
GROUPS = ["OASIS-TRT-20", "NKI-RS-22", "NKI-TRT-20", "MMRR-21", "Extra-18"]
STAGING = REPO / "datasets/_mindboggle101_raw"
STD = REPO / "datasets/standardized/Mindboggle101"
MANIFEST = REPO / "datasets/manifests/Mindboggle101.csv"


def extract_group(group: str) -> None:
    """Stream-extrait UNIQUEMENT t1weighted.nii.gz + labels.DKT31.manual+aseg.nii.gz."""
    dest = STAGING
    dest.mkdir(parents=True, exist_ok=True)
    # déjà extrait ? (Extra-18 contient des noms hétérogènes: Colin27-1, HLN-12-*, ...)
    if list((STAGING / f"{group}_volumes").glob("*/t1weighted.nii.gz")):
        print(f"  [{group}] déjà extrait", flush=True)
        return
    print(f"  [{group}] extraction…", flush=True)
    cmd = (
        f"unzip -p '{ZIP}' '{group}_volumes.tar.gz' | "
        f"tar xz -C '{dest}' --wildcards --no-anchored "
        f"'*/t1weighted.nii.gz' '*/labels.DKT31.manual+aseg.nii.gz' 2>/dev/null"
    )
    subprocess.run(["bash", "-c", cmd], check=False)


def brain_mask_from_labels(label_path: Path) -> nib.Nifti1Image:
    lab = nib.load(str(label_path))
    ld = np.asarray(lab.dataobj)
    m = ld > 0
    # plus grosse composante puis remplissage des trous (ventricules, etc.)
    lbl, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
        m = lbl == (int(np.argmax(sizes)) + 1)
    m = ndi.binary_closing(m, iterations=1)
    m = ndi.binary_fill_holes(m)
    return nib.Nifti1Image(m.astype(np.uint8), lab.affine, lab.header)


def wholehead_ratio(t1_path: Path, mask: np.ndarray) -> float:
    td = np.asarray(nib.load(str(t1_path)).dataobj, dtype=np.float32)
    fg = td > np.percentile(td[td > 0], 30)
    inb = (fg & mask).sum()
    return float((fg & ~mask).sum()) / max(int(inb), 1)


def main() -> int:
    STD.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    for g in GROUPS:
        extract_group(g)

    rows = []
    flagged = []
    for g in GROUPS:
        # tout sous-dossier contenant les 2 fichiers (Extra-18 = noms hétérogènes)
        subj_dirs = sorted(p.parent for p in (STAGING / f"{g}_volumes").glob("*/t1weighted.nii.gz"))
        for sd in subj_dirs:
            t1 = sd / "t1weighted.nii.gz"
            lab = sd / "labels.DKT31.manual+aseg.nii.gz"
            if not (t1.exists() and lab.exists()):
                continue
            sub = sd.name.replace("-", "_")  # OASIS-TRT-20-1 -> OASIS_TRT_20_1
            anat = STD / sub / "anat"
            anat.mkdir(parents=True, exist_ok=True)
            t1_out = anat / f"sub-{sub}_T1w.nii.gz"
            mask_out = anat / f"sub-{sub}_desc-brain_mask.nii.gz"

            img = nib.load(str(t1))
            if t1_out.exists() and mask_out.exists():
                mimg = nib.load(str(mask_out))
            else:
                nib.save(img, str(t1_out))
                mimg = brain_mask_from_labels(lab)
                if mimg.shape != img.shape:
                    flagged.append((sub, f"shape T1{img.shape} != mask{mimg.shape}"))
                    continue
                nib.save(mimg, str(mask_out))

            md = np.asarray(mimg.dataobj) > 0
            vox = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
            ratio = wholehead_ratio(t1_out, md)
            if ratio < 0.15:
                flagged.append((sub, f"PAS whole-head (ratio {ratio:.2f})"))
            rows.append({
                "dataset": "Mindboggle101", "subject": sub, "session": "",
                "split": "gold", "has_gold": True, "qc_only": False,
                "atlas_eligible": True, "modality": "T1w", "profile": "adult_t1",
                "t1_path": str(t1_out), "t2_path": "", "input_path": str(t1_out),
                "brain_mask_path": str(mask_out), "label_path": "",
                "source_path": str(t1), "standardized_dir": str(anat),
                "shape": "x".join(map(str, img.shape[:3])),
                "zooms": "x".join(f"{z:.3g}" for z in img.header.get_zooms()[:3]),
                "mask_voxels": int(md.sum()), "mask_volume_cm3": round(md.sum() * vox, 2),
                "wholehead_ratio": round(ratio, 3), "group": g, "notes": "",
            })
            print(f"  {sub:20} vol={md.sum()*vox:7.0f}cm3 ratio={ratio:.2f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(MANIFEST, index=False)
    print(f"\nmanifest: {MANIFEST}  (N={len(df)})")
    print(f"groupes: {df['group'].value_counts().to_dict()}")
    print(f"volume médian = {df['mask_volume_cm3'].median():.0f} cm3 | "
          f"whole-head ratio médian = {df['wholehead_ratio'].median():.2f}")
    if flagged:
        print(f"\n⚠️ {len(flagged)} sujets à vérifier:")
        for s, r in flagged[:20]:
            print(f"   {s}: {r}")
    else:
        print("✅ tous whole-head, masques cohérents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
