"""Qualifie un dataset comme VRAI benchmark de skull-stripping (whole-head) ou
le rejette comme PRÉ-EXTRAIT (invalide pour le skull-stripping).

Origine : le 2026-06-21 on a découvert qu'iSeg2017 (ratio 0.00) et MCRIB (~0.08)
sont distribués DÉJÀ skull-strippés — évaluer un "skull-stripping" dessus est une
tâche dégénérée et gonfle le Dice. Seuls les datasets whole-head (BOBs 1.12,
SynthStripInfant 2.22, FetalFMRI 12.0, IBSR 2.22…) sont de vrais benchmarks.

⚠️ Le RATIO calculé ici (hors-masque / dans-masque, non borné) n'est PAS la FRACTION
publiée dans l'article (hors-masque / tête entière, dans [0,1], calculée par
`make_ablation_table.wholehead_eligibility()`). Les valeurs ci-dessus sont des ratios :
le MCRIB à 0.08 correspond à une fraction de 0.022. Ne pas citer l'une pour l'autre —
c'est l'erreur qu'une version antérieure du manuscrit contenait.

Métrique : ratio = volume(signal hors-masque-cerveau) / volume(signal dans-cerveau)
sur l'image T1/T2 ORIGINALE. Un crâne+scalp ajoute autant ou plus de signal que le
cerveau lui-même (ratio ≳ 0.5) ; une image pré-extraite a un ratio ≈ 0.

Usage :
    python -m scripts.validate_wholehead --manifest datasets/manifests/BOBs.csv
    python -m scripts.validate_wholehead --manifest <m.csv> --n 8 --json
    python -m scripts.validate_wholehead --selftest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

# Seuil de décision sur le ratio hors-cerveau/cerveau.
WHOLEHEAD_THRESHOLD = 0.15   # < seuil = pré-extrait ; ≥ seuil = whole-head
FOREGROUND_FRAC = 0.05       # fraction du max pour binariser le "signal"


def head_outside_ratio(t1_path: str | Path, mask_path: str | Path) -> float | None:
    """Ratio volume(signal hors masque) / volume(signal dans masque). None si échec."""
    try:
        t1 = nib.load(str(t1_path)).get_fdata().squeeze()
        gt = nib.load(str(mask_path)).get_fdata().squeeze() > 0
    except Exception:
        return None
    if t1.shape[:3] != gt.shape[:3]:
        return None
    fg = t1 > (t1.max() * FOREGROUND_FRAC)
    inside = float((fg & gt).sum())
    if inside < 1.0:
        return None
    return float((fg & ~gt).sum()) / inside


def classify(ratio: float | None) -> str:
    if ratio is None:
        return "inconnu"
    return "whole-head" if ratio >= WHOLEHEAD_THRESHOLD else "pré-extrait"


def validate_manifest(manifest_csv: Path, n: int | None = None) -> dict:
    """Évalue les premiers n sujets d'un manifest. Retourne un rapport agrégé."""
    import pandas as pd

    df = pd.read_csv(manifest_csv)
    # colonnes candidates pour l'image originale et le masque GT
    img_cols = [c for c in ("original_t1_path", "source_t1_path", "t1_path",
                            "input_path", "original_t2_path", "t2_path") if c in df.columns]
    mask_cols = [c for c in ("original_gold_mask_path", "gold_mask_path",
                            "brain_mask_path") if c in df.columns]
    if not img_cols or not mask_cols:
        return {"error": f"colonnes image/masque introuvables dans {manifest_csv.name}",
                "columns": list(df.columns)}
    rows = df.head(n) if n else df
    ratios, per_subject = [], []
    for _, r in rows.iterrows():
        img = next((r[c] for c in img_cols if isinstance(r[c], str) and Path(str(r[c])).exists()), None)
        msk = next((r[c] for c in mask_cols if isinstance(r[c], str) and Path(str(r[c])).exists()), None)
        if not img or not msk:
            continue
        ratio = head_outside_ratio(img, msk)
        if ratio is None:
            continue
        ratios.append(ratio)
        per_subject.append({"subject": str(r.get("subject", "?")),
                            "ratio": round(ratio, 3), "type": classify(ratio)})
    if not ratios:
        return {"error": "aucune paire image/masque exploitable", "manifest": str(manifest_csv)}
    mean = float(np.mean(ratios))
    verdict = classify(mean)
    return {
        "manifest": str(manifest_csv),
        "n_evaluated": len(ratios),
        "mean_ratio": round(mean, 3),
        "median_ratio": round(float(np.median(ratios)), 3),
        "verdict": verdict,
        "valid_for_skull_stripping": verdict == "whole-head",
        "per_subject": per_subject,
    }


def _selftest() -> None:
    """Self-test synthétique : sphère 'cerveau' + coquille 'crâne' optionnelle."""
    import tempfile

    zz, yy, xx = np.mgrid[0:40, 0:40, 0:40]
    r = np.sqrt((zz - 20) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2)
    brain = r < 12
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        aff = np.eye(4)

        def save(arr, name):
            p = d / name
            nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), str(p))
            return p

        mask_p = save(brain, "mask.nii.gz")
        # 1. image pré-extraite : signal = cerveau seul → ratio ≈ 0
        stripped = save(brain.astype(float) * 100, "stripped.nii.gz")
        rs = head_outside_ratio(stripped, mask_p)
        assert rs is not None and rs < 0.05, f"pré-extrait ratio={rs}"
        assert classify(rs) == "pré-extrait"

        # 2. image whole-head : cerveau + coquille crâne (16<r<19) → ratio élevé
        skull = (r >= 14) & (r < 19)
        whole = save((brain | skull).astype(float) * 100, "whole.nii.gz")
        rw = head_outside_ratio(whole, mask_p)
        assert rw is not None and rw >= WHOLEHEAD_THRESHOLD, f"whole-head ratio={rw}"
        assert classify(rw) == "whole-head"

    print("validate_wholehead: self-tests OK")
    print(f"  pré-extrait ratio={rs:.3f} (<{WHOLEHEAD_THRESHOLD}) | whole-head ratio={rw:.3f} (≥{WHOLEHEAD_THRESHOLD})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, help="CSV manifest (BIDS-like, colonnes *t1_path/*mask_path)")
    p.add_argument("--n", type=int, default=8, help="nb sujets à échantillonner (défaut 8)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0
    if not args.manifest:
        p.error("--manifest requis (ou --selftest)")
    rep = validate_manifest(args.manifest, args.n)
    if args.json:
        import json
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0 if rep.get("valid_for_skull_stripping") else 1
    if "error" in rep:
        print(f"✗ {rep['error']}")
        return 2
    icon = "✅" if rep["valid_for_skull_stripping"] else "❌"
    print(f"{icon} {Path(rep['manifest']).name}: {rep['verdict'].upper()} "
          f"(ratio moyen={rep['mean_ratio']}, n={rep['n_evaluated']})")
    if not rep["valid_for_skull_stripping"]:
        print("   ⚠ PRÉ-EXTRAIT — invalide comme benchmark de skull-stripping (cf. iSeg2017).")
    return 0 if rep["valid_for_skull_stripping"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
