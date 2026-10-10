"""ASSD/HD95 surface-à-surface pour les configurations du tableau d'ablation (T2).

Le CSV `ablation_cascade.csv` produit par `make_ablation_table.py` contient bien des
colonnes `assd_mean`/`hd95_mean`, mais calculées avec une définition POOLÉE (moyenne et
95ᵉ centile sur la concaténation des deux jeux de distances). T1 publie, lui, la
définition usuelle : moyenne des deux moyennes directionnelles pour l'ASSD, maximum des
deux 95ᵉ centiles pour HD95. Mélanger les deux dans un même article reproduirait
exactement l'incohérence documentée en §5 de REPRODUCIBILITY_KNOWN_ISSUES.md.

Ce script recalcule donc les distances de T2 avec la fonction de T1
(`recompute_surface_metrics.surface_metrics`) et écrit le détail **par sujet** dans
`outputs/ablation_2026-07/tables/t2_surface_metrics.csv`.

Résolution des masques — deux univers distincts, à ne pas confondre :
  * `atlas_alone` vient de `make_ablation_table.py` (les sujets du run complet : BOBs 71,
    IBSR18 20, Mindboggle-101 76) ;
  * `no_atlas` vient **directement des CSV du run d'ablation**. Sur Mindboggle-101
    l'ablation porte sur un sous-échantillon stratifié de 35 sujets dont 10 seulement
    appartiennent aux 76 du run complet : passer par `make_ablation_table.py` ne
    retrouverait que ces 10-là et décrirait une autre ligne que celle publiée dans T2.

Deux optimisations, sans effet sur les valeurs (égalité vérifiée à 1e-9 près contre
`recompute_surface_metrics.surface_metrics`) :
  * la transformée de distance de la référence est calculée une seule fois par sujet,
    partagée entre les configurations ;
  * le calcul est restreint à la boîte englobante des deux masques, élargie d'un voxel.
    L'EDT y reste exacte : les deux surfaces sont entièrement contenues dans la boîte,
    donc le point le plus proche de tout voxel interrogé l'est aussi.

Le script est **reprenable** : les couples (benchmark, sujet, configuration) déjà présents
dans le CSV de sortie ne sont pas recalculés.

    .venv/bin/python scripts/make_table2_surface.py [--force]
"""
from __future__ import annotations

import glob
import os
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, ".")
from scripts.make_ablation_table import bobs_stages, gt_map, ibsr18_stages, mbg_stages

OUT = "outputs/ablation_2026-07/tables"
CSV = f"{OUT}/t2_surface_metrics.csv"
ABL = "outputs/ablation_2026-07/noatlas"

# Étage portant la ligne « atlas apparié seul » de T2, par benchmark.
# BOBs n'a pas d'étage « adaptatif » distinct : son atlas apparié est l'atlas recalé.
ATLAS_STAGE = {"BOBs": "A_atlas_registered",
               "IBSR18": "B_atlas_adaptive",
               "Mindboggle-101": "B_atlas_adaptive"}

# CSV des runs sans atlas, et dossier de données standardisées pour retrouver une vérité
# terrain absente du CSV (un sujet qui échoue tôt n'a pas de `gold_mask_path` renseigné).
NOATLAS_GLOB = {"BOBs": f"{ABL}/bobs/**/sam_neuroadapt_*results.csv",
                "IBSR18": f"{ABL}/ibsr18/**/sam_neuroadapt_*results.csv",
                "Mindboggle-101": f"{ABL}/mbg101/**/sam_neuroadapt_*results.csv"}


def _surface(m: np.ndarray) -> np.ndarray:
    return m & ~ndi.binary_erosion(m, iterations=1, border_value=0)


def _bbox(m: np.ndarray, pad: int = 1) -> tuple[slice, ...]:
    idx = np.nonzero(m)
    return tuple(slice(max(int(i.min()) - pad, 0), int(i.max()) + pad + 1) for i in idx)


def surface_metrics(pred: np.ndarray, ref_surf: np.ndarray, dt_ref: np.ndarray,
                    spacing) -> tuple[float, float]:
    """(ASSD, HD95) en mm — définition de T1 (cf. recompute_surface_metrics.py).

    `ref_surf` et `dt_ref` sont fournis par l'appelant pour éviter de les recalculer à
    chaque configuration du même sujet.
    """
    ps = _surface(pred)
    if not ps.any() or not ref_surf.any():
        return float("nan"), float("nan")
    sl = _bbox(ps | ref_surf)                # l'EDT de `pred` y est exacte
    dt_pred = ndi.distance_transform_edt(~ps[sl], sampling=spacing)
    d_p2r = dt_ref[ps]                       # prédiction -> surface de référence
    d_r2p = dt_pred[ref_surf[sl]]            # référence  -> surface prédite
    assd = float((d_p2r.mean() + d_r2p.mean()) / 2.0)
    hd95 = float(max(np.percentile(d_p2r, 95), np.percentile(d_r2p, 95)))
    return assd, hd95


def build_tasks() -> list[dict]:
    """(benchmark, sujet, configuration) -> chemins masque prédit / vérité terrain."""
    tasks: list[dict] = []

    for bench, stages_fn in (("BOBs", bobs_stages), ("IBSR18", ibsr18_stages),
                             ("Mindboggle-101", mbg_stages)):
        stages, gts = stages_fn(), gt_map(bench)
        key = ATLAS_STAGE[bench]
        for sub, st in stages.items():
            tasks.append(dict(benchmark=bench, subject=sub, config="atlas_alone",
                              pred=st.get(key), gt=gts.get(sub)))

        # `no_atlas` : lu dans les CSV du run d'ablation (univers de sujets distinct).
        frames = [pd.read_csv(c) for c in sorted(glob.glob(NOATLAS_GLOB[bench], recursive=True))]
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True)
        col = "output_mask_path" if "output_mask_path" in d.columns else "mask_path"
        for _, r in d.iterrows():
            gt = r.get("gold_mask_path")
            if not isinstance(gt, str):
                gt = gts.get(r["subject"])
            tasks.append(dict(benchmark=bench, subject=r["subject"], config="no_atlas",
                              pred=r.get(col), gt=gt))

    return [t for t in tasks
            if isinstance(t["pred"], str) and os.path.exists(t["pred"])
            and isinstance(t["gt"], str) and os.path.exists(t["gt"])]


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    force = "--force" in sys.argv
    done = pd.read_csv(CSV) if (os.path.exists(CSV) and not force) else pd.DataFrame()
    seen = set(zip(done.benchmark, done.subject, done.config)) if len(done) else set()

    tasks = build_tasks()
    todo = [t for t in tasks if (t["benchmark"], t["subject"], t["config"]) not in seen]
    print(f"{len(tasks)} couples résolus ; {len(tasks) - len(todo)} déjà calculés, "
          f"{len(todo)} à faire.", flush=True)

    # regroupé par sujet : la transformée de la référence est partagée
    by_subject: dict[tuple[str, str], list[dict]] = {}
    for t in todo:
        by_subject.setdefault((t["benchmark"], t["subject"]), []).append(t)

    rows: list[dict] = []
    t0 = time.time()
    for i, ((bench, sub), group) in enumerate(sorted(by_subject.items()), 1):
        gt_img = nib.load(group[0]["gt"])
        gt = np.asarray(gt_img.dataobj) > 0
        if not gt.any():
            continue
        sp = gt_img.header.get_zooms()[:3]
        rs = _surface(gt)
        dt_ref = ndi.distance_transform_edt(~rs, sampling=sp)
        for t in group:
            pr = np.asarray(nib.load(t["pred"]).dataobj) > 0
            if pr.shape != gt.shape:
                print(f"  !! shape mismatch {bench}/{sub}/{t['config']}", flush=True)
                continue
            s = int(pr.sum()) + int(gt.sum())
            assd, hd95 = surface_metrics(pr, rs, dt_ref, sp)
            rows.append(dict(benchmark=bench, subject=sub, config=t["config"],
                             dice=float(2 * (pr & gt).sum() / s) if s else 0.0,
                             assd=assd, hd95=hd95))
        if i % 20 == 0:
            print(f"  ... {i}/{len(by_subject)}  ({time.time() - t0:.0f} s)", flush=True)

    df = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["benchmark", "subject", "config"], keep="last")
    df.to_csv(CSV, index=False)

    agg = (df.groupby(["benchmark", "config"])
             .agg(n=("dice", "size"), dice=("dice", "mean"),
                  assd=("assd", "mean"), assd_sd=("assd", "std"),
                  hd95=("hd95", "mean"), hd95_sd=("hd95", "std")).round(3))
    print("\n=== ASSD / HD95 des configurations de T2 (mm, définition de T1) ===")
    print(agg.to_string())
    print(f"\nécrit : {CSV}   ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
