"""Tableau d'ablation en cascade + métriques étendues + statistiques participant.

Produit les chiffres du NOUVEAU cadrage de l'article : l'appariement d'atlas est
le levier dominant, SAM2 et les raffinements sont des composants de robustesse.

Trois sorties dans outputs/ablation_2026-07/tables/ :
  ablation_cascade.csv    par benchmark × étage du pipeline : Dice, ASSD, HD95,
                          taux d'échec, cas Dice<0.80, volumes FP/FN relatifs
  bobs_participant.csv    statistiques BOBs corrigées au niveau PARTICIPANT
                          (71 visites = 51 enfants → l'appariement par visite
                          viole l'indépendance du Wilcoxon)
  wholehead_eligibility.csv  fraction hors-cerveau par dataset (formalise la
                          découverte « benchmarks pré-extraits »)

Sémantique des masques VÉRIFIÉE dans le code (ne pas modifier sans relire) :
  - atlases_advanced/*_aff*.nii.gz et *_syn*.nii.gz : atlas recalé, écrit AVANT
    l'appel SAM2 (iseg_style_pipeline.py:1432 vs _extract_brain:1476) → atlas pur.
  - *_brain_mask_adaptive.nii.gz : atlas adaptatif, aucune GT lue
    (iseg2017_adaptive_atlas.py, docstring) → atlas pur.
  - *_pre-adaptive-fusion.nii.gz : PAS du SAM2 brut (3 étages de raffinement
    déjà appliqués). Le snapshot le plus proche du SAM2 guidé est _pre-refine.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/ablation_2026-07/tables"
ABL = REPO / "outputs/ablation_2026-07"

FAIL_DICE = 0.80  # seuil « échec » (cf. évaluation : nombre de cas Dice<0.80)


# ------------------------------------------------------------------ métriques ----
def lb(p) -> np.ndarray:
    return np.asarray(nib.load(str(p)).dataobj) > 0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 0.0


def _surface(m: np.ndarray) -> np.ndarray:
    return m & ~ndi.binary_erosion(m, iterations=1, border_value=0)


def surface_metrics(pred: np.ndarray, ref: np.ndarray, spacing) -> tuple[float, float]:
    """(ASSD, HD95) en mm ; (nan, nan) si une surface est vide."""
    ps, rs = _surface(pred), _surface(ref)
    if not ps.any() or not rs.any():
        return float("nan"), float("nan")
    dt_r = ndi.distance_transform_edt(~rs, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~ps, sampling=spacing)
    d = np.concatenate([dt_r[ps], dt_p[rs]])
    if d.size == 0:
        return float("nan"), float("nan")
    return float(d.mean()), float(np.percentile(d, 95))


def all_metrics(pred_path, gt_path) -> dict | None:
    if not (pred_path and os.path.exists(str(pred_path)) and os.path.exists(str(gt_path))):
        return None
    gt_img = nib.load(str(gt_path))
    gt = np.asarray(gt_img.dataobj) > 0
    pr = lb(pred_path)
    if pr.shape != gt.shape:
        return {"shape_mismatch": True, "pred_shape": str(pr.shape), "gt_shape": str(gt.shape)}
    assd, hd95 = surface_metrics(pr, gt, gt_img.header.get_zooms()[:3])
    n_gt = max(int(gt.sum()), 1)
    return {
        "dice": dice(pr, gt), "assd": assd, "hd95": hd95,
        "fp_rel": float((pr & ~gt).sum() / n_gt),
        "fn_rel": float((gt & ~pr).sum() / n_gt),
        "shape_mismatch": False,
    }


# ------------------------------------------------------- résolution des masques ----
def bobs_stages() -> dict[str, dict[str, str]]:
    """{sujet: {étage: chemin}} pour BOBs (run figé + ablation si présente)."""
    base = REPO / "outputs/validation_runs/bobs_determ"
    df = pd.read_csv(base / "sam_neuroadapt_bobs_results.csv")
    abl = ABL / "noatlas/bobs"
    abl_csv = sorted(abl.glob("sam_neuroadapt_*results.csv"))
    abl_map: dict[str, str] = {}
    if abl_csv:
        a = pd.read_csv(abl_csv[0])
        col = "mask_path" if "mask_path" in a.columns else "output_mask_path"
        abl_map = {r.subject: str(r[col]) for _, r in a.iterrows()
                   if isinstance(r.get(col), str)}
    out: dict[str, dict[str, str]] = {}
    for _, r in df.iterrows():
        s = r["subject"]
        d = base / s / "anat"
        st = {
            "A_atlas_registered": str(r["atlas_path"]) if isinstance(r.get("atlas_path"), str) else "",
            "B_atlas_adaptive": str(r["adaptive_atlas_path"]) if isinstance(r.get("adaptive_atlas_path"), str) else "",
            "D_sam2_atlas_guided": str(d / f"{s}_T1w_desc-brain_mask_pre-refine.nii.gz"),
            "E_plus_refinements": str(d / f"{s}_T1w_desc-brain_mask_pre-adaptive-fusion.nii.gz"),
            "F_full_chain": str(r["mask_path"]),
        }
        if s in abl_map:
            st["C_sam2_no_atlas"] = abl_map[s]
        out[s] = st
    return out


def _adult_stages(base: Path, ours_csv: Path, mask_col: str, abl_dirs: list[Path],
                  final_resolver=None) -> dict[str, dict[str, str]]:
    df = pd.read_csv(ours_csv)
    abl_map: dict[str, str] = {}
    for ad in abl_dirs:
        for c in sorted(ad.glob("**/sam_neuroadapt_*results.csv")):
            try:
                a = pd.read_csv(c)
            except Exception:
                continue
            col = mask_col if mask_col in a.columns else "output_mask_path"
            for _, r in a.iterrows():
                if isinstance(r.get(col), str):
                    abl_map[r["subject"]] = str(r[col])
    out: dict[str, dict[str, str]] = {}
    for _, r in df.iterrows():
        s = r["subject"]
        d = base / s / "anat"
        aa = d / "atlases_advanced"
        st = {
            "B_atlas_adaptive": str(aa / "external_adult_t1_adaptive_atlas.nii.gz"),
            "B2_atlas_consensus": str(aa / "external_atlas_consensus.nii.gz"),
            "F_full_chain": (final_resolver(s) if final_resolver else str(r[mask_col])),
        }
        syn = sorted(aa.glob("*_syn.nii.gz")) or sorted(aa.glob("*_aff.nii.gz"))
        if syn:
            st["A_atlas_registered"] = str(syn[0])
        for key, pat in (("D_sam2_atlas_guided", "*_desc-brain_mask_pre-adult-refine.nii.gz"),
                         ("E_plus_refinements", "*_desc-brain_mask_pre-adult-candidate-adjudicator.nii.gz")):
            hits = sorted(d.glob(pat))
            if hits:
                st[key] = str(hits[0])
        if s in abl_map:
            st["C_sam2_no_atlas"] = abl_map[s]
        out[s] = st
    return out


def ibsr18_stages() -> dict[str, dict[str, str]]:
    base = REPO / "outputs/validation_runs/repro/ibsr18_s20260517"
    return _adult_stages(base, base / "sam_neuroadapt_ibsr18_adultt1_results.csv",
                         "output_mask_path", [ABL / "noatlas/ibsr18"])


def mbg_stages() -> dict[str, dict[str, str]]:
    FULL = REPO / "outputs/validation_runs/mindboggle101_full"
    bdirs = sorted(FULL.glob("b*/"))

    def final(s: str) -> str:
        for cand in (FULL / "_rescue_tmp" / f"{s}.nii.gz", FULL / "_corr25" / f"{s}.nii.gz"):
            if cand.exists():
                return str(cand)
        for b in bdirs:
            p = b / s / "anat" / f"sub-{s}_desc-brain_mask.nii.gz"
            if p.exists():
                return str(p)
        return ""

    frames = [pd.read_csv(c) for c in sorted(FULL.glob("b*/sam_neuroadapt_*results.csv"))]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out: dict[str, dict[str, str]] = {}
    abl_map: dict[str, str] = {}
    for c in sorted((ABL / "noatlas/mbg101").glob("**/sam_neuroadapt_*results.csv")):
        try:
            a = pd.read_csv(c)
        except Exception:
            continue
        col = "output_mask_path" if "output_mask_path" in a.columns else "mask_path"
        for _, r in a.iterrows():
            if isinstance(r.get(col), str):
                abl_map[r["subject"]] = str(r[col])
    for _, r in df.iterrows():
        s = r["subject"]
        d = None
        for b in bdirs:
            if (b / s / "anat").is_dir():
                d = b / s / "anat"
                break
        st = {"F_full_chain": final(s)}
        if d is not None:
            aa = d / "atlases_advanced"
            st["B_atlas_adaptive"] = str(aa / "external_adult_t1_adaptive_atlas.nii.gz")
            st["B2_atlas_consensus"] = str(aa / "external_atlas_consensus.nii.gz")
            aff = sorted(aa.glob("*_aff.nii.gz"))
            if aff:
                st["A_atlas_registered"] = str(aff[0])
            for key, pat in (("D_sam2_atlas_guided", "*_desc-brain_mask_pre-adult-refine.nii.gz"),
                             ("E_plus_refinements", "*_desc-brain_mask_pre-adult-candidate-adjudicator.nii.gz")):
                hits = sorted(d.glob(pat))
                if hits:
                    st[key] = str(hits[0])
        if s in abl_map:
            st["C_sam2_no_atlas"] = abl_map[s]
        out[s] = st
    return out


# ------------------------------------------------------------------ vérités ----
def gt_map(bench: str) -> dict[str, str]:
    if bench == "BOBs":
        df = pd.read_csv(REPO / "outputs/validation_runs/bobs_determ/sam_neuroadapt_bobs_results.csv")
        return {r["subject"]: str(REPO / f"datasets/standardized/BOBs/{r['subject']}/anat/{r['subject']}_desc-brain_mask.nii.gz")
                for _, r in df.iterrows()}
    src = {"IBSR18": "outputs/ibsr18_sota_eval_v4_external_templates_frozen/all_methods_long.csv",
           "Mindboggle-101": "outputs/validation_runs/mindboggle101_full/sota_eval/all_methods_long.csv"}[bench]
    df = pd.read_csv(REPO / src)
    return {r["subject"]: str(r["gold_mask_path"]) for _, r in df[df.method == "HD-BET"].iterrows()}


STAGE_LABEL = {
    "A_atlas_registered": "Atlas recalé seul (sans SAM2)",
    "B_atlas_adaptive": "Atlas adaptatif / apparié seul (sans SAM2)",
    "B2_atlas_consensus": "Consensus multi-atlas seul (sans SAM2)",
    "C_sam2_no_atlas": "SAM2 seul (sans guidance atlas)",
    "D_sam2_atlas_guided": "SAM2 guidé par atlas",
    "E_plus_refinements": "+ raffinements",
    "F_full_chain": "Chaîne complète",
}
STAGE_ORDER = list(STAGE_LABEL)


def build_cascade() -> pd.DataFrame:
    rows = []
    for bench, stages_fn in (("BOBs", bobs_stages), ("IBSR18", ibsr18_stages),
                             ("Mindboggle-101", mbg_stages)):
        stages = stages_fn()
        gts = gt_map(bench)
        per_stage: dict[str, list[dict]] = {k: [] for k in STAGE_ORDER}
        per_stage_sub: dict[str, list[str]] = {k: [] for k in STAGE_ORDER}
        for sub, st in stages.items():
            gt = gts.get(sub)
            if not gt:
                continue
            for key in STAGE_ORDER:
                m = all_metrics(st.get(key), gt)
                if m and not m.get("shape_mismatch"):
                    per_stage[key].append(m)
                    per_stage_sub[key].append(sub)
                elif m and m.get("shape_mismatch"):
                    print(f"  !! shape mismatch {bench}/{sub}/{key}: {m['pred_shape']} vs {m['gt_shape']}")
        final_by_sub = {s: m for s, m in zip(per_stage_sub["F_full_chain"], per_stage["F_full_chain"])}
        for key in STAGE_ORDER:
            ms = per_stage[key]
            if not ms:
                continue
            d = np.array([m["dice"] for m in ms])
            subs = per_stage_sub[key]
            paired = [(m["dice"], final_by_sub[s]["dice"]) for s, m in zip(subs, ms) if s in final_by_sub]
            p = np.nan
            if key != "F_full_chain" and len(paired) >= 6:
                a = np.array([x for x, _ in paired]); b = np.array([y for _, y in paired])
                if not np.allclose(a, b):
                    p = wilcoxon(a, b).pvalue
            rows.append({
                "benchmark": bench, "stage": key, "label": STAGE_LABEL[key], "n": len(ms),
                "dice_mean": d.mean(), "dice_sd": d.std(ddof=1) if len(d) > 1 else np.nan,
                "dice_median": np.median(d), "dice_min": d.min(),
                "assd_mean": np.nanmean([m["assd"] for m in ms]),
                "hd95_mean": np.nanmean([m["hd95"] for m in ms]),
                "fp_rel_mean": np.mean([m["fp_rel"] for m in ms]),
                "fn_rel_mean": np.mean([m["fn_rel"] for m in ms]),
                "n_fail_dice_lt_080": int((d < FAIL_DICE).sum()),
                "fail_rate": float((d < FAIL_DICE).mean()),
                "p_vs_full_chain": p,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------- BOBs niveau participant ----
def bobs_participant_stats(n_boot: int = 10000, seed: int = 20260517) -> pd.DataFrame:
    ours = pd.read_csv(REPO / "outputs/validation_runs/bobs_determ/sam_neuroadapt_bobs_results.csv")
    ped = pd.read_csv(REPO / "outputs/pediatric_sota_eval/all_pediatric_sota_results_long.csv")
    hd = ped[(ped.dataset.astype(str) == "BOBs") & (ped.method == "HD-BET")].set_index("subject")["dice"]
    df = ours[["subject", "dice"]].copy()
    df["hd"] = df.subject.map(hd)
    df = df.dropna()
    df["participant"] = df.subject.str.extract(r"sub-(\d+)")
    df["delta"] = df.dice - df.hd

    rng = np.random.default_rng(seed)
    parts = df.participant.unique()
    by_part = {p: g["delta"].values for p, g in df.groupby("participant")}

    # 1) bootstrap groupé par participant (respecte la non-indépendance des visites)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(parts, size=len(parts), replace=True)
        boots[i] = np.mean(np.concatenate([by_part[p] for p in pick]))
    ci = np.percentile(boots, [2.5, 97.5])

    # 2) sensibilité : une seule visite par enfant (la première, déterministe)
    first = df.sort_values("subject").groupby("participant").first()
    p_first = wilcoxon(first.dice, first.hd).pvalue

    # 3) moyenne par participant puis test (chaque enfant compte une fois)
    per_part = df.groupby("participant")[["dice", "hd"]].mean()
    p_mean = wilcoxon(per_part.dice, per_part.hd).pvalue

    naive_p = wilcoxon(df.dice, df.hd).pvalue
    return pd.DataFrame([
        {"analysis": "Naïve, par visite (INCORRECT : viole l'indépendance)",
         "unit": "71 visites", "n": len(df), "delta_mean": df.delta.mean(),
         "ci_low": np.nan, "ci_high": np.nan, "p": naive_p},
        {"analysis": "Bootstrap groupé par participant",
         "unit": "51 enfants", "n": len(parts), "delta_mean": df.delta.mean(),
         "ci_low": ci[0], "ci_high": ci[1], "p": np.nan},
        {"analysis": "Une visite par enfant (première)",
         "unit": "51 enfants", "n": len(first), "delta_mean": (first.dice - first.hd).mean(),
         "ci_low": np.nan, "ci_high": np.nan, "p": p_first},
        {"analysis": "Moyenne par enfant puis Wilcoxon",
         "unit": "51 enfants", "n": len(per_part), "delta_mean": (per_part.dice - per_part.hd).mean(),
         "ci_low": np.nan, "ci_high": np.nan, "p": p_mean},
    ])


# ------------------------------------------------------- éligibilité datasets ----
def wholehead_eligibility() -> pd.DataFrame:
    """Fraction de voxels hors-cerveau dans la référence : ~0 ⇒ image pré-extraite.

    ⚠️ Ne pas confondre avec le *ratio* de `scripts/validate_wholehead.py`, qui divise le
    signal hors-masque par le signal DANS le masque (donc non borné : 1.12 pour BOBs).
    Ici on divise par le signal total de la tête, donc la valeur est dans [0, 1]. Les deux
    répondent à la même question mais ne sont pas comparables : une version antérieure du
    manuscrit citait le ratio de M-CRIB (0.08) dans une phrase qui définissait la fraction
    (mesurée à 0.02). C'est la fraction qui est publiée.
    """
    specs = [
        ("BOBs", "outputs/validation_runs/bobs_determ/sam_neuroadapt_bobs_results.csv",
         "source_t1_path", "gold_mask_path"),
        ("IBSR18", "outputs/ibsr18_sota_eval_v4_external_templates_frozen/all_methods_long.csv",
         "source_t1_path", "gold_mask_path"),
        ("Mindboggle-101", "outputs/validation_runs/mindboggle101_full/sota_eval/all_methods_long.csv",
         "source_t1_path", "gold_mask_path"),
        ("iSeg2017", "outputs/iseg2017_m1_pediatric_isegstyle_eval/sam_neuroadapt_iseg2017_results.csv",
         "source_t1_path", "gold_mask_path"),
        # Run d'évaluation complet (10 sujets) et non le run de diagnostic `_uncbcp_diag`,
        # qui n'en contient qu'un — l'éligibilité d'un dataset ne se juge pas sur n=1.
        # Les variantes MCRIB2 et MCRIB_WM partagent T1 et vérité terrain avec MCRIB :
        # la fraction y est identique au chiffre près, elles ne sont donc pas listées.
        ("MCRIB", "outputs/mcrib_m1_pediatric_isegstyle_eval/sam_neuroadapt_mcrib_results.csv",
         "source_t1_path", "gold_mask_path"),
        # Jeux exclus ou perdants : le supplément annonce « every dataset considered »,
        # il faut donc les mesurer aussi, pas seulement les trois benchmarks retenus.
        ("MCRIB2", "outputs/mcrib2_m1_pediatric_isegstyle_eval/sam_neuroadapt_mcrib2_results.csv",
         "source_t1_path", "gold_mask_path"),
        ("MCRIB_WM", "outputs/mcribwm_m1_pediatric_isegstyle_eval/sam_neuroadapt_mcribwm_results.csv",
         "source_t1_path", "gold_mask_path"),
        ("SynthStripInfant",
         "outputs/synthstripinfant_m1_pediatric_isegstyle_eval/sam_neuroadapt_synthstripinfant_results.csv",
         "source_t1_path", "gold_mask_path"),
        ("FetalFMRI", "outputs/fetalfmri_m1_pediatric_isegstyle_eval/sam_neuroadapt_fetalfmri_results.csv",
         "source_t1_path", "gold_mask_path"),
        ("LPBA40", "outputs/lpba40_m1_eval_v4_external_templates_frozen/sam_neuroadapt_lpba40_adultt1_results.csv",
         "source_t1_path", "gold_mask_path"),
        ("NFBS", "outputs/nfbs_m1_eval_v4_external_templates_frozen/sam_neuroadapt_nfbs_adultt1_results.csv",
         "source_t1_path", "gold_mask_path"),
    ]
    rows = []
    for name, rel, tcol, gcol in specs:
        p = REPO / rel
        if not p.exists():
            rows.append({"dataset": name, "n": 0, "nonbrain_fraction": np.nan, "note": "CSV absent"})
            continue
        df = pd.read_csv(p)
        if "method" in df.columns:
            df = df[df.method == "HD-BET"]
        # Une mesure par SUJET : le manifeste FetalFMRI liste une entrée par frame BOLD,
        # mais toutes les frames d'un fœtus pointent vers la même image anatomique.
        if "subject" in df.columns:
            df = df.drop_duplicates(subset="subject", keep="first")
        fr = []
        for _, r in df.iterrows():
            t, g = r.get(tcol), r.get(gcol)
            if not (isinstance(t, str) and isinstance(g, str) and os.path.exists(t) and os.path.exists(g)):
                continue
            img = np.asarray(nib.load(t).dataobj)
            m = np.asarray(nib.load(g).dataobj) > 0
            if img.shape != m.shape:
                continue
            head = img > (np.percentile(img[img > 0], 15) if (img > 0).any() else 0)
            n_head = int(head.sum())
            if n_head:
                fr.append(float((head & ~m).sum() / n_head))
        rows.append({"dataset": name, "n": len(fr),
                     "nonbrain_fraction": float(np.mean(fr)) if fr else np.nan,
                     "note": "whole-head" if fr and np.mean(fr) > 0.25 else
                             ("PRÉ-EXTRAIT (dégénéré)" if fr else "non mesurable")})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Tableau d'ablation en cascade ===")
    casc = build_cascade()
    casc.to_csv(OUT / "ablation_cascade.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(casc[["benchmark", "label", "n", "dice_mean", "dice_sd", "assd_mean",
                    "hd95_mean", "n_fail_dice_lt_080", "p_vs_full_chain"]].round(4).to_string(index=False))

    print("\n=== BOBs — statistiques au niveau participant ===")
    bp = bobs_participant_stats()
    bp.to_csv(OUT / "bobs_participant.csv", index=False)
    print(bp.round(5).to_string(index=False))

    print("\n=== Éligibilité whole-head des benchmarks ===")
    we = wholehead_eligibility()
    we.to_csv(OUT / "wholehead_eligibility.csv", index=False)
    print(we.round(4).to_string(index=False))
    print(f"\nÉcrit dans {OUT}")


if __name__ == "__main__":
    main()
