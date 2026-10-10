"""Mindboggle-101 COMPLET (101 sujets) : pipeline corrigé (raffinement intensité) vs HD-BET.

RÉSUMABLE PAR BATCH (survit aux extinctions PC — relancer le script reprend).
- 25 sujets déjà en cache (ours_s20260517, SAM2 brut) → corrigés OFFLINE par le module.
- 76 restants → run_m1_adult par batches de 8, FLAG intensité ON (masques corrigés in-pipeline).
- HD-BET sur les 101 (m2, CPU) → baseline.
- Combine + Wilcoxon par groupe → SUMMARY_101.md.

Lancer / reprendre : bash scripts/_run_mbg101.sh
"""
from __future__ import annotations
import json, os, subprocess, sys, time, shutil, glob, warnings
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")

REPO = Path(".")
MAN = pd.read_csv(REPO / "datasets/manifests/Mindboggle101.csv")
CACHE25 = REPO / "outputs/validation_runs/mindboggle/ours_s20260517"
STD = REPO / "datasets/standardized/Mindboggle101"
FULL = REPO / "outputs/validation_runs/mindboggle101_full"
FULL.mkdir(parents=True, exist_ok=True)
LOG = FULL / "mbg101.log"
HDBET_CSV = FULL / "hdbet_101.csv"
SEED = 20260517
BATCH = 8

def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"; print(line, flush=True)
    open(LOG, "a").write(line + "\n")

def loadb(p): return np.asarray(nib.load(p).dataobj) > 0
def dice(a, b): s = a.sum() + b.sum(); return 2 * (a & b).sum() / s if s else 0.0

ALL = sorted(MAN["subject"].astype(str))
DONE25 = sorted(s for s in (os.path.basename(d.rstrip("/")) for d in glob.glob(str(CACHE25 / "*/"))) if s in set(ALL))
REM = [s for s in ALL if s not in set(DONE25)]
GROUP = dict(zip(MAN["subject"].astype(str), MAN["group"].astype(str)))
HDM_GROUPS = MAN.set_index(MAN["subject"].astype(str))["group"].to_dict()

def batch_csv(i): return FULL / f"b{i}" / "sam_neuroadapt_mindboggle101_adultt1_results.csv"

def batch_done(i, subs):
    p = batch_csv(i)
    if not p.exists(): return False
    try: d = pd.read_csv(p).dropna(subset=["dice"])
    except Exception: return False
    return set(subs).issubset(set(d["subject"].astype(str)))

def run_batches():
    batches = [REM[i:i + BATCH] for i in range(0, len(REM), BATCH)]
    for i, subs in enumerate(batches):
        if batch_done(i, subs):
            log(f"  SKIP batch {i} ({len(subs)} sujets, déjà fait)"); continue
        env = dict(os.environ); env["SAM_NEUROADAPT_INTENSITY_REFINE"] = "1"
        cmd = [sys.executable, "-m", "scripts.run_m1_adult", "--dataset", "Mindboggle101",
               "--subjects", ",".join(subs), "--out-base", str(FULL / f"b{i}"), "--seed", str(SEED)]
        log(f"  RUN batch {i}/{len(batches)-1} ({len(subs)} sujets)…")
        t0 = time.time(); r = subprocess.run(cmd, cwd=str(REPO), env=env)
        log(f"  batch {i} rc={r.returncode} en {(time.time()-t0)/60:.1f} min")
        if r.returncode != 0: raise RuntimeError(f"batch {i} échec")

def ensure_hdbet():
    if HDBET_CSV.exists():
        log(f"  SKIP HD-BET (déjà: {HDBET_CSV})"); return pd.read_csv(HDBET_CSV)
    log("  GÉNÈRE HD-BET sur les 101 (m2, CPU)…")
    mini = pd.DataFrame({"subject": MAN["subject"].astype(str),
                         "source_t1_path": MAN["input_path"].fillna(MAN["t1_path"]),
                         "gold_mask_path": MAN["brain_mask_path"], "error": ""})
    mini_csv = FULL / "_mini_m1.csv"; mini.to_csv(mini_csv, index=False)
    os.environ["HDBET_DEVICE"] = "cpu"; os.environ.setdefault("HDBET_DISABLE_TTA", "1")
    from scripts.m2_external_sota_runner import run_m2_external_sota
    run_m2_external_sota(dataset="Mindboggle101", m1_csv=str(mini_csv), out_base=str(FULL / "sota_eval"),
                         subjects=ALL, force_rerun=False, require_all_tools=False,
                         require_all_complete=False, require_clean_m1=False)
    lon = pd.read_csv(FULL / "sota_eval" / "all_methods_long.csv")
    hd = lon[lon["method"].astype(str).str.upper().str.replace("_", "-") == "HD-BET"][["subject", "dice"]].dropna()
    hd.to_csv(HDBET_CSV, index=False); log(f"  HD-BET N={len(hd)} mean={hd['dice'].mean():.4f}")
    return hd

def corrected_dice_for_cache25():
    """Applique le module de raffinement intensité aux 25 cachés (offline) → dice corrigé."""
    from scripts.adult_intensity_refinement import apply_adult_intensity_refinement, DEFAULTS
    res = [x for x in glob.glob(str(CACHE25 / "sam_neuroadapt_*results*.csv")) if "live" not in x]
    dis = pd.read_csv(res[0]).set_index("subject")["atlas_disagreement_score"]
    tmp = FULL / "_corr25"; tmp.mkdir(exist_ok=True)
    out = {}
    for S in DONE25:
        b = CACHE25 / S / "anat"
        g = STD / S / "anat" / f"sub-{S}_desc-brain_mask.nii.gz"
        o = b / f"sub-{S}_desc-brain_mask.nii.gz"
        a = b / "atlases_advanced" / "external_adult_t1_adaptive_atlas.nii.gz"
        n = b / f"sub-{S}_desc-n4_T1w.nii.gz"
        if not all(p.exists() for p in [g, o, a, n]): continue
        w = tmp / f"{S}.nii.gz"; shutil.copyfile(o, w)
        apply_adult_intensity_refinement(w, a, n, float(dis.get(S, 0.0)), config={**DEFAULTS, "enabled": True})
        out[S] = dice(loadb(g), loadb(w))
    return out

def main():
    log("=" * 70); log(f"MINDBOGGLE-101 COMPLET | 25 cache + {len(REM)} à passer (batches de {BATCH})")
    run_batches()
    hd = ensure_hdbet(); hdm = dict(zip(hd["subject"].astype(str).str.replace("-", "_"), hd["dice"]))
    # dice corrigé : 76 depuis les batches + 25 offline
    our = {}
    for i in range(0, len(REM), BATCH):
        p = batch_csv(i // BATCH)
        if p.exists():
            d = pd.read_csv(p).dropna(subset=["dice"])
            for _, r in d.iterrows(): our[str(r["subject"])] = float(r["dice"])
    our.update(corrected_dice_for_cache25())
    rows = []
    for S in ALL:
        if S in our and S.replace("-", "_") in hdm:
            rows.append({"subject": S, "group": GROUP.get(S, "?"), "our": our[S], "hd": hdm[S.replace("-", "_")]})
    df = pd.DataFrame(rows)
    df.to_csv(FULL / "comparison_101.csv", index=False)
    a, b = df["our"].values, df["hd"].values
    p = wilcoxon(a, b).pvalue if np.any(a - b != 0) else float("nan")
    log(f"\n=== MINDBOGGLE-101 (N={len(df)}) : nous={a.mean():.4f} vs HD-BET={b.mean():.4f} "
        f"D={a.mean()-b.mean():+.4f} Wilcoxon p={p:.4f} >=HD {int((a>=b).sum())}/{len(df)} ===")
    lines = ["# Mindboggle-101 COMPLET — pipeline corrigé vs HD-BET", "",
             f"N={len(df)} | nous **{a.mean():.4f}** vs HD-BET **{b.mean():.4f}** | "
             f"D={a.mean()-b.mean():+.4f} | Wilcoxon p={p:.4f} | >=HD {int((a>=b).sum())}/{len(df)}",
             "", "## Par groupe", "| groupe | N | nous | HD-BET |", "|---|---|---|---|"]
    for g, gg in df.groupby("group"):
        lines.append(f"| {g} | {len(gg)} | {gg['our'].mean():.4f} | {gg['hd'].mean():.4f} |")
    (FULL / "SUMMARY_101.md").write_text("\n".join(lines) + "\n")
    log("TERMINÉ. Voir SUMMARY_101.md")

if __name__ == "__main__":
    raise SystemExit(main())
