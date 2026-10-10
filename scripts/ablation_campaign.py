"""Campagne de ré-exécutions pour l'article Brain Informatics — RÉSUMABLE.

Deux lots (cf. plan) :
  P0  ablation « SAM2 sans guidance atlas » (SAM_NEUROADAPT_NO_ATLAS=1) sur les
      3 benchmarks — mesure la contribution PROPRE de SAM2, jamais mesurée.
  P1  multi-seed IBSR18, 5 graines sur la config gelée — remplace le chiffre
      « meilleure des 5 exécutions » par une moyenne ± écart-type honnête.

CONÇU POUR SURVIVRE À : extinction du laptop, mise en veille prolongée, Ctrl-C,
coupure de courant. Relancer exactement la même commande reprend où ça s'est
arrêté, au SUJET près (jamais au benchmark près).

Mécanique de reprise, à trois niveaux :
  1. tâche  — une tâche dont le CSV final contient N lignes `dice` non nulles est
              marquée done et sautée (état dans CAMPAIGN_STATE.json) ;
  2. sujet  — chaque runner tourne avec --no-force-extraction sur un out-base
              PROPRE à l'ablation : les sujets déjà traités sont réutilisés, les
              autres recalculés. Sûr ici car l'out-base ne contient aucun masque
              guidé par atlas (il est créé par cette campagne) ;
  3. lot    — Mindboggle est découpé en lots de 8 sujets, chaque lot ayant son
              propre CSV : une coupure ne perd qu'un lot au maximum.

Usage :
    .venv/bin/python -m scripts.ablation_campaign            # lance / reprend
    .venv/bin/python -m scripts.ablation_campaign --status   # état seul, ne lance rien
    .venv/bin/python -m scripts.ablation_campaign --only p0  # un seul lot
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/ablation_2026-07"
STATE_PATH = ROOT / "CAMPAIGN_STATE.json"
LOG_PATH = ROOT / "campaign.log"
PROGRESS_MD = ROOT / "PROGRESS.md"
PY = str(REPO / ".venv/bin/python")

SEED = 20260517
MULTISEED_SEEDS = [20260517, 20260518, 20260519, 20260520, 20260521]

# Sous-échantillon Mindboggle stratifié par cohorte (35 sujets) — déterministe :
# les N premiers de chaque cohorte par ordre alphabétique, pas de tirage aléatoire.
MBG_PER_COHORT = {
    "NKI_RS_22": 6, "MMRR_21": 6, "OASIS_TRT_20": 6, "NKI_TRT_20": 6,
    "HLN_12": 5, "MMRR_3T7T_2": 2, "Twins_2": 2, "Afterthought": 1, "Colin27": 1,
}
MBG_BATCH = 8


def log(msg: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------- état ----
def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log("!! CAMPAIGN_STATE.json illisible (coupure pendant écriture ?) — réinitialisé")
    return {"tasks": {}, "created": datetime.now(timezone.utc).isoformat()}


def save_state(state: dict) -> None:
    """Écriture atomique : une coupure ne peut pas corrompre l'état."""
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def mbg_subjects() -> list[str]:
    df = pd.read_csv(REPO / "datasets/manifests/Mindboggle101.csv")
    df["cohort"] = df["subject"].str.replace(r"_\d+$", "", regex=True)
    out: list[str] = []
    for cohort, n in MBG_PER_COHORT.items():
        subs = sorted(df.loc[df["cohort"] == cohort, "subject"].astype(str))
        out.extend(subs[:n])
    return sorted(out)


# --------------------------------------------------------------------- tâches ----
def build_tasks() -> list[dict]:
    # Ordre = priorité scientifique. BOBs (benchmark phare, seul cas infant) en tête ;
    # multi-seed IBSR18 (point bloquant du reviewer) ensuite ; Mindboggle sans atlas
    # (confirmation redondante avec IBSR18, adulte) en dernier.
    tasks: list[dict] = []

    # ---- P0a : ablation SAM2 sans atlas — BOBs (PRIORITAIRE) ----
    tasks.append({
        "id": "p0_noatlas_bobs", "lot": "p0", "n": 71,
        "desc": "Ablation sans atlas — BOBs (71 sujets)",
        "out": str(ROOT / "noatlas/bobs"),
        "csv_glob": "sam_neuroadapt_*results.csv",
        "env": {"SAM_NEUROADAPT_NO_ATLAS": "1"},
        "cmd": [PY, "-m", "scripts.run_m1_pediatric_isegstyle", "--dataset", "BOBs",
                "--out-base", str(ROOT / "noatlas/bobs"),
                "--no-clean", "--seed", str(SEED), "--no-force-extraction"],
    })
    tasks.append({
        "id": "p0_noatlas_ibsr18", "lot": "p0", "n": 20,
        "desc": "Ablation sans atlas — IBSR18 (20 sujets)",
        "out": str(ROOT / "noatlas/ibsr18"),
        "csv_glob": "sam_neuroadapt_*results.csv",
        "env": {"SAM_NEUROADAPT_NO_ATLAS": "1"},
        "cmd": [PY, "-m", "scripts.run_m1_adult", "--dataset", "IBSR18",
                "--out-base", str(ROOT / "noatlas/ibsr18"),
                "--seed", str(SEED), "--no-force-extraction"],
    })

    # ---- P1 : multi-seed IBSR18 (point bloquant reviewer) ----
    # Config = référence figée (INFLATION_GUARD=1) + GARDE DE COHÉRENCE SPATIALE
    # (SPATIAL_GUARD=1, ajoutée 2026-07-24). La garde spatiale est do-no-harm :
    # elle ne revert QUE les téléportations catastrophiques (IoU<0.40, Dice=0.00)
    # des étages boundary-refine / ibsr-align ; sur un run sans téléport elle est
    # un no-op total (vérifié offline : seed 20260517 0.9026->0.9026 inchangé,
    # seed 20260518 0.8304->0.8770 en récupérant l'unique téléport IBSR_112_2).
    # NE PAS ajouter INTENSITY_REFINE (la référence ne l'active pas).
    for s in MULTISEED_SEEDS:
        tasks.append({
            "id": f"p1_multiseed_ibsr18_s{s}", "lot": "p1", "n": 20,
            "desc": f"Multi-seed IBSR18 — graine {s}",
            "out": str(ROOT / f"multiseed/ibsr18_s{s}"),
            "csv_glob": "sam_neuroadapt_*results.csv",
            "env": {"SAM_NEUROADAPT_INFLATION_GUARD": "1", "SAM_NEUROADAPT_SPATIAL_GUARD": "1"},
            "cmd": [PY, "-m", "scripts.run_m1_adult", "--dataset", "IBSR18",
                    "--out-base", str(ROOT / f"multiseed/ibsr18_s{s}"),
                    "--seed", str(s)],
        })

    # ---- P0b : ablation SAM2 sans atlas — Mindboggle (confirmation) ----
    subs = mbg_subjects()
    for i in range(0, len(subs), MBG_BATCH):
        batch = subs[i:i + MBG_BATCH]
        b = i // MBG_BATCH
        tasks.append({
            "id": f"p0_noatlas_mbg_b{b}", "lot": "p0", "n": len(batch),
            "desc": f"Ablation sans atlas — Mindboggle lot {b} ({len(batch)} sujets)",
            "out": str(ROOT / f"noatlas/mbg101/b{b}"),
            "csv_glob": "sam_neuroadapt_*results.csv",
            "env": {"SAM_NEUROADAPT_NO_ATLAS": "1", "SAM_NEUROADAPT_INTENSITY_REFINE": "1"},
            "cmd": [PY, "-m", "scripts.run_m1_adult", "--dataset", "Mindboggle101",
                    "--subjects", ",".join(batch),
                    "--out-base", str(ROOT / f"noatlas/mbg101/b{b}"),
                    "--seed", str(SEED), "--no-force-extraction"],
        })
    return tasks


def task_progress(task: dict) -> tuple[int, float | None, int]:
    """(sujets RÉSOLUS, dice moyen des réussis, nb d'échecs) — source de vérité : le disque.

    « Résolu » = le sujet a une ligne dans le CSV, qu'il ait produit un masque ou
    qu'il ait échoué. Un sujet qui échoue de façon déterministe (p. ex. SAM2 sans
    atlas ne produit aucun masque, ce qui EST le résultat mesuré ici) ne doit pas
    faire boucler la campagne indéfiniment : il est résolu, pas en attente.
    """
    out = Path(task["out"])
    best = (0, None, 0)
    for pat in (task["csv_glob"], task["csv_glob"].replace("results.csv", "results_live.csv")):
        for csv in out.glob(pat):
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            if "subject" not in df.columns:
                continue
            n_resolved = int(df["subject"].notna().sum())
            d = df["dice"].dropna() if "dice" in df.columns else pd.Series(dtype=float)
            if n_resolved > best[0]:
                best = (n_resolved, float(d.mean()) if len(d) else None, n_resolved - len(d))
    return best


def is_done(task: dict) -> bool:
    n, _, _ = task_progress(task)
    return n >= task["n"]


# --------------------------------------------------------------------- rendu ----
def write_progress_md(tasks: list[dict], state: dict) -> None:
    lots = {"p0": "P0 — Ablation SAM2 sans atlas", "p1": "P1 — Multi-seed IBSR18"}
    lines = [
        "# Avancement de la campagne de ré-exécutions",
        "",
        f"_Mis à jour : {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}_",
        "",
        "Reprise après coupure : relancer `.venv/bin/python -m scripts.ablation_campaign`.",
        "",
    ]
    for lot, title in lots.items():
        sel = [t for t in tasks if t["lot"] == lot]
        if not sel:
            continue
        tot_n = sum(t["n"] for t in sel)
        tot_done = sum(task_progress(t)[0] for t in sel)
        pct = 100.0 * tot_done / tot_n if tot_n else 0.0
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        lines += [f"## {title}", "", f"`{bar}` **{tot_done}/{tot_n} sujets ({pct:.0f} %)**", "",
                  "| Tâche | Sujets | Dice moyen | État |", "|---|---:|---:|---|"]
        for t in sel:
            n, mean, nfail = task_progress(t)
            st = state["tasks"].get(t["id"], {})
            if n >= t["n"]:
                label = "✅ terminé"
            elif st.get("status") == "running":
                label = "⏳ en cours"
            elif st.get("status") == "failed":
                label = f"❌ échec ({st.get('returncode')})"
            elif n > 0:
                label = "⏸ partiel (repris au prochain lancement)"
            else:
                label = "· en attente"
            fail_txt = f" ({nfail} échec{'s' if nfail > 1 else ''})" if nfail else ""
            lines.append(f"| {t['desc']} | {n}/{t['n']}{fail_txt} | "
                         f"{f'{mean:.4f}' if mean is not None else '—'} | {label} |")
        lines.append("")
    PROGRESS_MD.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_MD.write_text("\n".join(lines), encoding="utf-8")


def print_status(tasks: list[dict], state: dict) -> None:
    for lot in ("p0", "p1"):
        sel = [t for t in tasks if t["lot"] == lot]
        if not sel:
            continue
        tot_n = sum(t["n"] for t in sel)
        tot_done = sum(task_progress(t)[0] for t in sel)
        print(f"\n=== {lot.upper()} : {tot_done}/{tot_n} sujets ===")
        for t in sel:
            n, mean, nfail = task_progress(t)
            flag = "OK " if n >= t["n"] else ("...." if n else "    ")
            m = f"  dice={mean:.4f}" if mean is not None else ""
            f_txt = f"  [{nfail} échec(s)]" if nfail else ""
            print(f"  [{flag}] {t['desc']:52s} {n:3d}/{t['n']:<3d}{m}{f_txt}")


# --------------------------------------------------------------------- run ----
def run_task(task: dict, state: dict) -> bool:
    Path(task["out"]).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(task["env"])
    env.setdefault("PYTHONUNBUFFERED", "1")

    state["tasks"].setdefault(task["id"], {})
    state["tasks"][task["id"]].update(
        {"status": "running", "started": datetime.now(timezone.utc).isoformat(),
         "cmd": " ".join(task["cmd"]), "env": task["env"]})
    save_state(state)

    log(f">>> DÉBUT {task['id']} — {task['desc']}")
    log(f"    env  : {task['env']}")
    log(f"    cmd  : {' '.join(task['cmd'])}")
    task_log = Path(task["out"]) / "run.log"
    t0 = time.time()
    with open(task_log, "a", encoding="utf-8") as lf:
        lf.write(f"\n=== relance {datetime.now(timezone.utc).isoformat()} ===\n")
        lf.flush()
        proc = subprocess.run(task["cmd"], cwd=REPO, env=env, stdout=lf,
                              stderr=subprocess.STDOUT, check=False)
    dt = time.time() - t0
    n, mean, nfail = task_progress(task)
    ok = proc.returncode == 0 and n >= task["n"]

    state["tasks"][task["id"]].update(
        {"status": "done" if ok else "failed", "returncode": proc.returncode,
         "ended": datetime.now(timezone.utc).isoformat(),
         "elapsed_s": round(dt, 1), "n_done": n, "n_failed_subjects": nfail,
         "mean_dice": round(mean, 4) if mean is not None else None})
    save_state(state)

    log(f"<<< FIN   {task['id']} rc={proc.returncode} en {dt/3600:.2f} h — "
        f"{n}/{task['n']} sujets" + (f" ({nfail} en échec)" if nfail else "")
        + (f", dice={mean:.4f}" if mean is not None else ""))
    if not ok:
        log(f"!!! {task['id']} incomplet — sera repris au prochain lancement "
            f"(voir {task_log})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true", help="affiche l'état et sort")
    ap.add_argument("--only", choices=["p0", "p1"], help="ne traiter qu'un lot")
    args = ap.parse_args()

    tasks = build_tasks()
    state = load_state()

    if args.status:
        print_status(tasks, state)
        write_progress_md(tasks, state)
        print(f"\nDétail : {PROGRESS_MD}")
        return 0

    if args.only:
        tasks = [t for t in tasks if t["lot"] == args.only]

    log("=" * 72)
    log(f"CAMPAGNE — {len(tasks)} tâches ; reprise automatique des tâches inachevées")
    write_progress_md(tasks, state)

    for task in tasks:
        if is_done(task):
            log(f"--- SAUTÉ {task['id']} (déjà complet)")
            state["tasks"].setdefault(task["id"], {})["status"] = "done"
            save_state(state)
            continue
        run_task(task, state)
        write_progress_md(tasks, state)

    log("=== CAMPAGNE : passe terminée ===")
    print_status(tasks, state)
    n_left = sum(1 for t in tasks if not is_done(t))
    if n_left:
        log(f"!! {n_left} tâche(s) incomplète(s) — relancer la même commande pour reprendre")
    return 0


if __name__ == "__main__":
    sys.exit(main())
