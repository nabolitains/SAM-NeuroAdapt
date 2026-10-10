"""Reusable SOTA baseline runner for adult M1 result tables.

The runner compares SAM-NeuroAdapt against FSL BET, SynthStrip and HD-BET on
the same subjects present in a finished M1 CSV.  It never feeds SOTA masks back
into M1; gold labels are used only here for post-inference metrics.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import pandas as pd

from .m1_external_adaptive_runner import compute_metrics
from .sota_comparison_stats import add_sam_paired_stats


def _resolve_executable(env_name: str, candidates: Iterable[str | Path | None]) -> str | None:
    env_value = os.environ.get(env_name)
    ordered = ([env_value] if env_value else []) + list(candidates)
    for candidate in ordered:
        if not candidate:
            continue
        text = str(candidate)
        found = shutil.which(text) if os.path.sep not in text else text
        if found and Path(found).exists():
            return str(Path(found))
    return None


def _run_command(cmd: list[str | Path], timeout_s: int | float, log_prefix: str | Path) -> dict[str, Any]:
    log_prefix = Path(log_prefix)
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    meta: dict[str, Any] = {
        "cmd": " ".join(str(part) for part in cmd),
        "returncode": None,
        "runtime_s": None,
        "stdout_log": str(log_prefix.with_suffix(".stdout.txt")),
        "stderr_log": str(log_prefix.with_suffix(".stderr.txt")),
        "stderr_tail": "",
    }
    log_prefix.with_suffix(".cmd.txt").write_text(meta["cmd"] + "\n")
    try:
        proc = subprocess.run(
            [str(part) for part in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=int(timeout_s),
        )
        meta["returncode"] = int(proc.returncode)
        meta["runtime_s"] = round(time.time() - started, 2)
        log_prefix.with_suffix(".stdout.txt").write_text(proc.stdout or "")
        log_prefix.with_suffix(".stderr.txt").write_text(proc.stderr or "")
        meta["stderr_tail"] = (proc.stderr or "")[-1000:]
    except subprocess.TimeoutExpired as exc:
        meta["returncode"] = -999
        meta["runtime_s"] = round(time.time() - started, 2)
        meta["stderr_tail"] = f"timeout after {timeout_s}s"
        log_prefix.with_suffix(".stderr.txt").write_text(str(exc))
    return meta


def _canonicalize_mask(candidate_paths: Iterable[str | Path], canonical_mask: str | Path) -> Path | None:
    canonical_mask = Path(canonical_mask)
    for candidate in candidate_paths:
        candidate = Path(candidate)
        if candidate.exists():
            if candidate.resolve() != canonical_mask.resolve():
                if canonical_mask.exists():
                    canonical_mask.unlink()
                shutil.move(str(candidate), str(canonical_mask))
            return canonical_mask
    return None


def _clean_subject(out_dir: str | Path, subject: str, tag: str) -> None:
    for path in Path(out_dir).glob(f"{subject}_{tag}*"):
        if path.is_file():
            path.unlink()


def _write_progress(path: str | Path, stage: str, method: str | None = None, subject: str | None = None, index: int | None = None, total: int | None = None) -> None:
    payload = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "method": method,
        "subject": subject,
        "index": index,
        "total": total,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _safe_subject(subject: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(subject)).strip("_") or "subject"


def _find_synthstrip() -> dict[str, Any]:
    binary = _resolve_executable(
        "SYNTHSTRIP_BIN",
        [Path("resources/tools/mri_synthstrip"), shutil.which("mri_synthstrip")],
    )
    script = Path(os.environ.get("SYNTHSTRIP_PY", "resources/synthstrip_standalone.py"))
    weights = Path(os.environ.get("SYNTHSTRIP_WT", "resources/models/synthstrip.1.pt"))
    if binary:
        return {"mode": "bin", "executable": binary, "weights": str(weights) if weights.exists() else ""}
    if script.exists() and weights.exists():
        return {"mode": "script", "executable": str(script), "weights": str(weights)}
    return {"mode": "missing", "executable": "", "weights": str(weights) if weights.exists() else ""}


def _repair_mask_spatial_reference(mask_path: str | Path | None, reference_path: str | Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "spatial_repair_checked": False,
        "spatial_repair_used": False,
        "spatial_repair_reason": "not_checked",
    }
    if not mask_path or not Path(mask_path).exists() or not Path(reference_path).exists():
        info["spatial_repair_reason"] = "missing_mask_or_reference"
        return info
    try:
        mask_img = nib.load(str(mask_path))
        ref_img = nib.load(str(reference_path))
    except Exception as exc:
        info["spatial_repair_reason"] = f"load_failed:{exc}"
        return info

    info["spatial_repair_checked"] = True
    info["spatial_repair_old_axcodes"] = "".join(nib.aff2axcodes(mask_img.affine))
    info["spatial_repair_ref_axcodes"] = "".join(nib.aff2axcodes(ref_img.affine))
    if mask_img.shape[:3] != ref_img.shape[:3]:
        info["spatial_repair_reason"] = "shape_mismatch"
        return info
    mask_zooms = tuple(float(v) for v in mask_img.header.get_zooms()[:3])
    ref_zooms = tuple(float(v) for v in ref_img.header.get_zooms()[:3])
    if not np.allclose(mask_zooms, ref_zooms, atol=1e-3):
        info["spatial_repair_reason"] = "spacing_mismatch"
        return info
    if np.allclose(mask_img.affine, ref_img.affine, atol=1e-3):
        info["spatial_repair_reason"] = "already_aligned"
        return info

    data = np.asarray(mask_img.dataobj)
    header = ref_img.header.copy()
    header.set_data_dtype(mask_img.get_data_dtype())
    repaired = nib.Nifti1Image(data, ref_img.affine, header)
    repaired.set_qform(ref_img.affine, code=int(ref_img.header["qform_code"]) or 1)
    repaired.set_sform(ref_img.affine, code=int(ref_img.header["sform_code"]) or 1)
    nib.save(repaired, str(mask_path))
    info["spatial_repair_used"] = True
    info["spatial_repair_reason"] = "affine_repaired"
    info["spatial_repair_new_axcodes"] = "".join(nib.aff2axcodes(ref_img.affine))
    return info


def _prepare_m1_dataframe(
    m1_csv: str | Path,
    subjects: Iterable[str] | None = None,
    n_subjects: int | None = None,
    require_clean_m1: bool = True,
) -> pd.DataFrame:
    m1_csv = Path(m1_csv)
    if not m1_csv.exists():
        raise FileNotFoundError(f"M1 CSV not found: {m1_csv}")
    df = pd.read_csv(m1_csv)
    for col in ("subject", "source_t1_path", "gold_mask_path"):
        if col not in df.columns:
            raise RuntimeError(f"Invalid M1 CSV, missing column: {col}")
    if require_clean_m1:
        error = (
            df["error"].fillna("").astype(str).str.strip()
            if "error" in df.columns
            else pd.Series([""] * len(df), index=df.index)
        )
        df = df[error.eq("")].copy()
        if "dice" in df.columns:
            df = df[pd.to_numeric(df["dice"], errors="coerce").notna()].copy()
    if subjects:
        requested = [str(subject) for subject in subjects]
        requested_set = set(requested)
        df = df[df["subject"].astype(str).isin(requested_set)].copy()
        missing = sorted(requested_set - set(df["subject"].astype(str)))
        if missing:
            raise RuntimeError(f"Requested subjects missing from clean M1 CSV: {missing}")
        order = {subject: i for i, subject in enumerate(requested)}
        df["_subject_order"] = df["subject"].astype(str).map(order)
        df = df.sort_values("_subject_order").drop(columns=["_subject_order"])
    if n_subjects is not None:
        df = df.head(int(n_subjects)).copy()
    if df.empty:
        raise RuntimeError(f"No usable M1 subject found in {m1_csv}")
    return df.reset_index(drop=True)


def _m1_run_id(m1_csv: str | Path, df: pd.DataFrame) -> str:
    path = Path(m1_csv)
    stat = path.stat()
    subjects = ",".join(df["subject"].astype(str).tolist())
    return f"{path.resolve()}::{stat.st_mtime_ns}::{len(df)}::{subjects}"


def _sam_rows(dataset: str, df_m1: pd.DataFrame, m1_csv: str | Path, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mtime_ns = Path(m1_csv).stat().st_mtime_ns
    for row in df_m1.itertuples(index=False):
        rd = row._asdict()
        rows.append(
            {
                "dataset": dataset,
                "subject": str(rd["subject"]),
                "method": "SAM-NeuroAdapt",
                "source_t1_path": str(rd["source_t1_path"]),
                "gold_mask_path": str(rd["gold_mask_path"]),
                "mask_path": str(rd.get("output_mask_path", "")),
                "runtime_s": rd.get("runtime_s", np.nan),
                "returncode": np.nan,
                "dice": rd.get("dice", np.nan),
                "hd95": rd.get("hd95", np.nan),
                "assd": rd.get("assd", np.nan),
                "vol_pred_cm3": rd.get("vol_pred_cm3", np.nan),
                "vol_gs_cm3": rd.get("vol_gs_cm3", np.nan),
                "error": "",
                "m1_run_id": run_id,
                "source_m1_csv": str(m1_csv),
                "source_m1_mtime_ns": int(mtime_ns),
            }
        )
    return rows


def _summarize(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in ("dice", "hd95", "assd", "vol_pred_cm3", "vol_gs_cm3", "runtime_s"):
        if col in long_df.columns:
            long_df[col] = pd.to_numeric(long_df[col], errors="coerce")
    ok = long_df[(long_df["error"].fillna("").astype(str).str.strip() == "") & long_df["dice"].notna()].copy()
    summary = (
        ok.groupby("method")
        .agg(
            n=("dice", "count"),
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            dice_min=("dice", "min"),
            hd95_mean=("hd95", "mean"),
            hd95_std=("hd95", "std"),
            assd_mean=("assd", "mean"),
            assd_std=("assd", "std"),
            runtime_mean_s=("runtime_s", "mean"),
        )
        .reset_index()
    )
    long_df, summary = add_sam_paired_stats(long_df, summary, baseline_method="SAM-NeuroAdapt")
    return long_df, summary.sort_values("dice_mean", ascending=False).reset_index(drop=True)


def run_m2_external_sota(
    dataset: str,
    m1_csv: str | Path,
    out_base: str | Path | None = None,
    subjects: Iterable[str] | None = None,
    n_subjects: int | None = None,
    force_rerun: bool = True,
    require_all_tools: bool = True,
    require_all_complete: bool = True,
    require_clean_m1: bool = True,
) -> dict[str, Any]:
    out_base = Path(out_base) if out_base else Path("outputs") / f"{dataset.lower()}_sota_eval_v4_external_templates_frozen"
    out_base.mkdir(parents=True, exist_ok=True)
    log_dir = out_base / "logs"
    long_csv = out_base / "all_methods_long.csv"
    live_csv = out_base / "all_methods_long_live.csv"
    summary_csv = out_base / "method_summary.csv"
    comparison_csv = out_base / "comparison_table.csv"
    progress_json = out_base / "m2_external_sota_progress.json"

    df_m1 = _prepare_m1_dataframe(m1_csv, subjects=subjects, n_subjects=n_subjects, require_clean_m1=require_clean_m1)
    run_id = _m1_run_id(m1_csv, df_m1)

    if force_rerun:
        for path in (long_csv, live_csv, summary_csv, comparison_csv, progress_json):
            if path.exists():
                path.unlink()
    elif long_csv.exists():
        cached = pd.read_csv(long_csv)
        if "m1_run_id" in cached.columns and cached["m1_run_id"].dropna().astype(str).eq(run_id).any():
            cached, summary = _summarize(cached)
            print(f"M2 cache synced with current M1 CSV: {long_csv}")
            return {
                "long": cached,
                "summary": summary,
                "long_csv": long_csv,
                "summary_csv": summary_csv,
                "comparison_csv": comparison_csv,
                "progress_json": progress_json,
                "out_base": out_base,
            }
        print("M2 cache ignored: source M1 CSV changed since the cached comparison.")

    bet_bin = _resolve_executable("FSL_BET_BIN", [shutil.which("bet"), os.environ.get("FSLDIR", "") and os.path.join(os.environ["FSLDIR"], "share/fsl/bin/bet")])
    synthstrip = _find_synthstrip()
    hdbet_bin = _resolve_executable(
        "HDBET_BIN",
        [
            Path(".venv/bin/hd-bet"),
            Path(".venv/bin/hd_bet"),
            Path("venv_wsl/bin/hd-bet"),
            shutil.which("hd-bet"),
            shutil.which("hd_bet"),
            "/mnt/d/CHARKO-RESEARCH/FEDERATED-LEARNING/V2/venv_wsl/bin/hd-bet",
        ],
    )
    hdbet_device = os.environ.get("HDBET_DEVICE")
    if hdbet_device is None:
        try:
            import torch

            hdbet_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            hdbet_device = "cuda"
    hdbet_timeout_s = int(os.environ.get("HDBET_TIMEOUT_S", "900"))
    hdbet_disable_tta = os.environ.get("HDBET_DISABLE_TTA", "0").strip().lower() in {"1", "true", "yes", "y", "on"}

    missing_tools: list[str] = []
    if bet_bin is None:
        missing_tools.append("FSL BET")
    if not synthstrip.get("executable"):
        missing_tools.append("SynthStrip")
    if hdbet_bin is None:
        missing_tools.append("HD-BET")

    print("M2-EXTERNAL-SOTA preflight")
    print(f"FSL BET    : {bet_bin}")
    print(
        "SynthStrip : "
        + (
            f"{synthstrip['mode']}={synthstrip['executable']} | weights={synthstrip.get('weights', '')}"
            if synthstrip.get("executable")
            else "missing"
        )
    )
    print(f"HD-BET     : {hdbet_bin} | device={hdbet_device} | TTA={'off' if hdbet_disable_tta else 'on'}")
    if missing_tools and require_all_tools:
        raise FileNotFoundError("Missing SOTA tools: " + ", ".join(missing_tools))

    def run_bet(t1_path: Path, out_dir: Path, subject: str) -> tuple[Path | None, dict[str, Any]]:
        if bet_bin is None:
            return None, {"error": "FSL BET missing", "runtime_s": np.nan, "returncode": np.nan}
        _clean_subject(out_dir, subject, "bet")
        out_brain = out_dir / f"{subject}_bet_brain.nii.gz"
        out_mask = out_dir / f"{subject}_bet_mask.nii.gz"
        cmd = [bet_bin, str(t1_path), str(out_brain), "-m", "-R"]
        meta = _run_command(cmd, 240, log_dir / f"{subject}_fsl_bet")
        mask = _canonicalize_mask([out_mask, out_dir / f"{subject}_bet_brain_mask.nii.gz"], out_mask)
        if meta.get("returncode") != 0 and mask is None:
            meta["error"] = f"FSL BET failed rc={meta.get('returncode')}: {meta.get('stderr_tail', '')}"
        return mask, meta

    def run_synthstrip(t1_path: Path, out_dir: Path, subject: str) -> tuple[Path | None, dict[str, Any]]:
        if not synthstrip.get("executable"):
            return None, {"error": "SynthStrip missing", "runtime_s": np.nan, "returncode": np.nan}
        _clean_subject(out_dir, subject, "synthstrip")
        out_brain = out_dir / f"{subject}_synthstrip_brain.nii.gz"
        out_mask = out_dir / f"{subject}_synthstrip_mask.nii.gz"
        if synthstrip["mode"] == "script":
            cmd: list[str | Path] = [sys.executable, synthstrip["executable"], "-i", t1_path, "-o", out_brain, "-m", out_mask]
        else:
            cmd = [synthstrip["executable"], "-i", t1_path, "-o", out_brain, "-m", out_mask]
        if synthstrip.get("weights"):
            cmd += ["--model", synthstrip["weights"]]
        if os.environ.get("SYNTHSTRIP_GPU", "").strip().lower() in {"1", "true", "yes", "y", "on"}:
            cmd.append("-g")
        meta = _run_command(cmd, 420, log_dir / f"{subject}_synthstrip")
        mask = out_mask if out_mask.exists() else None
        if meta.get("returncode") != 0 and mask is None:
            meta["error"] = f"SynthStrip failed rc={meta.get('returncode')}: {meta.get('stderr_tail', '')}"
        return mask, meta

    def run_hdbet(t1_path: Path, out_dir: Path, subject: str) -> tuple[Path | None, dict[str, Any]]:
        if hdbet_bin is None:
            return None, {"error": "HD-BET missing", "runtime_s": np.nan, "returncode": np.nan}
        _clean_subject(out_dir, subject, "hdbet")
        out_stub = out_dir / f"{subject}_hdbet.nii.gz"
        out_mask = out_dir / f"{subject}_hdbet_mask.nii.gz"
        cmd: list[str | Path] = [
            hdbet_bin,
            "-i",
            t1_path,
            "-o",
            out_stub,
            "-device",
            str(hdbet_device),
            "--save_bet_mask",
            "--no_bet_image",
        ]
        if hdbet_disable_tta:
            cmd.append("--disable_tta")
        meta = _run_command(cmd, hdbet_timeout_s, log_dir / f"{subject}_hdbet")
        mask = _canonicalize_mask(
            [
                out_mask,
                out_dir / f"{subject}_hdbet_bet.nii.gz",
                out_dir / f"{subject}_hdbet_brain_bet.nii.gz",
                *sorted(out_dir.glob(f"{subject}_hdbet*_bet.nii.gz")),
            ],
            out_mask,
        )
        if mask is not None:
            meta.update(_repair_mask_spatial_reference(mask, t1_path))
        if meta.get("returncode") != 0 and mask is None:
            meta["error"] = f"HD-BET failed rc={meta.get('returncode')}: {meta.get('stderr_tail', '')}"
        elif mask is None:
            meta["error"] = "HD-BET finished but mask not found"
        return mask, meta

    baselines: dict[str, Any] = {}
    if bet_bin is not None:
        baselines["FSL BET"] = run_bet
    if synthstrip.get("executable"):
        baselines["SynthStrip"] = run_synthstrip
    if hdbet_bin is not None:
        baselines["HD-BET"] = run_hdbet

    long_rows = _sam_rows(dataset, df_m1, m1_csv, run_id)
    baseline_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in baselines}
    total = len(df_m1)
    _write_progress(progress_json, "started", total=total)
    print(f"M2-EXTERNAL-SOTA {dataset}: {total} subjects from {m1_csv}")

    for index, row in enumerate(df_m1.itertuples(index=False), 1):
        rd = row._asdict()
        subject = str(rd["subject"])
        safe_subject = _safe_subject(subject)
        t1_path = Path(str(rd["source_t1_path"]))
        gold_path = Path(str(rd["gold_mask_path"]))
        subject_dir = out_base / safe_subject
        subject_dir.mkdir(parents=True, exist_ok=True)

        for method, runner in baselines.items():
            _write_progress(progress_json, "running", method, subject, index, total)
            result: dict[str, Any] = {
                "dataset": dataset,
                "subject": subject,
                "method": method,
                "source_t1_path": str(t1_path),
                "gold_mask_path": str(gold_path),
                "mask_path": "",
                "runtime_s": np.nan,
                "returncode": np.nan,
                "error": "",
                "m1_run_id": run_id,
                "source_m1_csv": str(m1_csv),
                "source_m1_mtime_ns": int(Path(m1_csv).stat().st_mtime_ns),
            }
            try:
                mask_path, meta = runner(t1_path, subject_dir, safe_subject)
                result.update({k: v for k, v in meta.items() if k not in {"stderr_tail"}})
                if meta.get("error"):
                    result["error"] = str(meta["error"])
                if mask_path and Path(mask_path).exists():
                    result["mask_path"] = str(mask_path)
                    result.update(compute_metrics(mask_path, gold_path))
                elif not result["error"]:
                    result["error"] = "mask_not_found"
            except Exception as exc:
                result["error"] = str(exc)

            baseline_rows[method].append(result)
            long_rows.append(result)
            pd.DataFrame(baseline_rows[method]).to_csv(out_base / f"{method.replace(' ', '_')}_results_live.csv", index=False)
            pd.DataFrame(long_rows).to_csv(live_csv, index=False)
            if result["error"]:
                print(f"{method} | {subject}: ERROR {result['error']}")
            else:
                print(f"{method} | {subject}: Dice={result['dice']:.4f} HD95={result['hd95']:.3f}")

    for method, rows in baseline_rows.items():
        pd.DataFrame(rows).to_csv(out_base / f"{method.replace(' ', '_')}_results.csv", index=False)

    long_df = pd.DataFrame(long_rows)
    long_df, summary = _summarize(long_df)
    long_df.to_csv(long_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary.to_csv(comparison_csv, index=False)
    _write_progress(progress_json, "done", total=total)

    if require_all_complete:
        failures = long_df[
            (long_df["method"].astype(str) != "SAM-NeuroAdapt")
            & (long_df["error"].fillna("").astype(str).str.strip() != "")
        ]
        if len(failures):
            raise RuntimeError(
                "M2-EXTERNAL-SOTA incomplete: "
                + "; ".join(f"{r.method}/{r.subject}: {r.error}" for r in failures.itertuples(index=False))
            )

    print(f"\nM2-EXTERNAL-SOTA done: {long_csv}")
    return {
        "long": long_df,
        "summary": summary,
        "long_csv": long_csv,
        "summary_csv": summary_csv,
        "comparison_csv": comparison_csv,
        "progress_json": progress_json,
        "out_base": out_base,
    }

