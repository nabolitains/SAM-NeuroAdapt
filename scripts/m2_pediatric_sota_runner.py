#!/usr/bin/env python3
"""Run external SOTA skull-stripping baselines on pediatric manifests.

This is an M2 baseline runner.  It never feeds FSL BET, SynthStrip or HD-BET
outputs back into M1.  Gold masks are loaded only after each SOTA mask has been
written, for Dice/HD95/ASSD reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import nibabel as nib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.m1_external_adaptive_runner import compute_metrics
from scripts.m2_external_sota_runner import (
    _canonicalize_mask,
    _clean_subject,
    _find_synthstrip,
    _repair_mask_spatial_reference,
    _resolve_executable,
    _run_command,
    _safe_subject,
)


DEFAULT_PEDIATRIC_DATASETS = (
    "iSeg2017",
    "BOBs",
    "MCRIB",
    "MCRIB_WM",
    "MCRIB2",
    "SynthStripT1",
)
DEFAULT_METHODS = ("fsl_bet", "synthstrip", "hdbet")
METHOD_LABELS = {
    "fsl_bet": "FSL BET",
    "synthstrip": "SynthStrip",
    "hdbet": "HD-BET",
}


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on", "oui"}


def _parse_csv_list(value: str | Iterable[str] | None, default: Iterable[str] = ()) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [chunk.strip() for chunk in value.replace(";", ",").split(",") if chunk.strip()]
    return [str(chunk).strip() for chunk in value if str(chunk).strip()]


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else float("nan")
    except Exception:
        return float("nan")


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _squeeze_metric_image(path: str | Path, out_path: str | Path) -> Path:
    """Return a 3D NIfTI path for metrics, preserving the source file when possible."""

    path = Path(path)
    img = nib.load(str(path))
    shape = tuple(int(v) for v in img.shape)
    if len(shape) == 3:
        return path
    squeezed = np.asarray(img.dataobj)
    while squeezed.ndim > 3 and squeezed.shape[-1] == 1:
        squeezed = squeezed[..., 0]
    if squeezed.ndim != 3:
        return path
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = img.header.copy()
    header.set_data_dtype(img.get_data_dtype())
    out = nib.Nifti1Image(squeezed, img.affine, header)
    out.set_qform(img.affine, code=int(img.header["qform_code"]) or 1)
    out.set_sform(img.affine, code=int(img.header["sform_code"]) or 1)
    nib.save(out, str(out_path))
    return out_path


def _compute_metrics_3d(pred_mask_path: str | Path, gold_mask_path: str | Path, metric_dir: str | Path) -> dict[str, float]:
    metric_dir = Path(metric_dir)
    pred = _squeeze_metric_image(pred_mask_path, metric_dir / "pred_metric_3d.nii.gz")
    gold = _squeeze_metric_image(gold_mask_path, metric_dir / "gold_metric_3d.nii.gz")
    return compute_metrics(pred, gold)


def _load_records(
    dataset: str,
    manifest_root: str | Path = "datasets/manifests",
    manifest: str | Path | None = None,
    *,
    subjects: Iterable[str] | None = None,
    n_subjects: int | None = None,
    profile_filter: str | None = None,
) -> list[dict[str, Any]]:
    manifest_path = Path(manifest) if manifest else Path(manifest_root) / f"{dataset}.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest introuvable pour {dataset}: {manifest_path}")
    df = pd.read_csv(manifest_path)
    required = {"subject", "input_path", "brain_mask_path", "has_gold"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Manifest {manifest_path} invalide, colonnes manquantes: {missing}")

    work = df.copy()
    work = work[work["has_gold"].astype(str).str.lower().isin({"true", "1", "yes", "y"})].copy()
    if "qc_only" in work.columns:
        work = work[~work["qc_only"].astype(str).str.lower().isin({"true", "1", "yes", "y"})].copy()
    if profile_filter:
        wanted = {item.strip() for item in profile_filter.split(",") if item.strip()}
        work = work[work.get("profile", "").astype(str).isin(wanted)].copy()
    else:
        profile = work.get("profile", pd.Series([""] * len(work), index=work.index)).astype(str)
        pediatric = profile.str.contains("infant|pediatric", case=False, regex=True, na=False)
        if pediatric.any():
            work = work[pediatric].copy()

    if subjects:
        ordered = [str(subject) for subject in subjects]
        wanted = set(ordered)
        work = work[work["subject"].astype(str).isin(wanted)].copy()
        missing_subjects = sorted(wanted - set(work["subject"].astype(str)))
        if missing_subjects:
            raise RuntimeError(f"Sujets absents du manifest {dataset}: {missing_subjects}")
        order = {subject: idx for idx, subject in enumerate(ordered)}
        work["_subject_order"] = work["subject"].astype(str).map(order)
        work = work.sort_values("_subject_order").drop(columns=["_subject_order"])
    if n_subjects is not None:
        work = work.head(int(n_subjects)).copy()

    records: list[dict[str, Any]] = []
    for row in work.to_dict("records"):
        input_path = Path(str(row.get("input_path") or row.get("t1_path") or ""))
        gold_path = Path(str(row.get("brain_mask_path") or ""))
        if not input_path.exists() or not gold_path.exists():
            continue
        records.append(
            {
                "dataset": dataset,
                "subject": str(row["subject"]),
                "session": str(row.get("session", "")),
                "profile": str(row.get("profile", "")),
                "modality": str(row.get("modality", "")),
                "input_path": input_path,
                "gold_mask_path": gold_path,
                "label_path": str(row.get("label_path", "")),
                "manifest": str(manifest_path),
                "gt_type": "derived_from_labels" if "label" in str(row.get("notes", "")).lower() else "brain_mask",
                "gt_volume_cm3_manifest": _safe_float(row.get("mask_volume_cm3", np.nan)),
            }
        )
    if not records:
        raise RuntimeError(f"Aucun sujet pediatrique gold exploitable pour {dataset} dans {manifest_path}")
    return records


def _tool_registry(require_all_tools: bool) -> dict[str, Any]:
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
        ],
    )
    hdbet_device = os.environ.get("HDBET_DEVICE")
    if hdbet_device is None:
        try:
            import torch

            hdbet_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            hdbet_device = "cuda"

    missing = []
    if bet_bin is None:
        missing.append("FSL BET")
    if not synthstrip.get("executable"):
        missing.append("SynthStrip")
    if hdbet_bin is None:
        missing.append("HD-BET")
    if missing and require_all_tools:
        raise FileNotFoundError("SOTA manquants: " + ", ".join(missing))

    return {
        "bet_bin": bet_bin,
        "synthstrip": synthstrip,
        "hdbet_bin": hdbet_bin,
        "hdbet_device": hdbet_device,
        "hdbet_timeout_s": int(os.environ.get("HDBET_TIMEOUT_S", "900")),
        "hdbet_disable_tta": _bool_env("HDBET_DISABLE_TTA", False),
        "synthstrip_gpu": _bool_env("SYNTHSTRIP_GPU", False),
    }


def _run_fsl_bet(record: dict[str, Any], out_dir: Path, log_dir: Path, tools: dict[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    bet_bin = tools.get("bet_bin")
    if not bet_bin:
        return None, {"error": "FSL BET missing", "runtime_s": np.nan, "returncode": np.nan}
    subject = _safe_subject(record["subject"])
    _clean_subject(out_dir, subject, "fsl_bet")
    out_brain = out_dir / f"{subject}_fsl_bet_brain.nii.gz"
    out_mask = out_dir / f"{subject}_fsl_bet_mask.nii.gz"
    cmd = [bet_bin, str(record["input_path"]), str(out_brain), "-m", "-R"]
    meta = _run_command(cmd, 240, log_dir / f"{subject}_fsl_bet")
    mask = _canonicalize_mask([out_mask, out_dir / f"{subject}_fsl_bet_brain_mask.nii.gz"], out_mask)
    if meta.get("returncode") != 0 and mask is None:
        meta["error"] = f"FSL BET failed rc={meta.get('returncode')}: {meta.get('stderr_tail', '')}"
    return mask, meta


def _run_synthstrip(record: dict[str, Any], out_dir: Path, log_dir: Path, tools: dict[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    synthstrip = tools.get("synthstrip") or {}
    if not synthstrip.get("executable"):
        return None, {"error": "SynthStrip missing", "runtime_s": np.nan, "returncode": np.nan}
    subject = _safe_subject(record["subject"])
    _clean_subject(out_dir, subject, "synthstrip")
    out_brain = out_dir / f"{subject}_synthstrip_brain.nii.gz"
    out_mask = out_dir / f"{subject}_synthstrip_mask.nii.gz"
    exe = synthstrip["executable"]
    local_script = False
    try:
        exe_path = Path(str(exe))
        if exe_path.is_file():
            head = exe_path.read_bytes()[:500]
            local_script = b"python" in head[:200] or b"import " in head
    except Exception:
        local_script = False
    if synthstrip.get("mode") == "script" or local_script:
        cmd: list[str | Path] = [sys.executable, exe, "-i", record["input_path"], "-o", out_brain, "-m", out_mask]
    else:
        cmd = [exe, "-i", record["input_path"], "-o", out_brain, "-m", out_mask]
    if synthstrip.get("weights"):
        cmd += ["--model", synthstrip["weights"]]
    if tools.get("synthstrip_gpu"):
        cmd.append("-g")
    meta = _run_command(cmd, 420, log_dir / f"{subject}_synthstrip")
    mask = out_mask if out_mask.exists() else None
    if meta.get("returncode") != 0 and mask is None:
        meta["error"] = f"SynthStrip failed rc={meta.get('returncode')}: {meta.get('stderr_tail', '')}"
    return mask, meta


def _run_hdbet(record: dict[str, Any], out_dir: Path, log_dir: Path, tools: dict[str, Any]) -> tuple[Path | None, dict[str, Any]]:
    hdbet_bin = tools.get("hdbet_bin")
    if not hdbet_bin:
        return None, {"error": "HD-BET missing", "runtime_s": np.nan, "returncode": np.nan}
    subject = _safe_subject(record["subject"])
    _clean_subject(out_dir, subject, "hdbet")
    out_stub = out_dir / f"{subject}_hdbet.nii.gz"
    out_mask = out_dir / f"{subject}_hdbet_mask.nii.gz"
    cmd: list[str | Path] = [
        hdbet_bin,
        "-i",
        record["input_path"],
        "-o",
        out_stub,
        "-device",
        str(tools["hdbet_device"]),
        "--save_bet_mask",
        "--no_bet_image",
    ]
    if tools.get("hdbet_disable_tta"):
        cmd.append("--disable_tta")
    meta = _run_command(cmd, tools["hdbet_timeout_s"], log_dir / f"{subject}_hdbet")
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
        meta.update(_repair_mask_spatial_reference(mask, record["input_path"]))
    if meta.get("returncode") != 0 and mask is None:
        meta["error"] = f"HD-BET failed rc={meta.get('returncode')}: {meta.get('stderr_tail', '')}"
    elif mask is None:
        meta["error"] = "HD-BET finished but mask not found"
    return mask, meta


METHOD_RUNNERS = {
    "fsl_bet": _run_fsl_bet,
    "synthstrip": _run_synthstrip,
    "hdbet": _run_hdbet,
}


def _summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("dice", "hd95", "assd", "vol_pred_cm3", "vol_gs_cm3", "runtime_s"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    ok = df[(df["error"].fillna("").astype(str).str.strip() == "") & df["dice"].notna()].copy()
    if ok.empty:
        return pd.DataFrame(columns=["method", "n", "dice_mean", "hd95_mean", "assd_mean"])
    return (
        ok.groupby("method")
        .agg(
            n=("dice", "count"),
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            dice_min=("dice", "min"),
            dice_max=("dice", "max"),
            hd95_mean=("hd95", "mean"),
            assd_mean=("assd", "mean"),
            runtime_mean_s=("runtime_s", "mean"),
        )
        .reset_index()
        .sort_values("dice_mean", ascending=False)
    )


def run_m2_pediatric_sota_dataset(
    dataset: str,
    *,
    manifest: str | Path | None = None,
    manifest_root: str | Path = "datasets/manifests",
    output_root: str | Path = "outputs/pediatric_sota_eval",
    methods: Iterable[str] = DEFAULT_METHODS,
    subjects: Iterable[str] | None = None,
    n_subjects: int | None = None,
    profile_filter: str | None = None,
    force: bool = False,
    require_all_tools: bool = True,
    require_all_complete: bool = True,
) -> dict[str, Any]:
    methods = _parse_csv_list(methods, DEFAULT_METHODS)
    unknown = sorted(set(methods) - set(METHOD_RUNNERS))
    if unknown:
        raise ValueError(f"Methodes inconnues: {unknown}. Valides: {sorted(METHOD_RUNNERS)}")

    tools = _tool_registry(require_all_tools=require_all_tools)
    records = _load_records(
        dataset,
        manifest_root=manifest_root,
        manifest=manifest,
        subjects=subjects,
        n_subjects=n_subjects,
        profile_filter=profile_filter,
    )
    out_base = Path(output_root) / dataset
    log_dir = out_base / "logs"
    pred_root = out_base / "predictions"
    long_csv = out_base / "sota_results_long.csv"
    live_csv = out_base / "sota_results_live.csv"
    summary_csv = out_base / "sota_method_summary.csv"
    progress_json = out_base / "m2_pediatric_sota_progress.json"
    out_base.mkdir(parents=True, exist_ok=True)
    if force:
        for path in (long_csv, live_csv, summary_csv, progress_json):
            path.unlink(missing_ok=True)

    rows: list[dict[str, Any]] = []
    total = len(records) * len(methods)
    done = 0
    print(f"M2-PEDIATRIC-SOTA {dataset}: {len(records)} sujets, methodes={','.join(methods)}")
    print(f"FSL BET    : {tools.get('bet_bin')}")
    ss = tools.get("synthstrip") or {}
    print(f"SynthStrip : {ss.get('mode')}={ss.get('executable')} | weights={ss.get('weights', '')}")
    print(
        f"HD-BET     : {tools.get('hdbet_bin')} | device={tools.get('hdbet_device')} "
        f"| TTA={'off' if tools.get('hdbet_disable_tta') else 'on'}"
    )
    for index, record in enumerate(records, 1):
        subject = record["subject"]
        subject_dir = pred_root / _safe_subject(subject)
        subject_dir.mkdir(parents=True, exist_ok=True)
        for method in methods:
            done += 1
            label = METHOD_LABELS[method]
            _write_progress(
                progress_json,
                {
                    "stage": "running",
                    "dataset": dataset,
                    "subject": subject,
                    "method": label,
                    "index": index,
                    "subjects": len(records),
                    "done": done,
                    "total": total,
                },
            )
            row = {
                "dataset": dataset,
                "subject": subject,
                "session": record.get("session", ""),
                "profile": record.get("profile", ""),
                "modality": record.get("modality", ""),
                "method": label,
                "input_path": str(record["input_path"]),
                "gold_mask_path": str(record["gold_mask_path"]),
                "label_path": record.get("label_path", ""),
                "gt_type": record.get("gt_type", ""),
                "gt_volume_cm3_manifest": record.get("gt_volume_cm3_manifest", np.nan),
                "mask_path": "",
                "runtime_s": np.nan,
                "returncode": np.nan,
                "error": "",
            }
            try:
                mask_path, meta = METHOD_RUNNERS[method](record, subject_dir, log_dir, tools)
                row.update({k: v for k, v in meta.items() if k not in {"stderr_tail"}})
                if meta.get("error"):
                    row["error"] = str(meta["error"])
                if mask_path and Path(mask_path).exists():
                    row["mask_path"] = str(mask_path)
                    row.update(_compute_metrics_3d(mask_path, record["gold_mask_path"], subject_dir / "_metric_inputs"))
                elif not row["error"]:
                    row["error"] = "mask_not_found"
            except Exception as exc:
                row["error"] = str(exc)
            rows.append(row)
            pd.DataFrame(rows).to_csv(live_csv, index=False)
            if row["error"]:
                print(f"  {label} | {subject}: ERROR {row['error']}")
            else:
                print(f"  {label} | {subject}: Dice={row['dice']:.4f} HD95={row['hd95']:.3f} ASSD={row['assd']:.3f}")

    long_df = pd.DataFrame(rows)
    summary = _summarize(rows)
    long_df.to_csv(long_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _write_progress(progress_json, {"stage": "done", "dataset": dataset, "subjects": len(records), "total": total})

    if require_all_complete:
        failures = long_df[long_df["error"].fillna("").astype(str).str.strip() != ""]
        if len(failures):
            raise RuntimeError(
                f"M2-PEDIATRIC-SOTA {dataset} incomplet: "
                + "; ".join(f"{r.method}/{r.subject}: {r.error}" for r in failures.itertuples(index=False))
            )
    return {
        "dataset": dataset,
        "long": long_df,
        "summary": summary,
        "long_csv": long_csv,
        "summary_csv": summary_csv,
        "progress_json": progress_json,
        "out_base": out_base,
    }


def run_m2_pediatric_sota_all(
    datasets: Iterable[str] = DEFAULT_PEDIATRIC_DATASETS,
    *,
    manifest_root: str | Path = "datasets/manifests",
    output_root: str | Path = "outputs/pediatric_sota_eval",
    methods: Iterable[str] = DEFAULT_METHODS,
    n_subjects: int | None = None,
    profile_filter: str | None = None,
    force: bool = False,
    require_all_tools: bool = True,
    require_all_complete: bool = True,
) -> dict[str, Any]:
    datasets = _parse_csv_list(datasets, DEFAULT_PEDIATRIC_DATASETS)
    methods = _parse_csv_list(methods, DEFAULT_METHODS)
    results: dict[str, Any] = {}
    all_rows: list[pd.DataFrame] = []
    for dataset in datasets:
        result = run_m2_pediatric_sota_dataset(
            dataset,
            manifest_root=manifest_root,
            output_root=output_root,
            methods=methods,
            n_subjects=n_subjects,
            profile_filter=profile_filter,
            force=force,
            require_all_tools=require_all_tools,
            require_all_complete=require_all_complete,
        )
        results[dataset] = result
        all_rows.append(result["long"])
    output_root = Path(output_root)
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined_summary = (
            combined[combined["error"].fillna("").astype(str).str.strip().eq("")]
            .groupby(["dataset", "method"])
            .agg(n=("dice", "count"), dice_mean=("dice", "mean"), hd95_mean=("hd95", "mean"), assd_mean=("assd", "mean"))
            .reset_index()
            .sort_values(["dataset", "dice_mean"], ascending=[True, False])
        )
    else:
        combined = pd.DataFrame()
        combined_summary = pd.DataFrame()
    output_root.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_root / "all_pediatric_sota_results_long.csv", index=False)
    combined_summary.to_csv(output_root / "all_pediatric_sota_summary.csv", index=False)
    return {
        "datasets": results,
        "long": combined,
        "summary": combined_summary,
        "long_csv": output_root / "all_pediatric_sota_results_long.csv",
        "summary_csv": output_root / "all_pediatric_sota_summary.csv",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_PEDIATRIC_DATASETS))
    parser.add_argument("--manifest-root", default="datasets/manifests")
    parser.add_argument("--output-root", default="outputs/pediatric_sota_eval")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="Comma list: fsl_bet,synthstrip,hdbet")
    parser.add_argument("--n-subjects", type=int)
    parser.add_argument("--profile-filter")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-missing-tools", dest="require_all_tools", action="store_false")
    parser.add_argument("--allow-failures", dest="require_all_complete", action="store_false")
    parser.set_defaults(require_all_tools=True, require_all_complete=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_m2_pediatric_sota_all(
        datasets=_parse_csv_list(args.datasets, DEFAULT_PEDIATRIC_DATASETS),
        manifest_root=args.manifest_root,
        output_root=args.output_root,
        methods=_parse_csv_list(args.methods, DEFAULT_METHODS),
        n_subjects=args.n_subjects,
        profile_filter=args.profile_filter,
        force=args.force,
        require_all_tools=bool(args.require_all_tools),
        require_all_complete=bool(args.require_all_complete),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
