#!/usr/bin/env python3
"""Runtime diagnostics for long SAM-NeuroAdapt notebook runs.

The monitor writes JSONL snapshots that survive notebook/kernel crashes.  It is
intentionally dependency-light so it can run in a separate terminal while M1/M2
cells execute inside Jupyter.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_STOP = False


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(obj: Any) -> str:
    return str(obj)


def _read_meminfo() -> dict[str, float]:
    out: dict[str, float] = {}
    path = Path("/proc/meminfo")
    if not path.exists():
        return out
    for line in path.read_text(errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        value = rest.strip().split()[0]
        try:
            out[key] = float(value)
        except ValueError:
            pass
    return out


def _read_loadavg() -> dict[str, Any]:
    try:
        parts = Path("/proc/loadavg").read_text().split()
        return {
            "load1": float(parts[0]),
            "load5": float(parts[1]),
            "load15": float(parts[2]),
            "running": parts[3],
            "last_pid": int(parts[4]),
        }
    except Exception:
        return {}


def _disk(path: str | Path) -> dict[str, float] | None:
    try:
        usage = shutil.disk_usage(path)
    except Exception:
        return None
    total = usage.total / 1024**3
    used = usage.used / 1024**3
    free = usage.free / 1024**3
    return {
        "total_gib": round(total, 3),
        "used_gib": round(used, 3),
        "free_gib": round(free, 3),
        "used_percent": round((used / total) * 100.0, 2) if total else 0.0,
    }


def _run(cmd: list[str], timeout: float = 5.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except Exception:
        return None


def _gpu_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {"available": False, "gpus": [], "processes": []}
    smi = shutil.which("nvidia-smi")
    if not smi:
        return out
    query = [
        smi,
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    proc = _run(query)
    if proc and proc.returncode == 0:
        out["available"] = True
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            out["gpus"].append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mib": _float(parts[2]),
                    "memory_used_mib": _float(parts[3]),
                    "memory_free_mib": _float(parts[4]),
                    "gpu_util_percent": _float(parts[5]),
                    "mem_util_percent": _float(parts[6]),
                    "temperature_c": _float(parts[7]),
                    "power_w": _float(parts[8]),
                }
            )
    elif proc:
        out["query_stderr"] = proc.stderr.strip()

    proc_apps = _run(
        [
            smi,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc_apps and proc_apps.returncode == 0:
        for line in proc_apps.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                out["processes"].append(
                    {
                        "pid": _int(parts[0]),
                        "name": parts[1],
                        "used_memory_mib": _float(parts[2]),
                    }
                )
    elif proc_apps:
        out["process_query_stderr"] = proc_apps.stderr.strip()
    return out


def _float(value: str | int | float | None) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _int(value: str | int | None) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _process_snapshot() -> list[dict[str, Any]]:
    proc = _run(["ps", "-eo", "pid,ppid,stat,etime,%cpu,%mem,rss,vsz,cmd"], timeout=3.0)
    if not proc or proc.returncode != 0:
        return []
    keep = ("python", "jupyter", "ipykernel", "ants", "sam", "nvidia", "git", "hd-bet", "synthstrip")
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines()[1:]:
        if not any(k in line.lower() for k in keep):
            continue
        parts = line.split(None, 8)
        if len(parts) < 9:
            continue
        pid, ppid, stat, etime, pcpu, pmem, rss, vsz, cmd = parts
        rows.append(
            {
                "pid": _int(pid),
                "ppid": _int(ppid),
                "stat": stat,
                "etime": etime,
                "pcpu": _float(pcpu),
                "pmem": _float(pmem),
                "rss_mib": round((_float(rss) or 0.0) / 1024.0, 2),
                "vsz_mib": round((_float(vsz) or 0.0) / 1024.0, 2),
                "cmd": cmd,
            }
        )
    rows.sort(key=lambda r: r.get("rss_mib") or 0.0, reverse=True)
    return rows[:30]


def _read_progress(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {"exists": False, "path": str(p)}
    try:
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            data["exists"] = True
            data["path"] = str(p)
            return data
    except Exception as exc:
        return {"exists": True, "path": str(p), "read_error": str(exc)}
    return {"exists": True, "path": str(p), "read_error": "not a JSON object"}


def snapshot(tag: str = "runtime", progress: str | Path | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    disks: dict[str, Any] = {}
    for p in (".", "/tmp", "/dev/shm", "/mnt/wsl"):
        d = _disk(p)
        if d is not None:
            disks[p] = d
    return {
        "type": "snapshot",
        "timestamp": _now(),
        "tag": tag,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "cwd": str(Path.cwd()),
        "host": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "load": _read_loadavg(),
        "memory": _read_meminfo(),
        "disk": disks,
        "gpu": _gpu_snapshot(),
        "processes": _process_snapshot(),
        "progress": _read_progress(progress),
        "extra": extra or {},
    }


def log_event(path: str | Path, event: str, tag: str = "runtime", snapshot_data: dict[str, Any] | None = None, **fields: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "event",
        "timestamp": _now(),
        "tag": tag,
        "pid": os.getpid(),
        "event": event,
    }
    payload.update(fields)
    if snapshot_data is not None:
        payload["snapshot"] = snapshot_data
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _write_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _handle_signal(signum, _frame) -> None:
    global _STOP
    _STOP = True


def monitor(args: argparse.Namespace) -> None:
    global _STOP
    _STOP = False
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    started = time.time()
    _write_jsonl(args.output, {"type": "event", "event": "monitor_started", "timestamp": _now(), "tag": args.tag, "pid": os.getpid()})
    while not _STOP:
        snap = snapshot(args.tag, args.progress)
        _write_jsonl(args.output, snap)
        progress = snap.get("progress") or {}
        if args.stop_when_finished and progress.get("stage") == "finished":
            break
        if args.max_seconds and (time.time() - started) >= args.max_seconds:
            break
        time.sleep(max(0.2, float(args.interval)))
    if _STOP:
        _write_jsonl(args.output, {"type": "event", "event": "monitor_signal", "signal": signal.SIGTERM, "timestamp": _now(), "tag": args.tag, "pid": os.getpid()})
    _write_jsonl(args.output, {"type": "event", "event": "monitor_stopped", "timestamp": _now(), "tag": args.tag, "pid": os.getpid(), "snapshot": snapshot(args.tag, args.progress)})


def summarize(args: argparse.Namespace) -> None:
    path = Path(args.input)
    if not path.exists():
        print(f"missing: {path}")
        return
    entries: list[dict[str, Any]] = []
    events: dict[str, int] = {}
    min_mem = None
    max_gpu = None
    last_progress = {}
    for line in path.read_text(errors="ignore").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        entries.append(obj)
        if obj.get("type") == "event":
            name = obj.get("event") or "event"
            events[name] = events.get(name, 0) + 1
        snap = obj if obj.get("type") == "snapshot" else obj.get("snapshot") or {}
        mem = (snap.get("memory") or {}).get("MemAvailable_kb")
        if mem is not None:
            gib = float(mem) / 1024**2
            min_mem = gib if min_mem is None else min(min_mem, gib)
        for gpu in (snap.get("gpu") or {}).get("gpus") or []:
            used = gpu.get("memory_used_mib")
            if used is not None:
                max_gpu = float(used) if max_gpu is None else max(max_gpu, float(used))
        progress = snap.get("progress") or obj.get("progress") or {}
        if progress:
            last_progress = progress
    print(f"entries: {len(entries)}")
    if entries:
        print(f"first: {entries[0].get('timestamp')} | last: {entries[-1].get('timestamp')}")
    if events:
        print(f"events: {events}")
    if min_mem is not None:
        print(f"min MemAvailable: {min_mem:.2f} GiB")
    if max_gpu is not None:
        print(f"max GPU memory used: {max_gpu:.0f} MiB")
    if last_progress:
        print(
            "last progress: "
            f"stage={last_progress.get('stage')} subject={last_progress.get('subject')} "
            f"index={last_progress.get('index')}/{last_progress.get('total')} "
            f"updated={last_progress.get('updated_at')}"
        )
    print("last 5 entries:")
    for obj in entries[-5:]:
        print(json.dumps(obj, ensure_ascii=False, default=_json_default)[:2000])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_monitor = sub.add_parser("monitor")
    p_monitor.add_argument("--tag", default="runtime")
    p_monitor.add_argument("--progress")
    p_monitor.add_argument("--output", required=True)
    p_monitor.add_argument("--interval", type=float, default=5.0)
    p_monitor.add_argument("--max-seconds", type=float, default=None)
    p_monitor.add_argument("--stop-when-finished", action="store_true")
    p_snap = sub.add_parser("snapshot")
    p_snap.add_argument("--tag", default="runtime")
    p_snap.add_argument("--progress")
    p_snap.add_argument("--output", required=True)
    p_sum = sub.add_parser("summarize")
    p_sum.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "monitor":
        monitor(args)
    elif args.cmd == "snapshot":
        _write_jsonl(args.output, snapshot(args.tag, args.progress))
    elif args.cmd == "summarize":
        summarize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

