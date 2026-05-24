#!/usr/bin/env python3
"""Batch supervisor v2.2: per-video subprocess, cooldown, auto-recovery.

Each video runs in its own subprocess via --video mode.
Cooldown between videos lets WDDM/CUDA reclaim resources.
Automatic RECOVERY_MODE after consecutive severe errors.
"""

import argparse, json, re, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

PYTHON_EXE = Path(sys.executable)
MAIN_SCRIPT = Path(__file__).resolve().parent / "batch_system_c_cut_v2.py"
RUN_TAG = "SYSTEM_C_CUT_V2_01"

SEVERE_KEYWORDS = [
    "CUDA out of memory",
    "CUDA error",
    "DLL load failed",
    "fatal error",
    "Segmentation fault",
    "OSError",
    "ACCESS_VIOLATION",
]

MEMORY_LOAD_MAX = 90
PAGEFILE_LOAD_MAX = 90
GPU_MEM_MAX = 95


def safe_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\s]+', "_", name).strip("._")
    return cleaned or "video"


def parse_group_name(filename: str) -> str:
    """Parse group name from filename: 'kt (5).ts' -> 'kt'."""
    return filename.split("(")[0].strip().lower()


def first_match(base: Path, pattern: str) -> Path | None:
    found = sorted(base.glob(pattern))
    return found[-1] if found else None


def load_status(status_path: Path) -> dict:
    if status_path.exists():
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_status(status_path: Path, status: dict) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_path.replace(status_path)


def log_recovery(path: Path, message: str) -> None:
    ts = datetime.now().isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


# -- fast skip ----------------------------------------------------------

def check_pass_complete(batch_dir: Path, video_stem: str, backup_dir: Path) -> bool:
    """6-condition check without spawning Python.  True  => safe to skip."""
    tag = safe_name(video_stem)
    base = batch_dir / f"{tag}_{RUN_TAG}"
    if not base.exists():
        return False

    mp4 = first_match(base, "06_output_video/*_cut_final_CLEAN_*.mp4")
    if not mp4 or mp4.stat().st_size == 0:
        return False

    verify = first_match(base, "07_reports/*_final_verify_*.txt")
    if not verify:
        return False

    text = verify.read_text(encoding="utf-8", errors="replace")
    if "result: PASS" not in text and "result:PASS" not in text:
        return False

    backup = backup_dir / mp4.name
    if not backup.exists():
        return False
    if backup.stat().st_size != mp4.stat().st_size:
        return False

    return True


# -- ffprobe duration (CPU only, no CUDA) -------------------------------

def get_video_duration(video_path: Path) -> float | None:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            return float(r.stdout.strip().split("\n")[-1])
    except Exception:
        pass
    return None


def get_video_fps(video_path: Path) -> float | None:
    """Read r_frame_rate via ffprobe, fallback to avg_frame_rate if 0/0.

    Returns fps as float, or None if unreadable.
    """
    for entry in ("stream=r_frame_rate", "stream=avg_frame_rate"):
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", entry,
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                continue
            for line in r.stdout.strip().split("\n"):
                line = line.strip()
                if not line or "/" not in line:
                    continue
                parts = line.split("/")
                if len(parts) == 2:
                    try:
                        num, den = int(parts[0]), int(parts[1])
                    except ValueError:
                        continue
                    if den != 0 and num != 0:
                        return num / den
        except Exception:
            continue
    return None


# -- severe-error detection ---------------------------------------------

def detect_severe_error(exit_code: int, log_path: Path | None) -> bool:
    if exit_code == 3221225477:
        return True
    if exit_code < 0:
        return True

    if log_path and log_path.exists():
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            text_lower = text.lower()
            for kw in SEVERE_KEYWORDS:
                if kw.lower() in text_lower:
                    return True
            if "Loading checkpoint shards".lower() in text_lower and exit_code != 0:
                return True
        except Exception:
            pass
    return False


def detect_silent_severe(log_path: Path | None) -> bool:
    """Model loading never reached 100% -- GPU likely hung.

    True when log mentions checkpoint loading but never shows
    "Loading checkpoint shards: 100%", meaning the model failed
    to load and the script bailed out without doing any ASR work.
    """
    if not log_path or not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        text_lower = text.lower()
    except Exception:
        return False

    if "loading checkpoint shards" not in text_lower:
        return False
    if "loading checkpoint shards: 100%" in text_lower:
        return False
    return True


# -- strict post-subprocess PASS check ----------------------------------

def check_pass_strict(batch_dir: Path, video_stem: str,
                      exit_code: int, backup_dir: Path) -> tuple[bool, str, Path | None]:
    """Returns (passed, reason, mp4_path_or_None)."""
    if exit_code != 0:
        return False, f"exit_code={exit_code}", None

    tag = safe_name(video_stem)
    base = batch_dir / f"{tag}_{RUN_TAG}"

    mp4 = first_match(base, "06_output_video/*_cut_final_CLEAN_*.mp4")
    if not mp4 or mp4.stat().st_size == 0:
        return False, "MP4 missing or empty", None

    verify = first_match(base, "07_reports/*_final_verify_*.txt")
    if not verify:
        return False, "verify report missing", mp4

    text = verify.read_text(encoding="utf-8", errors="replace")
    if "result: PASS" not in text and "result:PASS" not in text:
        return False, "verify result not PASS", mp4

    backup_path = backup_dir / mp4.name
    try:
        shutil.copy2(mp4, backup_path)
    except Exception as e:
        return False, f"backup copy failed: {e}", mp4

    if backup_path.stat().st_size != mp4.stat().st_size:
        return False, "backup size mismatch", mp4

    return True, "PASS", mp4


# -- health probe (no model loading) ------------------------------------

def check_residual_procs() -> list[str]:
    bad = []
    try:
        r = subprocess.run(
            ["tasklist", "/fo", "csv", "/nh"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.lower().split("\n"):
            if any(x in line for x in ["batch_system_c_cut_v2", "ffprobe.exe", "ffmpeg.exe"]):
                bad.append(line.strip())
    except Exception:
        pass
    return bad


def nvidia_smi_ok() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            return False
        parts = r.stdout.strip().split(",")
        if len(parts) >= 3:
            used = int(parts[0].strip())
            total = int(parts[2].strip())
            return total > 0 and (used / total * 100) < GPU_MEM_MAX
        return False
    except Exception:
        return False


def cuda_probe() -> bool:
    probe_code = "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'CUDA_NOK')"
    try:
        r = subprocess.run(
            [str(PYTHON_EXE), "-c", probe_code],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0 and "CUDA_OK" in r.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def check_memory_pressure() -> bool:
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]
    ms = MEMORYSTATUSEX()
    ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))

    mem_load = ms.dwMemoryLoad
    pagefile_load = ((1 - ms.ullAvailPageFile / ms.ullTotalPageFile) * 100
                     if ms.ullTotalPageFile > 0 else 0)
    return mem_load < MEMORY_LOAD_MAX and pagefile_load < PAGEFILE_LOAD_MAX


def run_health_probe() -> tuple[bool, str]:
    residual = check_residual_procs()
    if residual:
        return False, f"residual processes: {len(residual)}"
    if not nvidia_smi_ok():
        return False, "nvidia-smi check failed"
    if not check_memory_pressure():
        return False, "memory / pagefile pressure too high"
    if not cuda_probe():
        return False, "CUDA probe failed"
    return True, "all probes passed"


# -- cooldown -----------------------------------------------------------

def do_cooldown(seconds: int, reason: str) -> None:
    print(f"COOLING_DOWN {seconds}s ({reason})", flush=True)
    time.sleep(seconds)
    print("COOLING_DONE", flush=True)


# -- sorting ------------------------------------------------------------

def sort_by_priority(videos: list[Path], supervisor_status: dict) -> list[Path]:
    """Historical severe failures first, then by duration desc."""
    scored = []
    for v in videos:
        entry = supervisor_status.get(v.name, {})
        prev_status = entry.get("status", "")
        prev_exit = entry.get("exit_code", 0)

        if prev_status in ("FAIL_NEED_REVIEW", "FAIL") and prev_exit == 3221225477:
            score = 0
        elif prev_status in ("FAIL_NEED_REVIEW", "FAIL"):
            score = 1
        else:
            score = 2

        dur = get_video_duration(v)
        scored.append((score, v, dur))

    scored.sort(key=lambda x: (x[0], -(x[2] or 0)))
    return [s[1] for s in scored]


# -- dry-run ------------------------------------------------------------

def dry_run_report(videos: list[Path], batch_dir: Path,
                   backup_dir: Path, supervisor_status: dict,
                   group_targets: dict | None = None) -> None:
    print("=" * 60)
    print("DRY RUN -- no video processing will start")
    print("=" * 60)

    skip_list = []
    process_list = []

    for v in videos:
        key = v.name
        entry = supervisor_status.get(key, {})
        prev_status = entry.get("status", "")
        prev_exit = entry.get("exit_code", 0)

        if check_pass_complete(batch_dir, v.stem, backup_dir):
            skip_list.append((key, prev_status))
        else:
            duration = get_video_duration(v)
            dur_str = f"{duration / 3600:.1f}h" if duration else "unknown"
            is_long = duration and duration > 10800

            risks = []
            if prev_exit == 3221225477:
                risks.append("prev ACCESS_VIOLATION")
            if prev_status in ("FAIL_NEED_REVIEW", "FAIL"):
                risks.append(f"prev {prev_status}")
            if duration is None:
                risks.append("unknown duration")
            if is_long:
                risks.append(f"long video {duration / 3600:.1f}h")

            cooldown_hint = "60s"
            if risks or is_long:
                cooldown_hint = "300s"
            elif prev_status == "FAIL":
                cooldown_hint = "180s"

            process_list.append({
                "name": key, "stem": v.stem,
                "duration": duration, "dur_str": dur_str,
                "high_risk": bool(risks), "risks": risks,
                "prev_status": prev_status, "prev_exit": prev_exit,
                "cooldown_hint": cooldown_hint,
            })

    print(f"\nSKIP ({len(skip_list)} videos):")
    for name, st in skip_list:
        print(f"  {name}  [{st}]")

    print(f"\nPROCESS ({len(process_list)} videos):")
    for p in process_list:
        tag = "HIGH RISK" if p["high_risk"] else "ok"
        print(f"  {p['name']}")
        print(f"    duration={p['dur_str']}  status={p['prev_status']}"
              f"  exit={p['prev_exit']}  cooldown={p['cooldown_hint']}")
        if p["risks"]:
            print(f"    risks: {', '.join(p['risks'])}")

    # --- group target fps ---
    if group_targets:
        print(f"\n{'─' * 60}")
        print("GROUP TARGET FPS")
        print(f"{'─' * 60}")
        for gname in sorted(group_targets):
            gt = group_targets[gname]
            fps_list = [f"{v['source_fps']:.0f}" if v['source_fps'] else '?' for v in gt['videos']]
            print(f"  [{gname}] target_fps={gt['target_fps']}  reason={gt['reason']}")
            print(f"          videos={gt['video_count']}  source_fps={fps_list}")

    print(f"\n{len(skip_list)} SKIP / {len(process_list)} PROCESS / {len(videos)} total")


# -- pre-flight input spec check -----------------------------------------

def run_input_spec_check(videos: list[Path], batch_dir: Path,
                         backup_dir: Path, status: dict,
                         logs_dir: Path) -> list[dict]:
    """Check FPS of all PROCESS videos before any processing starts.

    Only checks videos that are NOT already PASS.
    Returns list of anomaly dicts (empty list = all OK).
    """
    anomalies: list[dict] = []

    for v in videos:
        key = v.name

        # skip already PASS
        if check_pass_complete(batch_dir, v.stem, backup_dir):
            continue
        if isinstance(status.get(key, {}), dict):
            if status[key].get("status") == "PASS":
                continue

        fps = get_video_fps(v)

        if fps is None:
            anomalies.append({
                "video": key,
                "path": str(v),
                "fps": None,
                "issue": "FPS_UNKNOWN",
                "detail": "ffprobe cannot read fps from video stream",
            })
            continue

        if 44.5 <= fps <= 45.5:
            continue
        if 59.5 <= fps <= 60.5:
            continue

        anomalies.append({
            "video": key,
            "path": str(v),
            "fps": round(fps, 4),
            "issue": "unsupported input fps",
            "detail": f"detected fps={fps:.4f}, expected 45 or 60",
        })

    return anomalies


def write_input_spec_report(anomalies: list[dict], inbox: Path,
                            logs_dir: Path) -> Path:
    """Write pre-flight anomaly report. Returns report path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = logs_dir / f"input_spec_check_{ts}.txt"
    logs_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append("INPUT SPEC CHECK REPORT")
    lines.append("=" * 70)
    lines.append(f"Timestamp:  {datetime.now().isoformat()}")
    lines.append(f"Inbox:      {inbox}")
    lines.append(f"Mode:       PRE-FLIGHT (before any processing)")
    lines.append("")
    lines.append(f"RESULT: BATCH_BLOCKED_UNSUPPORTED_FPS")
    lines.append(f"Anomaly count: {len(anomalies)}")
    lines.append("")
    lines.append("--- Anomaly videos ---")
    lines.append("")
    for i, a in enumerate(anomalies, 1):
        lines.append(f"  [{i}] {a['video']}")
        lines.append(f"      Path:   {a['path']}")
        lines.append(f"      FPS:    {a['fps']}" if a["fps"] is not None
                     else f"      FPS:    UNAVAILABLE")
        lines.append(f"      Issue:  {a['issue']}")
        lines.append(f"      Detail: {a['detail']}")
        lines.append("")
    lines.append("=" * 70)
    lines.append("USER_ACTION_REQUIRED: Unsupported input fps detected.")
    lines.append("Batch was NOT started.")
    lines.append("No subprocess spawned. No GPU used. No ASR run. No output generated.")
    lines.append("Fix the above videos and re-run the supervisor command.")
    lines.append("=" * 70)
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# -- group target fps ----------------------------------------------------

def build_group_target_fps(videos: list[Path]) -> tuple[dict, list[dict]]:
    """Group videos by filename prefix, compute target_fps per group.

    Rules:
      - All 45fps       -> target_fps = 45
      - Mixed 45+60     -> target_fps = 45 (unify to lower)
      - All 60fps       -> target_fps = 60
      - Non-standard fps or unreadable -> anomaly (batch blocked)

    Returns (group_targets, anomalies).
    """
    groups: dict[str, list[dict]] = {}
    for v in videos:
        gname = parse_group_name(v.name)
        fps = get_video_fps(v)
        if gname not in groups:
            groups[gname] = []
        groups[gname].append({"path": v, "name": v.name, "fps": fps})

    group_targets: dict[str, dict] = {}
    anomalies: list[dict] = []

    for gname, vlist in sorted(groups.items()):
        valid_fps = [v["fps"] for v in vlist if v["fps"] is not None]
        unreadable = [v for v in vlist if v["fps"] is None]

        for v in unreadable:
            anomalies.append({
                "video": v["name"],
                "path": str(v["path"]),
                "fps": None,
                "issue": "FPS_UNKNOWN",
                "detail": f"ffprobe cannot read fps from video stream (group={gname})",
            })

        if not valid_fps:
            continue

        has_45 = any(abs(f - 45.0) < 0.1 for f in valid_fps)
        has_60 = any(abs(f - 60.0) < 1.0 for f in valid_fps)
        has_other = any(abs(f - 45.0) >= 0.1 and abs(f - 60.0) >= 1.0 for f in valid_fps)

        if has_other:
            for v in vlist:
                f = v["fps"]
                if f is not None and abs(f - 45.0) >= 0.1 and abs(f - 60.0) >= 1.0:
                    anomalies.append({
                        "video": v["name"],
                        "path": str(v["path"]),
                        "fps": round(f, 4),
                        "issue": "unsupported fps in group",
                        "detail": f"detected fps={f:.4f} in group={gname}, expected 45 or 60",
                    })
            continue

        if has_45 and has_60:
            target_fps = 45
            n_45 = sum(1 for f in valid_fps if abs(f - 45.0) < 0.1)
            n_60 = sum(1 for f in valid_fps if abs(f - 60.0) < 1.0)
            reason = f"mixed {n_45}x45fps + {n_60}x60fps, unifying to 45fps"
        elif has_45:
            target_fps = 45
            reason = f"all {len(valid_fps)} videos are 45fps"
        elif has_60:
            target_fps = 60
            reason = f"all {len(valid_fps)} videos are 60fps"
        else:
            continue

        group_targets[gname] = {
            "group_name": gname,
            "target_fps": target_fps,
            "reason": reason,
            "video_count": len(vlist),
            "videos": [{"name": v["name"], "source_fps": v["fps"]} for v in vlist],
        }

    return group_targets, anomalies


def write_group_target_fps_manifest(group_targets: dict, logs_dir: Path) -> Path:
    """Write group_target_fps.json manifest to logs_dir."""
    manifest_path = logs_dir / "group_target_fps.json"
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "groups": list(group_targets.values()),
    }
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


# -- main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch supervisor v2.2: per-video subprocess, cooldown, auto-recovery"
    )
    parser.add_argument("--inbox", required=True)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    inbox = Path(args.inbox)
    batch_dir = Path(args.batch_dir)

    if not inbox.is_dir():
        print(f"ERROR: inbox not found: {inbox}", flush=True)
        sys.exit(1)

    batch_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        (p for p in inbox.iterdir() if p.suffix.lower() in (".mp4", ".ts")),
        key=lambda p: p.name,
    )
    if not videos:
        print(f"No mp4/ts videos found in {inbox}", flush=True)
        sys.exit(1)

    logs_dir = batch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = batch_dir / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    recovery_log = logs_dir / "supervisor_recovery.log"
    pause_reason = logs_dir / "supervisor_pause_reason.txt"
    status_path = batch_dir / "supervisor_status.json"

    status = load_status(status_path)

    # --- initialise per-video entries ---
    for v in videos:
        key = v.name
        if key not in status:
            status[key] = {
                "status": "PENDING",
                "timestamp": "",
                "exit_code": None,
                "error": "",
            }

    # --- initialise / migrate _supervisor block ---
    sup = status.get("_supervisor", {})
    if not isinstance(sup, dict):
        sup = {}
    sup.setdefault("state", "STARTING")
    sup.setdefault("severe_count", 0)
    sup.setdefault("recovery_entries", 0)
    sup.setdefault("resume_index", 0)
    status["_supervisor"] = sup

    # --- resume from PAUSED_SYSTEM_UNSAFE ---
    if sup["state"] == "PAUSED_SYSTEM_UNSAFE":
        print("RESUMING from PAUSED_SYSTEM_UNSAFE -- resetting severe counters", flush=True)
        sup["state"] = "RUNNING"
        sup["severe_count"] = 0
        sup["recovery_entries"] = 0
        save_status(status_path, status)

    # --- sort & dry-run ---
    videos = sort_by_priority(videos, status)

    # --- pre-flight input spec check ---
    anomalies = run_input_spec_check(videos, batch_dir, backup_dir, status, logs_dir)
    if anomalies:
        report_path = write_input_spec_report(anomalies, inbox, logs_dir)
        sup["state"] = "BATCH_BLOCKED_UNSUPPORTED_FPS"
        sup["user_action_required"] = True
        sup["message"] = "Unsupported input fps detected. Batch was not started."
        sup["blocked_at"] = datetime.now().isoformat()
        sup["affected_videos"] = [a["video"] for a in anomalies]
        sup["report_file"] = str(report_path)
        status["_supervisor"] = sup
        save_status(status_path, status)

        print("", flush=True)
        print("=" * 70, flush=True)
        print("USER_ACTION_REQUIRED: Unsupported input fps detected", flush=True)
        print("  Batch was NOT started.", flush=True)
        print("  No subprocess spawned. No GPU used. No ASR run. No output generated.", flush=True)
        print(f"  Full report: {report_path}", flush=True)
        print("=" * 70, flush=True)
        for a in anomalies:
            fps_str = f"{a['fps']}" if a["fps"] is not None else "UNAVAILABLE"
            print(f"  ANOMALY: {a['video']}  fps={fps_str}  issue={a['issue']}", flush=True)
        print("", flush=True)
        return

    # --- group target fps scan ---
    group_targets, group_anomalies = build_group_target_fps(videos)
    if group_anomalies:
        report_path = write_input_spec_report(group_anomalies, inbox, logs_dir)
        sup["state"] = "BATCH_BLOCKED_UNSUPPORTED_FPS"
        sup["user_action_required"] = True
        sup["message"] = "Unsupported fps in group scan. Batch was not started."
        sup["blocked_at"] = datetime.now().isoformat()
        sup["affected_videos"] = [a["video"] for a in group_anomalies]
        sup["report_file"] = str(report_path)
        status["_supervisor"] = sup
        save_status(status_path, status)

        print("", flush=True)
        print("=" * 70, flush=True)
        print("USER_ACTION_REQUIRED: Group fps scan failed", flush=True)
        print("  Batch was NOT started.", flush=True)
        print(f"  Full report: {report_path}", flush=True)
        print("=" * 70, flush=True)
        for a in group_anomalies:
            fps_str = f"{a['fps']}" if a["fps"] is not None else "UNAVAILABLE"
            print(f"  ANOMALY: {a['video']}  fps={fps_str}  issue={a['issue']}", flush=True)
        print("", flush=True)
        return

    # Write group_target_fps manifest
    manifest_path = write_group_target_fps_manifest(group_targets, logs_dir)
    print(f"Group target FPS manifest: {manifest_path}", flush=True)
    for gname in sorted(group_targets):
        gt = group_targets[gname]
        print(f"  [{gname}] target_fps={gt['target_fps']}  {gt['reason']}", flush=True)

    # Build per-video target_fps lookup
    video_target_fps: dict[str, int] = {}
    for gname, gt in group_targets.items():
        for v in gt["videos"]:
            video_target_fps[v["name"]] = gt["target_fps"]

    total = len(videos)
    print(f"SUPERVISOR START: {total} videos", flush=True)
    print(f"  inbox     = {inbox}", flush=True)
    print(f"  batch_dir = {batch_dir}", flush=True)
    print(f"  logs      = {logs_dir}", flush=True)
    print(f"  backup    = {backup_dir}", flush=True)
    print(f"  status    = {status_path}", flush=True)

    if args.dry_run:
        dry_run_report(videos, batch_dir, backup_dir, status, group_targets)
        return

    # --- main loop ---
    active_index = 0  # always scan from start; check_pass_complete() handles skip
    prev_severe = False
    severe_count = sup["severe_count"]
    recovery_entries = sup["recovery_entries"]

    for i in range(active_index, len(videos)):
        video = videos[i]
        key = video.name
        entry = status.get(key, {})
        prev_status = entry.get("status", "")

        # -- fast skip --
        if check_pass_complete(batch_dir, video.stem, backup_dir):
            print(f"\n{'=' * 60}", flush=True)
            print(f"[{i + 1}/{total}] SKIP (complete PASS): {key}", flush=True)
            status[key] = {
                "status": "PASS",
                "timestamp": datetime.now().isoformat(),
                "exit_code": entry.get("exit_code", 0),
                "error": "",
            }
            sup["resume_index"] = i + 1
            save_status(status_path, status)
            continue

        # -- duration & risk flags --
        duration = get_video_duration(video)
        is_long = duration is not None and duration > 10800
        is_hist_access_violation = (
            prev_status in ("FAIL_NEED_REVIEW", "FAIL")
            and entry.get("exit_code") == 3221225477
        )

        # -- process --
        print(f"\n{'=' * 60}", flush=True)
        print(f"[{i + 1}/{total}] PROCESS: {key}", flush=True)
        dur_display = f"{duration / 3600:.1f}h" if duration else "unknown"
        print(f"  duration={dur_display}  prev_status={prev_status}", flush=True)
        print(f"{'=' * 60}", flush=True)

        sup["state"] = "RUNNING"
        status[key] = {
            "status": "RUNNING",
            "timestamp": datetime.now().isoformat(),
            "exit_code": None,
            "error": "",
        }
        save_status(status_path, status)

        log_path = logs_dir / f"{safe_name(video.stem)}.log"
        target_fps = video_target_fps.get(video.name)
        if target_fps is None:
            print(f"ERROR: no target_fps for {video.name}, skipping", flush=True)
            status[key] = {
                "status": "FAIL_NEED_REVIEW",
                "timestamp": datetime.now().isoformat(),
                "exit_code": -1,
                "error": "no target_fps in group mapping",
            }
            sup["resume_index"] = i + 1
            save_status(status_path, status)
            continue

        cmd = [
            str(PYTHON_EXE), str(MAIN_SCRIPT),
            "--video", str(video),
            "--batch-dir", str(batch_dir),
            "--target-fps", str(target_fps),
        ]
        if args.force:
            cmd.append("--force")
        if args.keep_temp:
            cmd.append("--keep-temp")

        # --- spawn subprocess ---
        exit_code = -1
        try:
            with open(log_path, "w", encoding="utf-8") as log_fh:
                proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            exit_code = proc.returncode
        except Exception as e:
            exit_code = -1
            status[key] = {
                "status": "FAIL_NEED_REVIEW",
                "timestamp": datetime.now().isoformat(),
                "exit_code": -1,
                "error": f"supervisor spawn failed: {e}",
            }

        # --- classify ---
        is_severe = detect_severe_error(exit_code, log_path)
        passed, reason, mp4_path = check_pass_strict(
            batch_dir, video.stem, exit_code, backup_dir
        )

        # Silent severe: exit=0 but model loading never completed (GPU hung)
        if not passed and not is_severe and exit_code == 0:
            if detect_silent_severe(log_path):
                is_severe = True

        # --- write result ---
        if passed:
            status[key] = {
                "status": "PASS",
                "timestamp": datetime.now().isoformat(),
                "exit_code": exit_code,
                "error": "",
            }
            backup_name = backup_dir / mp4_path.name if mp4_path else None
            print(f"  BACKUP OK: {backup_name}", flush=True)
            print(f"[{i + 1}/{total}] RESULT: {key} -> PASS", flush=True)
        else:
            status[key] = {
                "status": "FAIL_NEED_REVIEW",
                "timestamp": datetime.now().isoformat(),
                "exit_code": exit_code,
                "error": reason,
            }
            print(f"[{i + 1}/{total}] RESULT: {key} -> FAIL_NEED_REVIEW ({reason})",
                  flush=True)

        sup["resume_index"] = i + 1

        # -- cooldown / recovery --
        if passed:
            prev_severe = False
            secs = 300 if (is_long or is_hist_access_violation) else 60
            save_status(status_path, status)
            do_cooldown(secs, f"PASS {'long' if secs == 300 else 'normal'}")
            continue

        # --- not passed ---
        if is_severe:
            severe_count += 1
            sup["severe_count"] = severe_count

            if severe_count >= 3 and prev_severe:
                # ======= LONG_RECOVERY_MODE =======
                print(f"\nWARN: SEVERE_COUNT={severe_count} -- entering LONG_RECOVERY_MODE",
                      flush=True)
                sup["state"] = "LONG_RECOVERY_MODE"
                save_status(status_path, status)
                log_recovery(recovery_log,
                             f"LONG_RECOVERY_MODE severe_count={severe_count} video={key}")

                long_ok = False
                for la in range(1, 7):
                    print(f"LONG_RECOVERY_MODE: probe {la}/6 (waiting 1800s)...",
                          flush=True)
                    do_cooldown(1800, f"long-recovery probe {la}/6")
                    probe_ok, probe_reason = run_health_probe()
                    print(f"LONG HEALTH PROBE {la}/6: {'PASS' if probe_ok else 'FAIL'}"
                          f" -- {probe_reason}", flush=True)
                    log_recovery(recovery_log,
                                 f"long-probe {la}/6: {'PASS' if probe_ok else 'FAIL'} -- {probe_reason}")
                    if probe_ok:
                        sup["state"] = "RECOVERY_OK"
                        print("RECOVERY_OK from LONG_RECOVERY_MODE -- resuming", flush=True)
                        log_recovery(recovery_log, "RECOVERY_OK from LONG_RECOVERY_MODE")
                        long_ok = True
                        prev_severe = False
                        do_cooldown(300, "post-long-recovery")
                        break

                if not long_ok:
                    sup["state"] = "PAUSED_SYSTEM_UNSAFE"
                    save_status(status_path, status)
                    reason_text = (
                        f"PAUSED_SYSTEM_UNSAFE at {datetime.now().isoformat()}\n"
                        f"severe_count={severe_count}  recovery_entries={recovery_entries}\n"
                        f"last_video={key}  last_error={reason}\n"
                        f"resume_index={i + 1}  ({len(videos) - i - 1} videos remaining)\n"
                        "Action: reboot computer, then re-run same command.\n"
                        "supervisor will auto-resume from resume_index.\n"
                    )
                    pause_reason.write_text(reason_text, encoding="utf-8")
                    print(f"\n{'=' * 60}", flush=True)
                    print("PAUSED_SYSTEM_UNSAFE -- LONG_RECOVERY_MODE exhausted (3hrs)",
                          flush=True)
                    print(f"  resume_index = {i + 1}", flush=True)
                    print(f"  Please reboot and re-run same command.", flush=True)
                    print(f"{'=' * 60}", flush=True)
                    log_recovery(recovery_log, "PAUSED_SYSTEM_UNSAFE (long) -- exiting")
                    return

            elif prev_severe:
                # ======= RECOVERY_MODE =======
                print(f"\nWARN: CONSECUTIVE SEVERE ERRORS -- entering RECOVERY_MODE",
                      flush=True)
                sup["state"] = "RECOVERY_MODE"
                recovery_entries += 1
                sup["recovery_entries"] = recovery_entries
                save_status(status_path, status)
                log_recovery(recovery_log,
                             f"RECOVERY_MODE severe_count={severe_count} entry={recovery_entries} video={key}")

                recovered = False
                for attempt in range(1, 4):
                    print(f"RECOVERY_MODE: probe {attempt}/3 (waiting 600s)...",
                          flush=True)
                    do_cooldown(600, f"recovery probe {attempt}/3")
                    probe_ok, probe_reason = run_health_probe()
                    print(f"HEALTH PROBE {attempt}/3: {'PASS' if probe_ok else 'FAIL'}"
                          f" -- {probe_reason}", flush=True)
                    log_recovery(recovery_log,
                                 f"probe {attempt}/3: {'PASS' if probe_ok else 'FAIL'} -- {probe_reason}")
                    if probe_ok:
                        sup["state"] = "RECOVERY_OK"
                        print("RECOVERY_OK -- resuming processing", flush=True)
                        log_recovery(recovery_log, "RECOVERY_OK -- resuming")
                        recovered = True
                        prev_severe = False
                        do_cooldown(300, "post-recovery")
                        break

                if not recovered:
                    if severe_count >= 3:
                        # escalate to LONG_RECOVERY_MODE
                        print(f"\nWARN: RECOVERY_MODE failed, severe_count={severe_count}"
                              f" -- escalating to LONG_RECOVERY_MODE", flush=True)
                        sup["state"] = "LONG_RECOVERY_MODE"
                        save_status(status_path, status)
                        log_recovery(recovery_log,
                                     f"LONG_RECOVERY_MODE after RECOVERY_MODE fail severe_count={severe_count}")

                        long_ok = False
                        for la in range(1, 7):
                            print(f"LONG_RECOVERY_MODE: probe {la}/6 (waiting 1800s)...",
                                  flush=True)
                            do_cooldown(1800, f"long-recovery probe {la}/6")
                            probe_ok, probe_reason = run_health_probe()
                            print(f"LONG HEALTH PROBE {la}/6:"
                                  f" {'PASS' if probe_ok else 'FAIL'} -- {probe_reason}",
                                  flush=True)
                            log_recovery(recovery_log,
                                         f"long-probe {la}/6: {'PASS' if probe_ok else 'FAIL'} -- {probe_reason}")
                            if probe_ok:
                                sup["state"] = "RECOVERY_OK"
                                print("RECOVERY_OK from LONG_RECOVERY_MODE -- resuming",
                                      flush=True)
                                log_recovery(recovery_log,
                                             "RECOVERY_OK from LONG_RECOVERY_MODE")
                                long_ok = True
                                prev_severe = False
                                do_cooldown(300, "post-long-recovery")
                                break

                        if not long_ok:
                            sup["state"] = "PAUSED_SYSTEM_UNSAFE"
                            save_status(status_path, status)
                            reason_text = (
                                f"PAUSED_SYSTEM_UNSAFE at {datetime.now().isoformat()}\n"
                                f"severe_count={severe_count}  recovery_entries={recovery_entries}\n"
                                f"last_video={key}  last_error={reason}\n"
                                f"resume_index={i + 1}  ({len(videos) - i - 1} videos remaining)\n"
                                "Action: reboot computer, then re-run same command.\n"
                            )
                            pause_reason.write_text(reason_text, encoding="utf-8")
                            print(f"\n{'=' * 60}", flush=True)
                            print("PAUSED_SYSTEM_UNSAFE -- LONG_RECOVERY_MODE exhausted",
                                  flush=True)
                            print(f"  resume_index = {i + 1}", flush=True)
                            print(f"  Please reboot and re-run same command.", flush=True)
                            print(f"{'=' * 60}", flush=True)
                            log_recovery(recovery_log, "PAUSED_SYSTEM_UNSAFE (long) -- exiting")
                            return
                    else:
                        sup["state"] = "PAUSED_SYSTEM_UNSAFE"
                        save_status(status_path, status)
                        reason_text = (
                            f"PAUSED_SYSTEM_UNSAFE at {datetime.now().isoformat()}\n"
                            f"severe_count={severe_count}  recovery_entries={recovery_entries}\n"
                            f"last_video={key}  last_error={reason}\n"
                            f"resume_index={i + 1}  ({len(videos) - i - 1} videos remaining)\n"
                            "Action: reboot computer, then re-run same command.\n"
                        )
                        pause_reason.write_text(reason_text, encoding="utf-8")
                        print(f"\n{'=' * 60}", flush=True)
                        print("PAUSED_SYSTEM_UNSAFE -- RECOVERY_MODE failed 3 probes",
                              flush=True)
                        print(f"  resume_index = {i + 1}", flush=True)
                        print(f"  Please reboot and re-run same command.", flush=True)
                        print(f"{'=' * 60}", flush=True)
                        log_recovery(recovery_log, "PAUSED_SYSTEM_UNSAFE -- exiting")
                        return
            else:
                # first severe (non-consecutive)
                prev_severe = True
                secs = 300
                save_status(status_path, status)
                do_cooldown(secs, "severe error")
        else:
            # non-severe failure
            prev_severe = False
            secs = 300 if (is_long or is_hist_access_violation) else 180
            save_status(status_path, status)
            do_cooldown(secs, f"non-severe FAIL {'long' if secs == 300 else 'normal'}")

        sup["severe_count"] = severe_count
        sup["recovery_entries"] = recovery_entries
        save_status(status_path, status)

    # ======= DONE =======
    pass_count = sum(1 for v in status.values()
                     if isinstance(v, dict) and v.get("status") == "PASS")
    fail_count = sum(1 for v in status.values()
                     if isinstance(v, dict) and v.get("status") == "FAIL_NEED_REVIEW")
    pending_count = sum(1 for v in status.values()
                        if isinstance(v, dict) and v.get("status") in ("PENDING", "RUNNING"))

    print(f"\n{'=' * 60}", flush=True)
    print(f"SUPERVISOR DONE: {pass_count} PASS / {fail_count} FAIL_NEED_REVIEW / "
          f"{pending_count} PENDING ({total} total)", flush=True)
    print(f"  status  = {status_path}", flush=True)
    print(f"  logs    = {logs_dir}", flush=True)
    print(f"  backup  = {backup_dir}", flush=True)

    sup["state"] = "DONE"
    save_status(status_path, status)


if __name__ == "__main__":
    main()
