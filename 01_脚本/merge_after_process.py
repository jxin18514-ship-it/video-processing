"""merge_after_process.py — System C merge stage.

Default (no --merge): dry-run only, prints plan and ffmpeg preview.
With --merge: executes concat demuxer + h264_nvenc GPU re-encode for all groups.

Modes:
  FULL_HQ       — strict all-pass per group, no partial merge, formal output naming
  CURRENT_SCOPE — legacy mode, allows partial groups (user-confirmed skips)
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# probe helpers
# ---------------------------------------------------------------------------

def get_video_fps(video_path: Path) -> float | None:
    """Read r_frame_rate first, avg_frame_rate fallback."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,avg_frame_rate",
           "-of", "json", str(video_path)]
    try:
        raw = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace",
                                      creationflags=subprocess.CREATE_NO_WINDOW)
        info = json.loads(raw)
        streams = info.get("streams", [])
        if not streams:
            return None
        vs = streams[0]
        for key in ("r_frame_rate", "avg_frame_rate"):
            val = vs.get(key, "")
            if val and "/" in val:
                num, den = val.split("/", 1)
                if float(den) != 0:
                    return float(num) / float(den)
        return None
    except Exception:
        return None


def probe_mp4(path: Path) -> dict | None:
    """Full probe of an MP4 file; returns dict or None on failure."""
    cmd_v = ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,"
             "pix_fmt,color_space,color_transfer,color_primaries,color_range,bit_rate",
             "-of", "json", str(path)]
    cmd_a = ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name,bit_rate,sample_rate,channels",
             "-of", "json", str(path)]
    cmd_f = ["ffprobe", "-v", "error",
             "-show_entries", "format=duration,size,bit_rate",
             "-of", "json", str(path)]
    try:
        v_raw = subprocess.check_output(cmd_v, text=True, encoding="utf-8", errors="replace",
                                        creationflags=subprocess.CREATE_NO_WINDOW)
        a_raw = subprocess.check_output(cmd_a, text=True, encoding="utf-8", errors="replace",
                                        creationflags=subprocess.CREATE_NO_WINDOW)
        f_raw = subprocess.check_output(cmd_f, text=True, encoding="utf-8", errors="replace",
                                        creationflags=subprocess.CREATE_NO_WINDOW)
        vinfo = json.loads(v_raw)
        ainfo = json.loads(a_raw)
        finfo = json.loads(f_raw)

        vs = vinfo.get("streams", [{}])[0] if vinfo.get("streams") else {}
        as_ = ainfo.get("streams", [{}])[0] if ainfo.get("streams") else {}
        fmt = finfo.get("format", {})

        fps = get_video_fps(path)

        return {
            "fps": fps,
            "width": vs.get("width"),
            "height": vs.get("height"),
            "pix_fmt": vs.get("pix_fmt"),
            "color_space": vs.get("color_space"),
            "color_transfer": vs.get("color_transfer"),
            "color_primaries": vs.get("color_primaries"),
            "color_range": vs.get("color_range"),
            "video_codec": vs.get("codec_name"),
            "video_bitrate": vs.get("bit_rate"),
            "audio_codec": as_.get("codec_name"),
            "audio_bitrate": as_.get("bit_rate"),
            "audio_sample_rate": as_.get("sample_rate"),
            "audio_channels": as_.get("channels"),
            "duration": float(fmt.get("duration", 0)),
            "size": int(fmt.get("size", 0)),
            "format_bit_rate": fmt.get("bit_rate"),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# segment scanning
# ---------------------------------------------------------------------------

def parse_segment_key(dirname: str) -> tuple[str, int] | None:
    """Parse 'kt_(5)_SYSTEM_C_CUT_V2_01' -> ('kt', 5)."""
    m = re.match(r'([a-z]+)_\((\d+)\)_SYSTEM_C_CUT_V2_01', dirname)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def load_group_manifest(batch_dir: Path) -> dict | None:
    """Load group_target_fps.json from batch_dir/logs/.

    Returns the full manifest dict, or None if file missing/unparseable.
    """
    manifest_path = batch_dir / "logs" / "group_target_fps.json"
    if not manifest_path.exists():
        print(f"ERROR: group_target_fps.json not found at {manifest_path}", flush=True)
        print("Run supervisor dry-run first to generate the manifest.", flush=True)
        return None
    try:
        raw = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(raw)
        if "groups" not in manifest:
            print(f"ERROR: group_target_fps.json missing 'groups' key", flush=True)
            return None
        return manifest
    except Exception as e:
        print(f"ERROR: cannot parse group_target_fps.json: {e}", flush=True)
        return None


def build_expected_groups(manifest_groups: list[dict]) -> dict[str, list[int]]:
    """Derive expected segment numbers per group from manifest groups list.

    Each video name like 'kt (5).ts' yields group='kt', num=5.
    Returns {group_name: sorted list of expected nums}.
    """
    expected: dict[str, list[int]] = {}
    for g in manifest_groups:
        gname = g["group_name"]
        nums = []
        for v in g.get("videos", []):
            m = re.match(r'[a-z]+ \((\d+)\)\.ts', v["name"], re.IGNORECASE)
            if m:
                nums.append(int(m.group(1)))
        expected[gname] = sorted(nums)
    return expected


def lookup_supervisor_status(supervisor: dict, group: str, num: int) -> str:
    """Look up status in supervisor_status.json. Returns 'PASS','FAIL','RUNNING','UNKNOWN'."""
    for key in (f"{group} ({num}).ts", f"{group}({num}).ts"):
        if key in supervisor:
            return supervisor[key].get("status", "UNKNOWN")
    return "UNKNOWN"


def scan_segments(batch_dir: Path, manifest_groups: list[dict]) -> tuple[dict, dict]:
    """Scan batch_dir for all groups listed in manifest.

    Returns (groups, excluded) where each is {group_name: [dict, ...]}.
    """
    supervisor_path = batch_dir / "supervisor_status.json"
    supervisor = {}
    if supervisor_path.exists():
        try:
            supervisor = json.loads(supervisor_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    group_names = [g["group_name"] for g in manifest_groups]
    groups = {gn: [] for gn in group_names}
    excluded = {gn: [] for gn in group_names}

    for d in sorted(batch_dir.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_segment_key(d.name)
        if not parsed:
            continue
        group, num = parsed
        key = f"{group}({num})"

        # 1. Find cut_final MP4
        mp4_dir = d / "06_output_video"
        mp4s = sorted(mp4_dir.glob("*_cut_final_CLEAN_*.mp4")) if mp4_dir.exists() else []
        if not mp4s:
            excluded[group].append({"key": key, "reason": "no cut_final MP4"})
            continue
        mp4_path = mp4s[-1]

        # 2. ffprobe must succeed
        info = probe_mp4(mp4_path)
        if info is None:
            excluded[group].append({"key": key, "reason": "ffprobe failed (moov atom corrupt?)",
                                     "mp4": str(mp4_path)})
            continue

        fps = info.get("fps")
        if fps is None:
            excluded[group].append({"key": key, "reason": "cannot determine fps",
                                     "mp4": str(mp4_path)})
            continue

        # 3. fps must be 45 or 60
        is_45 = abs(fps - 45.0) < 0.1
        is_60 = abs(fps - 60.0) < 1.0
        if not is_45 and not is_60:
            excluded[group].append({"key": key, "reason": f"unsupported fps={fps:.2f}",
                                     "mp4": str(mp4_path)})
            continue

        # 4. Must have final_verify report
        verify_dir = d / "07_reports"
        verifies = sorted(verify_dir.glob("*_final_verify_*.txt")) if verify_dir.exists() else []
        if not verifies:
            excluded[group].append({"key": key, "reason": "no final_verify report",
                                     "mp4": str(mp4_path)})
            continue
        verify_path = verifies[-1]

        # 5. Verify report must say PASS
        try:
            verify_text = verify_path.read_text(encoding="utf-8", errors="replace")
            if "result: PASS" not in verify_text and "result:PASS" not in verify_text:
                excluded[group].append({"key": key, "reason": "final_verify not PASS",
                                         "mp4": str(mp4_path)})
                continue
        except Exception:
            excluded[group].append({"key": key, "reason": "cannot read verify report",
                                     "mp4": str(mp4_path)})
            continue

        # 6. Supervisor status must not be FAIL or RUNNING
        status = lookup_supervisor_status(supervisor, group, num)
        if status == "RUNNING":
            excluded[group].append({"key": key, "reason": "supervisor status=RUNNING",
                                     "mp4": str(mp4_path)})
            continue
        if status == "FAIL":
            excluded[group].append({"key": key, "reason": "supervisor status=FAIL",
                                     "mp4": str(mp4_path)})
            continue

        # Passed all checks
        groups[group].append({
            "key": key,
            "num": num,
            "mp4": str(mp4_path),
            "verify": str(verify_path),
            "fps": fps,
            "duration": info["duration"],
            "size": info["size"],
            "width": info["width"],
            "height": info["height"],
            "info": info,
        })

    # Sort each group by num
    for g in groups:
        groups[g].sort(key=lambda x: x["num"])
        excluded[g].sort(key=lambda x: x["key"])

    return groups, excluded


# ---------------------------------------------------------------------------
# FULL_HQ validation
# ---------------------------------------------------------------------------

def validate_full_hq(groups: dict, excluded: dict, expected: dict[str, list[int]]) -> dict:
    """Check each group has ALL expected segments PASS.

    Returns {group_name: {"valid": bool, "missing": [str], "extra_excluded": [str]}}
    """
    result = {}
    for group_name in sorted(expected.keys()):
        expected_nums = expected[group_name]
        passed_nums = {s["num"] for s in groups.get(group_name, [])}
        excluded_items = excluded.get(group_name, [])

        missing = [n for n in expected_nums if n not in passed_nums]
        # Which missing are in excluded (with reason) vs completely absent
        excluded_keys = {e["key"] for e in excluded_items}
        missing_details = []
        for n in missing:
            key = f"{group_name}({n})"
            found = False
            for e in excluded_items:
                if e["key"] == key:
                    missing_details.append(f"{key} -> {e['reason']}")
                    found = True
                    break
            if not found:
                missing_details.append(f"{key} -> segment directory not found")

        result[group_name] = {
            "valid": len(missing) == 0,
            "missing": missing_details,
            "passed": sorted(passed_nums),
            "expected": expected_nums,
        }
    return result


# ---------------------------------------------------------------------------
# ffmpeg command builder
# ---------------------------------------------------------------------------

def build_bufsize(bitrate_str: str) -> str:
    """Given '10M' or '12M', return '20M' or '24M' (2x)."""
    m = re.match(r'(\d+)M', bitrate_str)
    if m:
        return f"{int(m.group(1)) * 2}M"
    return bitrate_str


def build_ffmpeg_cmd(concat_path: Path, output_path: Path, fps: int,
                     video_bitrate: str, audio_bitrate: str) -> list[str]:
    """Build ffmpeg command line for concat demuxer + h264_nvenc re-encode."""
    bufsize = build_bufsize(video_bitrate)
    # NVENC: maxrate higher than target bitrate for VBR quality headroom
    if abs(fps - 45.0) < 0.1:
        maxrate = "15M"
    elif abs(fps - 60.0) < 1.0:
        maxrate = "18M"
    else:
        raise RuntimeError(f"Unsupported fps for NVENC merge: {fps} (only 45/60 supported)")
    return [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
        "-b:v", video_bitrate,
        "-maxrate", maxrate,
        "-bufsize", bufsize,
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-color_range", "tv",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ar", "48000",
        "-ac", "2",
        "-movflags", "+faststart",
        str(output_path),
    ]


def cmd_to_preview(cmd: list[str]) -> str:
    """Pretty-print a ffmpeg command with line continuations."""
    parts = []
    for x in cmd:
        s = str(x)
        if " " in s:
            parts.append(f'"{s}"')
        else:
            parts.append(s)
    return " \\\n    ".join(parts)


# ---------------------------------------------------------------------------
# copy-merge support (FULL_HQ only)
# ---------------------------------------------------------------------------

def check_copy_eligible(segs: list[dict]) -> tuple[bool, list[str]]:
    """Check if all segments have identical parameters for -c copy merge.

    Returns (allow_copy_merge, issues_list).
    """
    if len(segs) < 2:
        return len(segs) == 1, [] if len(segs) == 1 else ["no segments"]

    issues: list[str] = []
    ref = segs[0]["info"]
    ref_fps = segs[0]["fps"]

    # Fields that must match exactly across all segments
    checks = [
        ("video_codec", "h264", "video codec"),
        ("width", 1080, "width"),
        ("height", 1920, "height"),
        ("pix_fmt", "yuv420p", "pix_fmt"),
        ("color_space", "bt709", "color_space"),
        ("color_transfer", "bt709", "color_transfer"),
        ("color_primaries", "bt709", "color_primaries"),
        ("color_range", "tv", "color_range"),
        ("audio_codec", "aac", "audio codec"),
        ("audio_sample_rate", 48000, "audio sample_rate"),
        ("audio_channels", 2, "audio channels"),
    ]

    for i, seg in enumerate(segs):
        info = seg["info"]
        for field, expected, label in checks:
            actual = info.get(field)
            if actual != expected and str(actual) != str(expected):
                issues.append(
                    f"{seg['key']}: {label}={actual} (expected {expected})")

        # FPS must all be identical
        if seg["fps"] is not None and ref_fps is not None:
            if abs(seg["fps"] - ref_fps) > 0.1:
                issues.append(
                    f"{seg['key']}: fps={seg['fps']:.1f} (ref={ref_fps:.1f})")

        # Audio channel_layout
        layout = info.get("audio_channels")
        # channel_layout is not directly probed by current probe_mp4,
        # but stereo == 2 channels is checked above

    # Also verify all codecs are h264 (separate check for clarity)
    for seg in segs:
        vc = seg["info"].get("video_codec")
        if vc != "h264":
            issues.append(f"{seg['key']}: video_codec={vc} (expected h264)")

    # Deduplicate
    issues = list(dict.fromkeys(issues))

    if issues:
        return False, issues
    return True, []


def execute_copy_merge(segs: list[dict], output_path: Path,
                       output_dir: Path, group_name: str,
                       faststart: bool = False) -> int:
    """Merge segments with -c copy (no re-encode). Two-step process.

    Step 1: ffmpeg concat -> .ts (intermediate)
    Step 2: ffmpeg remux -> .mp4 (final, +faststart only if faststart=True)

    On failure, preserves concat_list, concat_output.ts, and ffmpeg log.
    Returns ffmpeg exit code (0 = success).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    concat_list = output_dir / f".{output_path.name}.concat.txt"
    concat_ts = output_dir / f".{output_path.name}.concat.ts"
    ffmpeg_log = output_dir / f".{output_path.name}.ffmpeg.log"

    # Write concat list
    concat_lines = [f"file '{Path(s['mp4']).as_posix()}'" for s in segs]
    concat_list.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    # Step 1: concat to .ts with stream copy
    cmd1 = [
        "ffmpeg", "-y", "-hide_banner",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy",
        "-f", "mpegts", str(concat_ts),
    ]

    with open(ffmpeg_log, "w", encoding="utf-8") as log_fh:
        proc1 = subprocess.Popen(cmd1, stderr=subprocess.PIPE, text=True,
                                 encoding="utf-8", errors="replace", bufsize=1,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        for line in proc1.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            log_fh.write(line)
        exit1 = proc1.wait()

    if exit1 != 0:
        print(f"  ERROR: concat step 1 failed (exit={exit1})", flush=True)
        print(f"  Preserved: {concat_list}", flush=True)
        print(f"  Preserved: {ffmpeg_log}", flush=True)
        return exit1

    # Step 2: remux to .mp4 (faststart only if requested)
    faststart_label = "+faststart" if faststart else "pure -c copy (no faststart)"
    cmd2 = [
        "ffmpeg", "-y", "-hide_banner",
        "-i", str(concat_ts),
        "-c", "copy",
    ]
    if faststart:
        cmd2 += ["-movflags", "+faststart"]
    cmd2.append(str(output_path))

    with open(ffmpeg_log, "a", encoding="utf-8") as log_fh:
        log_fh.write(f"\n{'='*60}\nSTEP 2: remux ({faststart_label})\n{'='*60}\n")
        proc2 = subprocess.Popen(cmd2, stderr=subprocess.PIPE, text=True,
                                 encoding="utf-8", errors="replace", bufsize=1,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        for line in proc2.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            log_fh.write(line)
        exit2 = proc2.wait()

    if exit2 != 0:
        print(f"  ERROR: remux step 2 failed (exit={exit2})", flush=True)
        print(f"  Preserved: {concat_list}", flush=True)
        print(f"  Preserved: {concat_ts}", flush=True)
        print(f"  Preserved: {ffmpeg_log}", flush=True)
        return exit2

    # Cleanup intermediate on success
    try: concat_list.unlink()
    except OSError: pass
    try: concat_ts.unlink()
    except OSError: pass

    # Write minimal verify report
    verify_name = output_path.name.replace(".mp4", "_verify.txt")
    verify_path = output_dir / verify_name
    info = probe_mp4(output_path)
    vlines = [
        "=== COPY MERGE VERIFY REPORT ===",
        f"group: {group_name}",
        f"output: {output_path}",
        f"mode: FULL_HQ -c copy (no re-encode)",
        f"segments: {len(segs)}",
        f"timestamp: {ts}",
        f"ffmpeg_exit: step1={exit1} step2={exit2}",
        f"status: {'PASS' if exit2 == 0 else 'FAIL'}",
    ]
    if info:
        vlines += [
            f"duration: {info.get('duration', 0):.1f}s",
            f"size: {info.get('size', 0)}",
            f"video_codec: {info.get('video_codec', '?')}",
            f"audio_codec: {info.get('audio_codec', '?')}",
            f"pix_fmt: {info.get('pix_fmt', '?')}",
            f"fps: {info.get('fps', '?')}",
        ]
    verify_path.write_text("\n".join(vlines) + "\n", encoding="utf-8")
    print(f"  Verify report: {verify_path}", flush=True)

    return 0


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------

def output_name(group_name: str, mode: str, fps: int, br_tag: str, ts: str) -> str:
    """Build output filename based on mode."""
    if mode == "FULL_HQ":
        return f"{group_name}_FULL_HQ_{fps}fps_{br_tag}M_{ts}.mp4"
    else:
        return f"{group_name}_CURRENT_SCOPE_{fps}fps_{br_tag}M_{ts}.mp4"


def dry_run(groups: dict, excluded: dict, args,
            expected: dict[str, list[int]] | None = None,
            manifest_group_names: list[str] | None = None) -> None:
    """Print merge plan without executing ffmpeg."""
    mode = args.mode
    print("=" * 90)
    print(f"MERGE DRY-RUN — {mode}")
    print(f"  mode:           {mode}")
    print(f"  output_fps:     {args.fps}")
    print(f"  video_bitrate:  {args.video_bitrate}")
    print(f"  audio_bitrate:  {args.audio_bitrate}")
    print(f"  batch_dir:      {args.batch_dir}")
    print(f"  output_dir:     {args.output_dir}")
    print(f"  merge_mode:     DRY-RUN (no --merge flag)")
    print("=" * 90)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    br_tag = args.video_bitrate.replace("M", "")

    if manifest_group_names is None:
        manifest_group_names = ["kt", "xp", "zh"]

    # FULL_HQ validation
    if mode == "FULL_HQ" and expected is not None:
        hq_result = validate_full_hq(groups, excluded, expected)
        print(f"\n{'─' * 90}")
        print("FULL_HQ GROUP COMPLETENESS CHECK:")
        for group_name in sorted(manifest_group_names):
            r = hq_result[group_name]
            status = "PASS (all segments present)" if r["valid"] else "FAIL (incomplete group)"
            print(f"  {group_name}: {status}")
            print(f"    expected: {r['expected']}")
            print(f"    passed:   {r['passed']}")
            if r["missing"]:
                for m in r["missing"]:
                    print(f"    MISSING:  {m}")

    for group_name in sorted(manifest_group_names):
        segs = groups.get(group_name, [])
        excs = excluded.get(group_name, [])

        print(f"\n{'─' * 90}")
        print(f"GROUP: {group_name}")
        print(f"  Included ({len(segs)} segments):")
        if segs:
            for s in segs:
                dur_min = s["duration"] / 60
                size_mb = s["size"] / 1e6
                print(f"    {s['key']:10s}  fps={s['fps']:.0f}  "
                      f"dur={s['duration']:.1f}s ({dur_min:.1f}min)  "
                      f"size={size_mb:.0f}MB  {s['width']}x{s['height']}")
            total_dur = sum(s["duration"] for s in segs)
            total_size = sum(s["size"] for s in segs)
            est_out_gb = (int(args.video_bitrate.replace("M", "")) * 1e6 * total_dur) / (8 * 1e9)
            print(f"    TOTAL:  dur={total_dur/3600:.2f}h  "
                  f"input_size={total_size/1e9:.2f}GB  "
                  f"est_output={est_out_gb:.1f}GB @ {args.video_bitrate}")
        else:
            print("    (none)")

        print(f"  Excluded ({len(excs)} segments):")
        if excs:
            for e in excs:
                print(f"    {e['key']:10s}  ->  {e['reason']}")
        else:
            print("    (none)")

        # FULL_HQ: check group is complete
        if mode == "FULL_HQ" and not hq_result[group_name]["valid"]:
            print(f"  -> SKIP: FULL_HQ requires all expected segments PASS")
            continue

        if not segs:
            print(f"  -> SKIP: no eligible segments")
            continue

        # FULL_HQ: check copy eligibility
        copy_ok = False
        copy_issues: list[str] = []
        if mode == "FULL_HQ":
            copy_ok, copy_issues = check_copy_eligible(segs)
            if copy_ok:
                print(f"  Copy merge eligible: YES (-c copy, no re-encode)")
            else:
                print(f"  Copy merge eligible: NO")
                for issue in copy_issues:
                    print(f"    ! {issue}")

        # Output filename preview
        out_name = output_name(group_name, mode, args.fps, br_tag, ts)
        out_path = args.output_dir / out_name
        print(f"  Output file:  {out_name}")
        print(f"  Output path:  {out_path}")
        print(f"  Already exists: {out_path.exists()}")

        # Build concat content preview
        concat_preview_path = args.output_dir / f".dry_run_preview_concat.txt"
        concat_lines = [f"file '{Path(s['mp4']).as_posix()}'" for s in segs]

        print(f"  Concat list ({len(concat_lines)} files):")
        for cl in concat_lines:
            print(f"    {cl}")

        # ffmpeg command preview
        if copy_ok:
            fs_label = "+faststart" if args.faststart else "no faststart"
            print(f"  FFmpeg command (copy merge, 2-step):")
            print(f"    faststart: {'ON' if args.faststart else 'OFF'}")
            print(f"    Step1: ffmpeg -f concat -safe 0 -i concat.txt -c copy -f mpegts output.ts")
            step2_extra = " -movflags +faststart" if args.faststart else ""
            print(f"    Step2: ffmpeg -i output.ts -c copy{step2_extra} output.mp4")
        else:
            cmd = build_ffmpeg_cmd(concat_preview_path, out_path, args.fps,
                                   args.video_bitrate, args.audio_bitrate)
            print(f"  FFmpeg command:")
            print(f"    {cmd_to_preview(cmd)}")
        print(f"  Will overwrite existing: NO (timestamp ensures unique name)")

    print(f"\n{'=' * 90}")
    print("DRY-RUN COMPLETE — no ffmpeg executed, no files created.")
    print("Run with --merge to execute the merge.")
    print("=" * 90)


# ---------------------------------------------------------------------------
# merge execution
# ---------------------------------------------------------------------------

def execute_merge(groups: dict, excluded: dict, args,
                  expected: dict[str, list[int]] | None = None,
                  manifest_group_names: list[str] | None = None) -> None:
    """Execute concat + re-encode for all groups."""
    mode = args.mode
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    br_tag = args.video_bitrate.replace("M", "")

    if manifest_group_names is None:
        manifest_group_names = ["kt", "xp", "zh"]

    # FULL_HQ pre-validation
    if mode == "FULL_HQ" and expected is not None:
        hq_result = validate_full_hq(groups, excluded, expected)
        print(f"\n{'=' * 90}")
        print("FULL_HQ GROUP COMPLETENESS CHECK:")
        for group_name in sorted(manifest_group_names):
            r = hq_result[group_name]
            status = "PASS" if r["valid"] else "FAIL"
            print(f"  {group_name}: {status}  expected={r['expected']}  passed={r['passed']}")
            if r["missing"]:
                for m in r["missing"]:
                    print(f"    MISSING: {m}")
        print(f"{'=' * 90}")

    for group_name in sorted(manifest_group_names):
        segs = groups.get(group_name, [])
        excs = excluded.get(group_name, [])

        # FULL_HQ: block partial merge
        if mode == "FULL_HQ" and not hq_result[group_name]["valid"]:
            print(f"\nSKIP {group_name}: FULL_HQ requires all expected segments PASS "
                  f"(missing: {hq_result[group_name]['missing']})")
            continue

        if not segs:
            print(f"\nSKIP {group_name}: no eligible segments")
            continue

        # FULL_HQ: check copy eligibility
        if mode == "FULL_HQ":
            copy_ok, copy_issues = check_copy_eligible(segs)
            if copy_ok:
                print(f"\n{group_name}: copy merge eligible — using -c copy (no re-encode)")
            else:
                print(f"\n{group_name}: copy merge NOT eligible — USER_ACTION_REQUIRED")
                print(f"  Inconsistent parameters detected:")
                for issue in copy_issues:
                    print(f"    ! {issue}")
                print(f"  Batch will NOT auto-fallback to re-encode.")
                print(f"  Fix the inconsistencies or confirm manual re-encode before proceeding.")
                continue

        out_name = output_name(group_name, mode, args.fps, br_tag, ts)
        out_path = args.output_dir / out_name

        # Never overwrite: append suffix if collision
        if out_path.exists():
            resolved = False
            for i in range(2, 100):
                alt = args.output_dir / output_name(group_name, mode, args.fps, br_tag,
                                                    f"{ts}_v{i}")
                if not alt.exists():
                    out_path = alt
                    out_name = alt.name
                    resolved = True
                    break
            if not resolved:
                raise RuntimeError(
                    f"Cannot find unique output name in {args.output_dir} (tried up to v99)")

        # FULL_HQ copy merge path
        if mode == "FULL_HQ" and copy_ok:
            print(f"\n{'=' * 90}")
            print(f"COPY MERGING: {group_name} -> {out_path}")
            print(f"  mode:           FULL_HQ -c copy (no re-encode)")
            print(f"  segments:       {len(segs)}")
            print(f"  total input duration: {sum(s['duration'] for s in segs)/3600:.2f}h")
            print(f"{'=' * 90}")

            exit_code = execute_copy_merge(segs, out_path, args.output_dir, group_name,
                                            faststart=args.faststart)
            verify_path = args.output_dir / out_name.replace(".mp4", "_verify.txt")
            status = "PASS" if exit_code == 0 else "FAIL"
            print(f"  RESULT: {status} (ffmpeg exit_code={exit_code})")
            print(f"  Verify report: {verify_path}")
            continue

        # Build concat file (re-encode path)
        concat_path = args.output_dir / f".{out_name}.concat.txt"
        concat_lines = [f"file '{Path(s['mp4']).as_posix()}'" for s in segs]
        concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

        # Build and run ffmpeg
        cmd = build_ffmpeg_cmd(concat_path, out_path, args.fps,
                               args.video_bitrate, args.audio_bitrate)

        print(f"\n{'=' * 90}")
        print(f"MERGING: {group_name} -> {out_path}")
        print(f"  mode:           {mode}")
        print(f"  segments:       {len(segs)}")
        print(f"  total input duration: {sum(s['duration'] for s in segs)/3600:.2f}h")
        print(f"  command:        {cmd_to_preview(cmd)[:300]}...")
        print(f"{'=' * 90}")

        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace", bufsize=1,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        stderr_lines = []
        for line in proc.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
            stderr_lines.append(line)
        exit_code = proc.wait()

        # Write verify report
        verify_name = out_name.replace(".mp4", "_verify.txt")
        verify_path = args.output_dir / verify_name
        write_verify_report(out_path, verify_path, group_name, segs, excs,
                            args, exit_code, stderr_lines)

        # Cleanup concat temp file
        try:
            concat_path.unlink()
        except OSError:
            pass

        status = "PASS" if exit_code == 0 else "FAIL"
        print(f"  RESULT: {status} (ffmpeg exit_code={exit_code})")
        print(f"  Verify report: {verify_path}")


def write_verify_report(mp4_path: Path, verify_path: Path, group_name: str,
                        included: list, excluded: list, args,
                        ffmpeg_exit_code: int,
                        stderr_lines: list[str]) -> None:
    """Write detailed verify report after merge."""
    info = probe_mp4(mp4_path)
    mode = args.mode

    lines = ["=== VERIFY REPORT ==="]

    if mode == "FULL_HQ":
        lines += [
            "OUTPUT_TYPE = FULL_HQ",
            "FULL_HQ_ALL_SEGMENTS_PASSED = YES",
        ]
    else:
        lines += [
            "OUTPUT_TYPE = CURRENT_SCOPE",
            "USER_CONFIRMED_REMAINING_FAILS_SKIPPED = YES",
            "THIS_IS_NOT_FULL_GROUP_MERGE = YES",
        ]

    lines += [
        "",
        f"group:              {group_name}",
        f"mode:               {mode}",
        f"batch_dir:          {args.batch_dir}",
        f"output_dir:         {args.output_dir}",
        f"input_segments:     {', '.join(s['key'] for s in included)}",
        f"output_path:        {mp4_path}",
        f"output_fps:         {args.fps}",
        f"output_bitrate:     {args.video_bitrate}",
        f"audio_bitrate:      {args.audio_bitrate}",
        ""]

    if info:
        lines += [
            f"video_codec:        {info.get('video_codec', '?')}",
            f"audio_codec:        {info.get('audio_codec', '?')}",
            f"resolution:         {info.get('width', '?')}x{info.get('height', '?')}",
            f"pix_fmt:            {info.get('pix_fmt', '?')}",
            f"color_space:        {info.get('color_space', '?')}",
            f"color_transfer:     {info.get('color_transfer', '?')}",
            f"color_primaries:    {info.get('color_primaries', '?')}",
            f"color_range:        {info.get('color_range', '?')}",
            f"total_duration:     {info.get('duration', 0):.1f}s",
            f"format_bit_rate:    {info.get('format_bit_rate', '?')}",
            f"video_bit_rate:     {info.get('video_bitrate', '?')}",
            f"audio_bit_rate:     {info.get('audio_bitrate', '?')}",
            f"audio_sample_rate:  {info.get('audio_sample_rate', '?')}",
            f"audio_channels:     {info.get('audio_channels', '?')}",
            f"avg_frame_rate:     r_frame_rate used for fps={info.get('fps', '?')}",
        ]

    lines += [
        f"ffmpeg_exit_code:   {ffmpeg_exit_code}",
        f"status:             {'PASS' if ffmpeg_exit_code == 0 else 'FAIL'}",
        "",
        "included_segments:",
    ]
    for s in included:
        lines.append(
            f"  {s['key']:10s} fps={s['fps']:.0f} dur={s['duration']:.1f}s "
            f"size={s['size']} mp4={s['mp4']}")

    lines.append("")
    lines.append("excluded_segments:")
    if excluded:
        for e in excluded:
            lines.append(f"  {e['key']:10s} -> {e['reason']}")
    else:
        lines.append("  (none)")

    if mode == "CURRENT_SCOPE":
        lines += ["",
                  "CURRENT_SCOPE: True"]

    lines += [
        "",
        f"timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "",
        "=== FFMPEG STDERR (last 50 lines) ==="]
    for line in stderr_lines[-50:]:
        lines.append(line.rstrip())

    verify_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="merge_after_process.py — System C merge stage (FULL_HQ or CURRENT_SCOPE)"
    )
    parser.add_argument("--batch-dir", type=str, required=True,
                        dest="batch_dir",
                        help="Batch output directory (e.g. D:\\处理结果\\batch_output)")
    parser.add_argument("--output-dir", type=str, default=None,
                        dest="output_dir",
                        help="Merged output directory (default: <batch_dir>\\merged_hq)")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["FULL_HQ", "CURRENT_SCOPE"],
                        help="Merge mode: FULL_HQ (strict all-pass per group) | CURRENT_SCOPE (legacy)")
    parser.add_argument("--fps", type=int, required=True,
                        help="Output frame rate (e.g. 45)")
    parser.add_argument("--video-bitrate", type=str, required=True,
                        dest="video_bitrate",
                        help="Video bitrate (e.g. 10M, 12M)")
    parser.add_argument("--audio-bitrate", type=str, default="128k",
                        dest="audio_bitrate",
                        help="Audio bitrate (default: 128k)")
    parser.add_argument("--merge", action="store_true", default=False,
                        help="Execute merge (default: dry-run only)")
    parser.add_argument("--faststart", action="store_true", default=False,
                        help="Enable +faststart moov relocation for web streaming "
                             "(default: OFF, not needed for local playback/upload/archive)")
    args = parser.parse_args()

    # Resolve paths
    args.batch_dir = Path(args.batch_dir)
    if args.output_dir:
        args.output_dir = Path(args.output_dir)
    else:
        args.output_dir = args.batch_dir / "merged_hq"

    print("merge_after_process.py — System C merge stage", flush=True)
    print(f"  mode:           {args.mode}", flush=True)
    print(f"  batch_dir:      {args.batch_dir}", flush=True)
    print(f"  output_dir:     {args.output_dir}", flush=True)
    print(f"  fps:            {args.fps}", flush=True)
    print(f"  video_bitrate:  {args.video_bitrate}", flush=True)
    print(f"  audio_bitrate:  {args.audio_bitrate}", flush=True)
    print(f"  merge:          {args.merge}", flush=True)
    print(flush=True)

    if not args.batch_dir.exists():
        print(f"ERROR: batch_dir does not exist: {args.batch_dir}", flush=True)
        sys.exit(1)

    # Load group manifest (required for FULL_HQ, beneficial for CURRENT_SCOPE)
    manifest = load_group_manifest(args.batch_dir)
    manifest_groups = manifest["groups"] if manifest else []
    expected = build_expected_groups(manifest_groups) if manifest_groups else None
    manifest_group_names = [g["group_name"] for g in manifest_groups] if manifest_groups else []

    if args.mode == "FULL_HQ" and not manifest:
        print("ERROR: FULL_HQ requires a valid group_target_fps.json manifest.", flush=True)
        print("Run supervisor dry-run first to generate the manifest.", flush=True)
        sys.exit(1)

    if args.mode == "FULL_HQ" and manifest_groups:
        print(f"  FULL_HQ groups (from manifest):", flush=True)
        for g in manifest_groups:
            nums = expected.get(g["group_name"], []) if expected else []
            print(f"    {g['group_name']}: {nums} ({len(nums)} segments, "
                  f"target_fps={g['target_fps']})", flush=True)
        print(flush=True)

    groups, excluded = scan_segments(args.batch_dir, manifest_groups if manifest_groups else [])

    if not args.merge:
        dry_run(groups, excluded, args, expected, manifest_group_names)
    else:
        execute_merge(groups, excluded, args, expected, manifest_group_names)


if __name__ == "__main__":
    main()
