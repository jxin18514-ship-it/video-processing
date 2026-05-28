import argparse, csv, gc, importlib.util, json, os, re, subprocess, sys, urllib.request
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
from opencc import OpenCC

BASE_DIR = Path(__file__).resolve().parent.parent  # 部署包根目录

INBOX = BASE_DIR / "输入视频"
PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN", "")

def notify_pushplus(title: str, content: str) -> None:
    try:
        req = urllib.request.Request(
            "http://www.pushplus.plus/send",
            data=json.dumps({"token": PUSHPLUS_TOKEN, "title": title, "content": content}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass
WORK_ROOT = BASE_DIR / "输出"
# v2.0: Use v2 pipeline module
PIPELINE_PATH = BASE_DIR / "01_脚本" / "wanxiang_recheck_asr_v2_pipeline.py"
RUN_TAG = "SYSTEM_C_CUT_V2_01"
SKIP_SENSEVOICE = True  # 跳过SenseVoice复核，只用Qwen3结果
SKIP_BOUNDARY_VERIFY = False  # 开启边界验证（Stage 10）
FASTSTART = False  # 默认不加 +faststart，需要时用 --faststart 开启
_MODEL_LEAK = []  # 阻止模型析构崩溃：把 model 引用永久保留
WORDLIST_DIR = BASE_DIR / "02_词库"

# ── v2.0 Model paths ──
# Deployment dir may not have models; set SYSTEM_C_MODEL_BASE or put under 03_模型
_MODEL_BASE_ENV = os.environ.get("SYSTEM_C_MODEL_BASE")
if _MODEL_BASE_ENV:
    _MODEL_BASE = Path(_MODEL_BASE_ENV)
elif (BASE_DIR / "03_模型").exists():
    _MODEL_BASE = BASE_DIR
else:
    raise FileNotFoundError(
        "Model directory not found. Put models under BASE_DIR/03_模型 "
        "or set SYSTEM_C_MODEL_BASE environment variable to the deployment root."
    )
QWEN3_MODEL_PATH = _MODEL_BASE / "03_模型" / "qwen3-asr-1.7b"
QWEN3_ALIGNER_PATH = _MODEL_BASE / "03_模型" / "qwen3-forced-aligner-0.6b"
SENSEVOICE_MODEL_PATH = _MODEL_BASE / "03_模型" / "models" / "iic" / "SenseVoiceSmall"

def write_status(batch_dir: Path, video_name: str, stage: str, detail: str) -> None:
    """每阶段写入状态文件，供仪表盘直接读取"""
    (batch_dir / "current_status.json").write_text(
        json.dumps({"video": video_name, "stage": stage, "detail": detail,
                    "timestamp": datetime.now().isoformat()}, ensure_ascii=False),
        encoding="utf-8")

def safe_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\s]+', "_", name).strip("._")
    return cleaned or "video"

def load_pipeline():
    spec = importlib.util.spec_from_file_location("wanxiang_pipeline", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def cleanup_models() -> None:
    # 只清理缓存，不删模型引用（_MODEL_LEAK 正是为了防止析构崩溃）
    torch.cuda.empty_cache()
    gc.collect()

def set_env() -> None:
    os.environ.setdefault("HF_HOME", str(BASE_DIR / "hf_cache"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(BASE_DIR / "hf_cache" / "hub"))
    os.environ.setdefault("TEMP", str(BASE_DIR / "tmp"))
    os.environ.setdefault("TMP", str(BASE_DIR / "tmp"))
    Path(os.environ["TEMP"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TMP"]).mkdir(parents=True, exist_ok=True)

def run(cmd: list[str]) -> None:
    print("RUN:", " ".join(str(x) for x in cmd), flush=True)
    if cmd[0] == "ffmpeg":
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=subprocess.CREATE_NO_WINDOW)
        for line in proc.stderr:
            sys.stderr.write(line)
            sys.stderr.flush()
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)
    else:
        subprocess.run(cmd, check=True, creationflags=subprocess.CREATE_NO_WINDOW)

def ffprobe(path: Path, fast: bool = False, timeout: int = 120) -> dict:
    entries = "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate"
    cmd = ["ffprobe","-v","error","-show_entries",entries,"-of","json",str(path)]
    try:
        raw = subprocess.check_output(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
            timeout=timeout,
        )
        return json.loads(raw)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffprobe timeout after {timeout}s: {path}") from e

def get_video_fps(video_path: Path) -> float | None:
    """Read true frame rate, r_frame_rate first, avg_frame_rate fallback."""
    entries = "stream=r_frame_rate,avg_frame_rate"
    cmd = ["ffprobe","-v","error","-select_streams","v:0",
           "-show_entries",entries,"-of","json",str(video_path)]
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

def build_video_encode_args(target_fps: int) -> list[str]:
    """Return NVENC encode args including -r/-fps_mode cfr. RuntimeError for unsupported fps."""
    if target_fps == 45:
        return ["-r","45","-fps_mode","cfr",
                "-c:v","h264_nvenc","-preset","p4","-rc","vbr",
                "-b:v","12M","-maxrate","15M","-bufsize","24M",
                "-pix_fmt","yuv420p",
                "-colorspace","bt709","-color_primaries","bt709",
                "-color_trc","bt709","-color_range","tv",
                "-c:a","aac","-b:a","128k"]
    if target_fps == 60:
        return ["-r","60","-fps_mode","cfr",
                "-c:v","h264_nvenc","-preset","p4","-rc","vbr",
                "-b:v","15M","-maxrate","18M","-bufsize","30M",
                "-pix_fmt","yuv420p",
                "-colorspace","bt709","-color_primaries","bt709",
                "-color_trc","bt709","-color_range","tv",
                "-c:a","aac","-b:a","128k"]
    raise RuntimeError(f"Unsupported target_fps: {target_fps} (only 45 and 60 are supported)")

def dirs(base: Path) -> None:
    for name in ["01_video_info","02_asr","03_recheck","04_detection","05_cut_plan","06_output_video","07_reports","08_logs"]:
        (base / name).mkdir(parents=True, exist_ok=True)

def get_or_load_qwen3_model():
    """复用 _MODEL_LEAK 中已有的 Qwen3ASRModel，避免重复加载导致页面文件不足。"""
    from qwen_asr import Qwen3ASRModel
    for m in _MODEL_LEAK:
        if isinstance(m, Qwen3ASRModel):
            return m
    model = Qwen3ASRModel.from_pretrained(
        str(QWEN3_MODEL_PATH), dtype=torch.float16, device_map="cuda:0",
        max_inference_batch_size=32, max_new_tokens=256,
        forced_aligner=str(QWEN3_ALIGNER_PATH),
    )
    _MODEL_LEAK.append(model)
    return model


def transcribe_qwen3(pipe, video: Path, base: Path, tag: str, bad_words: list[str], force: bool) -> pd.DataFrame:
    """Stage 1 v2.0: Full transcription with Qwen3-ASR-1.7B + hotword context biasing."""
    from qwen_asr import Qwen3ASRModel

    out = base / "02_asr" / f"{tag}_main_asr_qwen3_full_{RUN_TAG}.csv"
    tmp_out = base / "02_asr" / f"{tag}_main_asr_qwen3_full_{RUN_TAG}.tmp.csv"
    if out.exists() and not force:
        print(f"{tag} CHECKPOINT reuse existing Qwen3 ASR: {out}", flush=True)
        return pd.read_csv(out)
    # If only temp file exists (previous run crashed mid-ASR), resume from temp
    if tmp_out.exists() and not force:
        print(f"{tag} RESUME from partial ASR temp file, will continue...", flush=True)

    import soundfile as sf

    tmp_wav = base / "02_asr" / f"{tag}_tmp_full_audio.wav"
    want_duration = float(ffprobe(video)["format"]["duration"])
    if not tmp_wav.exists():
        run(["ffmpeg","-y","-i",str(video),"-vn","-ac","1","-ar","16000","-af","aresample=async=1:min_hard_comp=0.100000",str(tmp_wav)])
    # Validate: if WAV was truncated by a previous crash, re-extract
    try:
        audio_info = sf.info(str(tmp_wav))
        if audio_info.duration < want_duration - 5.0:  # >5s short = truncated
            print(f"{tag} WAV truncated ({audio_info.duration:.0f}s < {want_duration:.0f}s), re-extracting...", flush=True)
            tmp_wav.unlink()
            run(["ffmpeg","-y","-i",str(video),"-vn","-ac","1","-ar","16000","-af","aresample=async=1:min_hard_comp=0.100000",str(tmp_wav)])
            audio_info = sf.info(str(tmp_wav))
    except Exception:
        print(f"{tag} WAV corrupt, re-extracting...", flush=True)
        try: tmp_wav.unlink()
        except OSError: pass
        run(["ffmpeg","-y","-i",str(video),"-vn","-ac","1","-ar","16000","-af","aresample=async=1:min_hard_comp=0.100000",str(tmp_wav)])
        audio_info = sf.info(str(tmp_wav))
    total_duration = audio_info.duration

    model = get_or_load_qwen3_model()

    hotword_context = " ".join(bad_words[:200])

    chunk_duration = 30.0
    overlap = 2.0
    stride = chunk_duration - overlap
    rows = []
    failed_chunks = []
    fieldnames = ["文件名","ASR来源","segment_id","开始时间秒","结束时间秒","开始时间","结束时间","识别文本","模型","语言"]
    seg_counter = 0

    # Resume logic: read temp CSV if it exists from a crashed run
    resume_from = 0.0
    if tmp_out.exists():
        try:
            existing = pd.read_csv(tmp_out, on_bad_lines="skip")
            assert set(fieldnames).issubset(existing.columns)
        except Exception:
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            try: tmp_out.rename(tmp_out.with_suffix(f".broken_{ts}.csv"))
            except OSError: pass
            print(f"{tag} tmp CSV corrupt, backed up, starting fresh", flush=True)
            existing = pd.DataFrame()

        if not existing.empty:
            existing["开始时间秒"] = pd.to_numeric(existing["开始时间秒"], errors="coerce")
            existing["结束时间秒"] = pd.to_numeric(existing["结束时间秒"], errors="coerce")
            existing = existing.dropna(subset=["开始时间秒", "结束时间秒"])

            if not existing.empty:
                last_start = float(existing["开始时间秒"].max())
                resume_from = max(0.0, last_start - overlap)
                # Delete old overlapping rows that will be re-transcribed
                existing = existing[existing["结束时间秒"].astype(float) <= resume_from].copy()
                existing.to_csv(tmp_out, index=False, encoding="utf-8-sig")
                rows = existing.to_dict("records")
                seg_counter = len(rows)
                print(f"{tag} RESUME ASR from {resume_from:.0f}s (cleaned {len(rows)} segments)", flush=True)

    mode = "a" if resume_from > 0 else "w"
    with tmp_out.open(mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        chunk_start = resume_from

        while chunk_start < total_duration:
            chunk_end = min(chunk_start + chunk_duration, total_duration)
            if chunk_end - chunk_start < 1.0:
                break

            chunk_wav = base / "02_asr" / f"{tag}_chunk_{chunk_start:.0f}_{chunk_end:.0f}.wav"
            if not chunk_wav.exists():
                run(["ffmpeg","-y","-hide_banner","-loglevel","error",
                     "-ss",f"{chunk_start:.3f}","-t",f"{chunk_end-chunk_start:.3f}",
                     "-i",str(tmp_wav),"-vn","-ac","1","-ar","16000",str(chunk_wav)])

            try:
                results = model.transcribe(audio=str(chunk_wav), language="Chinese",
                                          context=hotword_context, return_time_stamps=True)
                r = results[0]
                ts = r.time_stamps

                if ts:
                    # Overlap midpoint split: keep only the unique portion of each chunk
                    if chunk_start == 0:
                        keep_start = 0.0
                    else:
                        keep_start = chunk_start + overlap / 2

                    if chunk_end >= total_duration - 0.001:
                        keep_end = total_duration
                    else:
                        keep_end = chunk_end - overlap / 2

                    for seg_text, seg_start, seg_end in _segment_qwen3_output(r.text, ts, chunk_start):
                        if not seg_text.strip():
                            continue
                        seg_mid = (seg_start + seg_end) / 2
                        if not (keep_start <= seg_mid < keep_end):
                            continue
                        seg_counter += 1
                        row = {"文件名": video.name, "ASR来源": "asr_qwen3_full_system_c",
                               "segment_id": f"main_{seg_counter:06d}",
                               "开始时间秒": round(seg_start, 3), "结束时间秒": round(seg_end, 3),
                               "开始时间": pipe.sec_to_tc(seg_start), "结束时间": pipe.sec_to_tc(seg_end),
                               "识别文本": seg_text.strip(), "模型": "Qwen3-ASR-1.7B", "语言": "zh"}
                        writer.writerow(row)
                        rows.append(row)
            except Exception as e:
                print(f"  WARN chunk {chunk_start:.0f}-{chunk_end:.0f} failed: {e}", flush=True)
                failed_chunks.append((chunk_start, chunk_end, str(e)))
                break
            finally:
                try: chunk_wav.unlink()
                except OSError: pass

            chunk_start += stride
            if seg_counter % 100 == 0:
                print(f"{tag} Qwen3 ASR segments: {seg_counter}", flush=True)
            torch.cuda.empty_cache()

    # Atomic rename: only promote temp→final when ALL chunks are done AND none failed
    if failed_chunks:
        failed_json = base / "02_asr" / f"{tag}_failed_chunks_{RUN_TAG}.json"
        failed_json.write_text(json.dumps(failed_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{tag} ASR FAILED: {len(failed_chunks)} chunks failed, tmp saved for resume", flush=True)
        raise RuntimeError(f"ASR failed: {len(failed_chunks)} chunks. See {failed_json}")
    elif chunk_start >= total_duration:
        tmp_out.replace(out)
    else:
        print(f"{tag} ASR INCOMPLETE: chunk_start={chunk_start:.0f} < total={total_duration:.0f} — temp saved for resume", flush=True)

    print(f"{tag} Qwen3 ASR complete: {len(rows)} segments", flush=True)
    write_status(base.parent, video.name, "Qwen3 ASR", f"{len(rows)} segments")
    return pd.DataFrame(rows, columns=fieldnames)


def _segment_qwen3_output(text: str, time_stamps, chunk_offset: float = 0.0):
    """Convert Qwen3 full text + per-character timestamps into punctuation-delimited segments."""
    ts_idx = 0
    char_to_ts = {}
    for i, ch in enumerate(text):
        if ts_idx < len(time_stamps) and ch not in '，。！？、；："""''!\n\r\t ':
            char_to_ts[i] = time_stamps[ts_idx]
            ts_idx += 1

    segments = []
    last_end = 0
    for m in re.finditer(r'[，。！？、；：\n]', text):
        seg_text = text[last_end:m.start()].strip()
        if seg_text:
            t_start, t_end = None, None
            for i in range(last_end, m.start()):
                if i in char_to_ts:
                    ts = char_to_ts[i]
                    if t_start is None:
                        t_start = ts.start_time
                    t_end = ts.end_time
            if t_start is not None and t_end is not None:
                segments.append((seg_text, round(t_start + chunk_offset, 3), round(t_end + chunk_offset, 3)))
        last_end = m.end()

    seg_text = text[last_end:].strip()
    if seg_text:
        t_start, t_end = None, None
        for i in range(last_end, len(text)):
            if i in char_to_ts:
                ts = char_to_ts[i]
                if t_start is None:
                    t_start = ts.start_time
                t_end = ts.end_time
        if t_start is not None and t_end is not None:
            segments.append((seg_text, round(t_start + chunk_offset, 3), round(t_end + chunk_offset, 3)))

    return segments

def same_text_runs(df: pd.DataFrame, source: str) -> list[dict]:
    runs = []
    cur = None; first = 0; start = 0.0; end = 0.0; count = 0
    for i, row in df.iterrows():
        text = str(row["识别文本"]).strip()
        s = float(row["开始时间秒"]); e = float(row["结束时间秒"])
        if text == cur:
            count += 1; end = e
        else:
            if cur is not None:
                duration = end - start
                if count >= 5 or (count >= 2 and duration >= 15):
                    runs.append({"source":source,"kind":"same_text_run","first_excel_row":first+2,
                                 "last_excel_row":i+1,"count":count,"start":round(start,3),
                                 "end":round(end,3),"duration":round(duration,3),"text":cur})
            cur = text; first = i; start = s; end = e; count = 1
    if cur is not None:
        duration = end - start
        if count >= 5 or (count >= 2 and duration >= 15):
            runs.append({"source":source,"kind":"same_text_run","first_excel_row":first+2,
                         "last_excel_row":len(df)+1,"count":count,"start":round(start,3),
                         "end":round(end,3),"duration":round(duration,3),"text":cur})
    return runs

def hallucination_segments(df: pd.DataFrame) -> list[dict]:
    templates = ["请不吝点赞","订阅","转发","打赏支持","谢谢大家"]
    items = []
    for i, row in df.iterrows():
        text = str(row["识别文本"]).strip()
        s = float(row["开始时间秒"]); e = float(row["结束时间秒"])
        duration = e - s
        asr_source = str(row.get("ASR来源", "recheck"))
        # SenseVoice outputs single long segments per window — skip duration-based hallucination check
        is_sensevoice = "sensevoice" in asr_source
        if (not is_sensevoice and duration >= 20) or abs(duration - 29.98) < 0.08 or any(t in text for t in templates):
            items.append({"source":asr_source,"kind":"long_or_template_segment",
                          "first_excel_row":i+2,"last_excel_row":i+2,"count":1,
                          "start":round(s,3),"end":round(e,3),"duration":round(duration,3),"text":text})
    return items

def merge_interval_items(items: list[dict], duration: float, pre: float = 60.0, post: float = 60.0) -> pd.DataFrame:
    intervals = []
    for item in items:
        s = max(0.0, float(item["start"]) - pre)
        e = min(duration, float(item["end"]) + post)
        if e > s:
            intervals.append([s, e, {item["source"]+":"+item["kind"]}])
    return merge_intervals(intervals, gap=15.0)

def merge_intervals(intervals: list[list], gap: float = 8.0) -> pd.DataFrame:
    intervals.sort(key=lambda x: (x[0],x[1]))
    merged = []
    for s, e, reasons in intervals:
        if not merged or s > merged[-1][1] + gap:
            merged.append([s, e, set(reasons)])
        else:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2].update(reasons)
    return pd.DataFrame([{"window_id":i,"window_start":round(s,3),"window_end":round(e,3),
                          "duration":round(e-s,3),"reason":"; ".join(sorted(reasons))}
                         for i,(s,e,reasons) in enumerate(merged, start=1)])

def merge_window_tables(tables: list[pd.DataFrame], duration: float) -> pd.DataFrame:
    intervals = []
    for table in tables:
        if table is None or table.empty:
            continue
        for _, row in table.iterrows():
            s = max(0.0, float(row["window_start"]))
            e = min(duration, float(row["window_end"]))
            if e > s:
                intervals.append([s, e, {str(row["reason"])}])
    return merge_intervals(intervals, gap=8.0)

def _parse_sensevoice_output(raw_text: str) -> tuple[str, str, str]:
    """Parse SenseVoice output format: <|zh|><|HAPPY|><|Speech|><|woitn|>text...

    Returns (emotion, audio_event, clean_text).
    """
    emotion = ""
    event = ""
    clean = raw_text
    for m in re.finditer(r'<\|([^|]+)\|>', raw_text):
        tag = m.group(1).strip()
        if tag in ("HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"):
            emotion = tag
        elif tag in ("Speech", "Music", "Applause", "Laughter", "Singing"):
            event = tag
    # Remove all tags to get clean text
    clean = re.sub(r'<\|[^|]+\|>', '', raw_text).strip()
    return emotion, event, clean


def run_sensevoice_recheck(pipe, video: Path, base: Path, tag: str, windows: pd.DataFrame, stage_name: str, force: bool) -> pd.DataFrame:
    """Stage 3 v2.0: Window recheck with SenseVoice-Small + emotion labels."""
    from funasr import AutoModel

    out = base / "03_recheck" / f"{tag}_{stage_name}_{RUN_TAG}.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    if windows.empty:
        return pd.DataFrame()

    model = AutoModel(
        model=str(SENSEVOICE_MODEL_PATH),
        device="cuda:0",
    )

    tmp = base / "03_recheck" / f"{stage_name}_tmp_audio"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    fieldnames = ["文件名","ASR来源","segment_id","window_id","window_start","window_end",
                  "开始时间秒","结束时间秒","开始时间","结束时间","识别文本","模型","语言","reason",
                  "emotion","audio_event"]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _, win in windows.iterrows():
            wid = int(win["window_id"])
            start = float(win["window_start"]); end = float(win["window_end"])
            wav = tmp / f"{stage_name}_{wid:04d}_{start:.3f}_{end:.3f}.wav"
            if force or not wav.exists():
                run(["ffmpeg","-y","-hide_banner","-loglevel","error",
                     "-ss",f"{start:.3f}","-t",f"{end-start:.3f}",
                     "-i",str(video),"-vn","-ac","1","-ar","16000",str(wav)])

            try:
                results = model.generate(input=str(wav), language="zh")
                if results and len(results) > 0:
                    r0 = results[0]
                    raw_text = (r0.get("text", "") or "").strip()
                    # Parse SenseVoice format: <|zh|><|HAPPY|><|Speech|><|woitn|>text...
                    emotion, event, clean_text = _parse_sensevoice_output(raw_text)
                else:
                    emotion, event, clean_text = "", "", ""

                if clean_text:
                    row = {"文件名": video.name,
                           "ASR来源": f"asr_sensevoice_{stage_name}",
                           "segment_id": f"{stage_name}_{wid:04d}_0001",
                           "window_id": wid, "window_start": start, "window_end": end,
                           "开始时间秒": round(start, 3), "结束时间秒": round(end, 3),
                           "开始时间": pipe.sec_to_tc(start), "结束时间": pipe.sec_to_tc(end),
                           "识别文本": clean_text, "模型": "SenseVoice-Small", "语言": "zh",
                           "reason": str(win["reason"]), "emotion": emotion, "audio_event": event}
                    writer.writerow(row)
                    rows.append(row)
            except Exception as e:
                print(f"  WARN SenseVoice window {wid} failed: {e}", flush=True)

            print(f"{tag} SenseVoice {stage_name} window {wid}/{len(windows)} emotion={emotion}", flush=True)
            write_status(base.parent, video.name, f"SenseVoice {stage_name}", f"窗口 {wid}/{len(windows)}")

    _MODEL_LEAK.append(model)
    return pd.DataFrame(rows, columns=fieldnames)

def to_detect_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    cols = ["文件名","ASR来源","segment_id","开始时间秒","结束时间秒","开始时间","结束时间","识别文本","模型","语言"]
    return df[cols].copy()

def build_cut_plan(review: pd.DataFrame) -> pd.DataFrame:
    yes = review[review["最终静音决定"].astype(str).str.strip().eq("是")].copy()
    intervals: list[dict] = []
    for idx, row in yes.iterrows():
        s0 = float(row["开始时间秒"]); e0 = float(row["结束时间秒"])
        intervals.append({"start":max(0.0,s0-6.0),"end":e0+3.0,"orig_start":s0,"orig_end":e0,
                          "word":str(row.get("词汇","")).strip(),"actual":str(row.get("实际命中词","")).strip(),
                          "source":str(row.get("ASR来源","")).strip(),"review_row":idx+2})
    intervals.sort(key=lambda x: (x["start"],x["end"]))
    merged: list[dict] = []
    for item in intervals:
        if not merged or item["start"] > merged[-1]["end"]:
            merged.append({**item, "items":[item]})
        else:
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
            merged[-1]["items"].append(item)
    rows = []
    for i, item in enumerate(merged, start=1):
        items = item["items"]
        rows.append({"cut_id":f"cut_{i:04d}","cut_start":round(item["start"],3),"cut_end":round(item["end"],3),
                     "cut_duration":round(item["end"]-item["start"],3),
                     "source_start":round(min(x["orig_start"] for x in items),3),
                     "source_end":round(max(x["orig_end"] for x in items),3),
                     "included_vocab":"; ".join(sorted(set(x["word"] for x in items if x["word"]))),
                     "included_actual_hit":"; ".join(sorted(set(x["actual"] for x in items if x["actual"]))),
                     "source_asr":"; ".join(sorted(set(x["source"] for x in items if x["source"]))),
                     "source_review_rows":", ".join(str(x["review_row"]) for x in items),
                     "merged_rows":len(items),"note":f"merged {len(items)} CLEAN review rows"})
    return pd.DataFrame(rows)

def build_cut_filter(cut_plan: pd.DataFrame, video_duration: float) -> str:
    cuts = [(float(cut_plan.iloc[i]["cut_start"]),float(cut_plan.iloc[i]["cut_end"])) for i in range(len(cut_plan)) if float(cut_plan.iloc[i]["cut_end"])>float(cut_plan.iloc[i]["cut_start"])]
    cuts.sort(key=lambda x: x[0])
    retain: list[tuple[float,float]] = []
    cursor = 0.0
    for start, end in cuts:
        start = max(0.0, start); end = min(video_duration, end)
        if start > cursor:
            retain.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < video_duration:
        retain.append((cursor, video_duration))
    retain = [(s,e) for s,e in retain if e-s >= 0.05]
    if not retain:
        raise RuntimeError("All content would be cut")
    parts = []
    for i, (start, end) in enumerate(retain):
        parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]")
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(retain)))
    parts.append(f"{concat_inputs}concat=n={len(retain)}:v=1:a=1[vout][aout]")
    return ";\n".join(parts) + "\n"

def write_full_review(pipe, main_asr: pd.DataFrame, review: pd.DataFrame, out: Path) -> pd.DataFrame:
    cc = OpenCC("t2s")
    rows = []
    for _, seg in main_asr.iterrows():
        s = float(seg["开始时间秒"]); e = float(seg["结束时间秒"])
        matched_review = review[(review["结束时间秒"].astype(float)>=s)&(review["开始时间秒"].astype(float)<=e)]
        yes = matched_review[matched_review["最终静音决定"].astype(str).str.strip().eq("是")]
        rows.append({"文件名":seg.get("文件名",""),"ASR来源":seg.get("ASR来源",""),
                     "segment_id":seg.get("segment_id",""),"开始时间秒":s,"结束时间秒":e,
                     "开始时间":seg.get("开始时间",pipe.sec_to_tc(s)),
                     "结束时间":seg.get("结束时间",pipe.sec_to_tc(e)),
                     "transcript_text":seg.get("识别文本",""),
                     "transcript_text_simplified":cc.convert(str(seg.get("识别文本",""))),
                     "overlap_review_rows":"; ".join(str(i+2) for i in matched_review.index),
                     "overlap_vocab":"; ".join(sorted(set(matched_review["词汇"].astype(str)))) if not matched_review.empty else "",
                     "overlap_actual_hit":"; ".join(sorted(set(matched_review["实际命中词"].astype(str)))) if not matched_review.empty else "",
                     "has_final_yes":"是" if not yes.empty else "否",
                     "final_yes_vocab":"; ".join(sorted(set(yes["词汇"].astype(str)))) if not yes.empty else "",
                     "final_yes_source_asr":"; ".join(sorted(set(yes["ASR来源"].astype(str)))) if not yes.empty else ""})
    full = pd.DataFrame(rows)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        full.to_excel(writer, index=False, sheet_name="FULL_transcript_review")
    return full

def generate_cut_video(video: Path, base: Path, tag: str, cut_plan: pd.DataFrame,
                      force: bool, target_fps: int) -> tuple[Path,Path,Path]:
    encode_args = build_video_encode_args(target_fps)
    plan_out = base / "05_cut_plan" / f"{tag}_cut_plan_CLEAN_{RUN_TAG}.xlsx"
    filter_out = base / "06_output_video" / f"{tag}_cut_filter_CLEAN_{RUN_TAG}.txt"
    video_out = base / "06_output_video" / f"{tag}_cut_final_CLEAN_{RUN_TAG}.mp4"
    verify_out = base / "07_reports" / f"{tag}_final_verify_{RUN_TAG}.txt"
    seg_dir = base / "06_output_video" / f"{tag}_segments"
    with pd.ExcelWriter(plan_out, engine="openpyxl") as writer:
        cut_plan.to_excel(writer, index=False, sheet_name="cut_plan")
    duration = float(ffprobe(video)["format"]["duration"])
    plan_df = pd.read_excel(plan_out)
    bypass_file = base / "_retain_bypass.json"
    if bypass_file.exists():
        import json as _json
        bypass = _json.loads(bypass_file.read_text(encoding="utf-8"))
        retain = [(float(s), float(e)) for s, e in bypass["retain"]]
    else:
        retain = _compute_retain(plan_df, duration, debug_out=base / "07_reports" / f"{tag}_retain_debug.json")
    # Reference only: actual cutting uses seg_file approach below, not this filter complex
    filter_out.write_text(build_cut_filter(pd.read_excel(plan_out), duration), encoding="utf-8")
    if force or not video_out.exists():
        seg_dir.mkdir(parents=True, exist_ok=True)
        concat_list = seg_dir / "concat_list.txt"
        concat_ts = seg_dir / "concat_output.ts"
        seg_files = []
        concat_entries = []
        for i, (start, end) in enumerate(retain):
            seg_file = seg_dir / f"seg_{i:05d}.ts"
            seg_tmp = seg_dir / f"seg_{i:05d}.ts.tmp"
            seg_files.append(seg_file)
            if not seg_file.exists() or seg_file.stat().st_size == 0:
                # Remove stale temp from a previous crash
                try: seg_tmp.unlink()
                except OSError: pass
                if start == 0.0:
                    # first segment: output-side seeking from beginning (accurate)
                    run(["ffmpeg","-y","-hide_banner","-loglevel","error",
                         "-i",str(video),"-ss","0","-t",str(end-start),
                         *encode_args,
                         "-avoid_negative_ts","make_zero",
                         "-f","mpegts",str(seg_tmp)])
                else:
                    # precise seek: -ss before -i for fast keyframe seek, -ss after -i for exact trim
                    pre_seek = max(0.0, start - 5.0)
                    run(["ffmpeg","-y","-hide_banner","-loglevel","error",
                         "-ss",str(pre_seek),"-i",str(video),
                         "-ss",str(start - pre_seek),"-t",str(end-start),
                         *encode_args,
                         "-avoid_negative_ts","make_zero",
                         "-f","mpegts",str(seg_tmp)])
                # Atomic rename: only promote temp→final on success
                seg_tmp.replace(seg_file)
                write_status(base.parent, video.name, "输出剪辑视频", f"seg_{i:05d}.ts ({i+1}/{len(retain)})")
            concat_entries.append(f"file '{seg_file.name}'")
        concat_list.write_text("\n".join(concat_entries), encoding="utf-8")
        missing = [f.name for f in seg_files if not f.exists() or f.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"Segments missing/empty, cannot concat: {missing}")
        run(["ffmpeg","-y","-hide_banner","-loglevel","error",
             "-f","concat","-safe","0","-i",str(concat_list),
             "-c","copy",str(concat_ts)])
        faststart_args = ["-movflags","+faststart"] if FASTSTART else []
        run(["ffmpeg","-y","-hide_banner","-loglevel","error",
             "-i",str(concat_ts),"-c","copy",
             *faststart_args,str(video_out)])
        # Verify moov atom; auto-retry once if missing
        try:
            ffprobe(video_out)
        except Exception:
            print("WARN: moov atom missing, retrying +faststart...", flush=True)
            run(["ffmpeg","-y","-hide_banner","-loglevel","error",
                 "-i",str(concat_ts),"-c","copy",
                 *faststart_args,str(video_out)])
            ffprobe(video_out)  # fail if still broken after retry
        for f in seg_files:
            if f.exists():
                try: f.unlink()
                except OSError: pass
        try: concat_list.unlink()
        except OSError: pass
        try: concat_ts.unlink()
        except OSError: pass
        try: seg_dir.rmdir()
        except OSError: pass
    original = ffprobe(video); output = ffprobe(video_out)
    orig_duration = float(original["format"]["duration"]); out_duration = float(output["format"]["duration"])
    planned_cut = orig_duration - sum(e - s for s, e in retain)
    actual_delta = orig_duration - out_duration
    out_video = next(s for s in output["streams"] if s["codec_type"]=="video")
    out_audio = next(s for s in output["streams"] if s["codec_type"]=="audio")
    tolerance = max(2.0, planned_cut * 0.05)
    ok = actual_delta > -0.25 and abs(actual_delta - planned_cut) <= tolerance and out_audio["codec_name"]=="aac"
    verify_out.write_text("\n".join([
        f"original: {video}",f"output: {video_out}",
        f"original_duration: {orig_duration:.3f}",f"output_duration: {out_duration:.3f}",
        f"planned_cut_duration: {planned_cut:.3f}",f"actual_duration_delta: {actual_delta:.3f}",
        f"duration_match_tolerance: {tolerance:.3f}",
        f"output_video_codec: {out_video['codec_name']}",f"output_audio_codec: {out_audio['codec_name']}",
        f"result: {'PASS' if ok else 'FAIL'}"
    ])+"\n", encoding="utf-8")
    if not ok:
        raise RuntimeError(f"cut video verification failed: {verify_out}")
    return plan_out, video_out, verify_out

def _compute_retain(cut_plan: pd.DataFrame, video_duration: float, debug_out: Path | None = None) -> list[tuple[float,float]]:
    cuts_raw = []
    for i in range(len(cut_plan)):
        s = float(cut_plan.iloc[i]["cut_start"])
        e = float(cut_plan.iloc[i]["cut_end"])
        if e > s:
            cuts_raw.append((s, e))
    cuts = sorted(cuts_raw, key=lambda x: x[0])
    # DEBUG: dump all cuts and retains to file for troubleshooting
    import json as _json
    if debug_out is None:
        debug_out = Path(os.environ.get("BATCH_DIR_OVERRIDE", str(Path(".").absolute())) + "/_compute_retain_debug.json")
    _all_cuts = [(float(cs), float(ce)) for cs, ce in cuts]
    retain = []
    cursor = 0.0
    for start, end in cuts:
        start = max(0.0, start); end = min(video_duration, end)
        if start >= end:
            continue  # cut entirely outside valid range (past video end)
        if start > cursor:
            retain.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < video_duration:
        retain.append((cursor, video_duration))
    retain = [(s,e) for s,e in retain if e-s >= 0.05]
    debug_out.write_text(_json.dumps({"n_cuts": len(cuts), "n_retain": len(retain),
        "cuts_95_105": [{"i":i,"s":round(cuts[i][0],6),"e":round(cuts[i][1],6)} for i in range(max(0,len(cuts)-30), len(cuts))],
        "retain_95_105": [{"i":i,"s":round(retain[i][0],6),"e":round(retain[i][1],6)} for i in range(max(0,len(retain)-30), len(retain))]},
        indent=2, ensure_ascii=False), encoding="utf-8")
    for i in range(1, len(retain)):
        if retain[i][0] < retain[i-1][1] - 0.01:
            raise RuntimeError(f"_compute_retain overlapping: retain[{i-1}]=({retain[i-1][0]:.3f},{retain[i-1][1]:.3f}) retain[{i}]=({retain[i][0]:.3f},{retain[i][1]:.3f}) — debug file: {debug_out}")
    return retain

def process_one(pipe, batch_dir: Path, video: Path, force: bool, target_fps: int,
                keep_temp: bool = False) -> dict:
    tag = safe_name(video.stem)
    base = batch_dir / f"{tag}_{RUN_TAG}"
    dirs(base)
    pipe.VIDEO = video
    info = ffprobe(video)
    duration = float(info["format"]["duration"])
    (base/"01_video_info"/f"{tag}_ffprobe.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")

    # v2.0: Load bad_words only (no aliases — all merged into bad_words.txt)
    bad_words = pipe.load_bad_words(WORDLIST_DIR / "bad_words.txt")

    # ── Stage 1: Qwen3-ASR full transcription ──
    main_asr = transcribe_qwen3(pipe, video, base, tag, bad_words, force)

    # ASR coverage validation
    if main_asr.empty:
        raise RuntimeError(f"ASR returned empty DataFrame for {video.name}")
    max_end = float(main_asr["结束时间秒"].max())
    if max_end < duration - 60.0:
        raise RuntimeError(f"ASR coverage incomplete: max_end={max_end:.0f}s < duration={duration:.0f}s (gap > 60s)")
    min_rows = max(100, int(duration / 5))
    if len(main_asr) < min_rows:
        raise RuntimeError(f"ASR rows too few: {len(main_asr)} < {min_rows} (expected)")

    # Keep Qwen3 on GPU — avoids CUDA fragmentation from reload cycles
    # cleanup_models()  # DISABLED: repeated load/unload causes ACCESS_VIOLATION

    # ── Stage 2: Hit detection (direct match only) ──
    main_hits = pipe.detect_hits(main_asr, bad_words, "asr_qwen3_full_system_c")
    initial_review = pipe.dedupe_review(main_hits)
    initial_plan = pipe.build_mute_plan(initial_review)

    # ── Stage 4: Bad block scan ──
    bad_blocks = same_text_runs(main_asr, "main_qwen3_full_system_c")
    bad_blocks_df = pd.DataFrame(bad_blocks)
    (base/"07_reports"/f"{tag}_qwen3_bad_blocks_{RUN_TAG}.csv").write_text(bad_blocks_df.to_csv(index=False),encoding="utf-8-sig")

    # ── Stage 5-7: Recheck (SenseVoice) ──
    if SKIP_SENSEVOICE:
        recheck_asr = pd.DataFrame()
        extra_recheck = pd.DataFrame()
        all_windows = pd.DataFrame()
        hallucination_df = pd.DataFrame()
    else:
        # ── Stage 5: Recheck windows (v2: ±30s instead of ±60s) ──
        normal_windows = pipe.build_recheck_windows(main_asr, initial_plan)
        bad_windows = merge_interval_items(bad_blocks, duration, pre=30.0, post=30.0)
        all_windows = merge_window_tables([normal_windows, bad_windows], duration)
        (base/"03_recheck"/f"{tag}_recheck_windows_{RUN_TAG}.csv").write_text(all_windows.to_csv(index=False),encoding="utf-8-sig")

        # ── Release GPU before SenseVoice (Qwen3 ~4.4GB → free for SenseVoice ~3GB) ──
        cleanup_models()

        # ── Stage 6: SenseVoice-Small recheck with emotion labels ──
        write_status(batch_dir, video.name, "SenseVoice 复核", f"窗口 0/{len(all_windows)}")
        recheck_asr = run_sensevoice_recheck(pipe, video, base, tag, all_windows, "wide_recheck", force)

        # ── Stage 7: Hallucination scan on recheck ──
        hallucinations = same_text_runs(recheck_asr, "wide_recheck_sensevoice") + hallucination_segments(recheck_asr)
        hallucination_df = pd.DataFrame(hallucinations)
        (base/"07_reports"/f"{tag}_recheck_hallucination_scan_{RUN_TAG}.csv").write_text(hallucination_df.to_csv(index=False),encoding="utf-8-sig")
        extra_recheck = pd.DataFrame()
        if hallucinations:
            write_status(batch_dir, video.name, "幻觉重试", f"{len(hallucinations)} 段 / {len(hallucination_windows := merge_interval_items(hallucinations, duration, pre=30.0, post=30.0))} 窗口")
            extra_recheck = run_sensevoice_recheck(pipe, video, base, tag, hallucination_windows, "hallucination_retry", force)

    # ── Release GPU (only after SenseVoice; Qwen3 stays if SKIP_SENSEVOICE) ──
    if not SKIP_SENSEVOICE:
        cleanup_models()

    # ── Stage 8: Final detection & CLEAN review ──
    detect_frames = [main_asr, to_detect_frame(recheck_asr)]
    if not extra_recheck.empty:
        detect_frames.append(to_detect_frame(extra_recheck))
    hit_frames = [pipe.detect_hits(df, bad_words, str(df["ASR来源"].iloc[0])) for df in detect_frames if not df.empty]
    raw_hits = pd.concat(hit_frames, ignore_index=True, sort=False) if hit_frames else pd.DataFrame()
    review = pipe.dedupe_review(raw_hits)

    # v2.0: Emotion-weighted review adjustment
    review = _apply_emotion_weighting(review, recheck_asr)

    cut_plan = build_cut_plan(review)
    write_status(batch_dir, video.name, "CLEAN 审核", f"review {len(review)} 行, cut {len(cut_plan)} 段")

    # ── Save reports ──
    raw_out = base/"04_detection"/f"{tag}_review_CLEAN_{RUN_TAG}_raw_hits.xlsx"
    review_out = base/"04_detection"/f"{tag}_review_CLEAN_{RUN_TAG}_all_fields.xlsx"
    jianying_out = base/"04_detection"/f"{tag}_transcript_review_timecode_for_jianying_{RUN_TAG}.xlsx"
    full_review_out = base/"04_detection"/f"{tag}_FULL_transcript_review_OVERLAP_MAPPED_with_simplified.xlsx"
    with pd.ExcelWriter(raw_out, engine="openpyxl") as w: raw_hits.to_excel(w, index=False, sheet_name="raw_hits")
    with pd.ExcelWriter(review_out, engine="openpyxl") as w: review.to_excel(w, index=False, sheet_name="CLEAN_review")
    jianying = review.copy()
    if not jianying.empty:
        jianying["剪映开始码"] = jianying["开始时间秒"].map(lambda x: pipe.sec_to_tc(float(x)))
        jianying["剪映结束码"] = jianying["结束时间秒"].map(lambda x: pipe.sec_to_tc(float(x)))
    with pd.ExcelWriter(jianying_out, engine="openpyxl") as w: jianying.to_excel(w, index=False, sheet_name="transcript_review_timecode")
    full_review = write_full_review(pipe, main_asr, review, full_review_out)
    plan_out, video_out, verify_out = generate_cut_video(video, base, tag, cut_plan, force, target_fps)

    # ── Stage 10: Boundary verification (re-ASR ±10s around splice points) ──
    if not SKIP_BOUNDARY_VERIFY:
        video_out, boundary_info = verify_boundaries(pipe, batch_dir, video_out, base, tag,
                                                      duration, cut_plan, bad_words, force,
                                                      target_fps, input_video=video)
    else:
        boundary_info = {
            "boundary_recut_triggered": False,
            "boundary_recut_segments": 0,
            "boundary_extra_cut_duration": 0.0,
            "boundary_verify_report": "",
            "final_duration_after_boundary": float(ffprobe(video_out)["format"]["duration"]),
        }

    # ── Pipeline report ──
    report_out = base/"07_reports"/f"{tag}_pipeline_report_{RUN_TAG}.txt"
    decision_counts = review["最终静音决定"].astype(str).value_counts().to_dict() if not review.empty else {}
    emotion_counts = review["命中来源"].astype(str).str.contains("sensevoice").value_counts().to_dict() if not review.empty else {}
    mode_str = "Qwen3-ASR only" if SKIP_SENSEVOICE else "Qwen3-ASR + SenseVoice-Small + emotion weighting"
    report = "\n".join([RUN_TAG,f"mode: System C v2.0 — {mode_str}",
        f"base: {base}",f"video: {video}",
        f"duration: {duration:.3f}",f"main_asr_rows: {len(main_asr)}",f"main_hit_rows: {len(main_hits)}",
        f"qwen3_bad_block_rows: {len(bad_blocks_df)}",f"merged_recheck_windows: {len(all_windows)}",
        f"wide_recheck_rows: {len(recheck_asr)}",f"recheck_hallucination_rows: {len(hallucination_df)}",
        f"hallucination_retry_rows: {len(extra_recheck)}",f"raw_hit_rows: {len(raw_hits)}",
        f"review_rows: {len(review)}",f"decision_counts: {decision_counts}",
        f"emotion_stats: {emotion_counts}",
        f"cut_intervals: {len(cut_plan)}",
        f"cut_duration: {float(cut_plan['cut_duration'].sum()) if not cut_plan.empty else 0.0:.3f}",
        f"full_review_rows: {len(full_review)}",
        f"cut_video_out: {video_out}",f"verify: {verify_out}",
        f"boundary_verify_enabled: {not SKIP_BOUNDARY_VERIFY}",
        f"boundary_recut_triggered: {boundary_info['boundary_recut_triggered']}",
        f"boundary_recut_segments: {boundary_info['boundary_recut_segments']}",
        f"boundary_extra_cut_duration: {boundary_info['boundary_extra_cut_duration']:.3f}",
        f"boundary_verify_report: {boundary_info['boundary_verify_report']}",
        f"final_duration_after_boundary: {boundary_info['final_duration_after_boundary']:.3f}"])+"\n"
    report_out.write_text(report, encoding="utf-8")
    print(report, flush=True)

    # Temp file cleanup (unless --keep-temp)
    if not keep_temp:
        tmp_wav = base / "02_asr" / f"{tag}_tmp_full_audio.wav"
        try: tmp_wav.unlink()
        except OSError: pass
        # SenseVoice temp directory
        sv_tmp = base / "03_recheck" / "wide_recheck_tmp_audio"
        if sv_tmp.exists():
            for f in sv_tmp.iterdir():
                try: f.unlink()
                except OSError: pass
            try: sv_tmp.rmdir()
            except OSError: pass

    return {"name":video.name,"source":str(video),"base":str(base),"duration":duration,
            "cut_video":str(video_out),"verify":str(verify_out),"full_review":str(full_review_out),
            "review":str(review_out),"cut_plan":str(plan_out),
            "cut_intervals":len(cut_plan),
            "cut_duration":float(cut_plan["cut_duration"].sum()) if not cut_plan.empty else 0.0,"result":"PASS"}


def verify_boundaries(pipe, batch_dir: Path, video_out: Path, base: Path, tag: str,
                      orig_duration: float, cut_plan: pd.DataFrame,
                      bad_words: list, force: bool, target_fps: int,
                      input_video: Path | None = None):
    """Stage 10: 边界验证 — 在输出视频每个拼接点前后各10秒重新ASR，发现漏裁就补裁。
    Returns (final_out, boundary_info)."""
    from qwen_asr import Qwen3ASRModel
    import soundfile as sf

    encode_args = build_video_encode_args(target_fps)

    boundary_info = {
        "boundary_recut_triggered": False,
        "boundary_recut_segments": 0,
        "boundary_extra_cut_duration": 0.0,
        "boundary_verify_report": "",
        "final_duration_after_boundary": 0.0,
    }

    write_status(batch_dir, Path(video_out).name, "边界验证", "计算拼接点...")

    # 1. 从 cut_plan 反算 retain 区间，再算输出视频中的拼接点
    cuts = sorted([(float(cut_plan.iloc[i]["cut_start"]), float(cut_plan.iloc[i]["cut_end"]))
                   for i in range(len(cut_plan)) if float(cut_plan.iloc[i]["cut_end"]) > float(cut_plan.iloc[i]["cut_start"])])
    retain = []
    cursor = 0.0
    for cs, ce in cuts:
        cs = max(0.0, cs); ce = min(orig_duration, ce)
        if cs > cursor:
            retain.append((cursor, cs))
        cursor = max(cursor, ce)
    if cursor < orig_duration:
        retain.append((cursor, orig_duration))
    retain = [(s, e) for s, e in retain if e - s >= 0.05]

    # 拼接点在输出视频中的时间 = 前面所有 retain 段的时长之和
    splice_points = []
    out_cursor = 0.0
    for i, (rs, re) in enumerate(retain):
        if i > 0:
            splice_points.append({"output_time": out_cursor, "cut_index": i - 1})
        out_cursor += (re - rs)

    if not splice_points:
        boundary_info["final_duration_after_boundary"] = float(ffprobe(video_out)["format"]["duration"])
        return video_out, boundary_info

    out_duration = float(ffprobe(video_out)["format"]["duration"])

    # 2. 每个拼接点前后各10秒提取音频
    snippet_dir = base / "08_boundary_verify"
    snippet_dir.mkdir(parents=True, exist_ok=True)
    snippets = []

    for i, sp in enumerate(splice_points):
        sp_time = sp["output_time"]
        ext_start = max(0.0, sp_time - 10.0)
        ext_end = min(out_duration, sp_time + 10.0)
        if ext_end - ext_start < 2.0:
            continue

        snip_wav = snippet_dir / f"{tag}_splice_{i:03d}_{sp_time:.1f}s.wav"
        if force or not snip_wav.exists():
            run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{ext_start:.3f}", "-t", f"{ext_end - ext_start:.3f}",
                 "-i", str(video_out), "-vn", "-ac", "1", "-ar", "16000",
                 str(snip_wav)])

        if snip_wav.exists() and snip_wav.stat().st_size > 1000:
            info = sf.info(str(snip_wav))
            if info.duration >= 1.0:
                snippets.append({"wav": snip_wav, "ext_start": ext_start,
                                 "output_time": sp_time, "cut_index": sp["cut_index"]})

    if not snippets:
        boundary_info["final_duration_after_boundary"] = float(ffprobe(video_out)["format"]["duration"])
        return video_out, boundary_info

    # 3. 复用主 ASR 已加载的 Qwen3 模型，逐个跑 snippet（都 < 30s，单 chunk 时间戳准）
    write_status(batch_dir, Path(video_out).name, "边界验证", f"ASR {len(snippets)} 段...")
    model = get_or_load_qwen3_model()

    # 3b. Assemble boundary ASR DataFrame and run through full detection chain
    boundary_cols = ["文件名","ASR来源","segment_id","开始时间秒","结束时间秒","开始时间","结束时间","识别文本","模型","语言"]

    boundary_rows = []
    for snip in snippets:
        try:
            results = model.transcribe(audio=str(snip["wav"]), language="Chinese",
                                       context=" ".join(bad_words[:200]), return_time_stamps=True)
            r = results[0]
            ts = r.time_stamps
            if not ts:
                continue

            for seg_text, seg_start, seg_end in _segment_qwen3_output(r.text, ts, 0.0):
                if not seg_text.strip():
                    continue
                out_start = snip["ext_start"] + seg_start
                out_end = snip["ext_start"] + seg_end
                boundary_rows.append({
                    "文件名": video_out.name,
                    "ASR来源": "asr_qwen3_boundary",
                    "segment_id": f"bnd_{snip['cut_index']:03d}_{len(boundary_rows):04d}",
                    "开始时间秒": round(out_start, 3),
                    "结束时间秒": round(out_end, 3),
                    "开始时间": pipe.sec_to_tc(out_start),
                    "结束时间": pipe.sec_to_tc(out_end),
                    "识别文本": seg_text.strip(),
                    "模型": "Qwen3-ASR-1.7B",
                    "语言": "zh",
                    "_splice_index": snip["cut_index"],
                })
        except Exception as e:
            print(f"  boundary ASR error on {snip['wav'].name}: {e}", flush=True)

    # 4. GPU cleanup
    cleanup_models()

    if not boundary_rows:
        report_path = base / "08_boundary_verify" / f"{tag}_boundary_verify_report_{RUN_TAG}.txt"
        report_lines = [f"boundary verify: {len(snippets)} splice points checked, 0 segments transcribed"]
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        boundary_info["boundary_verify_report"] = str(report_path)
        boundary_info["final_duration_after_boundary"] = float(ffprobe(video_out)["format"]["duration"])
        print(f"  边界验证通过: {len(snippets)} 个拼接点, 0 漏裁", flush=True)
        return video_out, boundary_info

    boundary_df = pd.DataFrame(boundary_rows)

    # Route through official detection chain
    detect_df = boundary_df[boundary_cols].copy()
    boundary_hits = pipe.detect_hits(detect_df, bad_words, "boundary_qwen3")
    boundary_review = pipe.dedupe_review(boundary_hits)

    # Save boundary review files
    boundary_hits.to_excel(base / "08_boundary_verify" / f"{tag}_boundary_raw_hits_{RUN_TAG}.xlsx", index=False)
    boundary_review.to_excel(base / "08_boundary_verify" / f"{tag}_boundary_review_{RUN_TAG}.xlsx", index=False)

    # Only recut confirmed hits (最终静音决定==是)
    yes_hits = boundary_review[boundary_review["最终静音决定"].astype(str).str.strip().eq("是")]
    if yes_hits.empty:
        report_path = base / "08_boundary_verify" / f"{tag}_boundary_verify_report_{RUN_TAG}.txt"
        report_lines = [f"boundary verify: {len(snippets)} splice points checked, {len(boundary_hits)} raw hits, 0 confirmed (after dedupe & context filter)"]
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        boundary_info["boundary_verify_report"] = str(report_path)
        boundary_info["final_duration_after_boundary"] = float(ffprobe(video_out)["format"]["duration"])
        print(f"  边界验证通过: {len(snippets)} 个拼接点, {len(boundary_hits)} raw hits, 0 confirmed", flush=True)
        return video_out, boundary_info

    # Build recut list from confirmed hits
    all_hits = []
    for _, row in yes_hits.iterrows():
        sid = str(row["segment_id"])
        splice_idx = 0
        mask = boundary_df["segment_id"] == sid
        if mask.any():
            splice_idx = int(boundary_df.loc[mask, "_splice_index"].values[0])
        all_hits.append({
            "output_position": float(row["开始时间秒"]),
            "word": str(row["词汇"]),
            "text": str(row["识别文本"]),
            "splice_index": splice_idx,
        })

    print(f"  边界验证发现 {len(all_hits)} 处漏裁(confirmed), 补裁中...", flush=True)
    for h in all_hits:
        print(f"    out={h['output_position']:.1f}s word={h['word']} text={h['text']}", flush=True)

    # 5. 构建补裁区间并重裁输出视频
    verify_cuts = []
    for h in all_hits:
        cs = max(0.0, h["output_position"] - 5.0)
        ce = min(out_duration, h["output_position"] + 3.0)
        verify_cuts.append((cs, ce))
    verify_cuts.sort()

    # 合并重叠
    merged_cuts = []
    for cs, ce in verify_cuts:
        if not merged_cuts or cs > merged_cuts[-1][1]:
            merged_cuts.append([cs, ce])
        else:
            merged_cuts[-1][1] = max(merged_cuts[-1][1], ce)

    # 算 retain
    verify_retain = []
    cursor2 = 0.0
    for cs, ce in merged_cuts:
        if cs > cursor2:
            verify_retain.append((cursor2, cs))
        cursor2 = max(cursor2, ce)
    if cursor2 < out_duration:
        verify_retain.append((cursor2, out_duration))
    verify_retain = [(s, e) for s, e in verify_retain if e - s >= 0.05]

    # ffmpeg concat 重裁 → 先输出到临时文件
    seg_dir = base / "08_boundary_verify" / f"{tag}_recheck_segs"
    seg_dir.mkdir(parents=True, exist_ok=True)
    concat_entries = []
    for i, (rs, re) in enumerate(verify_retain):
        seg_file = seg_dir / f"vseg_{i:04d}.ts"
        seg_tmp = seg_dir / f"vseg_{i:04d}.ts.tmp"
        if not seg_file.exists() or seg_file.stat().st_size == 0:
            try: seg_tmp.unlink()
            except OSError: pass
            if rs == 0.0:
                run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                     "-i", str(video_out), "-ss", "0", "-t", f"{re - rs:.3f}",
                     *encode_args,
                     "-avoid_negative_ts", "make_zero",
                     "-f", "mpegts", str(seg_tmp)])
            else:
                pre_seek = max(0.0, rs - 5.0)
                run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                     "-ss", f"{pre_seek:.3f}", "-i", str(video_out),
                     "-ss", f"{rs - pre_seek:.3f}", "-t", f"{re - rs:.3f}",
                     *encode_args,
                     "-avoid_negative_ts", "make_zero",
                     "-f", "mpegts", str(seg_tmp)])
            seg_tmp.replace(seg_file)
        concat_entries.append(f"file '{seg_file.name}'")

    concat_list = seg_dir / "concat_list.txt"
    concat_list.write_text("\n".join(concat_entries), encoding="utf-8")
    concat_ts = seg_dir / "concat_output.ts"
    final_out = base / "06_output_video" / f"{tag}_cut_final_CLEAN_{RUN_TAG}.mp4"
    final_tmp = base / "06_output_video" / f"{tag}_cut_final_CLEAN_{RUN_TAG}.boundary_tmp.mp4"

    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(concat_list),
         "-c", "copy", str(concat_ts)])
    faststart_args = ["-movflags","+faststart"] if FASTSTART else []
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(concat_ts), "-c", "copy",
         *faststart_args, str(final_tmp)])
    # Verify moov atom; auto-retry once if missing
    try:
        ffprobe(final_tmp)
    except Exception:
        print("WARN: boundary recut moov atom missing, retrying +faststart...", flush=True)
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(concat_ts), "-c", "copy",
             *faststart_args, str(final_tmp)])
        ffprobe(final_tmp)

    # Verify passed → replace original
    final_tmp.replace(final_out)
    total_recut = sum(ce - cs for cs, ce in merged_cuts)
    print(f"  补裁完成: {len(merged_cuts)} 段, 共 {total_recut:.1f}s → {final_out}", flush=True)

    # Write boundary verify report
    report_path = base / "08_boundary_verify" / f"{tag}_boundary_verify_report_{RUN_TAG}.txt"
    report_lines = [
        f"boundary verify: {len(all_hits)} leaks found at {len(set(h['splice_index'] for h in all_hits))} splice points",
        f"recut segments: {len(merged_cuts)}, total recut duration: {total_recut:.1f}s",
        f"final output: {final_out}",
    ]
    for h in all_hits:
        report_lines.append(f"  out={h['output_position']:.1f}s word={h['word']} text={h['text']}")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    boundary_info["boundary_recut_triggered"] = True
    boundary_info["boundary_recut_segments"] = len(merged_cuts)
    boundary_info["boundary_extra_cut_duration"] = total_recut
    boundary_info["boundary_verify_report"] = str(report_path)
    boundary_info["final_duration_after_boundary"] = float(ffprobe(final_out)["format"]["duration"])

    # 清理临时文件
    for f in seg_dir.iterdir():
        try: f.unlink()
        except OSError: pass
    try: seg_dir.rmdir()
    except OSError: pass
    for snip in snippets:
        try: snip["wav"].unlink()
        except OSError: pass

    return final_out, boundary_info


def _apply_emotion_weighting(review: pd.DataFrame, recheck_asr: pd.DataFrame) -> pd.DataFrame:
    """v2.0: Adjust review decisions using SenseVoice emotion labels.

    ANGRY + hit  → confidence↑ (keep YES)
    HAPPY  + hit → confidence↓ (likely福利 context, flip YES→NO)
    NEUTRAL + hit → no change
    """
    if review.empty or recheck_asr.empty:
        return review

    if "emotion" not in recheck_asr.columns:
        return review

    # Build emotion lookup by time window
    emotion_windows = []
    for _, row in recheck_asr.iterrows():
        emo = str(row.get("emotion", "")).strip()
        if emo and emo != "nan":
            emotion_windows.append({
                "start": float(row["开始时间秒"]),
                "end": float(row["结束时间秒"]),
                "emotion": emo,
            })

    if not emotion_windows:
        return review

    review = review.copy()
    for idx, row in review.iterrows():
        # Only apply to rows currently marked YES
        if str(row.get("最终静音决定", "")).strip() != "是":
            continue

        hit_source = str(row.get("命中来源", ""))
        is_sensevoice = "sensevoice" in hit_source

        hit_start = float(row["开始时间秒"])
        hit_end = float(row["结束时间秒"])

        # SenseVoice window cap (independent of emotion overlap)
        if is_sensevoice:
            seg_dur = hit_end - hit_start
            if seg_dur > 15.0:
                mid = (hit_start + hit_end) / 2
                review.at[idx, "开始时间秒"] = max(0.0, mid - 5.0)
                review.at[idx, "结束时间秒"] = mid + 5.0
                hit_start = max(0.0, mid - 5.0)
                hit_end = mid + 5.0
                cur = review.at[idx, "命中分类"]
                cur = "" if pd.isna(cur) else str(cur)
                review.at[idx, "命中分类"] = (cur + ";sensevoice_window_capped_10s").lstrip(";")

        # Find overlapping emotion windows
        overlap_emotions = []
        for ew in emotion_windows:
            if ew["start"] <= hit_end and ew["end"] >= hit_start:
                overlap_emotions.append(ew["emotion"])

        if not overlap_emotions:
            continue

        has_angry = any("ANGRY" in e.upper() for e in overlap_emotions)
        has_happy = any("HAPPY" in e.upper() for e in overlap_emotions)

        if is_sensevoice:
            # SenseVoice recheck hits: HAPPY → likely福利 context, demote to NO
            if has_happy and not has_angry:
                review.at[idx, "最终静音决定"] = "否"
                review.at[idx, "action"] = "review"
                cur = review.at[idx, "命中分类"]
                cur = "" if pd.isna(cur) else str(cur)
                review.at[idx, "命中分类"] = (cur + ";emotion_happy_demoted").lstrip(";")
            elif has_angry:
                cur = review.at[idx, "命中分类"]
                cur = "" if pd.isna(cur) else str(cur)
                review.at[idx, "命中分类"] = (cur + ";emotion_angry_boosted").lstrip(";")
        else:
            # Qwen3 primary ASR hits: emotion is supplementary, never demote
            if has_angry:
                cur = review.at[idx, "命中分类"]
                cur = "" if pd.isna(cur) else str(cur)
                review.at[idx, "命中分类"] = (cur + ";emotion_angry_confirmed").lstrip(";")
            elif has_happy:
                cur = review.at[idx, "命中分类"]
                cur = "" if pd.isna(cur) else str(cur)
                review.at[idx, "命中分类"] = (cur + ";emotion_happy_caution").lstrip(";")

    return review

def write_pending_merge(batch_dir: Path, results: list[dict]) -> None:
    manifest = batch_dir / "batch_system_c_outputs_PENDING_USER_REVIEW.json"
    manifest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = batch_dir / "batch_system_c_outputs_PENDING_USER_REVIEW.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")
    concat_list = batch_dir / "DO_NOT_RUN_YET_concat_inputs_sorted_by_filename.txt"
    pass_items = [item for item in results if item.get("result")=="PASS" and item.get("cut_video")]
    lines = [f"file '{Path(item['cut_video']).as_posix()}'" for item in sorted(pass_items, key=lambda x: x["name"])]
    concat_list.write_text("\n".join(lines)+"\n", encoding="utf-8")
    notice = batch_dir / "STOP_BEFORE_FINAL_MERGE_README.txt"
    notice.write_text("System C single-video processing is complete when all rows in the pending review CSV are PASS.\nThe final batch merge has NOT been run by design.\nUser asked to inspect outputs first. Only merge after explicit user confirmation.\n", encoding="utf-8")

def first_match(base: Path, pattern: str) -> Path | None:
    found = sorted(base.glob(pattern))
    return found[-1] if found else None

def existing_pass_result(batch_dir: Path, video: Path) -> dict | None:
    tag = safe_name(video.stem)
    base = batch_dir / f"{tag}_{RUN_TAG}"
    if not base.exists():
        return None
    cut_video = first_match(base, "06_output_video/*_cut_final_CLEAN_*.mp4")
    verify = first_match(base, "07_reports/*_final_verify_*.txt")
    full_review = first_match(base, "04_detection/*_FULL_transcript_review_OVERLAP_MAPPED_with_simplified.xlsx")
    review = first_match(base, "04_detection/*_review_CLEAN_*_all_fields.xlsx")
    cut_plan = first_match(base, "05_cut_plan/*_cut_plan_CLEAN_*.xlsx")
    required = [cut_video, verify, full_review, review, cut_plan]
    if not all(required):
        return None
    verify_text = verify.read_text(encoding="utf-8", errors="replace")
    if "result: PASS" not in verify_text and "result:PASS" not in verify_text:
        return None
    duration = ""; cut_intervals = ""; cut_duration = ""
    try:
        duration = float(ffprobe(video, fast=True)["format"]["duration"])
    except Exception:
        pass
    try:
        plan_df = pd.read_excel(cut_plan)
        cut_intervals = len(plan_df)
        if "cut_duration" in plan_df.columns:
            cut_duration = float(plan_df["cut_duration"].sum())
    except Exception:
        pass
    return {"name":video.name,"source":str(video),"base":str(base),"duration":duration,
            "cut_video":str(cut_video),"verify":str(verify),"full_review":str(full_review),
            "review":str(review),"cut_plan":str(cut_plan),
            "cut_intervals":cut_intervals,"cut_duration":cut_duration,"result":"PASS"}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inbox", default=str(INBOX))
    parser.add_argument("--batch-dir", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--faststart", action="store_true",
                        default=False,
                        help="Enable +faststart moov relocation (slower, for web streaming)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--video", default="")
    parser.add_argument("--target-fps", type=int, default=None,
                        choices=[45, 60],
                        help="Target output frame rate (required with --video)")
    args = parser.parse_args()
    global FASTSTART
    FASTSTART = args.faststart
    set_env()
    pipe = load_pipeline()
    if args.video:
        if args.target_fps is None:
            print("ERROR: --target-fps is required with --video", flush=True)
            sys.exit(1)
        videos = [Path(args.video)]
        print(f"SINGLE_VIDEO_MODE: {videos[0]}  target_fps={args.target_fps}", flush=True)
    else:
        inbox = Path(args.inbox)
        videos = sorted((p for p in inbox.iterdir() if p.suffix.lower() in (".mp4", ".ts")), key=lambda p: p.name)
        if args.limit:
            videos = videos[:args.limit]
        if not videos:
            raise FileNotFoundError(f"No mp4/ts videos found in {inbox}")
    batch_dir = Path(args.batch_dir) if args.batch_dir else WORK_ROOT / ("batch_system_c_cut_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"BATCH_DIR={batch_dir}", flush=True)
    print("FINAL MERGE GUARD: this script will not merge outputs.", flush=True)
    results: list[dict] = []
    for i, video in enumerate(videos, start=1):
        print(f"\n=== [{i}/{len(videos)}] PROCESS {video} ===", flush=True)
        try:
            result = None if args.force else existing_pass_result(batch_dir, video)
            if result is not None:
                print(f"SKIP existing PASS output: {video.name}", flush=True)
            else:
                result = process_one(pipe, batch_dir, video, args.force,
                                     args.target_fps, keep_temp=args.keep_temp)
            results.append(result)
            write_pending_merge(batch_dir, results)
            notify_pushplus(f"[{i}/{len(videos)}] {result['result']} {video.name[:30]}", f"status={result['result']}\ncut_duration={result.get('cut_duration','')}\ncut_intervals={result.get('cut_intervals','')}")
        except Exception as exc:
            import traceback
            fail = {"name":video.name,"source":str(video),"result":"FAIL",
                    "error":repr(exc),
                    "traceback":traceback.format_exc()}
            results.append(fail)
            write_pending_merge(batch_dir, results)
            notify_pushplus(f"[{i}/{len(videos)}] FAIL {video.name[:30]}", f"error={repr(exc)}")
            cleanup_models()
            if args.fail_fast:
                raise
    write_pending_merge(batch_dir, results)
    print(f"ALL_SINGLE_VIDEO_OUTPUTS_DONE_NO_MERGE batch_dir={batch_dir}", flush=True)
    print("STOP: user requested inspection before final merge.", flush=True)
    pass_count = sum(1 for r in results if r.get("result") == "PASS")
    fail_count = sum(1 for r in results if r.get("result") == "FAIL")
    notify_pushplus(f"ALL DONE {pass_count}PASS/{fail_count}FAIL ({len(videos)} total)", "请审核后确认是否合并。")

if __name__ == "__main__":
    main()
