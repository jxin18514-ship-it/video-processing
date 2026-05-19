import re
from pathlib import Path
import pandas as pd
from opencc import OpenCC

BASE_DIR = Path(__file__).resolve().parent.parent  # 部署包根目录
SMALL_MODEL = BASE_DIR / "03_模型" / "faster-whisper-small"
LARGE_MODEL = BASE_DIR / "03_模型" / "faster-whisper-large-v3"
VIDEO = Path()

CC = OpenCC("t2s")
RISK_CONTEXT_WORDS = [
    "价格","价","直接","最低","全网","活动","到手","优惠","便宜",
    "抄","福利","清仓","秒杀","亏本","只要","立减",
]
FIRST_AUTO_MUTE_PHRASES = [
    "全网第一","销量第一","排名第一","行业第一","品牌第一","第一名",
    "第一品牌","第一选择","第一梯队","全场第一","平台第一","直播间第一",
]
SOLE_WORDS = {
    # "底板" was added because ASR misrecognizes "地板" (floor price) as "底板".
    # Without price context, "底板" can be literal shoe sole — ignore.
    "底板",
}

FIRST_ORDINARY_PHRASES = [
    "第一次","第一波","第一个","第一件","第一只","第一脚感","第一眼",
    "第一天","第一款","第一双","第一批",
]

def sec_to_tc(seconds: float, fps: int = 30) -> str:
    total_frames = max(0, int(round(float(seconds) * fps)))
    frames = total_frames % fps
    total_seconds = total_frames // fps
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    return f"{hour}:{minute:02d}:{sec:02d}:{frames:02d}"

def normalize_text(text: str) -> str:
    normalized = CC.convert(str(text or ""))
    normalized = re.sub(r"\s+", "", normalized)
    return normalized

def load_bad_words(path: Path = None) -> list[str]:
    items: list[str] = []
    if path and path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            word = normalize_text(raw)
            if word:
                items.append(word)
    return sorted(set(items), key=lambda x: (-len(x), x))

def _context_words(text: str) -> list[str]:
    return [word for word in RISK_CONTEXT_WORDS if word in text]

def _contains_price_number_dense(text: str) -> bool:
    return len(re.findall(r"\d{2,4}", text)) >= 2

def _row_base(row: pd.Series, canonical: str, actual: str, note: str, source: str, action: str, match_mode: str) -> dict:
    text = str(row.get("识别文本",""))
    normalized = normalize_text(text)
    return {
        "文件名": row.get("文件名",""),"词汇": canonical,
        "开始时间": row.get("开始时间",""),"结束时间": row.get("结束时间",""),
        "识别文本": text,"归一化识别文本": normalized,
        "是否命中": "是","注释": note,"修正说明": source,
        "开始时间秒": float(row.get("开始时间秒",0.0)),"结束时间秒": float(row.get("结束时间秒",0.0)),
        "标准命中词": canonical,"实际命中词": actual,"命中来源": source,
        "action": action,"match_mode": match_mode,"最终静音决定": "否",
        "ASR来源": row.get("ASR来源",""),"segment_id": row.get("segment_id",""),
        "命中分类": "","错别字类型": "","正式别名命中": "否",
        "风险上下文词": "","不自动静音原因": "",
    }

def detect_hits(df: pd.DataFrame, bad_words: list[str], source_name: str) -> pd.DataFrame:
    """v2.0: Direct exact match only — all aliases already merged into bad_words.txt"""
    hits: list[dict] = []
    for _, row in df.iterrows():
        normalized = normalize_text(row.get("识别文本",""))
        if not normalized:
            continue
        row_hits: list[dict] = []
        context_words = _context_words(normalized)
        for word in bad_words:
            if word and word in normalized:
                if word in SOLE_WORDS and not context_words:
                    continue
                hit = _row_base(row, word, word, "direct_bad_word", source_name, "mute", "contains")
                hit["命中分类"] = "confirmed_bad_word"
                row_hits.append(hit)
        if not row_hits:
            continue
        for hit in row_hits:
            canonical = str(hit["词汇"])
            hit["风险上下文词"] = "; ".join(context_words)
            if canonical == "第一" or hit["实际命中词"] == "第一":
                if any(phrase in normalized for phrase in FIRST_AUTO_MUTE_PHRASES):
                    hit["action"] = "mute"; hit["最终静音决定"] = "是"
                    hit["命中分类"] = "first_absolute_claim"
                elif any(phrase in normalized for phrase in FIRST_ORDINARY_PHRASES):
                    hit["action"] = "review"; hit["最终静音决定"] = "否"
                    hit["命中分类"] = "first_ordinary_expression"
                else:
                    hit["action"] = "review"; hit["最终静音决定"] = "否"
                    hit["命中分类"] = "first_review_needed"
                continue
            hit["action"] = "mute"; hit["最终静音决定"] = "是"
        hits.extend(row_hits)
    columns = ["文件名","词汇","开始时间","结束时间","识别文本","归一化识别文本","是否命中",
               "注释","修正说明","开始时间秒","结束时间秒","标准命中词","实际命中词","命中来源",
               "action","match_mode","最终静音决定","ASR来源","segment_id","命中分类",
               "错别字类型","正式别名命中","风险上下文词","不自动静音原因"]
    return pd.DataFrame(hits, columns=columns)

def dedupe_review(raw_hits: pd.DataFrame) -> pd.DataFrame:
    if raw_hits is None or raw_hits.empty:
        return pd.DataFrame(columns=["文件名","词汇","开始时间","结束时间","识别文本","归一化识别文本",
            "是否命中","注释","修正说明","开始时间秒","结束时间秒","标准命中词","实际命中词",
            "命中来源","action","match_mode","最终静音决定","ASR来源","segment_id",
            "命中分类","错别字类型","正式别名命中","风险上下文词","不自动静音原因"])
    work = raw_hits.copy()
    work["开始时间秒"] = work["开始时间秒"].astype(float)
    work["结束时间秒"] = work["结束时间秒"].astype(float)
    work = work.sort_values(["开始时间秒","结束时间秒","segment_id","词汇","是否命中"]).reset_index(drop=True)
    rows: list[dict] = []
    for _, grp in work.groupby(["segment_id","词汇"], sort=False):
        first = grp.iloc[0].to_dict()
        first["实际命中词"] = "; ".join(sorted(set(x for x in grp["实际命中词"].astype(str) if x)))
        first["命中来源"] = "; ".join(sorted(set(x for x in grp["命中来源"].astype(str) if x)))
        first["ASR来源"] = "; ".join(sorted(set(x for x in grp["ASR来源"].astype(str) if x)))
        first["match_mode"] = "alias_longest_or_merged" if len(grp) > 1 else str(first.get("match_mode","contains"))
        first["正式别名命中"] = "是" if (grp["正式别名命中"].astype(str) == "是").any() else "否"
        first["风险上下文词"] = "; ".join(sorted(set(x for x in grp["风险上下文词"].astype(str) if x and x != "nan")))
        first["不自动静音原因"] = "; ".join(sorted(set(x for x in grp["不自动静音原因"].astype(str) if x and x != "nan")))
        first["命中分类"] = "; ".join(sorted(set(x for x in grp["命中分类"].astype(str) if x and x != "nan")))
        if (grp["最终静音决定"].astype(str) == "是").any():
            first["最终静音决定"] = "是"; first["action"] = "mute"
        elif (grp["action"].astype(str).str.contains("conditional", na=False)).any():
            first["最终静音决定"] = "否"; first["action"] = "conditional_review"
        else:
            first["最终静音决定"] = "否"; first["action"] = "review"
        rows.append(first)
    review = pd.DataFrame(rows)
    review = review.sort_values(["开始时间秒","结束时间秒","segment_id","词汇"]).reset_index(drop=True)
    return review

def build_mute_plan(review: pd.DataFrame) -> pd.DataFrame:
    yes = review[review["最终静音决定"].astype(str).str.strip().eq("是")].copy()
    rows: list[dict] = []
    for i, row in enumerate(yes.itertuples(), start=1):
        start = max(0.0, float(row.开始时间秒) - 1.5)
        end = float(row.结束时间秒) + 0.5
        rows.append({"mute_id": f"mute_{i:04d}","mute_start": round(start,3),"mute_end": round(end,3),
                     "duration": round(end-start,3),"source_start": round(float(row.开始时间秒),3),
                     "source_end": round(float(row.结束时间秒),3),"词汇": str(row.词汇),
                     "实际命中词": str(row.实际命中词),"ASR来源": str(row.ASR来源)})
    return pd.DataFrame(rows)

def build_recheck_windows(main_asr: pd.DataFrame, initial_plan: pd.DataFrame) -> pd.DataFrame:
    intervals: list[list] = []
    if initial_plan is not None and not initial_plan.empty:
        for _, row in initial_plan.iterrows():
            start = max(0.0, float(row["mute_start"]) - 15.0)
            end = float(row["mute_end"]) + 15.0
            intervals.append([start, end, {"around_initial_mute"}])
    for _, row in main_asr.iterrows():
        text = normalize_text(row.get("识别文本",""))
        if not text:
            continue
        reasons: set[str] = set()
        if _context_words(text):
            reasons.add("strong_risk_context")
        if _contains_price_number_dense(text):
            reasons.add("price_number_dense")
        if not reasons:
            continue
        start = max(0.0, float(row.get("开始时间秒",0.0)) - 20.0)
        end = float(row.get("结束时间秒",0.0)) + 20.0
        intervals.append([start, end, reasons])
    intervals.sort(key=lambda x: (x[0],x[1]))
    merged: list[list] = []
    for start, end, reasons in intervals:
        if not merged or start > merged[-1][1] + 8.0:
            merged.append([start, end, set(reasons)])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].update(reasons)
    rows = [{"window_id": i,"window_start": round(start,3),"window_end": round(end,3),
             "duration": round(end-start,3),"reason": "; ".join(sorted(reasons))}
            for i, (start, end, reasons) in enumerate(merged, start=1)]
    return pd.DataFrame(rows)
