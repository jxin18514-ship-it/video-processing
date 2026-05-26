# 系统 C v2.2 视频处理系统

基于 Qwen3-ASR-1.7B 的 9 阶段视频 ASR 转录 + 违禁词检测 + 自动静音管道。

## 硬件要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| GPU | NVIDIA 6GB VRAM | NVIDIA 8GB+ VRAM |
| 内存 | 16GB | 32GB |
| 磁盘 | 输入视频大小 × 3 | 输入视频大小 × 3 |
| 系统 | Windows 10/11 | Windows 10/11 |
| CUDA | 11.8+ | 12.x |

## 快速开始

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 下载模型（约 7 GB，仅首次）

```bash
python download_models.py
```

模型会下载到 `03_模型/` 目录：
- Qwen3-ASR-1.7B（语音识别，4.4 GB）
- Qwen3-Forced-Aligner-0.6B（强制对齐，1.7 GB）
- SenseVoice-Small（复核模型，0.9 GB）

> **提示：** 如果模型不在部署包的 `03_模型/` 目录，可设置环境变量 `SYSTEM_C_MODEL_BASE` 指向模型所在的部署根目录。

### 3. 环境自检

```bash
python check_env.py
```

全部 PASS 后再启动批处理。

### 4. 启动批处理（推荐：使用 supervisor）

**第一步 — dry-run 预览：**

```bash
python 01_脚本\batch_supervisor.py --inbox "你的视频目录" --batch-dir "输出目录" --dry-run
```

dry-run 会扫描视频、按文件名前缀自动分组、计算每组 target_fps，生成 `logs/group_target_fps.json`，不启动任何子进程。

**第二步 — 正式处理：**

```bash
python 01_脚本\batch_supervisor.py --inbox "你的视频目录" --batch-dir "输出目录"
```

supervisor 会自动：
- 按文件名前缀分组（如 `kt (1).ts`, `kt (2).ts` → 组 `kt`）
- 每组计算 target_fps：全 45 → 45，全 60 → 60，45/60 混合 → 45
- 逐视频调用 `batch_system_c_cut_v2.py --video --target-fps`
- 写 `supervisor_status.json` 追踪进度
- 支持中断后续跑（自动跳过已完成视频）

**第三步 — 全部 PASS 后合并：**

```bash
python 01_脚本\merge_after_process.py --batch-dir "输出目录" --mode FULL_HQ --fps 45 --video-bitrate 10M
```

合并流程：
- 从 `logs/group_target_fps.json` 自动读取分组
- FULL_HQ 模式：每组全部 PASS 才合并
- 参数一致走 `-c copy`（无重编码，秒级完成）
- 参数不一致不会自动回退重编码，需人工确认
- **默认不加 `+faststart`**（本地播放/上传不需要）
- 需要 `+faststart` 时加 `--faststart`

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--inbox` | 输入视频所在目录 | `./输入视频/` |
| `--batch-dir` | 输出目录（所有结果写入此处） | `./输出/` |
| `--force` | 强制重跑 ASR（已有 CSV 也会重跑） | 关闭 |
| `--fail-fast` | 遇到第一个错误就停止 | 关闭 |
| `--keep-temp` | 保留中间临时文件 | 关闭 |
| `--limit N` | 只处理前 N 个视频 | 0=全部 |

### 5. 示例

```bash
# === 推荐流程（supervisor + merge） ===

# 1. dry-run 预览分组和 target_fps
python 01_脚本\batch_supervisor.py --inbox "D:\待处理视频" --batch-dir "D:\处理结果" --dry-run

# 2. 正式处理 50 个视频
python 01_脚本\batch_supervisor.py --inbox "D:\待处理视频" --batch-dir "D:\处理结果"

# 3. 中断后续跑（相同命令）
python 01_脚本\batch_supervisor.py --inbox "D:\待处理视频" --batch-dir "D:\处理结果"

# 4. 全部 PASS 后合并（dry-run 预览）
python 01_脚本\merge_after_process.py --batch-dir "D:\处理结果" --mode FULL_HQ --fps 45 --video-bitrate 10M

# 5. 确认后执行合并
python 01_脚本\merge_after_process.py --batch-dir "D:\处理结果" --mode FULL_HQ --fps 45 --video-bitrate 10M --merge
```

## 管道流程（9 阶段）

```
1. 音频提取 → 2. Qwen3-ASR 转录 → 3. 违禁词检测 → 4. 去重审核
→ 5. 坏帧扫描 → 6. SenseVoice 复核 → 7. 幻觉扫描 → 8. CLEAN 审核 → 9. 视频输出
```

每个视频输出目录结构：
```
输出目录/
└── 视频名_SYSTEM_C_CUT_V2_01/
    ├── 02_asr/              ← ASR 转录 CSV + WAV
    ├── 03_detect/           ← 违禁词检测结果
    ├── 04_review/           ← 去重审核结果
    ├── 05_mute_plan/        ← 静音计划
    ├── 06_output_video/     ← 最终视频 + concat_output.ts
    └── 视频名_cut_final_CLEAN_SYSTEM_C_CUT_V2_01.mp4  ← 最终输出
```

## 崩溃恢复

如果处理中断：

```bash
# 续跑（跳过已完成 ASR 的视频）
python 01_脚本\batch_system_c_cut_v2.py --inbox "..." --batch-dir "同一个输出目录"
```

如果最终 MP4 无法播放，从中间文件重建：
```bash
ffmpeg -i 输出目录\视频名_SYSTEM_C_CUT_V2_01\06_output_video\视频名_segments\concat_output.ts -c copy -movflags +faststart 恢复.mp4
```

## 目录结构

```
系统C视频处理_v2.2/
├── README.md
├── requirements.txt
├── download_models.py       ← 一键下载模型
├── check_env.py             ← 环境自检
├── 00_长期规则.md            ← 操作规则手册
├── 01_脚本/
│   ├── batch_supervisor.py          ← 批处理调度（推荐入口）
│   ├── batch_system_c_cut_v2.py     ← 单视频处理
│   ├── merge_after_process.py       ← 合并输出
│   └── wanxiang_recheck_asr_v2_pipeline.py
├── 02_词库/
│   ├── bad_words.txt
│   └── high_risk_bad_word_aliases.csv
└── 03_模型/                  ← 模型下载到这里
    ├── qwen3-asr-1.7b/
    ├── qwen3-forced-aligner-0.6b/
    └── models/iic/SenseVoiceSmall/
```

## 注意事项

- 输入视频支持 `.mp4` 和 `.ts` 格式
- 视频文件名避免中文特殊字符，建议用英文/数字编号
- 磁盘空间：输入视频总大小 × 3 作为输出估算
- 长时间视频（3 小时+）可能需要 30 分钟以上处理时间
- GPU 驱动版本建议 535+（太旧的驱动可能导致 CUDA 错误）
- **启动批处理前，确保没有其他 GPU 进程占用显存**
