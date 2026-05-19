"""
环境自检：Python / CUDA / GPU / 模型文件 / 词库。
所有检查只读，不修改任何文件。
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OK = 0
WARN = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = "") -> None:
    global OK, WARN, FAIL
    if condition:
        OK += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")

print("=" * 60)
print("系统 C v2.2 环境检查")
print("=" * 60)

# 1. Python
print("\n[1] Python 版本")
v = sys.version_info
check(f"Python {v.major}.{v.minor}.{v.micro}", v >= (3, 10), f"需要 Python 3.10+")

# 2. CUDA / GPU
print("\n[2] GPU 检查")
try:
    import torch
    check("torch installed", True)
    cuda_ok = torch.cuda.is_available()
    check("CUDA available", cuda_ok, "GPU 批处理需要 NVIDIA GPU")
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        check(f"GPU: {gpu_name}", True)
        check(f"VRAM: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total", total >= 6*1024**3, "建议 8GB+ VRAM")
except ImportError:
    check("torch installed", False, "pip install torch")

# 3. 依赖
print("\n[3] Python 依赖")
for lib in ["modelscope", "opencc", "pandas", "soundfile", "transformers", "funasr"]:
    try:
        __import__(lib)
        check(lib, True)
    except ImportError:
        check(lib, False, f"pip install {lib}")
try:
    from qwen_asr import Qwen3ASRModel
    check("qwen-asr", True)
except ImportError:
    check("qwen-asr", False, "pip install qwen-asr>=0.0.6")

# 4. 脚本
print("\n[4] 脚本文件")
for f in ["batch_system_c_cut_v2.py", "wanxiang_recheck_asr_v2_pipeline.py"]:
    p = BASE_DIR / "01_脚本" / f
    check(f"01_脚本/{f}", p.exists())

# 5. 词库
print("\n[5] 词库文件")
for f in ["bad_words.txt", "high_risk_bad_word_aliases.csv"]:
    p = BASE_DIR / "02_词库" / f
    check(f"02_词库/{f}", p.exists())

# 6. 模型
print("\n[6] 模型文件 (未下载则显示 FAIL，运行 download_models.py)")
models = {
    "qwen3-asr-1.7b": BASE_DIR / "03_模型" / "qwen3-asr-1.7b",
    "qwen3-forced-aligner-0.6b": BASE_DIR / "03_模型" / "qwen3-forced-aligner-0.6b",
    "SenseVoiceSmall": BASE_DIR / "03_模型" / "models" / "iic" / "SenseVoiceSmall",
}
all_models_ok = True
for name, path in models.items():
    has_files = path.exists() and any(path.iterdir())
    if not has_files:
        all_models_ok = False
    check(name, has_files, "运行 python download_models.py")

# 7. py_compile
print("\n[7] 脚本语法检查")
import py_compile
main = BASE_DIR / "01_脚本" / "batch_system_c_cut_v2.py"
try:
    py_compile.compile(str(main), doraise=True)
    check("py_compile", True)
except py_compile.PyCompileError as e:
    check("py_compile", False, str(e))

# Summary
print("\n" + "=" * 60)
total_checks = OK + WARN + FAIL
if FAIL:
    print(f"结果: {OK} PASS / {FAIL} FAIL — 请先修复 FAIL 项再运行批处理")
elif not all_models_ok:
    print(f"结果: {OK} PASS — 模型未下载，运行: python download_models.py")
else:
    print(f"结果: {OK} PASS — 环境就绪，可以启动批处理")
print(f"\n启动命令:")
print(f'  python 01_脚本\\batch_system_c_cut_v2.py --inbox "你的视频目录" --batch-dir "输出目录"')
print("=" * 60)
