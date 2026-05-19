"""
一键下载 3 个必要模型到 03_模型/ 目录。
需要先安装依赖：pip install modelscope
"""
import os
from pathlib import Path
from modelscope import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "03_模型"
MODEL_DIR.mkdir(exist_ok=True)

MODELS = {
    "Qwen/Qwen3-ASR-1.7B": MODEL_DIR / "qwen3-asr-1.7b",
    "Qwen/Qwen3-Forced-Aligner-0.6B": MODEL_DIR / "qwen3-forced-aligner-0.6b",
    "iic/SenseVoiceSmall": MODEL_DIR / "models" / "iic" / "SenseVoiceSmall",
}

total = len(MODELS)
for i, (model_id, target) in enumerate(MODELS.items(), 1):
    print(f"[{i}/{total}] Downloading {model_id} → {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and any(target.iterdir()):
        print(f"  SKIP: already exists")
        continue
    downloaded = snapshot_download(model_id, cache_dir=MODEL_DIR)
    # Symlink or move if downloaded to cache dir
    if Path(downloaded).resolve() != target.resolve():
        if target.exists():
            target.unlink()
        # Use junction on Windows, symlink on Linux/Mac
        try:
            os.symlink(downloaded, target, target_is_directory=True)
        except OSError:
            import shutil
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(downloaded, str(target))
    print(f"  DONE")

print(f"\nAll {total} models downloaded to {MODEL_DIR}")
print(f"Total size: ~7 GB")
