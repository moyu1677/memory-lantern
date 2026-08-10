"""一键执行数据校验和模型训练。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="运行画钟评分独立实验")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-download", action="store_true")
    arguments = parser.parse_args()
    if not arguments.skip_download:
        subprocess.run([sys.executable, str(ROOT / "download_data.py")], check=True)
    subprocess.run([
        sys.executable,
        str(ROOT / "train_model.py"),
        "--epochs", str(arguments.epochs),
        "--batch-size", str(arguments.batch_size),
        "--device", arguments.device,
    ], check=True)


if __name__ == "__main__":
    main()
