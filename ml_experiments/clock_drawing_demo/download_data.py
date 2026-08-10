"""从论文作者公开仓库下载并校验画钟数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

try:
    from .dataset import discover_samples
except ImportError:  # 兼容在实验目录内直接运行脚本
    from dataset import discover_samples


SOURCE_URL = "https://github.com/cccnlab/CDT-API-Network/raw/refs/heads/main/clock_shulman.zip"
EXPECTED_SHA256 = "e7896ba9db5fca29fa574706b6c57f27c16cc0e7023fbee8ec52261a957811c5"
EXPECTED_COUNTS = {0: 13, 1: 20, 2: 53, 3: 352, 4: 1047, 5: 1623}
ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "raw"
ARCHIVE = RAW_ROOT / "clock_shulman.zip"
DATA_ROOT = RAW_ROOT / "clock_shulman"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(force=False):
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    if force or not ARCHIVE.exists():
        temporary = ARCHIVE.with_suffix(".part")
        print(f"正在下载：{SOURCE_URL}")
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(ARCHIVE)
    actual_hash = sha256(ARCHIVE)
    if actual_hash.lower() != EXPECTED_SHA256:
        raise RuntimeError(f"压缩包SHA256不一致：{actual_hash}")
    if force and DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    if not DATA_ROOT.exists():
        print("正在解压数据……")
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(RAW_ROOT)

    samples = discover_samples(DATA_ROOT)
    counts = {label: sum(sample.label == label for sample in samples) for label in range(6)}
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"类别数量不符合论文数据：{counts}")
    metadata = {
        "source_url": SOURCE_URL,
        "source_repository": "https://github.com/cccnlab/CDT-API-Network",
        "source_paper_doi": "10.1038/s41598-023-44723-1",
        "archive_sha256": actual_hash,
        "class_counts": counts,
        "total_images": len(samples),
        "license_note": (
            "源仓库页面未显示明确的数据许可证。数据仅保存在本机用于研究验证；"
            "重新分发、公开模型或商业使用前须向原作者确认授权。"
        ),
    }
    metadata_path = ROOT / "data" / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{len(samples)}张图片，元数据写入 {metadata_path}")
    return DATA_ROOT


def main():
    parser = argparse.ArgumentParser(description="下载并校验公开画钟评分数据")
    parser.add_argument("--force", action="store_true", help="重新下载和解压")
    arguments = parser.parse_args()
    download(force=arguments.force)


if __name__ == "__main__":
    main()
