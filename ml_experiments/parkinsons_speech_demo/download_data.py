"""下载并解压 UCI Parkinson's Speech 数据集。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/301/"
    "parkinson+speech+dataset+with+multiple+types+of+sound+recordings.zip"
)
DATASET_PAGE = (
    "https://archive.ics.uci.edu/dataset/301/"
    "parkinson+speech+dataset+with+multiple+types+of+sound+recordings"
)
DATASET_DOI = "https://doi.org/10.24432/C5NC8M"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REQUIRED_FILES = ("train_data.txt", "test_data.txt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "memory-lantern-research-demo/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def _write_metadata(download_sha256: str | None = None) -> None:
    metadata = {
        "name": "Parkinson's Speech with Multiple Types of Sound Recordings",
        "source_url": DATASET_URL,
        "dataset_page": DATASET_PAGE,
        "doi": DATASET_DOI,
        "license": "CC BY 4.0",
        "download_sha256": download_sha256,
        "files": {
            filename: _sha256(DATA_DIR / filename) for filename in REQUIRED_FILES
        },
    }
    (DATA_DIR / "dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_dataset(force: bool = False) -> tuple[Path, Path]:
    """确保训练文件和测试文件存在，返回它们的路径。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / REQUIRED_FILES[0]
    test_path = DATA_DIR / REQUIRED_FILES[1]

    if not force and train_path.exists() and test_path.exists():
        if not (DATA_DIR / "dataset_metadata.json").exists():
            _write_metadata()
        print(f"数据已存在：{DATA_DIR}")
        return train_path, test_path

    with tempfile.TemporaryDirectory(prefix="uci_parkinsons_") as temp_name:
        temp_dir = Path(temp_name)
        zip_path = temp_dir / "dataset.zip"
        rar_path = temp_dir / "Parkinson_Multiple_Sound_Recording.rar"
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        print(f"正在从UCI下载：{DATASET_URL}")
        _download(DATASET_URL, zip_path)

        with zipfile.ZipFile(zip_path) as archive:
            members = archive.namelist()
            if not members:
                raise RuntimeError("UCI压缩包为空。")
            rar_member = next((name for name in members if name.lower().endswith(".rar")), None)
            if rar_member is None:
                raise RuntimeError(f"UCI压缩包中没有找到RAR文件：{members}")
            with archive.open(rar_member) as source, rar_path.open("wb") as target:
                shutil.copyfileobj(source, target)

        try:
            subprocess.run(
                ["tar", "-xf", str(rar_path), "-C", str(extract_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "无法解压UCI提供的RAR文件。请确认系统可使用tar/bsdtar，"
                "或手动解压后把train_data.txt和test_data.txt放入data目录。"
            ) from exc

        for filename in REQUIRED_FILES:
            source = extract_dir / filename
            if not source.exists():
                raise RuntimeError(f"解压后缺少文件：{filename}")
            shutil.copy2(source, DATA_DIR / filename)

        _write_metadata(download_sha256=_sha256(zip_path))

    print(f"数据准备完成：{DATA_DIR}")
    return train_path, test_path


def main() -> None:
    ensure_dataset()


if __name__ == "__main__":
    main()
