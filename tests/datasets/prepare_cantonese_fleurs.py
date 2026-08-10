"""流式抽取 Google FLEURS 粤语测试样本并生成项目 ASR 清单。

默认只抽取 test split 的 50 条，避免下载完整数据集。音频统一转换为
16 kHz、单声道、16-bit PCM WAV，供腾讯云粤语识别基准脚本使用。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "cantonese_fleurs" / "generated"
DATASET_NAME = "google/fleurs"
DATASET_CONFIG = "yue_hant_hk"


def parse_args():
    parser = argparse.ArgumentParser(description="准备 FLEURS 粤语固定测试集。")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="从测试集第几条开始抽取。默认从第0条开始，避免为随机洗牌下载全集。",
    )
    parser.add_argument("--ffmpeg")
    return parser.parse_args()


def find_ffmpeg(explicit):
    if explicit:
        candidate = Path(explicit).resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"找不到指定的 ffmpeg：{candidate}")
    bundled = PROJECT_ROOT / "ffmpeg.exe"
    if bundled.exists():
        return bundled
    discovered = shutil.which("ffmpeg")
    if discovered:
        return Path(discovered)
    raise FileNotFoundError("找不到 ffmpeg，请使用 --ffmpeg 指定路径。")


def category_for(reference):
    length = len("".join(str(reference or "").split()))
    if length <= 12:
        return "short"
    if length <= 24:
        return "medium"
    return "long"


def write_source_audio(audio, destination):
    payload = audio.get("bytes") if isinstance(audio, dict) else None
    if payload:
        destination.write_bytes(payload)
        return
    source_path = audio.get("path") if isinstance(audio, dict) else None
    if source_path and Path(source_path).exists():
        shutil.copyfile(source_path, destination)
        return
    raise RuntimeError("FLEURS 样本没有可读取的音频 bytes/path。")


def convert_to_wav(source, destination, ffmpeg):
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-ar", "16000", "-ac", "1",
            "-sample_fmt", "s16", str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"音频转换失败：{result.stderr.strip()}")


def duration_seconds(path, ffmpeg):
    """利用 ffmpeg 的 WAV 输出大小稳定计算时长，不依赖额外音频包。"""
    probe = PROJECT_ROOT / "ffprobe.exe"
    if not probe.exists():
        discovered = shutil.which("ffprobe")
        probe = Path(discovered) if discovered else None
    if probe:
        result = subprocess.run(
            [
                str(probe), "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        try:
            return round(float(result.stdout.strip()), 3)
        except ValueError:
            pass
    return None


def main():
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count 必须大于0。")

    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "缺少 datasets。请先执行：python -m pip install datasets"
        ) from exc

    ffmpeg = find_ffmpeg(args.ffmpeg)
    output_root = Path(args.output_root).resolve()
    audio_dir = output_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=args.split,
        streaming=True,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))
    if args.offset:
        dataset = dataset.skip(args.offset)

    manifest_rows = []
    with tempfile.TemporaryDirectory(prefix="fleurs_yue_") as temporary:
        temporary_dir = Path(temporary)
        for index, item in enumerate(dataset.take(args.count), start=1):
            sample_id = f"FLEURS-YUE-{index:04d}"
            source_audio = temporary_dir / f"{sample_id}.source"
            destination = audio_dir / f"{sample_id}.wav"
            write_source_audio(item["audio"], source_audio)
            convert_to_wav(source_audio, destination, ffmpeg)
            reference = (
                item.get("transcription")
                or item.get("raw_transcription")
                or ""
            ).strip()
            if not reference:
                raise RuntimeError(f"样本 {sample_id} 缺少参考转写。")
            duration = duration_seconds(destination, ffmpeg)
            manifest_rows.append({
                "sample_id": sample_id,
                "language": "cantonese",
                "environment": "dataset_clean",
                "category": category_for(reference),
                "source": "Google FLEURS yue_hant_hk",
                "split": args.split,
                "gender": item.get("gender", ""),
                "duration_s": duration if duration is not None else "",
                "audio_path": f"audio/{destination.name}",
                "reference": reference,
            })
            print(f"准备 {index}/{args.count}: {sample_id}")

    if len(manifest_rows) != args.count:
        raise RuntimeError(
            f"只取得 {len(manifest_rows)} 条样本，少于请求的 {args.count} 条。"
        )

    manifest_path = output_root / "cantonese_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    category_counts = {}
    for row in manifest_rows:
        category = row["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    summary = {
        "source": "Google FLEURS",
        "dataset_config": DATASET_CONFIG,
        "split": args.split,
        "offset": args.offset,
        "selected_count": len(manifest_rows),
        "category_counts": category_counts,
        "total_duration_s": round(sum(
            float(row["duration_s"]) for row in manifest_rows if row["duration_s"] != ""
        ), 3),
        "audio_format": "WAV, 16 kHz, mono, signed 16-bit PCM",
        "clinical_scope": "只评估粤语语音转写性能，不评估认知障碍诊断能力。",
    }
    summary_path = output_root / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"完成：{manifest_path}")
    print(f"完成：{summary_path}")


if __name__ == "__main__":
    main()
