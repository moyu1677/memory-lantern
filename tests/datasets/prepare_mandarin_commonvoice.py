"""从 Common Voice 普通话数据中固定抽样并生成项目测试清单。

脚本不负责下载数据集，也不会改动原始文件。输出音频统一为 16 kHz、单声道、
16 bit PCM WAV，便于让不同批次使用完全相同的输入条件。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "mandarin_commonvoice" / "generated"
)
NUMBER_OR_DATE = re.compile(
    r"(?:\d|[零〇一二两三四五六七八九十百千万亿]{1,8}"
    r"(?:年|月|日|号|点|时|分|秒|岁|个|元)|(?:星期|周)[一二三四五六日天])"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="准备 Common Voice 普通话固定测试集。"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--metadata",
        help="TSV 路径；默认使用数据集目录下的 test.tsv。",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--max-scan",
        type=int,
        default=3000,
        help="为控制准备时间，最多探测多少个候选音频。",
    )
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    return parser.parse_args()


def find_binary(explicit, bundled_name, command_name):
    if explicit:
        candidate = Path(explicit).resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"找不到指定程序：{candidate}")
    bundled = PROJECT_ROOT / bundled_name
    if bundled.exists():
        return bundled
    discovered = shutil.which(command_name)
    if discovered:
        return Path(discovered)
    raise FileNotFoundError(
        f"找不到 {command_name}，请使用 --{command_name} 指定路径。"
    )


def resolve_audio_path(dataset_root, relative_path):
    relative = Path(str(relative_path or "").strip())
    direct = dataset_root / relative
    if direct.exists():
        return direct
    under_clips = dataset_root / "clips" / relative
    if under_clips.exists():
        return under_clips
    return under_clips


def read_candidates(metadata_path, dataset_root):
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    candidates = []
    for row in rows:
        reference = (row.get("sentence") or row.get("text") or "").strip()
        source_path = row.get("path", "").strip()
        if not reference or not source_path:
            continue
        audio_path = resolve_audio_path(dataset_root, source_path)
        if not audio_path.exists():
            continue
        candidates.append({
            "source_audio": audio_path,
            "reference": reference,
            "client_id": row.get("client_id", ""),
            "accent": row.get("accent") or row.get("accents", ""),
            "age": row.get("age", ""),
        })
    return candidates


def probe_duration(audio_path, ffprobe):
    result = subprocess.run(
        [
            str(ffprobe),
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if 0.5 <= duration <= 20 else None


def sample_category(item):
    if NUMBER_OR_DATE.search(item["reference"]):
        return "number_or_date"
    duration = float(item["duration_s"])
    if duration < 4:
        return "short"
    if duration < 8:
        return "medium"
    return "long"


def select_balanced(records, count, seed):
    """按固定种子抽取短、中、长及数字日期句，配额不足时自动补齐。"""
    if count <= 0:
        raise ValueError("抽样数量必须大于 0。")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    buckets = {name: [] for name in ("short", "medium", "long", "number_or_date")}
    for item in shuffled:
        buckets[sample_category(item)].append(item)

    quotas = {
        "number_or_date": round(count * 0.10),
        "long": round(count * 0.20),
        "medium": round(count * 0.40),
    }
    quotas["short"] = count - sum(quotas.values())
    selected = []
    selected_keys = set()
    for category in ("number_or_date", "long", "medium", "short"):
        for item in buckets[category][:quotas[category]]:
            copied = dict(item, category=category)
            selected.append(copied)
            selected_keys.add(str(item["source_audio"]))

    for item in shuffled:
        key = str(item["source_audio"])
        if len(selected) >= count:
            break
        if key not in selected_keys:
            selected.append(dict(item, category=sample_category(item)))
            selected_keys.add(key)
    return selected


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
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"音频转换失败：{source}\n{result.stderr.strip()}")


def anonymous_speaker_id(client_id):
    if not client_id:
        return "unknown"
    return hashlib.sha256(client_id.encode("utf-8")).hexdigest()[:12]


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    metadata_path = (
        Path(args.metadata).resolve()
        if args.metadata
        else dataset_root / "test.tsv"
    )
    output_root = Path(args.output_root).resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到数据集元数据：{metadata_path}")
    ffmpeg = find_binary(args.ffmpeg, "ffmpeg.exe", "ffmpeg")
    ffprobe = find_binary(args.ffprobe, "ffprobe.exe", "ffprobe")

    candidates = read_candidates(metadata_path, dataset_root)
    random.Random(args.seed).shuffle(candidates)
    measured = []
    for item in candidates[:args.max_scan]:
        duration = probe_duration(item["source_audio"], ffprobe)
        if duration is not None:
            measured.append(dict(item, duration_s=round(duration, 3)))
    selected = select_balanced(measured, args.count, args.seed)
    if len(selected) < args.count:
        raise RuntimeError(
            f"有效样本只有 {len(selected)} 条，少于请求的 {args.count} 条；"
            "请提高 --max-scan 或检查数据集。"
        )

    audio_directory = output_root / "audio"
    manifest_rows = []
    for index, item in enumerate(selected, start=1):
        sample_id = f"MCV-ZH-{index:04d}"
        destination = audio_directory / f"{sample_id}.wav"
        convert_to_wav(item["source_audio"], destination, ffmpeg)
        manifest_rows.append({
            "sample_id": sample_id,
            "speaker_id": anonymous_speaker_id(item["client_id"]),
            "language": "mandarin",
            "accent": item["accent"],
            "age_band": item["age"],
            "environment": "dataset_clean",
            "split": metadata_path.stem,
            "audio_path": f"audio/{destination.name}",
            "reference": item["reference"],
            "source": "Mozilla Common Voice zh-CN",
            "duration_s": item["duration_s"],
            "category": item["category"],
        })

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "mandarin_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    counts = {}
    for row in manifest_rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    summary = {
        "source": "Mozilla Common Voice zh-CN",
        "metadata": str(metadata_path),
        "seed": args.seed,
        "requested_count": args.count,
        "selected_count": len(manifest_rows),
        "category_counts": counts,
        "total_duration_s": round(sum(row["duration_s"] for row in manifest_rows), 3),
        "audio_format": "WAV, 16 kHz, mono, signed 16-bit PCM",
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"完成：{len(manifest_rows)} 条，测试清单写入 {manifest_path}")


if __name__ == "__main__":
    main()
