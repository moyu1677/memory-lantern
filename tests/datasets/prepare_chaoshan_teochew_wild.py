"""从已获授权的 Teochew-Wild 数据集中准备固定的潮汕话 ASR 测试集。

数据集需要先由研究者在 Hugging Face 页面同意使用条件并下载。本脚本不绕过
访问控制，只处理本地已经下载、解压的数据；默认跨说话人固定抽取 30 条。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "chaoshan_teochew_wild" / "generated"


def parse_args():
    parser = argparse.ArgumentParser(description="准备 Teochew-Wild 潮汕话固定测试集")
    parser.add_argument("--dataset-root", required=True, help="解压后的 Teochew-Wild 根目录")
    parser.add_argument("--annotation", help="annotation.txt 的路径；默认在数据根目录内查找")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--ffmpeg")
    return parser.parse_args()


def find_tool(explicit: str | None, bundled_name: str, system_name: str) -> Path:
    if explicit:
        candidate = Path(explicit).resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"找不到指定工具：{candidate}")
    bundled = PROJECT_ROOT / bundled_name
    if bundled.exists():
        return bundled
    discovered = shutil.which(system_name)
    if discovered:
        return Path(discovered)
    raise FileNotFoundError(f"找不到 {system_name}，请通过命令行参数指定路径。")


def locate_annotation(dataset_root: Path, explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit).resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"标注文件不存在：{candidate}")
    direct = dataset_root / "annotation.txt"
    if direct.exists():
        return direct
    matches = list(dataset_root.rglob("annotation.txt"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError("数据目录中没有找到 annotation.txt。")
    raise RuntimeError("发现多个 annotation.txt，请使用 --annotation 明确指定。")


def category_for(reference: str) -> str:
    length = len("".join(reference.split()))
    if length <= 12:
        return "short"
    if length <= 24:
        return "medium"
    return "long"


def parse_annotation(annotation_path: Path, dataset_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(
        annotation_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("|", maxsplit=3)
        if len(parts) != 4:
            raise ValueError(f"annotation.txt 第 {line_number} 行格式错误。")
        relative_audio, speaker, reference, pinyin = (part.strip() for part in parts)
        audio_path = dataset_root.joinpath(*relative_audio.split("/"))
        if not audio_path.exists():
            # 有些压缩包会在根目录外再套一层，按相对路径末端进行一次受限搜索。
            matches = list(dataset_root.rglob(Path(relative_audio).name))
            if len(matches) == 1:
                audio_path = matches[0]
            else:
                continue
        if not reference:
            continue
        rows.append({
            "source_audio": audio_path,
            "relative_audio": relative_audio,
            "speaker": speaker,
            "reference": reference,
            "pinyin": pinyin,
            "category": category_for(reference),
        })
    return rows


def select_across_speakers(rows: list[dict[str, object]], count: int, seed: int):
    """先覆盖尽可能多的说话人，再从各说话人中轮转补齐样本。"""
    generator = random.Random(seed)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["speaker"]), []).append(row)
    speakers = sorted(grouped)
    generator.shuffle(speakers)
    for samples in grouped.values():
        samples.sort(key=lambda item: str(item["relative_audio"]))
        generator.shuffle(samples)
    selected: list[dict[str, object]] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for speaker in speakers:
            samples = grouped[speaker]
            if round_index < len(samples):
                selected.append(samples[round_index])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        round_index += 1
    return selected


def convert_to_wav(source: Path, destination: Path, ffmpeg: Path):
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
        raise RuntimeError(f"音频转换失败：{source}\n{result.stderr.strip()}")


def duration_seconds(path: Path, ffprobe: Path | None):
    if ffprobe:
        result = subprocess.run(
            [
                str(ffprobe), "-v", "error", "-show_entries", "format=duration",
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
        raise ValueError("--count 必须大于 0。")
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"数据目录不存在：{dataset_root}")
    annotation_path = locate_annotation(dataset_root, args.annotation)
    ffmpeg = find_tool(args.ffmpeg, "ffmpeg.exe", "ffmpeg")
    try:
        ffprobe = find_tool(None, "ffprobe.exe", "ffprobe")
    except FileNotFoundError:
        ffprobe = None

    candidates = parse_annotation(annotation_path, dataset_root)
    if len(candidates) < args.count:
        raise RuntimeError(
            f"只找到 {len(candidates)} 条具有音频和文字标注的样本，少于请求的 {args.count} 条。"
        )
    selected = select_across_speakers(candidates, args.count, args.seed)
    if len(selected) != args.count:
        raise RuntimeError(f"只抽取到 {len(selected)} 条样本。")

    output_root = Path(args.output_root).resolve()
    audio_dir = output_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for index, item in enumerate(selected, start=1):
        sample_id = f"TEOCHEW-WILD-{index:04d}"
        destination = audio_dir / f"{sample_id}.wav"
        convert_to_wav(Path(item["source_audio"]), destination, ffmpeg)
        duration = duration_seconds(destination, ffprobe)
        manifest_rows.append({
            "sample_id": sample_id,
            "language": "chaoshan",
            "environment": "dataset_clean",
            "category": item["category"],
            "source": "Teochew-Wild",
            "speaker": item["speaker"],
            "duration_s": "" if duration is None else duration,
            "audio_path": f"audio/{destination.name}",
            "reference": item["reference"],
            "teochew_pinyin": item["pinyin"],
            "source_path": item["relative_audio"],
        })
        print(f"准备 {index}/{args.count}: {sample_id}（{item['speaker']}）")

    manifest_path = output_root / "chaoshan_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    categories: dict[str, int] = {}
    for row in manifest_rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    summary = {
        "source": "Teochew-Wild",
        "license_note": "仅按数据集访问条件用于非商业研究；不得把原始音频提交到 Git。",
        "seed": args.seed,
        "selected_count": len(manifest_rows),
        "speaker_count": len({row["speaker"] for row in manifest_rows}),
        "category_counts": categories,
        "total_duration_s": round(sum(
            float(row["duration_s"]) for row in manifest_rows if row["duration_s"] != ""
        ), 3),
        "audio_format": "WAV, 16 kHz, mono, signed 16-bit PCM",
        "clinical_scope": "只评估潮汕话语音转写性能，不评估认知障碍诊断能力。",
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
