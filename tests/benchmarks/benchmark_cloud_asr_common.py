"""普通话、粤语和潮汕话腾讯云 ASR 真实性能测试公共实现。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from time import perf_counter

from benchmark_common import (
    character_error_rate,
    login_session,
    raw_character_error_rate,
    utc_timestamp,
    write_csv,
)


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def run_language_benchmark(language, output_filename, args):
    manifest_path = Path(args.manifest).resolve()
    session, csrf_token = login_session(
        args.base_url,
        args.username,
        args.password,
    )
    rows = []

    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))

    selected = [
        row for row in manifest
        if row.get("language", "").strip().lower() == language
    ]
    if not selected:
        raise RuntimeError(f"清单中没有 language={language} 的样本。")

    for item in selected:
        audio_path = Path(item["audio_path"])
        if not audio_path.is_absolute():
            audio_path = manifest_path.parent / audio_path
        if not audio_path.exists():
            raise FileNotFoundError(f"音频不存在：{audio_path}")

        started = perf_counter()
        with audio_path.open("rb") as audio_stream:
            response = session.post(
                f"{args.base_url.rstrip('/')}/speech/recognize",
                data={"language": language},
                files={
                    "audio": (
                        audio_path.name,
                        audio_stream,
                        "application/octet-stream",
                    )
                },
                headers={"X-CSRF-Token": csrf_token},
                timeout=args.timeout,
            )
        latency = perf_counter() - started
        payload = response.json()
        prediction = payload.get("text", "")
        reference = item["reference"]
        success = response.status_code == 200 and bool(prediction)
        rows.append({
            "timestamp": utc_timestamp(),
            "sample_id": item["sample_id"],
            "language": language,
            "environment": item["environment"],
            "category": item.get("category", ""),
            "source": item.get("source", ""),
            "duration_s": item.get("duration_s", ""),
            "audio_path": str(audio_path),
            "reference": reference,
            "prediction": prediction,
            "http_status": response.status_code,
            "success": int(success),
            "latency_s": round(latency, 6),
            "cer": (
                round(character_error_rate(reference, prediction), 6)
                if success else ""
            ),
            "raw_cer": (
                round(raw_character_error_rate(reference, prediction), 6)
                if success else ""
            ),
            "normalized_cer": (
                round(character_error_rate(reference, prediction), 6)
                if success else ""
            ),
            "audio_duration_ms": payload.get("speech_observation", {}).get(
                "audio_duration_ms", ""
            ),
            "speech_duration_ms": payload.get("speech_observation", {}).get(
                "speech_duration_ms", ""
            ),
            "pause_count": payload.get("speech_observation", {}).get(
                "pause_count", ""
            ),
            "voiced_ratio": payload.get("speech_observation", {}).get(
                "voiced_ratio", ""
            ),
            "error": payload.get("error", ""),
        })
        print(
            f"{item['sample_id']}: HTTP {response.status_code}, "
            f"{latency:.3f}s"
        )

    output = write_csv(
        output_filename,
        rows,
        [
            "timestamp",
            "sample_id",
            "language",
            "environment",
            "category",
            "source",
            "duration_s",
            "audio_path",
            "reference",
            "prediction",
            "http_status",
            "success",
            "latency_s",
            "cer",
            "raw_cer",
            "normalized_cer",
            "audio_duration_ms",
            "speech_duration_ms",
            "pause_count",
            "voiced_ratio",
            "error",
        ],
    )
    print(f"完成：{len(rows)}条真实语音请求，结果写入 {output}")
