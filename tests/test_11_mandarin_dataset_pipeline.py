"""普通话公开数据集准备、计分和汇总工具的自动测试。"""

from pathlib import Path

from tests.benchmarks.benchmark_12_summarize_mandarin_dataset import summarize
from tests.benchmarks.benchmark_13_visualize_mandarin_asr import visualize_results
from tests.benchmarks.benchmark_common import (
    character_error_rate,
    raw_character_error_rate,
)
from tests.datasets.prepare_mandarin_commonvoice import select_balanced


def test_raw_and_basic_normalized_cer_are_both_preserved():
    assert raw_character_error_rate("您好。", "您好") == 1 / 3
    assert character_error_rate("您好。", "您好") == 0


def test_fixed_seed_selection_is_reproducible_and_balanced():
    records = []
    for index in range(20):
        records.append({
            "source_audio": Path(f"short-{index}.mp3"),
            "reference": "春暖花开",
            "duration_s": 3.0,
        })
        records.append({
            "source_audio": Path(f"medium-{index}.mp3"),
            "reference": "今天阳光很好适合散步",
            "duration_s": 6.0,
        })
        records.append({
            "source_audio": Path(f"long-{index}.mp3"),
            "reference": "老人每天保持规律作息并参加轻松活动",
            "duration_s": 10.0,
        })
        records.append({
            "source_audio": Path(f"number-{index}.mp3"),
            "reference": "今天是二零二六年八月六日",
            "duration_s": 5.0,
        })

    first = select_balanced(records, count=10, seed=20260806)
    second = select_balanced(records, count=10, seed=20260806)
    assert [item["source_audio"] for item in first] == [
        item["source_audio"] for item in second
    ]
    categories = [item["category"] for item in first]
    assert categories.count("number_or_date") == 1
    assert categories.count("long") == 2
    assert categories.count("medium") == 4
    assert categories.count("short") == 3


def test_mandarin_summary_uses_only_successful_requests():
    rows = [
        {
            "success": "1",
            "category": "short",
            "latency_s": "0.8",
            "raw_cer": "0.2",
            "normalized_cer": "0.1",
        },
        {
            "success": "1",
            "category": "short",
            "latency_s": "1.2",
            "raw_cer": "0.4",
            "normalized_cer": "0.2",
        },
        {
            "success": "0",
            "category": "long",
            "latency_s": "2.0",
            "raw_cer": "",
            "normalized_cer": "",
        },
    ]
    result = summarize(rows)
    assert result["overall"]["samples"] == 3
    assert result["overall"]["successful"] == 2
    assert result["overall"]["latency_mean_s"] == 1.0
    assert result["overall"]["normalized_cer_percent"] == 15.0
    assert result["overall"]["character_accuracy_percent"] == 85.0
    assert result["overall"]["perfect_transcription_rate_percent"] == 0.0
    assert result["overall"]["normalized_cer_mean_ci95_percent"] is not None
    assert result["by_environment"]["unspecified"]["samples"] == 3


def test_mandarin_visualization_generates_valid_png_and_worst_samples(tmp_path):
    rows = [
        {
            "sample_id": "sample-1",
            "success": "1",
            "category": "short",
            "environment": "quiet",
            "latency_s": "0.8",
            "raw_cer": "0.2",
            "normalized_cer": "0.1",
            "reference": "您好",
            "prediction": "你号",
        },
        {
            "sample_id": "sample-2",
            "success": "1",
            "category": "long",
            "environment": "noise",
            "latency_s": "1.4",
            "raw_cer": "0",
            "normalized_cer": "0",
            "reference": "今天天气很好",
            "prediction": "今天天气很好",
        },
    ]
    summary = summarize(rows)
    outputs = visualize_results(
        rows,
        summary,
        tmp_path,
        prefix="unit_test",
        title="单元测试",
    )
    assert {path.suffix for path in outputs} == {".png", ".csv"}
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_server_exposes_the_original_whisper_chinese_and_english_modes(app_module):
    assert app_module.ASR_LANGUAGE_OPTIONS["auto"]["whisper_language"] == "auto"
    assert app_module.ASR_LANGUAGE_OPTIONS["mandarin"]["whisper_language"] == "zh"
    assert app_module.ASR_LANGUAGE_OPTIONS["english"]["whisper_language"] == "en"
