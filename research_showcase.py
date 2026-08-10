"""研究数据集与验证记录展示页的数据装配。

只读取汇总统计，不向网页暴露音频、逐条转写、文件绝对路径或参与者信息。
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_ROOT / "data" / "research_registry.json"


def _read_json(relative_path: str | None):
    if not relative_path:
        return None
    candidate = (PROJECT_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _dataset_row(item):
    row = dict(item)
    preparation = _read_json(item.get("preparation_summary")) or {}
    result = _read_json(item.get("result_summary")) or {}
    overall = result.get("overall") if isinstance(result, dict) else None
    overall = overall if isinstance(overall, dict) else {}

    prepared_count = preparation.get("selected_count")
    row["prepared_count"] = (
        prepared_count if isinstance(prepared_count, int) else item.get("planned_count", 0)
    )
    row["duration_minutes"] = None
    try:
        row["duration_minutes"] = round(float(preparation["total_duration_s"]) / 60, 1)
    except (KeyError, TypeError, ValueError):
        pass
    row["metrics"] = {
        "samples": overall.get("samples"),
        "successful": overall.get("successful"),
        "failed": None,
        "success_rate": overall.get("success_rate_percent"),
        "accuracy": overall.get("character_accuracy_percent"),
        "cer": overall.get("normalized_cer_percent"),
        "raw_cer": overall.get("raw_cer_percent"),
        "perfect_rate": overall.get("perfect_transcription_rate_percent"),
        "latency_mean": overall.get("latency_mean_s"),
        "p95": overall.get("latency_p95_s"),
        "cer_ci95": None,
    }
    samples = row["metrics"]["samples"]
    successful = row["metrics"]["successful"]
    if isinstance(samples, int) and isinstance(successful, int):
        row["metrics"]["failed"] = max(0, samples - successful)
    ci95 = overall.get("normalized_cer_mean_ci95_percent")
    if isinstance(ci95, list) and len(ci95) == 2:
        row["metrics"]["cer_ci95"] = f"{ci95[0]}–{ci95[1]}%"
    row["has_result"] = bool(overall and overall.get("successful"))
    if row["has_result"]:
        row["status"] = "completed"
        row["status_label"] = "已完成"
    return row


def load_research_showcase():
    registry = _read_json(str(REGISTRY_PATH.relative_to(PROJECT_ROOT))) or {}
    datasets = [_dataset_row(item) for item in registry.get("datasets", [])]
    completed = [item for item in datasets if item["status"] == "completed"]
    prepared_total = sum(int(item.get("prepared_count") or 0) for item in datasets)
    measured_total = sum(
        int(item["metrics"].get("successful") or 0)
        for item in completed
    )
    return {
        "last_updated": registry.get("last_updated", "—"),
        "datasets": datasets,
        "verification_records": registry.get("verification_records", []),
        "summary": {
            "dataset_count": len(datasets),
            "completed_count": len(completed),
            "prepared_total": prepared_total,
            "measured_total": measured_total,
        },
    }
