"""汇总普通话公开数据集识别结果，生成 JSON 与论文粘贴用 Markdown。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

try:
    from benchmark_common import RESULTS_DIR, mean, p95
except ModuleNotFoundError:  # 允许 pytest 以包形式导入本模块。
    from tests.benchmarks.benchmark_common import RESULTS_DIR, mean, p95


INPUT = RESULTS_DIR / "asr_mandarin_dataset_results.csv"


def numeric(rows, field):
    values = []
    for row in rows:
        try:
            values.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def is_success(row):
    return str(row.get("success", "")).lower() in {"1", "true"}


def metric_block(rows):
    successful = [row for row in rows if is_success(row)]
    latency = numeric(successful, "latency_s")
    raw_cer = numeric(successful, "raw_cer")
    normalized_cer = numeric(successful, "normalized_cer")
    return {
        "samples": len(rows),
        "successful": len(successful),
        "success_rate_percent": (
            round(len(successful) / len(rows) * 100, 2) if rows else None
        ),
        "latency_mean_s": round(mean(latency), 4) if latency else None,
        "latency_p95_s": round(p95(latency), 4) if latency else None,
        "raw_cer_percent": round(mean(raw_cer) * 100, 2) if raw_cer else None,
        "normalized_cer_percent": (
            round(mean(normalized_cer) * 100, 2) if normalized_cer else None
        ),
    }


def summarize(rows):
    categories = {}
    for category in sorted({row.get("category", "unspecified") for row in rows}):
        category_name = category or "unspecified"
        categories[category_name] = metric_block([
            row
            for row in rows
            if (row.get("category") or "unspecified") == category_name
        ])
    return {"overall": metric_block(rows), "by_category": categories}


def display(value, suffix=""):
    return "无有效数据" if value is None else f"{value}{suffix}"


def build_markdown(summary):
    overall = summary["overall"]
    lines = [
        "# 普通话公开数据集识别测试汇总",
        "",
        "> 本结果是语音转写技术性能，不是认知障碍诊断结果。",
        "",
        "## 总体结果",
        "",
        "| 样本量 | 成功率 | 平均时延 | P95 时延 | 原始 CER | 基础规范化 CER |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {overall['samples']} | "
            f"{display(overall['success_rate_percent'], '%')} | "
            f"{display(overall['latency_mean_s'], ' s')} | "
            f"{display(overall['latency_p95_s'], ' s')} | "
            f"{display(overall['raw_cer_percent'], '%')} | "
            f"{display(overall['normalized_cer_percent'], '%')} |"
        ),
        "",
        "## 分组结果",
        "",
        "| 分组 | 样本量 | 成功率 | 基础规范化 CER | P95 时延 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for category, block in summary["by_category"].items():
        lines.append(
            f"| {category} | {block['samples']} | "
            f"{display(block['success_rate_percent'], '%')} | "
            f"{display(block['normalized_cer_percent'], '%')} | "
            f"{display(block['latency_p95_s'], ' s')} |"
        )
    lines.extend([
        "",
        "说明：基础规范化 CER 仅统一全角/半角、大小写、空白、标点和“〇/零”；",
        "不会自动替换同义词或任意改写数字，因此仍应抽查高错误样本。",
    ])
    return "\n".join(lines) + "\n"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"尚未找到测试结果：{INPUT}")
    with INPUT.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("结果文件为空。")
    summary = summarize(rows)
    json_path = RESULTS_DIR / "mandarin_dataset_summary.json"
    markdown_path = RESULTS_DIR / "mandarin_dataset_summary.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(build_markdown(summary), encoding="utf-8")
    print(f"完成：{json_path}")
    print(f"完成：{markdown_path}")


if __name__ == "__main__":
    main()
