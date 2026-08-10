"""汇总普通话公开数据集识别结果，生成 JSON 与论文粘贴用 Markdown。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import statistics

try:
    from benchmark_common import RESULTS_DIR, mean, p95
except ModuleNotFoundError:  # 允许 pytest 以包形式导入本模块。
    from tests.benchmarks.benchmark_common import RESULTS_DIR, mean, p95


INPUT = RESULTS_DIR / "asr_mandarin_dataset_results.csv"
DEFAULT_PREFIX = "mandarin_dataset"


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


def bootstrap_mean_ci(values, iterations=2000, seed=20260807):
    """使用固定随机种子计算均值的非参数95%置信区间。"""
    if not values:
        return None
    if len(values) == 1:
        value = round(values[0], 4)
        return [value, value]
    generator = random.Random(seed)
    sample_size = len(values)
    estimates = []
    for _ in range(iterations):
        sample = [values[generator.randrange(sample_size)] for _ in range(sample_size)]
        estimates.append(mean(sample))
    estimates.sort()
    lower = estimates[int(iterations * 0.025)]
    upper = estimates[min(iterations - 1, int(iterations * 0.975))]
    return [round(lower, 4), round(upper, 4)]


def metric_block(rows):
    successful = [row for row in rows if is_success(row)]
    latency = numeric(successful, "latency_s")
    raw_cer = numeric(successful, "raw_cer")
    normalized_cer = numeric(successful, "normalized_cer")
    perfect_count = sum(value == 0 for value in normalized_cer)
    mean_normalized_cer = mean(normalized_cer) if normalized_cer else None
    cer_ci = bootstrap_mean_ci(normalized_cer)
    return {
        "samples": len(rows),
        "successful": len(successful),
        "success_rate_percent": (
            round(len(successful) / len(rows) * 100, 2) if rows else None
        ),
        "latency_mean_s": round(mean(latency), 4) if latency else None,
        "latency_median_s": round(statistics.median(latency), 4) if latency else None,
        "latency_p95_s": round(p95(latency), 4) if latency else None,
        "raw_cer_percent": round(mean(raw_cer) * 100, 2) if raw_cer else None,
        "normalized_cer_percent": (
            round(mean_normalized_cer * 100, 2)
            if mean_normalized_cer is not None else None
        ),
        "normalized_cer_median_percent": (
            round(statistics.median(normalized_cer) * 100, 2)
            if normalized_cer else None
        ),
        "normalized_cer_p95_percent": (
            round(p95(normalized_cer) * 100, 2) if normalized_cer else None
        ),
        "normalized_cer_mean_ci95_percent": (
            [round(value * 100, 2) for value in cer_ci] if cer_ci else None
        ),
        "character_accuracy_percent": (
            round(max(0.0, 1 - mean_normalized_cer) * 100, 2)
            if mean_normalized_cer is not None else None
        ),
        "perfect_transcription_rate_percent": (
            round(perfect_count / len(normalized_cer) * 100, 2)
            if normalized_cer else None
        ),
    }


def summarize(rows):
    def grouped(field):
        result = {}
        names = sorted({row.get(field, "unspecified") or "unspecified" for row in rows})
        for name in names:
            result[name] = metric_block([
                row for row in rows if (row.get(field) or "unspecified") == name
            ])
        return result

    return {
        "overall": metric_block(rows),
        "by_category": grouped("category"),
        "by_environment": grouped("environment"),
    }


def display(value, suffix=""):
    return "无有效数据" if value is None else f"{value}{suffix}"


def build_markdown(summary):
    overall = summary["overall"]
    cer_ci = overall.get("normalized_cer_mean_ci95_percent")
    ci_text = (
        f"{cer_ci[0]}%–{cer_ci[1]}%" if cer_ci else "无有效数据"
    )
    lines = [
        "# 普通话公开数据集识别测试汇总",
        "",
        "> 本结果是语音转写技术性能，不是认知障碍诊断结果。",
        "",
        "## 总体结果",
        "",
        "| 样本量 | 成功率 | 字符准确率 | 完美转写率 | 平均时延 | P95 时延 | 规范化 CER | CER 95% CI |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {overall['samples']} | "
            f"{display(overall['success_rate_percent'], '%')} | "
            f"{display(overall['character_accuracy_percent'], '%')} | "
            f"{display(overall['perfect_transcription_rate_percent'], '%')} | "
            f"{display(overall['latency_mean_s'], ' s')} | "
            f"{display(overall['latency_p95_s'], ' s')} | "
            f"{display(overall['normalized_cer_percent'], '%')} | "
            f"{ci_text} |"
        ),
        "",
        "## 分组结果",
        "",
        "| 分组 | 样本量 | 成功率 | 字符准确率 | 规范化 CER | P95 时延 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, block in summary["by_category"].items():
        lines.append(
            f"| {category} | {block['samples']} | "
            f"{display(block['success_rate_percent'], '%')} | "
            f"{display(block['character_accuracy_percent'], '%')} | "
            f"{display(block['normalized_cer_percent'], '%')} | "
            f"{display(block['latency_p95_s'], ' s')} |"
        )
    lines.extend([
        "",
        "## 按环境统计",
        "",
        "| 环境 | 样本量 | 成功率 | 字符准确率 | 规范化 CER | P95 时延 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for environment, block in summary["by_environment"].items():
        lines.append(
            f"| {environment} | {block['samples']} | "
            f"{display(block['success_rate_percent'], '%')} | "
            f"{display(block['character_accuracy_percent'], '%')} | "
            f"{display(block['normalized_cer_percent'], '%')} | "
            f"{display(block['latency_p95_s'], ' s')} |"
        )
    lines.extend([
        "",
        "说明：基础规范化 CER 仅统一全角/半角、大小写、空白、标点和“〇/零”；",
        "不会自动替换同义词或任意改写数字，因此仍应抽查高错误样本。",
        "字符准确率按 max(0, 1 - 平均CER) 计算；95% CI使用固定随机种子的非参数Bootstrap。",
    ])
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="汇总并可视化普通话ASR结果")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--title", default="普通话语音识别技术测试")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"尚未找到测试结果：{input_path}")
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("结果文件为空。")
    summary = summarize(rows)
    json_path = RESULTS_DIR / f"{args.prefix}_summary.json"
    markdown_path = RESULTS_DIR / f"{args.prefix}_summary.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(build_markdown(summary), encoding="utf-8")

    try:
        from benchmark_13_visualize_mandarin_asr import visualize_results
    except ModuleNotFoundError:
        from tests.benchmarks.benchmark_13_visualize_mandarin_asr import (
            visualize_results,
        )
    generated = visualize_results(
        rows,
        summary,
        RESULTS_DIR,
        prefix=args.prefix,
        title=args.title,
    )
    print(f"完成：{json_path}")
    print(f"完成：{markdown_path}")
    for path in generated:
        print(f"完成：{path}")


if __name__ == "__main__":
    main()
