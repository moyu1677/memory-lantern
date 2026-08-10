"""汇总并可视化 Google FLEURS 粤语识别测试。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import statistics
from collections import Counter

import matplotlib.pyplot as plt

try:
    from benchmark_common import RESULTS_DIR, mean, p95
except ModuleNotFoundError:
    from tests.benchmarks.benchmark_common import RESULTS_DIR, mean, p95


DEFAULT_INPUT = RESULTS_DIR / "asr_cantonese_results.csv"
COLORS = {
    "green": "#39746d",
    "blue": "#4169a9",
    "orange": "#d18145",
    "cream": "#f5f1e8",
    "ink": "#1e2927",
    "muted": "#6f7775",
}
NAMES = {"short": "短句", "medium": "中句", "long": "长句"}


def numeric(rows, field):
    values = []
    for row in rows:
        try:
            values.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def successful(row):
    return str(row.get("success", "")).strip().lower() in {"1", "true"}


def bootstrap_ci(values, iterations=2000, seed=20260807):
    if not values:
        return None
    generator = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [values[generator.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    estimates.sort()
    return [estimates[int(iterations * 0.025)], estimates[int(iterations * 0.975)]]


def metric_block(rows):
    ok = [row for row in rows if successful(row)]
    latency = numeric(ok, "latency_s")
    cer = numeric(ok, "normalized_cer")
    mean_cer = mean(cer) if cer else None
    ci = bootstrap_ci(cer)
    return {
        "samples": len(rows),
        "successful": len(ok),
        "success_rate_percent": round(len(ok) / len(rows) * 100, 2) if rows else None,
        "character_accuracy_percent": round(max(0, 1 - mean_cer) * 100, 2) if mean_cer is not None else None,
        "perfect_transcription_rate_percent": round(sum(value == 0 for value in cer) / len(cer) * 100, 2) if cer else None,
        "normalized_cer_percent": round(mean_cer * 100, 2) if mean_cer is not None else None,
        "cer_ci95_percent": [round(value * 100, 2) for value in ci] if ci else None,
        "latency_mean_s": round(mean(latency), 4) if latency else None,
        "latency_median_s": round(statistics.median(latency), 4) if latency else None,
        "latency_p95_s": round(p95(latency), 4) if latency else None,
    }


def summarize(rows):
    categories = sorted({row.get("category") or "unspecified" for row in rows})
    failure_reasons = Counter(
        row.get("error", "未知错误").strip() or "未知错误"
        for row in rows
        if not successful(row)
    )
    return {
        "dataset": "Google FLEURS yue_hant_hk test split",
        "overall": metric_block(rows),
        "by_category": {
            category: metric_block([
                row for row in rows if (row.get("category") or "unspecified") == category
            ])
            for category in categories
        },
        "failure_reasons": dict(failure_reasons),
        "limitation": "朗读语音技术测试，不代表老年人群、自然对话或认知障碍筛查效度。",
    }


def display(value, suffix=""):
    return "无有效数据" if value is None else f"{value}{suffix}"


def markdown(summary):
    overall = summary["overall"]
    ci = overall.get("cer_ci95_percent")
    ci_text = f"{ci[0]}%–{ci[1]}%" if ci else "无有效数据"
    lines = [
        "# Google FLEURS 粤语识别测试汇总",
        "",
        "> 本结果只反映粤语语音转写技术性能，不构成认知障碍诊断结论。",
        "",
        "| 样本量 | 成功率 | 字符准确率 | 完美转写率 | 平均时延 | P95时延 | 平均CER | CER 95% CI |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {overall['samples']} | {display(overall['success_rate_percent'], '%')} | "
            f"{display(overall['character_accuracy_percent'], '%')} | "
            f"{display(overall['perfect_transcription_rate_percent'], '%')} | "
            f"{display(overall['latency_mean_s'], ' s')} | "
            f"{display(overall['latency_p95_s'], ' s')} | "
            f"{display(overall['normalized_cer_percent'], '%')} | {ci_text} |"
        ),
        "",
        "## 按句长分组",
        "",
        "| 分组 | 样本量 | 成功率 | 字符准确率 | 平均CER | P95时延 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, block in summary["by_category"].items():
        lines.append(
            f"| {NAMES.get(category, category)} | {block['samples']} | "
            f"{display(block['success_rate_percent'], '%')} | "
            f"{display(block['character_accuracy_percent'], '%')} | "
            f"{display(block['normalized_cer_percent'], '%')} | "
            f"{display(block['latency_p95_s'], ' s')} |"
        )
    lines.extend(["", f"局限性：{summary['limitation']}"])
    if summary.get("failure_reasons"):
        lines.extend(["", "## 失败原因", ""])
        for reason, count in summary["failure_reasons"].items():
            lines.append(f"- {count}条：{reason}")
    return "\n".join(lines) + "\n"


def configure_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def visualize(rows, summary):
    configure_font()
    overall = summary["overall"]
    ok = [row for row in rows if successful(row)]
    categories = list(summary["by_category"])
    fig = plt.figure(figsize=(14, 8.5), facecolor=COLORS["cream"])
    grid = fig.add_gridspec(3, 4, height_ratios=[0.7, 1.25, 1.35], hspace=0.46, wspace=0.4)
    fig.suptitle("Google FLEURS 粤语识别技术测试", fontsize=20, weight="bold", color=COLORS["ink"])
    fig.text(0.5, 0.93, "50条固定测试样本 · yue_hant_hk · 结果不代表临床诊断能力", ha="center", color=COLORS["muted"])
    cards = [
        ("请求成功率", overall.get("success_rate_percent"), "%", COLORS["green"]),
        ("字符准确率", overall.get("character_accuracy_percent"), "%", COLORS["blue"]),
        ("平均时延", overall.get("latency_mean_s"), "秒", COLORS["orange"]),
        ("P95时延", overall.get("latency_p95_s"), "秒", COLORS["orange"]),
    ]
    for index, (title, value, unit, color) in enumerate(cards):
        ax = fig.add_subplot(grid[0, index])
        ax.set_axis_off()
        ax.text(0.06, 0.72, title, fontsize=12, color=COLORS["muted"], transform=ax.transAxes)
        ax.text(0.06, 0.3, "—" if value is None else f"{value}{unit}", fontsize=24, weight="bold", color=color, transform=ax.transAxes)
        ax.set_facecolor("white")

    ax_cat = fig.add_subplot(grid[1, :2])
    values = [summary["by_category"][name].get("character_accuracy_percent") or 0 for name in categories]
    bars = ax_cat.barh([NAMES.get(name, name) for name in categories], values, color=COLORS["green"])
    ax_cat.set_xlim(0, 100)
    ax_cat.set_xlabel("字符准确率（%）")
    ax_cat.set_title("不同句长准确率", loc="left", weight="bold")
    ax_cat.bar_label(bars, fmt="%.1f%%", padding=3)
    ax_cat.spines[["top", "right"]].set_visible(False)

    ax_hist = fig.add_subplot(grid[1, 2:])
    cer_values = numeric(ok, "normalized_cer")
    if cer_values:
        ax_hist.hist([value * 100 for value in cer_values], bins=10, color=COLORS["blue"], edgecolor="white")
        ax_hist.axvline(overall.get("normalized_cer_percent") or 0, color=COLORS["orange"], linestyle="--", linewidth=2, label="平均CER")
        ax_hist.legend()
    else:
        ax_hist.text(0.5, 0.5, "接口未返回有效转写", ha="center", va="center", transform=ax_hist.transAxes, color=COLORS["muted"])
    ax_hist.set_xlabel("CER（%）")
    ax_hist.set_ylabel("样本数")
    ax_hist.set_title("逐样本CER分布", loc="left", weight="bold")
    ax_hist.spines[["top", "right"]].set_visible(False)

    ax_scatter = fig.add_subplot(grid[2, :2])
    ax_scatter.scatter(numeric(ok, "latency_s"), [value * 100 for value in cer_values], color=COLORS["blue"], alpha=0.75)
    ax_scatter.set_xlabel("端到端时延（秒）")
    ax_scatter.set_ylabel("CER（%）")
    ax_scatter.set_title("时延与识别错误率", loc="left", weight="bold")
    ax_scatter.spines[["top", "right"]].set_visible(False)
    if not ok:
        ax_scatter.text(0.5, 0.5, "无成功样本，无法计算CER与时延关系", ha="center", va="center", transform=ax_scatter.transAxes, color=COLORS["muted"])

    ranked = sorted(ok, key=lambda row: float(row.get("normalized_cer") or 0), reverse=True)[:10]
    worst_path = RESULTS_DIR / "cantonese_fleurs_worst_samples.csv"
    with worst_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = [
            "sample_id", "category", "reference", "prediction",
            "normalized_cer", "latency_s", "duration_s",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)
    ax_worst = fig.add_subplot(grid[2, 2:])
    labels = [row.get("sample_id", "") for row in ranked]
    worst_values = [float(row.get("normalized_cer") or 0) * 100 for row in ranked]
    ax_worst.barh(labels, worst_values, color=COLORS["orange"])
    ax_worst.invert_yaxis()
    ax_worst.set_xlabel("CER（%）")
    ax_worst.set_title("错误率最高的10条样本", loc="left", weight="bold")
    ax_worst.spines[["top", "right"]].set_visible(False)
    if not ranked:
        first_reason = next(iter(summary.get("failure_reasons", {})), "接口未返回有效结果")
        concise_reason = "腾讯云错误6001：当前网络需开通跨境流量" if "6001" in first_reason else first_reason
        ax_worst.text(0.5, 0.5, concise_reason, ha="center", va="center", wrap=True, transform=ax_worst.transAxes, color=COLORS["muted"])

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.07, top=0.88)
    output = RESULTS_DIR / "cantonese_fleurs_dashboard.png"
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output, worst_path


def main():
    parser = argparse.ArgumentParser(description="汇总 FLEURS 粤语ASR测试")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    with args.input.resolve().open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("测试结果为空。")
    result = summarize(rows)
    json_path = RESULTS_DIR / "cantonese_fleurs_summary.json"
    md_path = RESULTS_DIR / "cantonese_fleurs_summary.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown(result), encoding="utf-8")
    image_path, worst_path = visualize(rows, result)
    print(f"完成：{json_path}")
    print(f"完成：{md_path}")
    print(f"完成：{image_path}")
    print(f"完成：{worst_path}")


if __name__ == "__main__":
    main()
