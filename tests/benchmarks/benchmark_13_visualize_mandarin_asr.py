"""把普通话ASR逐样本结果转换为论文与汇报可用的可视化图表。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

try:
    from benchmark_common import RESULTS_DIR
except ModuleNotFoundError:
    from tests.benchmarks.benchmark_common import RESULTS_DIR


DEFAULT_INPUT = RESULTS_DIR / "asr_mandarin_dataset_results.csv"
COLORS = {
    "green": "#39746d",
    "blue": "#4169a9",
    "orange": "#d18145",
    "cream": "#f5f1e8",
    "ink": "#1e2927",
    "muted": "#6f7775",
    "red": "#b95d50",
}
DISPLAY_NAMES = {
    "short": "短句",
    "medium": "中句",
    "long": "长句",
    "number_or_date": "数字/日期",
    "semantic_fluency": "语义流畅性",
    "quiet": "安静环境",
    "noise": "噪声环境",
    "unspecified": "未分类",
}


def _success(row):
    return str(row.get("success", "")).strip().lower() in {"1", "true"}


def _number(row, field, default=None):
    try:
        return float(row.get(field, ""))
    except (TypeError, ValueError):
        return default


def _category(row):
    return row.get("category") or row.get("environment") or "unspecified"


def _display_name(value):
    return DISPLAY_NAMES.get(value, value)


def _set_plot_font():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _metric_card(ax, title, value, subtitle, color):
    ax.set_axis_off()
    card = FancyBboxPatch(
        (0.02, 0.08),
        0.96,
        0.84,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2,
        edgecolor="#d8dedb",
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(card)
    ax.text(0.08, 0.72, title, fontsize=12, color=COLORS["muted"], transform=ax.transAxes)
    ax.text(0.08, 0.36, value, fontsize=25, weight="bold", color=color, transform=ax.transAxes)
    ax.text(0.08, 0.16, subtitle, fontsize=9.5, color=COLORS["muted"], transform=ax.transAxes)


def _percent(value):
    return "—" if value is None else f"{value:.1f}%"


def _seconds(value):
    return "—" if value is None else f"{value:.2f}s"


def visualize_results(rows, summary, output_dir, prefix="mandarin_dataset", title="普通话语音识别技术测试"):
    """生成仪表板、错误分析图和最差样本CSV。"""
    _set_plot_font()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = [row for row in rows if _success(row)]
    overall = summary["overall"]

    dashboard_path = output_dir / f"{prefix}_dashboard.png"
    errors_path = output_dir / f"{prefix}_error_analysis.png"
    worst_csv_path = output_dir / f"{prefix}_worst_samples.csv"

    fig = plt.figure(figsize=(14, 9), facecolor=COLORS["cream"])
    grid = fig.add_gridspec(3, 4, height_ratios=[0.78, 1.45, 1.5], hspace=0.48, wspace=0.34)
    fig.suptitle(title, fontsize=19, weight="bold", color=COLORS["ink"], y=0.985)
    fig.text(
        0.5,
        0.945,
        "技术性能评估 · 结果不代表认知障碍诊断能力",
        ha="center",
        fontsize=10.5,
        color=COLORS["muted"],
    )

    cards = [
        (
            "请求成功率",
            _percent(overall.get("success_rate_percent")),
            f"{overall.get('successful', 0)}/{overall.get('samples', 0)} 条成功",
            COLORS["green"],
        ),
        (
            "字符准确率",
            _percent(overall.get("character_accuracy_percent")),
            "max(0, 1 - 平均CER)",
            COLORS["blue"],
        ),
        (
            "平均 / P95时延",
            f"{_seconds(overall.get('latency_mean_s'))} / {_seconds(overall.get('latency_p95_s'))}",
            "端到端接口响应时间",
            COLORS["orange"],
        ),
        (
            "完美转写率",
            _percent(overall.get("perfect_transcription_rate_percent")),
            "规范化CER = 0",
            COLORS["green"],
        ),
    ]
    for index, card in enumerate(cards):
        _metric_card(fig.add_subplot(grid[0, index]), *card)

    category_names = list(summary.get("by_category", {}))
    category_accuracy = [
        summary["by_category"][name].get("character_accuracy_percent") or 0
        for name in category_names
    ]
    ax_category = fig.add_subplot(grid[1, :2])
    bars = ax_category.barh(
        [_display_name(name) for name in category_names],
        category_accuracy,
        color=COLORS["green"],
    )
    ax_category.set_xlim(0, 100)
    ax_category.set_xlabel("字符准确率（%）")
    ax_category.set_title("不同句型/类别的字符准确率", loc="left", weight="bold")
    ax_category.bar_label(bars, fmt="%.1f%%", padding=4)
    ax_category.spines[["top", "right"]].set_visible(False)

    cer_values = [
        _number(row, "normalized_cer")
        for row in successful
        if _number(row, "normalized_cer") is not None
    ]
    ax_hist = fig.add_subplot(grid[1, 2:])
    if cer_values:
        ax_hist.hist(
            [value * 100 for value in cer_values],
            bins=min(12, max(4, len(cer_values) // 3)),
            color=COLORS["blue"],
            edgecolor="white",
        )
        ax_hist.axvline(
            overall.get("normalized_cer_percent") or 0,
            color=COLORS["orange"],
            linestyle="--",
            linewidth=2,
            label="平均CER",
        )
        ax_hist.legend()
    else:
        ax_hist.text(0.5, 0.5, "暂无成功样本", ha="center", va="center", transform=ax_hist.transAxes)
    ax_hist.set_xlabel("规范化CER（%）")
    ax_hist.set_ylabel("样本数")
    ax_hist.set_title("逐样本CER分布", loc="left", weight="bold")
    ax_hist.spines[["top", "right"]].set_visible(False)

    ax_scatter = fig.add_subplot(grid[2, :2])
    categories = sorted({_category(row) for row in successful})
    category_colors = [COLORS["green"], COLORS["blue"], COLORS["orange"], COLORS["red"]]
    for index, category in enumerate(categories):
        selected = [row for row in successful if _category(row) == category]
        latency = [_number(row, "latency_s") for row in selected]
        cer = [_number(row, "normalized_cer") for row in selected]
        points = [(x, y * 100) for x, y in zip(latency, cer) if x is not None and y is not None]
        if points:
            ax_scatter.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                s=55,
                alpha=0.78,
                label=_display_name(category),
                color=category_colors[index % len(category_colors)],
            )
    ax_scatter.set_xlabel("端到端时延（秒）")
    ax_scatter.set_ylabel("规范化CER（%）")
    ax_scatter.set_title("时延与识别错误率", loc="left", weight="bold")
    if categories:
        ax_scatter.legend(fontsize=9)
    ax_scatter.spines[["top", "right"]].set_visible(False)

    environments = list(summary.get("by_environment", {}))
    env_accuracy = [
        summary["by_environment"][name].get("character_accuracy_percent") or 0
        for name in environments
    ]
    env_p95 = [
        summary["by_environment"][name].get("latency_p95_s") or 0
        for name in environments
    ]
    ax_env = fig.add_subplot(grid[2, 2:])
    positions = list(range(len(environments)))
    width = 0.38
    ax_env.bar(
        [position - width / 2 for position in positions],
        env_accuracy,
        width,
        color=COLORS["green"],
        label="字符准确率（%）",
    )
    ax_env.set_ylabel("字符准确率（%）")
    ax_env.set_ylim(0, 100)
    ax_latency = ax_env.twinx()
    ax_latency.bar(
        [position + width / 2 for position in positions],
        env_p95,
        width,
        color=COLORS["orange"],
        label="P95时延（秒）",
    )
    ax_latency.set_ylabel("P95时延（秒）")
    ax_env.set_xticks(positions, [_display_name(name) for name in environments])
    ax_env.set_title("不同环境的准确率与时延", loc="left", weight="bold")
    handles_a, labels_a = ax_env.get_legend_handles_labels()
    handles_b, labels_b = ax_latency.get_legend_handles_labels()
    ax_env.legend(handles_a + handles_b, labels_a + labels_b, fontsize=9, loc="lower right")
    ax_env.spines["top"].set_visible(False)
    ax_latency.spines["top"].set_visible(False)

    fig.subplots_adjust(left=0.10, right=0.94, bottom=0.07, top=0.88)
    fig.savefig(dashboard_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    error_rows = [
        row
        for row in successful
        if (_number(row, "normalized_cer", 0) or 0) > 0
        or (_number(row, "raw_cer", 0) or 0) > 0
    ]
    ranked = sorted(
        error_rows,
        key=lambda row: _number(row, "normalized_cer", -1),
        reverse=True,
    )[:10]
    with worst_csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        fieldnames = [
            "sample_id",
            "category",
            "environment",
            "reference",
            "prediction",
            "raw_cer",
            "normalized_cer",
            "latency_s",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked)

    fig, ax = plt.subplots(figsize=(11, max(5.5, len(ranked) * 0.58)), facecolor=COLORS["cream"])
    if ranked:
        labels = [row.get("sample_id") or f"样本{index + 1}" for index, row in enumerate(ranked)]
        normalized = [(_number(row, "normalized_cer", 0) or 0) * 100 for row in ranked]
        raw = [(_number(row, "raw_cer", 0) or 0) * 100 for row in ranked]
        positions = list(range(len(ranked)))
        ax.barh(positions, raw, color="#c8cfcc", label="原始CER")
        ax.barh(positions, normalized, color=COLORS["orange"], label="规范化CER")
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlabel("CER（%）")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "暂无成功样本", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("高错误样本分析（最多10条）", loc="left", fontsize=17, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(errors_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    return [dashboard_path, errors_path, worst_csv_path]


def parse_args():
    parser = argparse.ArgumentParser(description="生成普通话ASR可视化")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prefix", default="mandarin_dataset")
    parser.add_argument("--title", default="普通话语音识别技术测试")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from benchmark_12_summarize_mandarin_dataset import summarize
    except ModuleNotFoundError:
        from tests.benchmarks.benchmark_12_summarize_mandarin_dataset import summarize

    with args.input.resolve().open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = summarize(rows)
    for output in visualize_results(
        rows,
        summary,
        RESULTS_DIR,
        prefix=args.prefix,
        title=args.title,
    ):
        print(f"完成：{output}")


if __name__ == "__main__":
    main()
