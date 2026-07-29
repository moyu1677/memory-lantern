"""汇总各独立测试文件的原始结果，生成论文表3/表4候选数据。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from benchmark_common import RESULTS_DIR, mean, p95


def read_csv(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def numeric(rows, field):
    values = []
    for row in rows:
        try:
            values.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def summarize_functional():
    path = RESULTS_DIR / "functional_results.xml"
    if not path.exists():
        return {"status": "未测试"}
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    failed = sum(
        case.find("failure") is not None or case.find("error") is not None
        for case in cases
    )
    skipped = sum(case.find("skipped") is not None for case in cases)
    passed = len(cases) - failed - skipped
    unique_names = {
        (
            case.get("classname", ""),
            re.sub(r"\[\d+(?:-\d+)?\]$", "", case.get("name", "")),
        )
        for case in cases
    }
    return {
        "status": "已测试",
        "unique_cases": len(unique_names),
        "executions": len(cases),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate_percent": (
            round(passed / len(cases) * 100, 2) if cases else None
        ),
    }


def summarize_pages():
    rows = read_csv("page_first_load_results.csv")
    if not rows:
        return {"status": "未测试"}
    by_path = {}
    for path in sorted({row["path"] for row in rows}):
        selected = [row for row in rows if row["path"] == path]
        fcp = numeric(selected, "fcp_ms")
        navigation = numeric(selected, "navigation_ms")
        by_path[path] = {
            "samples": len(selected),
            "fcp_mean_ms": round(mean(fcp), 2) if fcp else None,
            "fcp_p95_ms": round(p95(fcp), 2) if fcp else None,
            "navigation_mean_ms": (
                round(mean(navigation), 2) if navigation else None
            ),
            "navigation_p95_ms": (
                round(p95(navigation), 2) if navigation else None
            ),
        }
    all_fcp = numeric(rows, "fcp_ms")
    return {
        "status": "已测试",
        "samples": len(rows),
        "fcp_mean_ms": round(mean(all_fcp), 2) if all_fcp else None,
        "fcp_p95_ms": round(p95(all_fcp), 2) if all_fcp else None,
        "by_path": by_path,
    }


def summarize_asr(filename):
    rows = read_csv(filename)
    if not rows:
        return {"status": "未测试"}
    successful = [
        row for row in rows
        if str(row.get("success", "")).lower() in {"1", "true"}
    ]
    latency = numeric(successful, "latency_s")
    all_cer = numeric(successful, "cer")
    environments = {}
    for environment in ("quiet", "noise"):
        selected = [
            row for row in successful
            if row.get("environment") == environment
        ]
        cer_values = numeric(selected, "cer")
        environments[environment] = {
            "samples": len(selected),
            "mean_cer_percent": (
                round(mean(cer_values) * 100, 2) if cer_values else None
            ),
        }
    return {
        "status": "已测试",
        "samples": len(rows),
        "successful": len(successful),
        "success_rate_percent": round(
            len(successful) / len(rows) * 100,
            2,
        ),
        "latency_mean_s": round(mean(latency), 4) if latency else None,
        "latency_p95_s": round(p95(latency), 4) if latency else None,
        "mean_cer_percent": (
            round(mean(all_cer) * 100, 2) if all_cer else None
        ),
        "by_environment": environments,
    }


def summarize_report():
    rows = read_csv("report_generation_results.csv")
    if not rows:
        return {"status": "未测试"}
    values = numeric(rows, "latency_s")
    return {
        "status": "已测试",
        "requests": len(rows),
        "mean_latency_s": round(mean(values), 4) if values else None,
        "p95_latency_s": round(p95(values), 4) if values else None,
    }


def summarize_permissions():
    rows = read_csv("permission_results.csv")
    if not rows:
        return {"status": "未测试"}
    blocked = sum(int(row["blocked"]) for row in rows)
    return {
        "status": "已测试",
        "attempted": len(rows),
        "blocked": blocked,
        "blocked_rate_percent": round(blocked / len(rows) * 100, 2),
    }


def summarize_experience():
    rows = read_csv("user_experience_results.csv")
    if not rows:
        return {"status": "未测试"}
    completion = numeric(rows, "completed")
    minutes = numeric(rows, "completion_minutes")
    satisfaction = numeric(rows, "satisfaction")
    independent = sum(
        row.get("assistance") == "none" and row.get("completed") == "1"
        for row in rows
    )
    return {
        "status": "已测试",
        "participants": len(rows),
        "completion_rate_percent": round(mean(completion) * 100, 2),
        "independent_completion_rate_percent": round(
            independent / len(rows) * 100,
            2,
        ),
        "mean_completion_minutes": round(mean(minutes), 2),
        "mean_satisfaction": round(mean(satisfaction), 2),
    }


def display(value, suffix=""):
    return "未测试" if value is None else f"{value}{suffix}"


def build_markdown(summary):
    functional = summary["functional"]
    pages = summary["page_first_load"]
    report = summary["report_generation"]
    permissions = summary["permission_control"]
    lines = [
        "# 论文测试结果自动汇总",
        "",
        "## 表3 功能与性能测试候选数据",
        "",
        "| 测试项目 | 样本/次数 | 结果 |",
        "| --- | ---: | --- |",
        (
            f"| 核心业务功能 | {functional.get('executions', '未测试')} | "
            f"{display(functional.get('pass_rate_percent'), '%')} |"
        ),
        (
            f"| 页面首屏加载 | {pages.get('samples', '未测试')} | "
            f"均值 {display(pages.get('fcp_mean_ms'), ' ms')}，"
            f"P95 {display(pages.get('fcp_p95_ms'), ' ms')} |"
        ),
    ]
    for label, key in (
        ("普通话识别", "mandarin_asr"),
        ("粤语识别", "cantonese_asr"),
        ("潮汕话识别", "chaoshan_asr"),
    ):
        item = summary[key]
        lines.append(
            f"| {label} | {item.get('samples', '未测试')} | "
            f"均值 {display(item.get('latency_mean_s'), ' s')}，"
            f"P95 {display(item.get('latency_p95_s'), ' s')} |"
        )
    lines.extend([
        (
            f"| 报告生成 | {report.get('requests', '未测试')} | "
            f"均值 {display(report.get('mean_latency_s'), ' s')}，"
            f"P95 {display(report.get('p95_latency_s'), ' s')} |"
        ),
        (
            f"| 权限控制 | {permissions.get('attempted', '未测试')} | "
            f"拦截率 "
            f"{display(permissions.get('blocked_rate_percent'), '%')} |"
        ),
        "",
        "## 表4 多方言识别与用户体验候选数据",
        "",
        "| 项目 | 样本量 | 安静环境 | 噪声环境/结果 |",
        "| --- | ---: | --- | --- |",
    ])
    for label, key in (
        ("普通话CER", "mandarin_asr"),
        ("粤语CER", "cantonese_asr"),
        ("潮汕话CER", "chaoshan_asr"),
    ):
        item = summary[key]
        quiet = item.get("by_environment", {}).get("quiet", {})
        noise = item.get("by_environment", {}).get("noise", {})
        lines.append(
            f"| {label} | {item.get('samples', '未测试')} | "
            f"{display(quiet.get('mean_cer_percent'), '%')} | "
            f"{display(noise.get('mean_cer_percent'), '%')} |"
        )
    experience = summary["user_experience"]
    lines.extend([
        (
            f"| 任务完成率 | {experience.get('participants', '未测试')} | — | "
            f"{display(experience.get('completion_rate_percent'), '%')} |"
        ),
        (
            f"| 平均完成时间 | {experience.get('participants', '未测试')} | — | "
            f"{display(experience.get('mean_completion_minutes'), ' min')} |"
        ),
        (
            f"| 满意度 | {experience.get('participants', '未测试')} | — | "
            f"{display(experience.get('mean_satisfaction'), '/5')} |"
        ),
        "",
        "> “未测试”表示缺少对应原始结果文件，脚本不会自动补造数据。",
    ])
    return "\n".join(lines) + "\n"


def main():
    summary = {
        "functional": summarize_functional(),
        "page_first_load": summarize_pages(),
        "mandarin_asr": summarize_asr("asr_mandarin_results.csv"),
        "cantonese_asr": summarize_asr("asr_cantonese_results.csv"),
        "chaoshan_asr": summarize_asr("asr_chaoshan_results.csv"),
        "report_generation": summarize_report(),
        "permission_control": summarize_permissions(),
        "user_experience": summarize_experience(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "table3_summary.json"
    markdown_path = RESULTS_DIR / "table3_summary.md"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(build_markdown(summary), encoding="utf-8")
    print(f"已生成：{json_path}")
    print(f"已生成：{markdown_path}")


if __name__ == "__main__":
    main()

