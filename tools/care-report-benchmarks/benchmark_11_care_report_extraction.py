"""Run the frozen care-report cases against the real configured DeepSeek API.

The cases are synthetic and frozen before any model call.  This is a system
behaviour evaluation, not a claim of clinical accuracy or of knowing a real
elder's private needs.  It stores only case IDs and structured outcomes; no
chat transcript or model wording is written to the result files.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from care_report_engine import generate_care_report  # noqa: E402


DEFAULT_CASE_FILE = Path(__file__).with_name("care_report_frozen_cases.json")
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument(
        "--case-ids",
        help="Optional comma-separated frozen case IDs, used to split a long live run.",
    )
    return parser.parse_args()


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_cases(case_file: Path) -> list[dict]:
    with case_file.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("冻结案例文件中没有可运行的 cases")
    return cases


def report_signature(report: dict | None) -> str:
    if not report:
        return "invalid_or_rejected"
    areas = sorted({item["area"] for item in report["signals"]})
    return f"{report['status']}|{','.join(areas)}"


def evaluate_case(case: dict, report: dict | None) -> dict:
    expected_status = case["expected_status"]
    expected_areas = sorted(case["expected_areas"])
    if not report:
        return {
            "valid_report": 0,
            "status_pass": 0,
            "area_pass": 0,
            "overall_pass": 0,
            "actual_status": "invalid_or_rejected",
            "actual_areas": "",
        }
    actual_areas = sorted({item["area"] for item in report["signals"]})
    status_pass = report["status"] == expected_status
    area_pass = actual_areas == expected_areas
    return {
        "valid_report": 1,
        "status_pass": int(status_pass),
        "area_pass": int(area_pass),
        "overall_pass": int(status_pass and area_pass),
        "actual_status": report["status"],
        "actual_areas": ",".join(actual_areas),
    }


def percentage(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 1) if denominator else 0.0


def write_results(rows: list[dict], summary: dict) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = RESULTS_DIR / f"care_report_extraction_{stamp}.jsonl"
    summary_path = RESULTS_DIR / f"care_report_extraction_{stamp}_summary.json"
    with details_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return details_path, summary_path


def make_summary(rows: list[dict], case_count: int, runs: int, model: str) -> dict:
    total = len(rows)
    positives = [row for row in rows if row["expected_status"] == "signals_present"]
    negatives = [row for row in rows if row["expected_status"] != "signals_present"]
    by_case: dict[str, list[str]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row["signature"])
    stable_cases = sum(
        len(values) == runs
        and len(set(values)) == 1
        and values[0] != "invalid_or_rejected"
        for values in by_case.values()
    )
    errors = Counter(row["error_type"] for row in rows if row["error_type"])
    return {
        "timestamp": utc_timestamp(),
        "model": model,
        "case_count": case_count,
        "runs_per_case": runs,
        "total_api_calls": total,
        "valid_structured_report_rate_percent": percentage(
            sum(row["valid_report"] for row in rows), total
        ),
        "expected_status_pass_rate_percent": percentage(
            sum(row["status_pass"] for row in rows), total
        ),
        "expected_area_pass_rate_percent": percentage(
            sum(row["area_pass"] for row in rows), total
        ),
        "overall_behaviour_pass_rate_percent": percentage(
            sum(row["overall_pass"] for row in rows), total
        ),
        "positive_case_pass_rate_percent": percentage(
            sum(row["overall_pass"] for row in positives), len(positives)
        ),
        "negative_case_correct_refrain_rate_percent": percentage(
            sum(row["overall_pass"] for row in negatives), len(negatives)
        ),
        "case_level_stability_rate_percent": percentage(stable_cases, case_count),
        "api_error_counts": dict(errors),
        "scope": "合成冻结案例上的系统行为验证；不代表真实老人需求准确率或任何医疗结论。",
        "privacy_note": "结果文件只保存案例编号和结构化通过情况，不保存聊天原文或模型报告原文。",
    }


def main(args):
    if args.runs < 1:
        raise ValueError("--runs 至少为 1")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY；请通过环境变量或项目启动脚本提供密钥。")

    cases = load_cases(args.case_file)
    if args.case_ids:
        requested_ids = [item.strip() for item in args.case_ids.split(",") if item.strip()]
        requested = set(requested_ids)
        cases = [case for case in cases if case["id"] in requested]
        missing = requested - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"未找到冻结案例：{', '.join(sorted(missing))}")
        if not cases:
            raise ValueError("--case-ids 没有选中任何冻结案例")
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=90.0)
    rows = []
    for case_index, case in enumerate(cases, start=1):
        print(f"[{case_index}/{len(cases)}] {case['id']}")
        for run in range(1, args.runs + 1):
            started = perf_counter()
            error_type = ""
            try:
                report = generate_care_report(client, args.model, case["history"])
            except Exception as exc:  # Keep the remaining frozen cases runnable.
                report = None
                error_type = type(exc).__name__
            result = evaluate_case(case, report)
            rows.append(
                {
                    "timestamp": utc_timestamp(),
                    "case_id": case["id"],
                    "run": run,
                    "expected_status": case["expected_status"],
                    "expected_areas": ",".join(sorted(case["expected_areas"])),
                    **result,
                    "signature": report_signature(report),
                    "latency_s": round(perf_counter() - started, 3),
                    "error_type": error_type,
                }
            )

    summary = make_summary(rows, len(cases), args.runs, args.model)
    details_path, summary_path = write_results(rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"详细结果：{details_path}")
    print(f"汇总结果：{summary_path}")


if __name__ == "__main__":
    main(parse_args())
