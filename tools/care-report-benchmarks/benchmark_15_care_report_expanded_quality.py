"""Run the expanded task-aligned frozen suite against the configured API.

This runner combines two static JSON files prepared before model calls.  It is
an extension of benchmark_13, not training data and not a clinical study.
Only IDs, labels and timings are persisted in output files.
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

from benchmark_13_care_report_final_quality import (  # noqa: E402
    evaluate_case,
    make_summary,
    signal_signature,
)
from care_report_engine import generate_care_report  # noqa: E402


BASE_CASE_FILE = Path(__file__).with_name("care_report_final_extended_cases.json")
ADDITIONAL_CASE_FILE = Path(__file__).with_name("care_report_expanded_additional_cases.json")
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    )
    parser.add_argument(
        "--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_case_file(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No cases in {path}")
    return cases


def load_cases() -> list[dict]:
    cases = load_case_file(BASE_CASE_FILE) + load_case_file(ADDITIONAL_CASE_FILE)
    ids = [case.get("id") for case in cases]
    if len(cases) != 48 or len(set(ids)) != len(ids):
        raise ValueError("Expanded suite must contain 48 unique frozen cases.")
    required = {"id", "group", "expected_status", "expected_areas", "expected_action_types", "history"}
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ValueError(f"{case.get('id', '<unknown>')} missing {sorted(missing)}")
    return cases


def write_results(rows: list[dict], summary: dict) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = RESULTS_DIR / f"care_report_expanded_quality_{stamp}.jsonl"
    summary_path = RESULTS_DIR / f"care_report_expanded_quality_{stamp}_summary.json"
    with detail_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return detail_path, summary_path


def main(args):
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
    cases = load_cases()
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=90.0)
    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}")
        for run in range(1, args.runs + 1):
            started = perf_counter()
            error_type = ""
            try:
                report = generate_care_report(client, args.model, case["history"])
            except Exception as exc:
                report = None
                error_type = type(exc).__name__
            result = evaluate_case(case, report)
            rows.append(
                {
                    "timestamp": utc_timestamp(),
                    "case_id": case["id"],
                    "case_group": case["group"],
                    "run": run,
                    "expected_status": case["expected_status"],
                    "expected_areas": sorted(case["expected_areas"]),
                    "expected_action_types": sorted(case["expected_action_types"]),
                    **result,
                    "signature": signal_signature(report),
                    "latency_s": round(perf_counter() - started, 3),
                    "error_type": error_type,
                }
            )
    summary = make_summary(rows, cases, args.runs, args.model)
    summary["timestamp"] = utc_timestamp()
    summary["evaluation_name"] = "家庭关怀报告扩展任务对齐冻结集验证"
    summary["case_count"] = len(cases)
    summary["case_groups"] = dict(sorted(Counter(case["group"] for case in cases).items()))
    summary["source_case_files"] = [
        BASE_CASE_FILE.name,
        ADDITIONAL_CASE_FILE.name,
    ]
    summary["scope"] = (
        "48 个中文冻结案例上的系统行为验证，覆盖心理陪伴、日常协助、安全提示、"
        "克制、改写、干扰和后续解决；不代表真实老人群体准确率或医疗能力。"
    )
    details, summary_path = write_results(rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"DETAILS={details}")
    print(f"SUMMARY={summary_path}")


if __name__ == "__main__":
    main(parse_args())
