"""Run the final frozen care-report evaluation against the configured API.

The primary set has known, pre-written expectations and is therefore used for
the reported precision/recall, abstention, action-match and robustness
metrics.  Results store only IDs, labels and timings; neither prompts nor
model wording are written to disk.
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

from care_report_engine import generate_care_report, normalize_care_report  # noqa: E402


DEFAULT_CASE_FILE = Path(__file__).with_name("care_report_final_extended_cases.json")
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
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_cases(case_file: Path) -> list[dict]:
    with case_file.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 24:
        raise ValueError("Final case file must contain exactly 24 frozen cases.")
    required = {"id", "expected_status", "expected_areas", "expected_action_types", "history"}
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ValueError(f"{case.get('id', '<unknown>')} is missing {sorted(missing)}")
    return cases


def signal_signature(report: dict | None) -> str:
    if not report:
        return "invalid_or_rejected"
    pairs = sorted(
        f"{item['area']}:{item.get('action_type', '')}" for item in report["signals"]
    )
    return f"{report['status']}|{','.join(pairs)}"


def safe_values(report: dict | None, field: str) -> list[str]:
    if not report:
        return []
    return sorted({item[field] for item in report["signals"]})


def evaluate_case(case: dict, report: dict | None) -> dict:
    expected_areas = sorted(case["expected_areas"])
    expected_actions = sorted(case["expected_action_types"])
    if not report:
        return {
            "valid_report": 0,
            "status_pass": 0,
            "area_exact_pass": 0,
            "action_exact_pass": 0,
            "overall_pass": 0,
            "actual_status": "invalid_or_rejected",
            "actual_areas": [],
            "actual_action_types": [],
        }
    actual_areas = safe_values(report, "area")
    actual_actions = safe_values(report, "action_type")
    status_pass = report["status"] == case["expected_status"]
    area_exact_pass = actual_areas == expected_areas
    action_exact_pass = actual_actions == expected_actions
    return {
        "valid_report": 1,
        "status_pass": int(status_pass),
        "area_exact_pass": int(area_exact_pass),
        "action_exact_pass": int(action_exact_pass),
        "overall_pass": int(status_pass and area_exact_pass and action_exact_pass),
        "actual_status": report["status"],
        "actual_areas": actual_areas,
        "actual_action_types": actual_actions,
    }


def percentage(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 1) if denominator else 0.0


def category_counts(rows: list[dict]) -> dict:
    true_positive = false_positive = false_negative = 0
    for row in rows:
        expected = set(row["expected_areas"])
        actual = set(row["actual_areas"])
        true_positive += len(expected & actual)
        false_positive += len(actual - expected)
        false_negative += len(expected - actual)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision_percent": round(100 * precision, 1),
        "recall_percent": round(100 * recall, 1),
        "f1_percent": round(100 * f1, 1),
    }


def guardrail_checks() -> dict:
    """Exercise the deterministic output gate separately from model quality."""

    source = ["this is a long source sentence that must never be copied verbatim"]
    base = {
        "status": "signals_present",
        "summary": "consider a gentle contact",
        "signals": [
            {
                "area": "psychological",
                "action_type": "contact_companionship",
                "topic": "contact and companionship",
                "strength": "low",
                "trend": "new",
                "family_action": "make one gentle call this week",
                "source_turn_ids": [1],
            }
        ],
    }
    cases = []
    invalid_status = json.loads(json.dumps(base))
    invalid_status["status"] = "needs_immediate_intervention"
    cases.append(normalize_care_report(invalid_status, {1}, source) is None)

    diagnosis = json.loads(json.dumps(base))
    diagnosis["summary"] = "\u5df2\u7ecf\u786e\u8bca\u67d0\u79cd\u75be\u75c5\u3002"
    cases.append(normalize_care_report(diagnosis, {1}, source) is None)

    diagnosis_in_action = json.loads(json.dumps(base))
    diagnosis_in_action["signals"][0]["family_action"] = "\u8bf7\u5bb6\u4eba\u8bca\u65ad\u4e00\u4e0b\u3002"
    normalized = normalize_care_report(diagnosis_in_action, {1}, source)
    cases.append(
        bool(
            normalized
            and normalized["status"] == "insufficient_or_conflicting"
            and not normalized["signals"]
        )
    )

    copied_summary = json.loads(json.dumps(base))
    copied_summary["summary"] = source[0]
    cases.append(normalize_care_report(copied_summary, {1}, source) is None)

    copied = json.loads(json.dumps(base))
    copied["signals"][0]["topic"] = source[0]
    normalized = normalize_care_report(copied, {1}, source)
    cases.append(
        bool(
            normalized
            and normalized["status"] == "insufficient_or_conflicting"
            and not normalized["signals"]
        )
    )

    bad_source = json.loads(json.dumps(base))
    bad_source["signals"][0]["source_turn_ids"] = [99]
    normalized = normalize_care_report(bad_source, {1}, source)
    cases.append(bool(normalized and normalized["status"] == "insufficient_or_conflicting" and not normalized["signals"]))

    bad_area_action = json.loads(json.dumps(base))
    bad_area_action["signals"][0]["action_type"] = "safety_check"
    normalized = normalize_care_report(bad_area_action, {1}, source)
    cases.append(bool(normalized and normalized["status"] == "insufficient_or_conflicting" and not normalized["signals"]))

    missing_action = json.loads(json.dumps(base))
    del missing_action["signals"][0]["action_type"]
    normalized = normalize_care_report(missing_action, {1}, source)
    cases.append(bool(normalized and normalized["status"] == "insufficient_or_conflicting" and not normalized["signals"]))

    too_many = json.loads(json.dumps(base))
    too_many["signals"] = too_many["signals"] * 4
    normalized = normalize_care_report(too_many, {1}, source)
    cases.append(bool(normalized and len(normalized["signals"]) == 3))

    forced_signal_on_abstain = json.loads(json.dumps(base))
    forced_signal_on_abstain["status"] = "insufficient_or_conflicting"
    normalized = normalize_care_report(forced_signal_on_abstain, {1}, source)
    cases.append(bool(normalized and not normalized["signals"]))
    return {"passed": sum(cases), "total": len(cases)}


def make_summary(rows: list[dict], cases: list[dict], runs: int, model: str) -> dict:
    total = len(rows)
    negative_rows = [row for row in rows if row["expected_status"] == "insufficient_or_conflicting"]
    positive_rows = [row for row in rows if row["expected_status"] == "signals_present"]
    by_case: dict[str, list[str]] = {}
    by_case_run: dict[tuple[str, int], str] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(row["signature"])
        by_case_run[(row["case_id"], row["run"])] = row["signature"]
    stable_cases = sum(
        len(values) == runs and len(set(values)) == 1 and values[0] != "invalid_or_rejected"
        for values in by_case.values()
    )
    pairs = [case for case in cases if case.get("paired_with")]
    pair_passes = 0
    pair_total = len(pairs) * runs
    for case in pairs:
        for run in range(1, runs + 1):
            actual = by_case_run.get((case["id"], run), "invalid_or_rejected")
            if case["expected_relation"] == "same_signature":
                passed = actual == by_case_run.get((case["paired_with"], run))
            else:
                passed = actual == "insufficient_or_conflicting|"
            pair_passes += int(passed)
    errors = Counter(row["error_type"] for row in rows if row["error_type"])
    latencies = [row["latency_s"] for row in rows]
    return {
        "timestamp": utc_timestamp(),
        "evaluation_name": "家庭关怀报告最终冻结集验证",
        "model": model,
        "thinking_mode": "enabled",
        "case_count": len(cases),
        "runs_per_case": runs,
        "total_api_calls": total,
        "case_groups": {
            "clear_signal_cases": 8,
            "abstain_cases": 8,
            "paired_robustness_cases": 8,
        },
        "valid_structured_report_rate_percent": percentage(sum(row["valid_report"] for row in rows), total),
        "category": category_counts(rows),
        "abstention_correct_rate_percent": percentage(
            sum(row["overall_pass"] for row in negative_rows), len(negative_rows)
        ),
        "positive_exact_match_rate_percent": percentage(
            sum(row["overall_pass"] for row in positive_rows), len(positive_rows)
        ),
        "action_type_exact_match_rate_percent": percentage(
            sum(row["action_exact_pass"] for row in rows), total
        ),
        "paired_robustness_pass_rate_percent": percentage(pair_passes, pair_total),
        "case_level_stability_rate_percent": percentage(stable_cases, len(cases)),
        "guardrail_output_gate": guardrail_checks(),
        "mean_latency_seconds": round(sum(latencies) / total, 3) if total else 0.0,
        "max_latency_seconds": round(max(latencies), 3) if latencies else 0.0,
        "api_error_counts": dict(errors),
        "scope": "这是冻结受控案例上的系统行为验证，不代表真实老人需求准确率、临床结论或医疗诊断能力。",
        "privacy_note": "明细仅保存案例 ID、期望标签、结构标签、耗时和错误类型；不保存聊天原文或模型原文。",
    }


def write_results(rows: list[dict], summary: dict) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    details_path = RESULTS_DIR / f"care_report_final_quality_{stamp}.jsonl"
    summary_path = RESULTS_DIR / f"care_report_final_quality_{stamp}_summary.json"
    with details_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return details_path, summary_path


def main(args):
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
    cases = load_cases(args.case_file)
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=90.0)
    rows = []
    for case_index, case in enumerate(cases, start=1):
        print(f"[{case_index}/{len(cases)}] {case['id']}")
        for run in range(1, args.runs + 1):
            started = perf_counter()
            error_type = ""
            try:
                report = generate_care_report(client, args.model, case["history"])
            except Exception as exc:  # Keep later frozen cases runnable.
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
    details_path, summary_path = write_results(rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"DETAILS={details_path}")
    print(f"SUMMARY={summary_path}")


if __name__ == "__main__":
    main(parse_args())
