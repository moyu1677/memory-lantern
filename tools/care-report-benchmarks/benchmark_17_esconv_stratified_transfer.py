"""Run a stratified ESConv transfer sensitivity audit.

The selected public conversations have a negative-emotion label, a support
problem label and final seeker emotional intensity >= 3.  They are suitable for
testing psychological-signal sensitivity across several problem types, but are
not elderly-family-care labels and are never combined with primary F1 metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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


DEFAULT_SOURCE = Path(r"D:\CodexHome\memories\tmp\care-report-external-20260811\ESConv.json")
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
STRATA = {
    "ongoing_depression": ("ongoing depression", {"depression", "sadness"}),
    "job_crisis": ("job crisis", {"anxiety", "nervousness"}),
    "breakup": ("breakup with partner", {"sadness", "anger"}),
    "friendship_problem": ("problems with friends", {"sadness", "anger"}),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--per-stratum", type=int, default=8)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument(
        "--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    )
    parser.add_argument(
        "--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def final_intensity(record: dict) -> int:
    value = record.get("survey_score", {}).get("seeker", {}).get("final_emotion_intensity")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def history_from_record(record: dict) -> list[dict]:
    return [
        {
            "role": "user" if turn.get("speaker") == "seeker" else "assistant",
            "content": str(turn.get("content") or "").strip(),
        }
        for turn in record.get("dialog", [])
        if str(turn.get("content") or "").strip()
    ]


def load_cases(source: Path, per_stratum: int) -> list[dict]:
    with source.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    cases = []
    for stratum, (problem_type, emotions) in STRATA.items():
        selected = [
            record
            for record in records
            if record.get("problem_type") == problem_type
            and record.get("emotion_type") in emotions
            and final_intensity(record) >= 3
        ][:per_stratum]
        if len(selected) != per_stratum:
            raise ValueError(f"Not enough qualifying ESConv rows for {stratum}.")
        for index, record in enumerate(selected, start=1):
            history = history_from_record(record)
            if sum(turn["role"] == "user" for turn in history) < 2:
                raise ValueError(f"ESConv {stratum}:{index} has too few seeker turns.")
            cases.append({"id": f"{stratum}_{index:02d}", "stratum": stratum, "history": history})
    return cases


def main(args):
    if args.runs < 1 or args.per_stratum < 1:
        raise ValueError("--runs and --per-stratum must be positive")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
    cases = load_cases(args.source, args.per_stratum)
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
            areas = sorted({signal["area"] for signal in report["signals"]}) if report else []
            rows.append(
                {
                    "timestamp": utc_timestamp(),
                    "case_id": case["id"],
                    "stratum": case["stratum"],
                    "run": run,
                    "valid_report": int(report is not None),
                    "psychological_signal": int(
                        report is not None
                        and report["status"] == "signals_present"
                        and "psychological" in areas
                    ),
                    "actual_status": report["status"] if report else "invalid_or_rejected",
                    "actual_areas": areas,
                    "latency_s": round(perf_counter() - started, 3),
                    "error_type": error_type,
                }
            )
    total = len(rows)
    by_stratum = defaultdict(list)
    for row in rows:
        by_stratum[row["stratum"]].append(row)
    stratum_rates = {
        stratum: round(100 * sum(row["psychological_signal"] for row in values) / len(values), 1)
        for stratum, values in sorted(by_stratum.items())
    }
    summary = {
        "timestamp": utc_timestamp(),
        "evaluation_name": "ESConv 分层心理关怀迁移检查",
        "model": args.model,
        "source": "ESConv public corpus; 4 negative-emotion support-problem strata, final seeker emotional intensity >= 3",
        "case_count": len(cases),
        "runs_per_case": args.runs,
        "total_api_calls": total,
        "strata": {key: value[0] for key, value in STRATA.items()},
        "valid_structured_report_rate_percent": round(100 * sum(row["valid_report"] for row in rows) / total, 1),
        "psychological_signal_rate_percent": round(100 * sum(row["psychological_signal"] for row in rows) / total, 1),
        "psychological_signal_rate_by_stratum_percent": stratum_rates,
        "mean_latency_seconds": round(sum(row["latency_s"] for row in rows) / total, 3),
        "api_errors": sum(bool(row["error_type"]) for row in rows),
        "interpretation_boundary": "ESConv 的标签是情感支持问题和情绪，不是老人、家人行动或真实潜在需求真值。该结果只表示在多个未缓解的情感支持场景中的心理线索敏感性，不计入主集 F1。",
        "privacy_note": "不保存公开对话文字、situation 或模型原文。",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = RESULTS_DIR / f"care_report_esconv_stratified_{stamp}.jsonl"
    summary_path = RESULTS_DIR / f"care_report_esconv_stratified_{stamp}_summary.json"
    with detail_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"DETAILS={detail_path}")
    print(f"SUMMARY={summary_path}")


if __name__ == "__main__":
    main(parse_args())
