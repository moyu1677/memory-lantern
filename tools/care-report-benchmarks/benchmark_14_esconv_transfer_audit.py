"""Run a bounded transfer audit on public ESConv conversations.

ESConv is an emotional-support corpus, not an elderly-family-care ground truth.
This script therefore reports only whether explicit, public emotional-support
conversations are conservatively surfaced as psychological signals.  It does
not contribute to the primary precision/recall claim and does not save any
external dialogue text.
"""

from __future__ import annotations

import argparse
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


DEFAULT_SOURCE = Path(
    r"D:\CodexHome\memories\tmp\care-report-external-20260811\ESConv.json"
)
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sample-size", type=int, default=12)
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


def load_sample(source: Path, sample_size: int) -> list[dict]:
    with source.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    selected = [
        record
        for record in records
        if record.get("problem_type") == "ongoing depression"
        and record.get("emotion_type") in {"depression", "sadness"}
    ][:sample_size]
    if len(selected) != sample_size:
        raise ValueError("ESConv does not contain the requested bounded subset.")
    cases = []
    for index, record in enumerate(selected, start=1):
        history = [
            {"role": "user", "content": turn.get("content", "").strip()}
            for turn in record.get("dialog", [])
            if turn.get("speaker") == "seeker" and turn.get("content", "").strip()
        ]
        if len(history) < 2:
            raise ValueError(f"Selected ESConv record {index} has too few seeker turns.")
        cases.append({"id": f"esconv_ongoing_{index:02d}", "history": history})
    return cases


def main(args):
    if args.runs < 1 or args.sample_size < 1:
        raise ValueError("--runs and --sample-size must be positive")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
    cases = load_sample(args.source, args.sample_size)
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
            actual_status = report["status"] if report else "invalid_or_rejected"
            areas = sorted({item["area"] for item in report["signals"]}) if report else []
            rows.append(
                {
                    "timestamp": utc_timestamp(),
                    "case_id": case["id"],
                    "run": run,
                    "valid_report": int(report is not None),
                    "psychological_signal": int(
                        actual_status == "signals_present" and "psychological" in areas
                    ),
                    "actual_status": actual_status,
                    "actual_areas": areas,
                    "latency_s": round(perf_counter() - started, 3),
                    "error_type": error_type,
                }
            )
    total = len(rows)
    summary = {
        "timestamp": utc_timestamp(),
        "evaluation_name": "ESConv 公开情感支持对话迁移检查",
        "model": args.model,
        "source": "ESConv public corpus; filtered problem_type=ongoing depression and emotion_type in {depression, sadness}",
        "case_count": len(cases),
        "runs_per_case": args.runs,
        "total_api_calls": total,
        "valid_structured_report_rate_percent": round(100 * sum(row["valid_report"] for row in rows) / total, 1),
        "psychological_signal_rate_percent": round(100 * sum(row["psychological_signal"] for row in rows) / total, 1),
        "mean_latency_seconds": round(sum(row["latency_s"] for row in rows) / total, 3),
        "api_errors": sum(bool(row["error_type"]) for row in rows),
        "interpretation_boundary": "ESConv 标签描述情感支持语境，不是老人真实需求、家人行动或医疗真值；本结果仅是公开真实对话上的迁移检查，不纳入主评测精确率。",
        "privacy_note": "不保存外部对话文本、原始 situation 或模型原文。",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = RESULTS_DIR / f"care_report_esconv_transfer_{stamp}.jsonl"
    summary_path = RESULTS_DIR / f"care_report_esconv_transfer_{stamp}_summary.json"
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
