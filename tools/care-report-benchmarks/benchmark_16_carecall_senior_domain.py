"""Check the report pipeline on the elderly-oriented CareCall corpus.

CareCall's public translated sample contains senior-user/system dialogues and is
elderly-domain aligned, but it has no family-care-need gold labels.  This is
therefore a domain-coverage and safety/structure audit, never an accuracy
benchmark.  Output deliberately stores no dialogue text.
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


DEFAULT_SOURCE = Path(
    r"D:\CodexHome\memories\tmp\care-report-external-20260811\carecall_translated_samples.json"
)
RESULTS_DIR = PROJECT_ROOT / "tests" / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--per-slice", type=int, default=30)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--model", default=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    )
    parser.add_argument(
        "--base-url", default=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    return parser.parse_args()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def history_from_record(record: dict) -> list[dict]:
    history = []
    for turn in record.get("data", []):
        text = str(turn.get("utterance") or "").strip()
        if text:
            history.append(
                {"role": "user" if turn.get("role") == "user" else "assistant", "content": text}
            )
    return history


def load_cases(source: Path, per_slice: int) -> list[dict]:
    with source.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    slices = {"validated": [], "fixed": []}
    for record in records:
        guid = str(record.get("guid") or "")
        prefix = guid.split("-", 1)[0]
        history = history_from_record(record)
        if prefix in slices and sum(turn["role"] == "user" for turn in history) >= 2:
            slices[prefix].append({"id": guid, "slice": prefix, "history": history})
    cases = []
    for name in ("validated", "fixed"):
        if len(slices[name]) < per_slice:
            raise ValueError(f"CareCall {name} has fewer than {per_slice} usable dialogues.")
        cases.extend(slices[name][:per_slice])
    return cases


def main(args):
    if args.runs < 1 or args.per_slice < 1:
        raise ValueError("--runs and --per-slice must be positive")
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
    cases = load_cases(args.source, args.per_slice)
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=90.0)
    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['slice']}:{case['id']}")
        for run in range(1, args.runs + 1):
            started = perf_counter()
            error_type = ""
            try:
                report = generate_care_report(client, args.model, case["history"])
            except Exception as exc:
                report = None
                error_type = type(exc).__name__
            signals = report["signals"] if report else []
            rows.append(
                {
                    "timestamp": utc_timestamp(),
                    "case_id": case["id"],
                    "corpus_slice": case["slice"],
                    "run": run,
                    "valid_report": int(report is not None),
                    "actual_status": report["status"] if report else "invalid_or_rejected",
                    "actual_areas": sorted({signal["area"] for signal in signals}),
                    "actual_action_types": sorted({signal["action_type"] for signal in signals}),
                    "latency_s": round(perf_counter() - started, 3),
                    "error_type": error_type,
                }
            )
    total = len(rows)
    area_counts = Counter(area for row in rows for area in row["actual_areas"])
    status_counts = Counter(row["actual_status"] for row in rows)
    summary = {
        "timestamp": utc_timestamp(),
        "evaluation_name": "CareCall 老年对话领域覆盖检查",
        "model": args.model,
        "source": "CareCall translated samples; senior-user/system dialogues, LLM-generated with human support",
        "case_count": len(cases),
        "runs_per_case": args.runs,
        "total_api_calls": total,
        "slice_counts": dict(Counter(case["slice"] for case in cases)),
        "valid_structured_report_rate_percent": round(100 * sum(row["valid_report"] for row in rows) / total, 1),
        "status_counts": dict(status_counts),
        "area_presence_counts": dict(area_counts),
        "mean_latency_seconds": round(sum(row["latency_s"] for row in rows) / total, 3),
        "api_errors": sum(bool(row["error_type"]) for row in rows),
        "interpretation_boundary": "CareCall 与老年聊天场景相近，但发布样本没有家庭关怀需求真值，且语料由 LLM 生成并有人类支持；本检查只验证老年领域输入上的可运行性、输出结构和保守分布，不计入准确率。",
        "privacy_note": "结果只保存公开语料的 guid、切片、结构标签、耗时和错误类型，不保存对话或模型原文。",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = RESULTS_DIR / f"care_report_carecall_domain_{stamp}.jsonl"
    summary_path = RESULTS_DIR / f"care_report_carecall_domain_{stamp}_summary.json"
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
