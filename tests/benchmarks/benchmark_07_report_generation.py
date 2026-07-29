"""表3：通过真实HTTP请求测量专属报告端到端响应时间。"""

from __future__ import annotations

import argparse
from time import perf_counter

from benchmark_common import login_session, utc_timestamp, write_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--assessment-id", type=int, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main(args):
    session, _ = login_session(
        args.base_url,
        args.username,
        args.password,
    )
    url = (
        f"{args.base_url.rstrip('/')}/history/"
        f"{args.assessment_id}/report"
    )
    rows = []

    for run in range(1, args.runs + args.warmups + 1):
        started = perf_counter()
        response = session.get(url, timeout=args.timeout)
        elapsed = perf_counter() - started
        if response.status_code != 200:
            raise RuntimeError(
                f"报告请求失败：HTTP {response.status_code}"
            )
        if run > args.warmups:
            rows.append({
                "timestamp": utc_timestamp(),
                "run": run - args.warmups,
                "assessment_id": args.assessment_id,
                "http_status": response.status_code,
                "latency_s": round(elapsed, 6),
                "response_bytes": len(response.content),
            })

    output = write_csv(
        "report_generation_results.csv",
        rows,
        [
            "timestamp",
            "run",
            "assessment_id",
            "http_status",
            "latency_s",
            "response_bytes",
        ],
    )
    print(f"完成：{len(rows)}次报告请求，结果写入 {output}")


if __name__ == "__main__":
    main(parse_args())

