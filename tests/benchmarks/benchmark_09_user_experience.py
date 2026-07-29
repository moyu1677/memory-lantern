"""表4：匿名记录任务完成率、完成时间、协助程度与5级满意度。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchmark_common import RESULTS_DIR, utc_timestamp


FIELDS = (
    "timestamp",
    "participant_id",
    "completed",
    "assistance",
    "completion_minutes",
    "satisfaction",
    "notes",
)


def ask_choice(prompt, choices):
    while True:
        value = input(prompt).strip().lower()
        if value in choices:
            return value
        print(f"请输入：{', '.join(choices)}")


def ask_number(prompt, minimum, maximum):
    while True:
        try:
            value = float(input(prompt).strip())
        except ValueError:
            print("请输入数字。")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"请输入 {minimum} 至 {maximum} 之间的数值。")


def main(output):
    print("请只填写匿名编号，不要输入姓名、电话或地址。")
    participant_id = input("匿名参与者编号：").strip()
    if not participant_id:
        raise RuntimeError("匿名参与者编号不能为空。")

    completed = ask_choice("是否完成全部规定任务？(yes/no)：", {"yes", "no"})
    assistance = ask_choice(
        "协助程度 (none/some/full)：",
        {"none", "some", "full"},
    )
    completion_minutes = ask_number("完成时间（分钟）：", 0, 180)
    satisfaction = int(ask_number("满意度（1至5分）：", 1, 5))
    notes = input("匿名观察备注（可留空）：").strip()

    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": utc_timestamp(),
            "participant_id": participant_id,
            "completed": int(completed == "yes"),
            "assistance": assistance,
            "completion_minutes": completion_minutes,
            "satisfaction": satisfaction,
            "notes": notes,
        })
    print(f"记录完成：{output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "user_experience_results.csv",
    )
    main(parser.parse_args().output)

