"""从双人复核后的 SQLite 样本训练绘图评分基线模型。

示例：
    python tools/train_drawing_model.py --task clock
    python tools/train_drawing_model.py --task all --minimum-per-class 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drawing_scoring import (  # noqa: E402
    DrawingScoringError,
    TASKS,
    train_centroid_model,
)


def resolved_score(connection, submission_id):
    rows = connection.execute(
        """
        SELECT review_role, score
        FROM drawing_reviews
        WHERE submission_id = ?
        ORDER BY updated_at
        """,
        (submission_id,),
    ).fetchall()
    adjudicated = [int(row["score"]) for row in rows if row["review_role"] == "adjudicator"]
    if adjudicated:
        return adjudicated[-1], "adjudicated"
    raters = [int(row["score"]) for row in rows if row["review_role"] == "rater"]
    if len(raters) >= 2 and len(set(raters)) == 1:
        return raters[0], "rater_consensus"
    return None, "pending"


def load_samples(database_path, task_type):
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        try:
            submissions = connection.execute(
                """
                SELECT id, submission_id, task_type, features_json
                FROM drawing_submissions
                WHERE task_type = ? AND research_consent = 1
                ORDER BY created_at
                """,
                (task_type,),
            ).fetchall()
        except sqlite3.OperationalError as error:
            if "no such table" in str(error).lower():
                raise SystemExit(
                    "数据库尚未创建绘图表。请先启动一次 Flask 项目完成数据库升级。"
                ) from error
            raise
        samples = []
        for submission in submissions:
            score, label_source = resolved_score(connection, submission["id"])
            if score is None:
                continue
            samples.append(
                {
                    "submission_id": submission["submission_id"],
                    "features": json.loads(submission["features_json"]),
                    "score": score,
                    "label_source": label_source,
                }
            )
        return samples
    finally:
        connection.close()


def parse_args():
    parser = argparse.ArgumentParser(description="训练绘图机器辅助评分基线模型")
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "memory_lantern.db",
        help="SQLite 数据库路径",
    )
    parser.add_argument(
        "--task",
        choices=["all", *TASKS.keys()],
        default="all",
        help="要训练的任务",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "data" / "models",
        help="模型JSON输出目录",
    )
    parser.add_argument(
        "--minimum-per-class",
        type=int,
        default=5,
        help="每个得分类别的最少共识样本数",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.database.exists():
        raise SystemExit(f"数据库不存在：{args.database}")
    if args.minimum_per_class < 2:
        raise SystemExit("每类最少样本数不能小于2。")

    tasks = list(TASKS) if args.task == "all" else [args.task]
    args.output_directory.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    for task_type in tasks:
        samples = load_samples(args.database, task_type)
        try:
            model = train_centroid_model(
                samples,
                task_type,
                minimum_per_class=args.minimum_per_class,
            )
        except DrawingScoringError as error:
            print(f"[跳过] {TASKS[task_type]['label']}：{error}（可用样本{len(samples)}份）")
            continue
        output_path = args.output_directory / f"drawing_{task_type}.json"
        output_path.write_text(
            json.dumps(model, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"[完成] {TASKS[task_type]['label']}：{model['training_sample_count']}份，"
            f"版本 {model['model_version']}，输出 {output_path}"
        )
        succeeded += 1
    if not succeeded:
        raise SystemExit("没有任务满足训练条件；请继续完成双人复核样本。")


if __name__ == "__main__":
    main()
