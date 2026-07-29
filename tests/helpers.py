"""测试用例共用的小型辅助函数。"""

from __future__ import annotations

from datetime import datetime


WEEKDAYS = ("一", "二", "三", "四", "五", "六", "日")


def api_headers(token: str):
    return {"X-CSRF-Token": token}


def correct_answer(question, now=None):
    """根据服务端题目结构生成一个可接受的满分答案。"""
    now = now or datetime.now()
    answer_type = question["type"]

    if answer_type == "keywords":
        return "、".join(aliases[0] for aliases in question["concepts"])
    if answer_type == "year":
        return str(now.year)
    if answer_type == "month":
        return f"{now.month}月"
    if answer_type == "day":
        return f"{now.day}号"
    if answer_type == "weekday":
        return f"星期{WEEKDAYS[now.weekday()]}"
    return question["answers"][0]


def create_assessment(app_module, user_id, *, score=14, percentage=100):
    """直接创建一条报告和权限测试所需的最小测试记录。"""
    domain_scores = {
        "即时记忆": 3 if score else 0,
        "延迟记忆": 3 if score else 0,
        "时间定向": 3 if score else 0,
        "注意力": 2 if score else 0,
        "计算能力": 1 if score else 0,
        "语言能力": 1 if score else 0,
        "日常判断": 1 if score else 0,
    }
    with app_module.app.app_context():
        return app_module.database.save_assessment(
            test_session_id=f"test-session-{user_id}-{score}",
            user_id=user_id,
            score=score,
            maximum_score=14,
            percentage=percentage,
            level=app_module.result_level(percentage),
            domain_scores=domain_scores,
            input_language="browser",
            question_bank_version=app_module.QUESTION_BANK_VERSION,
            question_ids=[f"question-{index}" for index in range(10)],
        )

