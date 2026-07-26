"""认知表现测试题库与确定性判分规则。

这是项目原型题库，不是经过临床验证的诊断量表。正式使用前应由
老年医学、神经内科或心理测量专业人员审核题目和结果阈值。
"""

from datetime import datetime
import re


QUESTIONS = [
    {
        "id": "immediate_memory",
        "domain": "即时记忆",
        "prompt": "请记住这三个词：苹果、钥匙、河流。现在请把这三个词复述一遍。",
        "type": "keywords",
        "concepts": [["苹果"], ["钥匙"], ["河流", "小河"]],
        "points": 3,
    },
    {
        "id": "year",
        "domain": "时间定向",
        "prompt": "现在是哪一年？",
        "type": "year",
        "points": 1,
    },
    {
        "id": "month",
        "domain": "时间定向",
        "prompt": "现在是几月？",
        "type": "month",
        "points": 1,
    },
    {
        "id": "weekday",
        "domain": "时间定向",
        "prompt": "今天是星期几？",
        "type": "weekday",
        "points": 1,
    },
    {
        "id": "reverse_digits",
        "domain": "注意力",
        "prompt": "请把数字三、八、一倒过来说。",
        "type": "exact",
        "answers": ["一八三", "183"],
        "points": 1,
    },
    {
        "id": "calculation",
        "domain": "计算能力",
        "prompt": "十减三等于多少？",
        "type": "exact",
        "answers": ["七", "7"],
        "points": 1,
    },
    {
        "id": "sequence",
        "domain": "注意力",
        "prompt": "请听数字：二、四、六、八。下一个数字是什么？",
        "type": "exact",
        "answers": ["十", "10"],
        "points": 1,
    },
    {
        "id": "sentence",
        "domain": "语言能力",
        "prompt": "请完整复述这句话：今天的天气很好。",
        "type": "contains",
        "answers": ["今天的天气很好", "今天天气很好"],
        "points": 1,
    },
    {
        "id": "safety",
        "domain": "日常判断",
        "prompt": "过马路时红灯亮起，应该停下等待，还是继续通行？",
        "type": "contains",
        "answers": ["停下等待", "停下", "停止", "等待", "不能通行"],
        "points": 1,
    },
    {
        "id": "delayed_memory",
        "domain": "延迟记忆",
        "prompt": "还记得第一题让你记住的三个词吗？请尽量说出来。",
        "type": "keywords",
        "concepts": [["苹果"], ["钥匙"], ["河流", "小河"]],
        "points": 3,
    },
]


CHINESE_DIGITS = {
    0: "零",
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
}

WEEKDAY_NAMES = {
    0: ["星期一", "周一", "礼拜一"],
    1: ["星期二", "周二", "礼拜二"],
    2: ["星期三", "周三", "礼拜三"],
    3: ["星期四", "周四", "礼拜四"],
    4: ["星期五", "周五", "礼拜五"],
    5: ["星期六", "周六", "礼拜六"],
    6: ["星期日", "星期天", "周日", "周天", "礼拜日", "礼拜天"],
}


def normalize_answer(value):
    """去掉常见标点和空白，减少语音转文字格式差异造成的误判。"""
    text = str(value or "").strip().lower()
    return re.sub(r"[\s，。！？、,.!?；;：:（）()\"'“”‘’\-]", "", text)


def _year_answers(now):
    year = str(now.year)
    chinese_year = "".join(CHINESE_DIGITS[int(digit)] for digit in year)
    return {year, f"{year}年", chinese_year, f"{chinese_year}年"}


def _month_answers(now):
    month = now.month
    chinese_month = "十" if month == 10 else (
        f"十{CHINESE_DIGITS[month - 10]}" if month > 10 else CHINESE_DIGITS[month]
    )
    return {str(month), f"{month}月", chinese_month, f"{chinese_month}月"}


def score_answer(question, answer, now=None):
    """返回该题实际得分；不使用大模型参与判分。"""
    normalized = normalize_answer(answer)
    now = now or datetime.now()
    answer_type = question["type"]

    if answer_type == "keywords":
        earned = 0
        for aliases in question["concepts"]:
            if any(normalize_answer(alias) in normalized for alias in aliases):
                earned += 1
        return min(earned, question["points"])

    if answer_type == "year":
        accepted = {normalize_answer(item) for item in _year_answers(now)}
        return question["points"] if normalized in accepted else 0

    if answer_type == "month":
        accepted = {normalize_answer(item) for item in _month_answers(now)}
        return question["points"] if normalized in accepted else 0

    if answer_type == "weekday":
        accepted = {
            normalize_answer(item) for item in WEEKDAY_NAMES[now.weekday()]
        }
        return question["points"] if normalized in accepted else 0

    accepted = [normalize_answer(item) for item in question["answers"]]
    if answer_type == "exact":
        correct = normalized in accepted
    else:
        correct = any(item in normalized for item in accepted)
    return question["points"] if correct else 0


def total_points():
    return sum(question["points"] for question in QUESTIONS)


def domain_max_points():
    totals = {}
    for question in QUESTIONS:
        domain = question["domain"]
        totals[domain] = totals.get(domain, 0) + question["points"]
    return totals


def result_level(percentage):
    """原型反馈阈值，不是疾病诊断分界值。"""
    if percentage >= 80:
        return "本次认知任务表现较稳定"
    if percentage >= 60:
        return "本次部分任务存在困难，建议休息后再次测试"
    return "本次多项任务存在困难，建议由家属陪同进行专业评估"

