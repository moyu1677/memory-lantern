"""认知表现测试题库、随机组卷与确定性判分规则。

这是项目原型题库，不是经过临床验证的诊断量表。正式使用前应由
老年医学、神经内科或心理测量专业人员审核题目和结果阈值。
"""

from datetime import datetime
import re
import secrets


MEMORY_SETS = (
    (("苹果",), ("钥匙",), ("河流", "小河")),
    (("香蕉",), ("雨伞",), ("火车",)),
    (("橘子",), ("眼镜",), ("大桥", "桥")),
    (("西瓜",), ("手表",), ("公园",)),
    (("花生",), ("毛巾",), ("学校",)),
    (("鸡蛋",), ("茶杯", "杯子"), ("树林",)),
    (("面包",), ("剪刀",), ("市场", "菜市场")),
    (("葡萄",), ("帽子",), ("海边", "海滩")),
)

TIME_QUESTIONS = (
    {
        "id": "time_year",
        "domain": "时间定向",
        "prompt": "现在是哪一年？",
        "type": "year",
        "points": 1,
    },
    {
        "id": "time_month",
        "domain": "时间定向",
        "prompt": "现在是几月？",
        "type": "month",
        "points": 1,
    },
    {
        "id": "time_weekday",
        "domain": "时间定向",
        "prompt": "今天是星期几？",
        "type": "weekday",
        "points": 1,
    },
    {
        "id": "time_day",
        "domain": "时间定向",
        "prompt": "今天是这个月的几号？",
        "type": "day",
        "points": 1,
    },
)

ATTENTION_QUESTIONS = (
    {
        "id": "attention_reverse_381",
        "domain": "注意力",
        "prompt": "请把数字三、八、一倒过来说。",
        "type": "exact",
        "answers": ("一八三", "183"),
        "points": 1,
    },
    {
        "id": "attention_reverse_527",
        "domain": "注意力",
        "prompt": "请把数字五、二、七倒过来说。",
        "type": "exact",
        "answers": ("七二五", "725"),
        "points": 1,
    },
    {
        "id": "attention_reverse_694",
        "domain": "注意力",
        "prompt": "请把数字六、九、四倒过来说。",
        "type": "exact",
        "answers": ("四九六", "496"),
        "points": 1,
    },
    {
        "id": "attention_reverse_246",
        "domain": "注意力",
        "prompt": "请把数字二、四、六倒过来说。",
        "type": "exact",
        "answers": ("六四二", "642"),
        "points": 1,
    },
    {
        "id": "attention_sequence_even",
        "domain": "注意力",
        "prompt": "请听数字：二、四、六、八。下一个数字是什么？",
        "type": "exact",
        "answers": ("十", "10"),
        "points": 1,
    },
    {
        "id": "attention_sequence_odd",
        "domain": "注意力",
        "prompt": "请听数字：一、三、五、七。下一个数字是什么？",
        "type": "exact",
        "answers": ("九", "9"),
        "points": 1,
    },
    {
        "id": "attention_sequence_fives",
        "domain": "注意力",
        "prompt": "请听数字：五、十、十五。下一个数字是什么？",
        "type": "exact",
        "answers": ("二十", "20"),
        "points": 1,
    },
    {
        "id": "attention_sequence_down",
        "domain": "注意力",
        "prompt": "请听数字：十、九、八、七。下一个数字是什么？",
        "type": "exact",
        "answers": ("六", "6"),
        "points": 1,
    },
    {
        "id": "attention_target_word",
        "domain": "注意力",
        "prompt": "请听三个词：小猫、白云、红花。请只说中间那个词。",
        "type": "exact",
        "answers": ("白云",),
        "points": 1,
    },
    {
        "id": "attention_target_number",
        "domain": "注意力",
        "prompt": "请听三个数字：四、七、二。请只说最后一个数字。",
        "type": "exact",
        "answers": ("二", "2"),
        "points": 1,
    },
)

CALCULATION_QUESTIONS = (
    ("calculation_10_minus_3", "十减三等于多少？", ("七", "7")),
    ("calculation_12_minus_5", "十二减五等于多少？", ("七", "7")),
    ("calculation_15_minus_6", "十五减六等于多少？", ("九", "9")),
    ("calculation_20_minus_8", "二十减八等于多少？", ("十二", "12")),
    ("calculation_8_plus_4", "八加四等于多少？", ("十二", "12")),
    ("calculation_6_plus_7", "六加七等于多少？", ("十三", "13")),
    ("calculation_9_plus_5", "九加五等于多少？", ("十四", "14")),
    ("calculation_7_plus_8", "七加八等于多少？", ("十五", "15")),
    ("calculation_18_minus_9", "十八减九等于多少？", ("九", "9")),
    ("calculation_14_minus_4", "十四减四等于多少？", ("十", "10")),
    ("calculation_5_plus_5", "五加五等于多少？", ("十", "10")),
    ("calculation_16_minus_7", "十六减七等于多少？", ("九", "9")),
)

LANGUAGE_QUESTIONS = (
    ("language_weather", "今天的天气很好。", ("今天的天气很好", "今天天气很好")),
    ("language_walk", "我们一起去散步。", ("我们一起去散步",)),
    ("language_flower", "窗边开着一朵花。", ("窗边开着一朵花",)),
    ("language_breakfast", "早上记得吃早餐。", ("早上记得吃早餐",)),
    ("language_birds", "小鸟在树上唱歌。", ("小鸟在树上唱歌",)),
    ("language_family", "家人今晚回来吃饭。", ("家人今晚回来吃饭",)),
    ("language_water", "桌上放着一杯温水。", ("桌上放着一杯温水",)),
    ("language_sun", "太阳出来以后很暖和。", ("太阳出来以后很暖和",)),
    ("language_market", "我们明天早上去市场。", ("我们明天早上去市场",)),
    ("language_radio", "收音机正在播放音乐。", ("收音机正在播放音乐",)),
)

JUDGMENT_QUESTIONS = (
    {
        "id": "judgment_red_light",
        "prompt": "过马路时红灯亮起，应该停下等待，还是继续通行？",
        "answers": ("停下等待", "停下", "停止", "等待", "不能通行"),
    },
    {
        "id": "judgment_smell_gas",
        "prompt": "在家闻到很重的煤气味，应该继续开火，还是关闭火源并请人帮助？",
        "answers": ("关闭火源", "关火", "请人帮助", "找人帮助", "开窗", "报警"),
    },
    {
        "id": "judgment_unknown_medicine",
        "prompt": "看到一瓶不知道用途的药，应该随便服用，还是先询问医生或家属？",
        "answers": ("询问医生", "问医生", "询问家属", "问家属", "不能随便服用"),
    },
    {
        "id": "judgment_phone_scam",
        "prompt": "陌生电话要求立即转账，应该马上转账，还是先联系家人核实？",
        "answers": ("联系家人", "找家人", "先核实", "核实", "不能转账", "不转账"),
    },
    {
        "id": "judgment_dizzy",
        "prompt": "走路时突然头晕站不稳，应该继续独自行走，还是先坐下并呼叫帮助？",
        "answers": ("先坐下", "坐下", "呼叫帮助", "叫人", "找人帮助"),
    },
    {
        "id": "judgment_hot_water",
        "prompt": "准备喝刚烧开的水，应该马上大口喝，还是先放凉并小心试温？",
        "answers": ("先放凉", "放凉", "试温", "小心试温", "不能马上喝"),
    },
    {
        "id": "judgment_lost",
        "prompt": "外出时找不到回家的路，应该继续乱走，还是留在安全地点联系家人？",
        "answers": ("安全地点", "联系家人", "打电话", "找警察", "求助", "不能乱走"),
    },
    {
        "id": "judgment_wet_floor",
        "prompt": "看到地面有水很滑，应该快速走过去，还是放慢脚步并绕开？",
        "answers": ("放慢脚步", "慢慢走", "绕开", "小心", "不能快走"),
    },
    {
        "id": "judgment_door",
        "prompt": "陌生人敲门说要进屋检查，应该直接开门，还是先核实身份并联系家人？",
        "answers": ("核实身份", "联系家人", "不能开门", "不开门", "先确认"),
    },
    {
        "id": "judgment_power",
        "prompt": "发现电线破损露在外面，应该用手触碰，还是远离并请人处理？",
        "answers": ("远离", "不能触碰", "不能碰", "请人处理", "找人处理"),
    },
)

DOMAIN_MAXIMUMS = {
    "即时记忆": 3,
    "延迟记忆": 3,
    "时间定向": 3,
    "注意力": 2,
    "计算能力": 1,
    "语言能力": 1,
    "日常判断": 1,
}
TEST_QUESTION_COUNT = 10
QUESTION_BANK_VERSION = "random-cognitive-bank-v1"

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
    0: ("星期一", "周一", "礼拜一"),
    1: ("星期二", "周二", "礼拜二"),
    2: ("星期三", "周三", "礼拜三"),
    3: ("星期四", "周四", "礼拜四"),
    4: ("星期五", "周五", "礼拜五"),
    5: ("星期六", "周六", "礼拜六"),
    6: ("星期日", "星期天", "周日", "周天", "礼拜日", "礼拜天"),
}


def _memory_question(memory_set, delayed=False):
    words = [aliases[0] for aliases in memory_set]
    identifier = "_".join(words)
    if delayed:
        return {
            "id": f"delayed_memory_{identifier}",
            "domain": "延迟记忆",
            "prompt": "还记得第一题让您记住的三个词吗？请尽量说出来。",
            "type": "keywords",
            "concepts": memory_set,
            "points": 3,
        }
    return {
        "id": f"immediate_memory_{identifier}",
        "domain": "即时记忆",
        "prompt": (
            f"请记住这三个词：{'、'.join(words)}。"
            "现在请把这三个词复述一遍。"
        ),
        "type": "keywords",
        "concepts": memory_set,
        "points": 3,
    }


def _calculation_question(item):
    identifier, prompt, answers = item
    return {
        "id": identifier,
        "domain": "计算能力",
        "prompt": prompt,
        "type": "exact",
        "answers": answers,
        "points": 1,
    }


def _language_question(item):
    identifier, sentence, answers = item
    return {
        "id": identifier,
        "domain": "语言能力",
        "prompt": f"请完整复述这句话：{sentence}",
        "type": "contains",
        "answers": answers,
        "points": 1,
    }


def _judgment_question(item):
    return {
        "id": item["id"],
        "domain": "日常判断",
        "prompt": item["prompt"],
        "type": "contains",
        "answers": item["answers"],
        "points": 1,
    }


def build_test_questions(randomizer=None):
    """按固定维度配额随机生成一次10题测试。"""
    randomizer = randomizer or secrets.SystemRandom()
    memory_set = randomizer.choice(MEMORY_SETS)
    middle = [
        *randomizer.sample(TIME_QUESTIONS, 3),
        *randomizer.sample(ATTENTION_QUESTIONS, 2),
        _calculation_question(randomizer.choice(CALCULATION_QUESTIONS)),
        _language_question(randomizer.choice(LANGUAGE_QUESTIONS)),
        _judgment_question(randomizer.choice(JUDGMENT_QUESTIONS)),
    ]
    randomizer.shuffle(middle)
    return [
        _memory_question(memory_set),
        *middle,
        _memory_question(memory_set, delayed=True),
    ]


def normalize_answer(value):
    """去掉常见标点和空白，减少语音转文字格式差异造成的误判。"""
    text = str(value or "").strip().lower()
    return re.sub(r"[\s，。！？、,.!?；;：:（）()\"'“”‘’\-]", "", text)


def _chinese_number(number):
    if number < 10:
        return CHINESE_DIGITS[number]
    if number == 10:
        return "十"
    if number < 20:
        return f"十{CHINESE_DIGITS[number - 10]}"
    tens, ones = divmod(number, 10)
    return f"{CHINESE_DIGITS[tens]}十{CHINESE_DIGITS[ones] if ones else ''}"


def _number_answers(number, suffix):
    chinese = _chinese_number(number)
    return {
        str(number),
        f"{number}{suffix}",
        chinese,
        f"{chinese}{suffix}",
    }


def _year_answers(now):
    year = str(now.year)
    chinese_year = "".join(CHINESE_DIGITS[int(digit)] for digit in year)
    return {year, f"{year}年", chinese_year, f"{chinese_year}年"}


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
        accepted = {
            normalize_answer(item)
            for item in _number_answers(now.month, "月")
        }
        return question["points"] if normalized in accepted else 0

    if answer_type == "day":
        accepted = {
            normalize_answer(item)
            for item in _number_answers(now.day, "号")
            | _number_answers(now.day, "日")
        }
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
    return sum(DOMAIN_MAXIMUMS.values())


def domain_max_points():
    return dict(DOMAIN_MAXIMUMS)


def result_level(percentage):
    """原型反馈阈值，不是疾病诊断分界值。"""
    if percentage >= 80:
        return "本次认知任务表现较稳定"
    if percentage >= 60:
        return "本次部分任务存在困难，建议休息后再次测试"
    return "本次多项任务存在困难，建议由家属陪同进行专业评估"
