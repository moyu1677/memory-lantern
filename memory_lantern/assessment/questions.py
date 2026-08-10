"""认知表现测试题库、随机组卷与确定性判分规则。

这是项目原型题库，不是经过临床验证的诊断量表。正式使用前应由
老年医学、神经内科或心理测量专业人员审核题目和结果阈值。
"""

from datetime import datetime
import re
import secrets

from memory_lantern.assessment.judgment_scoring import analyze_judgment_answer


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
TEST_QUESTION_COUNT = 11
QUESTION_BANK_VERSION = "random-cognitive-bank-v3-safety-judgment"

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

ENGLISH_WEEKDAY_NAMES = {
    0: ("monday",),
    1: ("tuesday",),
    2: ("wednesday",),
    3: ("thursday",),
    4: ("friday",),
    5: ("saturday",),
    6: ("sunday",),
}

ENGLISH_DOMAINS = {
    "即时记忆": "Immediate memory",
    "延迟记忆": "Delayed recall",
    "时间定向": "Time orientation",
    "注意力": "Attention",
    "计算能力": "Calculation",
    "语言能力": "Language",
    "日常判断": "Everyday judgement",
    "视空间观察": "Visuospatial observation",
}

ENGLISH_MEMORY_WORDS = {
    "苹果": ("apple",), "钥匙": ("key", "keys"), "河流": ("river",),
    "香蕉": ("banana",), "雨伞": ("umbrella",), "火车": ("train",),
    "橘子": ("orange",), "眼镜": ("glasses", "spectacles"), "大桥": ("bridge",),
    "西瓜": ("watermelon",), "手表": ("watch",), "公园": ("park",),
    "花生": ("peanut", "peanuts"), "毛巾": ("towel",), "学校": ("school",),
    "鸡蛋": ("egg",), "茶杯": ("cup", "teacup"), "树林": ("wood", "woods", "forest"),
    "面包": ("bread",), "剪刀": ("scissors",), "市场": ("market",),
    "葡萄": ("grapes", "grape"), "帽子": ("hat",), "海边": ("seaside", "beach"),
}

ENGLISH_PROMPTS = {
    "time_year": "What year is it now?",
    "time_month": "What month is it now?",
    "time_weekday": "What day of the week is it today?",
    "time_day": "What is today's date?",
    "attention_reverse_381": "Please say the digits three, eight, one in reverse order.",
    "attention_reverse_527": "Please say the digits five, two, seven in reverse order.",
    "attention_reverse_694": "Please say the digits six, nine, four in reverse order.",
    "attention_reverse_246": "Please say the digits two, four, six in reverse order.",
    "attention_sequence_even": "Listen to these numbers: two, four, six, eight. What comes next?",
    "attention_sequence_odd": "Listen to these numbers: one, three, five, seven. What comes next?",
    "attention_sequence_fives": "Listen to these numbers: five, ten, fifteen. What comes next?",
    "attention_sequence_down": "Listen to these numbers: ten, nine, eight, seven. What comes next?",
    "attention_target_word": "Listen to three words: cat, cloud, flower. Say only the middle word.",
    "attention_target_number": "Listen to three numbers: four, seven, two. Say only the last number.",
    "calculation_10_minus_3": "What is ten minus three?",
    "calculation_12_minus_5": "What is twelve minus five?",
    "calculation_15_minus_6": "What is fifteen minus six?",
    "calculation_20_minus_8": "What is twenty minus eight?",
    "calculation_8_plus_4": "What is eight plus four?",
    "calculation_6_plus_7": "What is six plus seven?",
    "calculation_9_plus_5": "What is nine plus five?",
    "calculation_7_plus_8": "What is seven plus eight?",
    "calculation_18_minus_9": "What is eighteen minus nine?",
    "calculation_14_minus_4": "What is fourteen minus four?",
    "calculation_5_plus_5": "What is five plus five?",
    "calculation_16_minus_7": "What is sixteen minus seven?",
    "judgment_red_light": "At a red traffic light, should you stop and wait or keep crossing?",
    "judgment_smell_gas": "If you smell strong gas at home, should you keep using a flame or turn it off and ask for help?",
    "judgment_unknown_medicine": "If you find medicine you do not recognise, should you take it or ask a doctor or family member first?",
    "judgment_phone_scam": "If an unknown caller demands an immediate money transfer, should you transfer it or contact family to check first?",
    "judgment_dizzy": "If you suddenly feel dizzy and unsteady, should you keep walking alone or sit down and call for help?",
    "judgment_hot_water": "Should you drink freshly boiled water immediately or let it cool and test the temperature first?",
    "judgment_lost": "If you cannot find your way home, should you keep wandering or stay somewhere safe and contact family?",
    "judgment_wet_floor": "If the floor is wet and slippery, should you walk quickly or slow down and go around it?",
    "judgment_door": "If a stranger asks to enter your home for an inspection, should you open the door or verify their identity and contact family first?",
    "judgment_power": "If an electrical wire is damaged and exposed, should you touch it or stay away and ask someone to deal with it?",
}

ENGLISH_ANSWERS = {
    "attention_reverse_381": ("183", "one eight three"),
    "attention_reverse_527": ("725", "seven two five"),
    "attention_reverse_694": ("496", "four nine six"),
    "attention_reverse_246": ("642", "six four two"),
    "attention_sequence_even": ("10", "ten"),
    "attention_sequence_odd": ("9", "nine"),
    "attention_sequence_fives": ("20", "twenty"),
    "attention_sequence_down": ("6", "six"),
    "attention_target_word": ("cloud",),
    "attention_target_number": ("2", "two"),
    "judgment_red_light": ("stop", "wait", "stop and wait"),
    "judgment_smell_gas": ("turn off", "turn it off", "ask for help", "open a window", "call emergency services"),
    "judgment_unknown_medicine": ("ask a doctor", "ask family", "do not take it", "don't take it"),
    "judgment_phone_scam": ("contact family", "check first", "verify", "do not transfer", "don't transfer"),
    "judgment_dizzy": ("sit down", "call for help", "ask for help"),
    "judgment_hot_water": ("let it cool", "cool", "test the temperature", "do not drink immediately"),
    "judgment_lost": ("stay somewhere safe", "contact family", "call family", "ask for help", "police"),
    "judgment_wet_floor": ("slow down", "go around", "walk carefully", "be careful"),
    "judgment_door": ("verify", "check their identity", "contact family", "do not open", "don't open"),
    "judgment_power": ("stay away", "do not touch", "don't touch", "ask someone", "call for help"),
}

ENGLISH_CALCULATION_ANSWERS = {
    "calculation_10_minus_3": ("7", "seven"),
    "calculation_12_minus_5": ("7", "seven"),
    "calculation_15_minus_6": ("9", "nine"),
    "calculation_20_minus_8": ("12", "twelve"),
    "calculation_8_plus_4": ("12", "twelve"),
    "calculation_6_plus_7": ("13", "thirteen"),
    "calculation_9_plus_5": ("14", "fourteen"),
    "calculation_7_plus_8": ("15", "fifteen"),
    "calculation_18_minus_9": ("9", "nine"),
    "calculation_14_minus_4": ("10", "ten"),
    "calculation_5_plus_5": ("10", "ten"),
    "calculation_16_minus_7": ("9", "nine"),
}

ENGLISH_MONTHS = (
    "", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)

ENGLISH_LANGUAGE_ITEMS = {
    "language_weather": ("The weather is lovely today.", ("the weather is lovely today",)),
    "language_walk": ("Let us go for a walk together.", ("let us go for a walk together", "let's go for a walk together")),
    "language_flower": ("A flower is blooming by the window.", ("a flower is blooming by the window",)),
    "language_breakfast": ("Remember to eat breakfast in the morning.", ("remember to eat breakfast in the morning",)),
    "language_birds": ("A bird is singing in the tree.", ("a bird is singing in the tree",)),
    "language_family": ("The family is coming home for dinner tonight.", ("the family is coming home for dinner tonight",)),
    "language_water": ("There is a cup of warm water on the table.", ("there is a cup of warm water on the table",)),
    "language_sun": ("It feels warm after the sun comes out.", ("it feels warm after the sun comes out",)),
    "language_market": ("We will go to the market tomorrow morning.", ("we will go to the market tomorrow morning",)),
    "language_radio": ("The radio is playing music.", ("the radio is playing music",)),
}


def _memory_question(memory_set, delayed=False, language="zh"):
    words = [aliases[0] for aliases in memory_set]
    identifier = "_".join(words)
    concepts = memory_set
    if language == "en":
        concepts = tuple(
            ENGLISH_MEMORY_WORDS[aliases[0]] for aliases in memory_set
        )
    if delayed:
        return {
            "id": f"delayed_memory_{identifier}",
            "domain": "延迟记忆",
            "prompt": (
                "Can you remember the three words from the first task? "
                "Please say as many as you can."
                if language == "en"
                else "还记得第一题让您记住的三个词吗？请尽量说出来。"
            ),
            "type": "keywords",
            "concepts": concepts,
            "points": 3,
            "domain_label": ENGLISH_DOMAINS["延迟记忆"] if language == "en" else "延迟记忆",
        }
    display_words = (
        [ENGLISH_MEMORY_WORDS[aliases[0]][0] for aliases in memory_set]
        if language == "en" else words
    )
    return {
        "id": f"immediate_memory_{identifier}",
        "domain": "即时记忆",
        "prompt": (
            f"Please remember these three words: {', '.join(display_words)}. "
            "Now repeat the three words."
            if language == "en" else
            f"请记住这三个词：{'、'.join(words)}。现在请把这三个词复述一遍。"
        ),
        "type": "keywords",
        "concepts": concepts,
        "points": 3,
        "domain_label": ENGLISH_DOMAINS["即时记忆"] if language == "en" else "即时记忆",
    }


def _calculation_question(item, language="zh"):
    identifier, prompt, answers = item
    if language == "en":
        answers = ENGLISH_CALCULATION_ANSWERS[identifier]
    return {
        "id": identifier,
        "domain": "计算能力",
        "prompt": ENGLISH_PROMPTS.get(identifier, prompt) if language == "en" else prompt,
        "type": "exact",
        "answers": answers,
        "points": 1,
        "domain_label": ENGLISH_DOMAINS["计算能力"] if language == "en" else "计算能力",
    }


def _language_question(item, language="zh"):
    identifier, sentence, answers = item
    if language == "en":
        sentence, answers = ENGLISH_LANGUAGE_ITEMS[identifier]
    return {
        "id": identifier,
        "domain": "语言能力",
        "prompt": (
            f"Please repeat this sentence exactly: {sentence}"
            if language == "en" else f"请完整复述这句话：{sentence}"
        ),
        "type": "contains",
        "answers": answers,
        "points": 1,
        "domain_label": ENGLISH_DOMAINS["语言能力"] if language == "en" else "语言能力",
    }


def _judgment_question(item, language="zh"):
    return {
        "id": item["id"],
        "domain": "日常判断",
        "prompt": ENGLISH_PROMPTS.get(item["id"], item["prompt"]) if language == "en" else item["prompt"],
        "type": "judgment",
        "answers": ENGLISH_ANSWERS.get(item["id"], item["answers"]) if language == "en" else item["answers"],
        "points": 1,
        "domain_label": ENGLISH_DOMAINS["日常判断"] if language == "en" else "日常判断",
    }


def _clock_drawing_question(language="zh"):
    """实验性绘图观察题，不与现有14分生活化测试混算。"""
    return {
        "id": "clock_drawing_1110",
        "domain": "视空间观察",
        "prompt": (
            "Draw a complete clock below: include the clock face and all numbers, "
            "then draw the hands to show 11:10. Select “Submit clock drawing” when finished."
            if language == "en" else
            "请在下方画布上画一个完整的钟表：画出表盘和全部数字，再用指针表示11点10分。完成后点击“提交画钟”。"
        ),
        "type": "drawing_clock",
        "response_type": "drawing",
        "points": 0,
        "domain_label": ENGLISH_DOMAINS["视空间观察"] if language == "en" else "视空间观察",
    }


def build_test_questions(randomizer=None, language="zh"):
    """按固定维度配额生成10道计分题和1道绘图观察题。"""
    randomizer = randomizer or secrets.SystemRandom()
    memory_set = randomizer.choice(MEMORY_SETS)
    time_questions = [dict(item) for item in randomizer.sample(TIME_QUESTIONS, 3)]
    attention_questions = [dict(item) for item in randomizer.sample(ATTENTION_QUESTIONS, 2)]
    if language == "en":
        for question in time_questions + attention_questions:
            question["prompt"] = ENGLISH_PROMPTS.get(question["id"], question["prompt"])
            question["domain_label"] = ENGLISH_DOMAINS[question["domain"]]
            if question["id"] in ENGLISH_ANSWERS:
                question["answers"] = ENGLISH_ANSWERS[question["id"]]
    middle = [
        *time_questions,
        *attention_questions,
        _calculation_question(randomizer.choice(CALCULATION_QUESTIONS), language),
        _language_question(randomizer.choice(LANGUAGE_QUESTIONS), language),
        _judgment_question(randomizer.choice(JUDGMENT_QUESTIONS), language),
    ]
    randomizer.shuffle(middle)
    return [
        _memory_question(memory_set, language=language),
        *middle,
        _memory_question(memory_set, delayed=True, language=language),
        _clock_drawing_question(language),
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

    if answer_type == "drawing_clock":
        raise ValueError("画钟题必须通过绘图接口提交。")

    if answer_type == "judgment":
        language = "en" if question.get("domain_label") == "Everyday judgement" else "zh"
        return question["points"] * analyze_judgment_answer(
            question["id"],
            answer,
            language=language,
        ).earned

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
        if question.get("domain_label") == "Time orientation":
            accepted.add(ENGLISH_MONTHS[now.month])
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
            normalize_answer(item)
            for item in (
                ENGLISH_WEEKDAY_NAMES[now.weekday()]
                if question.get("domain_label") == "Time orientation"
                else WEEKDAY_NAMES[now.weekday()]
            )
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
