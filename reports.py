"""根据历史测试得分生成非诊断性的个性化建议报告。"""


REPORT_SOURCES = [
    {
        "title": "世界卫生组织：降低认知下降和痴呆风险指南（第二版）",
        "url": "https://www.who.int/publications/i/item/9789240123557",
    },
    {
        "title": "美国国家老龄研究所：认知健康与健康老龄化",
        "url": "https://www.nia.nih.gov/health/brain-health",
    },
]


DOMAIN_GUIDANCE = {
    "即时记忆": {
        "goal": "练习短时间记住并复述少量信息。",
        "activities": [
            "选择3个日常物品，观察后遮住，立即说出名称。",
            "把要记的内容组成一个小故事，或联想到熟悉的画面。",
            "一次练习5至10分钟，答错时给予提示，不催促、不责备。",
        ],
    },
    "延迟记忆": {
        "goal": "练习隔一段时间后回想刚刚接触的信息。",
        "activities": [
            "记住2至3个词，间隔2分钟和5分钟分别回想一次。",
            "回忆当天发生的一件小事，可借助照片或物品提示。",
            "先自由回忆，再逐步给类别或首字提示，避免直接公布答案。",
        ],
    },
    "时间定向": {
        "goal": "借助固定线索熟悉日期、星期和当天安排。",
        "activities": [
            "早晨一起查看大字日历，说出年月日和星期。",
            "在醒目位置放置时钟、日历和当天活动卡片。",
            "晚上简单回顾今天做过的事情和明天的安排。",
        ],
    },
    "注意力": {
        "goal": "练习在安静环境中专注完成一个短任务。",
        "activities": [
            "进行数字顺背、简单找不同或按颜色分类，每次5至10分钟。",
            "训练时关闭电视等干扰，一次只完成一项任务。",
            "出现疲劳或烦躁时立即休息，不用延长训练时间来补偿。",
        ],
    },
    "计算能力": {
        "goal": "在熟悉生活场景中保持简单数量处理能力。",
        "activities": [
            "用硬币、豆子或水果做10以内的加减练习。",
            "模拟买菜或购物，计算少量物品的数量和简单金额。",
            "允许使用实物辅助，重点是理解过程而不是追求速度。",
        ],
    },
    "语言能力": {
        "goal": "练习理解、表达和复述简短信息。",
        "activities": [
            "每天看一张熟悉照片，说出人物、地点和正在发生的事情。",
            "听一两句话后用自己的话复述，不要求一字不差。",
            "选择喜欢的短文、戏曲或新闻片段进行朗读和交流。",
        ],
    },
    "日常判断": {
        "goal": "通过熟悉情景巩固日常安全判断。",
        "activities": [
            "用图片讨论过马路、防跌倒和陌生来电等安全情景。",
            "把重要步骤做成一事一卡，放在对应位置提醒。",
            "涉及燃气、用药或外出安全时，由家属共同检查，不单靠记忆。",
        ],
    },
}


def _overall_section(percentage):
    if percentage >= 80:
        return {
            "tone": "steady",
            "title": "本次整体表现较稳定",
            "summary": (
                "多数任务完成情况较好。建议保持规律生活、社交交流和适量认知活动，"
                "重点维护相对薄弱的能力维度。"
            ),
        }
    if percentage >= 60:
        return {
            "tone": "watch",
            "title": "本次部分任务存在困难",
            "summary": (
                "建议在精神和体力状态较好时进行短时、低压力训练。"
                "如果困难持续出现或开始影响日常生活，可与家属一起咨询专业人员。"
            ),
        }
    return {
        "tone": "support",
        "title": "本次多项任务需要更多支持",
        "summary": (
            "一次测试可能受到听力、疲劳、情绪、教育经历和语音识别准确率影响。"
            "建议由家属陪同关注日常变化，并考虑到正规医疗机构进行专业评估。"
        ),
    }


def _domain_rows(domain_scores, domain_maximums):
    rows = []
    for order, (domain, maximum) in enumerate(domain_maximums.items()):
        score = int(domain_scores.get(domain, 0))
        ratio = score / maximum if maximum else 0
        rows.append({
            "domain": domain,
            "score": score,
            "maximum": maximum,
            "ratio": ratio,
            "percentage": round(ratio * 100),
            "order": order,
        })
    return rows


def build_assessment_report(assessment, domain_scores, domain_maximums):
    """返回供模板展示的确定性报告，不调用大模型。"""
    percentage = int(assessment["percentage"])
    rows = _domain_rows(domain_scores, domain_maximums)
    ranked = sorted(rows, key=lambda row: (row["ratio"], row["order"]))

    strengths = [
        row for row in sorted(rows, key=lambda row: (-row["ratio"], row["order"]))
        if row["ratio"] >= 0.75
    ][:3]

    priorities = [row for row in ranked if row["ratio"] < 0.75][:3]
    if not priorities:
        priorities = ranked[:2]

    training_cards = []
    for row in priorities:
        guidance = DOMAIN_GUIDANCE[row["domain"]]
        training_cards.append({
            **row,
            "goal": guidance["goal"],
            "activities": guidance["activities"],
        })

    priority_names = [card["domain"] for card in training_cards]
    weekly_plan = [
        {
            "when": "每天早晨",
            "task": "查看大字日历和当天安排，用2至3分钟确认日期与计划。",
        },
        {
            "when": "每周一、四",
            "task": f"练习{priority_names[0]}，每次5至10分钟，保持轻松。",
        },
        {
            "when": "每周二、五",
            "task": (
                f"练习{priority_names[1] if len(priority_names) > 1 else priority_names[0]}"
                "，出现疲劳时立即休息。"
            ),
        },
        {
            "when": "每周三或周末",
            "task": "选择散步、聊天、唱歌、下棋或熟悉的兴趣活动，与家人共同参与。",
        },
    ]

    return {
        "overall": _overall_section(percentage),
        "strengths": strengths,
        "priorities": priorities,
        "training_cards": training_cards,
        "weekly_plan": weekly_plan,
        "health_actions": [
            "在身体情况允许且医生没有限制的前提下，保持规律活动，可从短时间散步开始。",
            "保持相对固定的作息；如果长期睡不好、白天过度困倦，应咨询医生。",
            "保持与亲友交流，并参加本人喜欢且不会造成挫败感的活动。",
            "按医嘱管理血压、血糖、血脂和慢性病，不自行停药或更改药量。",
            "如有听力或视力困难，及时检查并使用合适的辅助设备。",
            "均衡饮食；没有明确缺乏时，不把保健品或维生素当作改善记忆的治疗。",
        ],
        "follow_up": (
            "不要每天重复刷同一套测试题，以免练习效应影响结果。"
            "更重要的是观察记忆和判断困难是否持续、加重，或影响用药、做饭、出行、"
            "财务等日常生活；出现这些情况时，建议由家属陪同进行专业评估。"
        ),
        "urgent_note": (
            "如果出现突然意识混乱、突然说话含糊或听不懂话、单侧肢体无力、"
            "突然行走不稳等情况，应立即寻求急诊医疗帮助，不要等待下一次测试。"
        ),
        "sources": REPORT_SOURCES,
    }

