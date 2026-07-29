"""论文3.1：程序一致性、随机组卷结构和规则判分测试。"""

from collections import Counter
import random

import questions


def all_question_ids():
    identifiers = {
        item["id"] for item in questions.TIME_QUESTIONS
    }
    identifiers.update(item["id"] for item in questions.ATTENTION_QUESTIONS)
    identifiers.update(item[0] for item in questions.CALCULATION_QUESTIONS)
    identifiers.update(item[0] for item in questions.LANGUAGE_QUESTIONS)
    identifiers.update(item["id"] for item in questions.JUDGMENT_QUESTIONS)
    for memory_set in questions.MEMORY_SETS:
        words = "_".join(aliases[0] for aliases in memory_set)
        identifiers.add(f"immediate_memory_{words}")
        identifiers.add(f"delayed_memory_{words}")
    return identifiers


def test_generate_10000_papers_with_stable_structure():
    expected_domains = {
        "即时记忆": 1,
        "延迟记忆": 1,
        "时间定向": 3,
        "注意力": 2,
        "计算能力": 1,
        "语言能力": 1,
        "日常判断": 1,
    }
    expected_ids = all_question_ids()
    observed_ids = set()
    unique_papers = set()
    randomizer = random.Random(20260729)

    for _ in range(10_000):
        paper = questions.build_test_questions(randomizer=randomizer)
        observed_ids.update(item["id"] for item in paper)
        unique_papers.add(tuple(item["id"] for item in paper))

        assert len(paper) == 10
        assert sum(item["points"] for item in paper) == 14
        assert Counter(item["domain"] for item in paper) == expected_domains
        assert paper[0]["domain"] == "即时记忆"
        assert paper[-1]["domain"] == "延迟记忆"
        assert paper[0]["concepts"] == paper[-1]["concepts"]

    assert len(expected_ids) == 62
    assert observed_ids == expected_ids
    assert len(unique_papers) > 9_000


def test_answer_normalization_and_rule_scoring():
    exact = {
        "type": "exact",
        "answers": ("一八三", "183"),
        "points": 1,
    }
    contains = {
        "type": "contains",
        "answers": ("联系家人", "先核实"),
        "points": 1,
    }
    memory = {
        "type": "keywords",
        "concepts": (("苹果",), ("钥匙",), ("河流", "小河")),
        "points": 3,
    }

    assert questions.score_answer(exact, " 1，8，3。") == 1
    assert questions.score_answer(exact, "381") == 0
    assert questions.score_answer(contains, "我会先联系家人核实") == 1
    assert questions.score_answer(memory, "苹果、钥匙、小河") == 3
    assert questions.score_answer(memory, "只记得苹果") == 1


def test_result_thresholds_are_deterministic():
    assert questions.result_level(80) == "本次认知任务表现较稳定"
    assert questions.result_level(79) == "本次部分任务存在困难，建议休息后再次测试"
    assert questions.result_level(60) == "本次部分任务存在困难，建议休息后再次测试"
    assert questions.result_level(59) == "本次多项任务存在困难，建议由家属陪同进行专业评估"

