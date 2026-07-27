"""适老化成语接龙的确定性规则与精选常用成语库。"""

import re
import secrets

from database import (
    idiom_exists,
    list_idioms_starting,
    random_chainable_idiom,
)

TRADITIONAL_TO_SIMPLIFIED = str.maketrans({
    "風": "风",
    "順": "顺",
    "虛": "虚",
    "頓": "顿",
    "開": "开",
    "門": "门",
    "見": "见",
    "長": "长",
    "歲": "岁",
    "樂": "乐",
    "業": "业",
    "於": "于",
    "學": "学",
    "練": "练",
    "達": "达",
    "萬": "万",
    "氣": "气",
    "發": "发",
    "揚": "扬",
    "報": "报",
})


def clean_idiom(value):
    """删除语音识别可能附带的标点和空格，只保留中文。"""
    chinese = "".join(
        re.findall(r"[\u4e00-\u9fff]", str(value or ""))
    )
    return chinese.translate(TRADITIONAL_TO_SIMPLIFIED)


def resolve_idiom(value):
    """从输入或一小句语音识别结果中找出成语。"""
    cleaned = clean_idiom(value)
    if idiom_exists(cleaned):
        return cleaned
    matches = {
        cleaned[index:index + 4]
        for index in range(max(0, len(cleaned) - 3))
        if idiom_exists(cleaned[index:index + 4])
    }
    return next(iter(matches)) if len(matches) == 1 else cleaned


def available_idioms(first_character, used):
    return list_idioms_starting(
        first_character,
        excluded=used,
        limit=80,
    )


def choose_computer_idiom(first_character, used):
    """优先选择后面仍有可接成语的回复。"""
    candidates = available_idioms(first_character, used)
    continuing = [
        idiom
        for idiom in candidates
        if available_idioms(idiom[-1], used | {idiom})
    ]
    return secrets.choice(continuing) if continuing else None


def new_game_state(user_id):
    first = random_chainable_idiom()
    if first is None:
        raise RuntimeError("成语数据库中没有可用于接龙的数据")
    return {
        "user_id": user_id,
        "computer_idiom": first,
        "required_character": first[-1],
        "used": {first},
        "rounds": 0,
        "history": [{"speaker": "computer", "idiom": first}],
    }


def hints_for(state, limit=3):
    return available_idioms(
        state["required_character"],
        state["used"],
    )[:limit]


def play_turn(state, submitted):
    idiom = resolve_idiom(submitted)
    if len(idiom) != 4:
        return {
            "ok": False,
            "error": "请输入四个字的成语。语音识别后也可以先修改再提交。",
        }
    if not idiom_exists(idiom):
        return {
            "ok": False,
            "error": "这个词暂时不在成语库中，可以检查文字或点击“给我提示”。",
        }
    if idiom in state["used"]:
        return {"ok": False, "error": "这个成语已经出现过了，换一个试试吧。"}
    if not idiom.startswith(state["required_character"]):
        return {
            "ok": False,
            "error": f"这一轮需要用“{state['required_character']}”字开头。",
        }

    state["used"].add(idiom)
    state["history"].append({"speaker": "user", "idiom": idiom})
    state["rounds"] += 1

    computer = choose_computer_idiom(idiom[-1], state["used"])
    if computer is None:
        return {
            "ok": True,
            "completed": True,
            "user_idiom": idiom,
            "rounds": state["rounds"],
            "message": "太棒了，这一轮把电脑难住了！",
        }

    state["used"].add(computer)
    state["history"].append({"speaker": "computer", "idiom": computer})
    state["computer_idiom"] = computer
    state["required_character"] = computer[-1]
    return {
        "ok": True,
        "completed": False,
        "user_idiom": idiom,
        "computer_idiom": computer,
        "required_character": computer[-1],
        "rounds": state["rounds"],
        "hints": hints_for(state),
        "message": f"接得好！电脑接“{computer}”。",
    }
