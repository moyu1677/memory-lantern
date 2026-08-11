"""聊天关怀报告的结构化提取与安全校验。

模型只整理老人和 AI 对话中出现的关怀信号；它不诊断疾病，也不把
逐字原话返回给家属页面。来源轮次仅用于服务端校验和审计。
"""

from __future__ import annotations

import json
import re
from typing import Iterable


AREAS = {"psychological", "physical_or_daily"}
STATUSES = {"signals_present", "insufficient_or_conflicting"}
STRENGTHS = {"low", "medium"}
TRENDS = {"new", "repeated", "improved", "conflicting"}
ACTION_TYPES = {
    "contact_companionship",
    "shared_activity",
    "daily_step_support",
    "safety_check",
}
ACTION_TYPES_BY_AREA = {
    "psychological": {"contact_companionship", "shared_activity"},
    "physical_or_daily": {"daily_step_support", "safety_check"},
}
DIAGNOSIS_RE = re.compile(
    r"(确诊|诊断|患有|疾病|病症|抑郁症|阿尔茨海默|老年痴呆|帕金森|癌症|"
    r"处方|停药|换药|调整剂量|用药|服药|药物|治疗|就医|医院|医生)"
)


def _clean_text(value: object, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())[:limit]


def build_elder_turns(history: Iterable[dict], limit: int = 40) -> list[dict]:
    """只取老人说过的话，并给每条话一个不向家属公开的轮次编号。"""

    turns = []
    for item in list(history):
        if item.get("role") != "user":
            continue
        text = _clean_text(item.get("content"), 900)
        if text:
            turns.append({"id": len(turns) + 1, "text": text})
    return turns[-limit:]


def _report_prompt(turns: list[dict]) -> str:
    transcript = "\n".join(
        f"[老人第{turn['id']}轮] {turn['text']}" for turn in turns
    )
    return (
        "你是家庭关怀信号整理助手。你只能依据下面的老人聊天内容工作。"
        "你的任务不是猜测老人内心，更不是做医疗或心理诊断；你要整理少量、"
        "值得家人在现实中主动关心的信号。\n\n"
        "严格规则：\n"
        "1. 不得输出疾病名、诊断、处方、用药调整或确定性的心理结论。\n"
        "2. 不得复述、引用或改写老人原话；家人页面不会展示原话。\n"
        "3. 每条信号必须有 source_turn_ids，且只能使用输入中存在的轮次编号。\n"
        "4. 最新一轮若明确表示问题已解决、当前不受影响、不需要家人操心，"
        "应覆盖更早的旧问题并输出 insufficient_or_conflicting；除非最新内容同时提出了新的未解决困难。"
        "不得仅因历史上曾出现过不适或孤单，就继续给家人推送提示。\n"
        "5. psychological 只用于陪伴、联系、孤独感、情绪支持或参与活动等信号；"
        "physical_or_daily 用于行动、睡眠、视听、日常自理、家务安全或身体不适等实际困难。"
        "普通做饭、兴趣、天气等日常描述本身不是 physical_or_daily 信号。\n"
        "6. 若线索不足、普通闲聊、或前后相互矛盾，必须保守地输出"
        "insufficient_or_conflicting，不要凑内容。\n"
        "7. 每条信号必须给出 action_type，用来说明家人下一步动作的类别：\n"
        "   - psychological 只能选 contact_companionship（主动联系、倾听、陪伴）或 shared_activity（一起参与一件具体活动）；\n"
        "   - physical_or_daily 只能选 daily_step_support（日常具体协助）或 safety_check（先确认安全、陪同处理）；\n"
        "8. 最多输出 3 条；family_action 是一句温和、可执行的家人行动建议。\n\n"
        "只输出 JSON：\n"
        "{\n"
        '  "status": "signals_present 或 insufficient_or_conflicting",\n'
        '  "summary": "不给出诊断的简短概览",\n'
        '  "signals": [\n'
        "    {\n"
        '      "area": "psychological 或 physical_or_daily",\n'
        '      "action_type": "contact_companionship、shared_activity、daily_step_support 或 safety_check",\n'
        '      "topic": "简短关怀主题，不能含原话",\n'
        '      "strength": "low 或 medium",\n'
        '      "trend": "new、repeated、improved 或 conflicting",\n'
        '      "family_action": "一句建议",\n'
        '      "source_turn_ids": [1]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "老人聊天内容：\n"
        f"{transcript}"
    )


def _contains_diagnosis(value: object) -> bool:
    return bool(DIAGNOSIS_RE.search(str(value or "")))


def _contains_long_source_span(
    value: object,
    source_texts: Iterable[str],
    minimum_length: int = 10,
) -> bool:
    """Reject a report field that copies a long contiguous span of a chat turn."""

    candidate = re.sub(r"[\W_]+", "", str(value or ""))
    if len(candidate) < minimum_length:
        return False
    for source in source_texts:
        normalized_source = re.sub(r"[\W_]+", "", str(source or ""))
        if len(normalized_source) < minimum_length:
            continue
        for start in range(len(normalized_source) - minimum_length + 1):
            if normalized_source[start : start + minimum_length] in candidate:
                return True
    return False


def normalize_care_report(
    raw: object,
    turn_ids: set[int],
    source_texts: Iterable[str] = (),
) -> dict | None:
    """只接收受限、可审计且不含诊断的报告结构。"""

    if not isinstance(raw, dict):
        return None
    status = _clean_text(raw.get("status"), 48)
    summary = _clean_text(raw.get("summary"), 180)
    if (
        status not in STATUSES
        or _contains_diagnosis(summary)
        or _contains_long_source_span(summary, source_texts)
    ):
        return None

    safe_signals = []
    for item in raw.get("signals") or []:
        if not isinstance(item, dict) or len(safe_signals) >= 3:
            continue
        area = _clean_text(item.get("area"), 32)
        action_type = _clean_text(item.get("action_type"), 32)
        topic = _clean_text(item.get("topic"), 80)
        strength = _clean_text(item.get("strength"), 16)
        trend = _clean_text(item.get("trend"), 24)
        family_action = _clean_text(item.get("family_action"), 160)
        source_turn_ids = item.get("source_turn_ids") or []
        if (
            area not in AREAS
            or action_type not in ACTION_TYPES_BY_AREA.get(area, set())
            or strength not in STRENGTHS
            or trend not in TRENDS
            or not topic
            or not family_action
            or not isinstance(source_turn_ids, list)
        ):
            continue
        try:
            source_ids = sorted({int(value) for value in source_turn_ids})
        except (TypeError, ValueError):
            continue
        if not source_ids or not set(source_ids).issubset(turn_ids):
            continue
        if any(
            _contains_diagnosis(value)
            or _contains_long_source_span(value, source_texts)
            for value in (topic, family_action)
        ):
            continue
        safe_signals.append(
            {
                "area": area,
                "action_type": action_type,
                "topic": topic,
                "strength": strength,
                "trend": trend,
                "family_action": family_action,
                "source_turn_ids": source_ids,
            }
        )

    if status == "signals_present" and not safe_signals:
        status = "insufficient_or_conflicting"
    if status == "insufficient_or_conflicting":
        safe_signals = []
        summary = summary or "近期聊天中的线索不足或彼此矛盾，暂不形成关怀提示。"
    if not summary:
        summary = "系统仅整理可供家人关心的线索，不作医疗或心理诊断。"

    return {"status": status, "summary": summary, "signals": safe_signals}


def generate_care_report(client, model: str, history: Iterable[dict]) -> dict | None:
    """调用后台推理，生成不包含原话的家庭关怀报告。"""

    turns = build_elder_turns(history)
    if len(turns) < 2:
        return None
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "请遵守用户消息中的 JSON 格式和安全规则。"},
            {"role": "user", "content": _report_prompt(turns)},
        ],
        response_format={"type": "json_object"},
        stream=False,
        extra_body={"thinking": {"type": "enabled"}},
    )
    content = completion.choices[0].message.content or "{}"
    try:
        raw = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    return normalize_care_report(
        raw,
        {turn["id"] for turn in turns},
        [turn["text"] for turn in turns],
    )
