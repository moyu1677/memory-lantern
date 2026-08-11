"""Bounded, consent-first extraction for conversation-derived need cards."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable


CATEGORIES = {
    "居家事务", "出行与生活安排", "身体不适原话", "家庭联系",
    "社交与情感表达", "兴趣和愿望", "服务或物品", "互动偏好",
}
POLARITIES = {"needs_help", "declines_help", "mentions_interest", "uncertain"}
LIFE_CUE_RE = re.compile(
    r"(灯|水管|煤气|门锁|电器|买菜|吃药|医院|出门|下楼|看不清|听不清|"
    r"不舒服|疼|累|麻烦|修|想家|没人)"
)
IMPACT_CUE_RE = re.compile(
    r"(不方便|麻烦|看不清|听不清|走不动|够不着|害怕|累|疼|难受|"
    r"不敢|没办法|希望|想让|帮忙|顺便)"
)
DECLINE_RE = re.compile(r"(不想|不用|别|算了|不麻烦|不用告诉|不需要)")
SHARE_YES_RE = re.compile(r"^(可以|好|同意|行|告诉|让.*知道|麻烦你).{0,12}$")


@dataclass(frozen=True)
class DialogueDecision:
    next_stage: str
    instruction: str = ""
    should_extract: bool = False


def decide_next_action(stage: str, user_text: str) -> DialogueDecision:
    """Allow at most one impact question and one explicit share confirmation."""

    text = str(user_text or "").strip()
    if DECLINE_RE.search(text):
        return DialogueDecision("normal")
    if stage == "normal" and LIFE_CUE_RE.search(text):
        return DialogueDecision(
            "awaiting_impact",
            "老人刚提到一件具体生活事情。只顺着这件事温和追问一次，"
            "了解它是否带来不便；不要连续提问、不要诊断，也不要替老人下结论。",
        )
    if stage == "awaiting_impact" and IMPACT_CUE_RE.search(text):
        return DialogueDecision(
            "awaiting_share",
            "老人已经说出不便或个人想法。请只用一句话询问：是否愿意把"
            "简短摘要告诉已绑定的家人；明确说明不会分享完整聊天或录音。",
        )
    if stage == "awaiting_share" and SHARE_YES_RE.match(text):
        return DialogueDecision("normal", should_extract=True)
    return DialogueDecision("normal")


def bounded_history(history: Iterable[dict], limit: int = 8) -> list[dict]:
    """Keep only the latest safe turns for a short-lived model request."""

    safe = []
    for item in list(history)[-limit:]:
        role = item.get("role")
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            safe.append({"role": role, "content": content[:800]})
    return safe


def _normalize_card(raw: dict, user_text: str) -> dict | None:
    category = str(raw.get("category") or "").strip()
    subject = re.sub(r"\s+", " ", str(raw.get("subject") or "").strip())[:80]
    polarity = str(raw.get("polarity") or "").strip()
    evidence = str(raw.get("evidence") or "").strip()
    directness = str(raw.get("directness") or "indirect").strip()
    if category not in CATEGORIES or polarity not in POLARITIES:
        return None
    if not subject or not evidence or evidence not in user_text:
        return None
    return {
        "category": category,
        "subject": subject,
        "polarity": polarity,
        "directness": directness if directness in {"direct", "indirect"} else "indirect",
        "evidence": evidence[:240],
    }


def extract_session_card(client, model: str, history: Iterable[dict]) -> dict | None:
    """Extract a verifiable card only after the caller has received consent."""

    messages = bounded_history(history, limit=10)
    user_text = "\n".join(
        item["content"] for item in messages if item["role"] == "user"
    )
    if not user_text:
        return None
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你只做信息整理，不作医疗或心理判断。根据老人实际说过的话，"
                    "输出一个 JSON 对象，字段严格为 category、subject、polarity、"
                    "directness、evidence。category 只能是："
                    f"{'、'.join(sorted(CATEGORIES))}。polarity 只能是："
                    "needs_help、declines_help、mentions_interest、uncertain。"
                    "evidence 必须逐字摘自老人说的话；如果没有可靠内容，返回 {}。"
                ),
            },
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    try:
        raw = json.loads(completion.choices[0].message.content or "{}")
    except (TypeError, json.JSONDecodeError):
        return None
    return _normalize_card(raw, user_text) if isinstance(raw, dict) else None
