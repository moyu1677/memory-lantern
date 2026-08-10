"""日常安全判断题的可解释规则评分。

规则同时描述安全行为和危险行为，并识别行为是否被否定。该模块只返回
任务级判断，不输出疾病风险，也不依赖大模型。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


RULE_VERSION = "judgment-safety-v1"


@dataclass(frozen=True)
class SafetyRule:
    safe_zh: tuple[str, ...]
    dangerous_zh: tuple[str, ...]
    safe_en: tuple[str, ...]
    dangerous_en: tuple[str, ...]


RULES = {
    "judgment_red_light": SafetyRule(
        ("停下等待", "停下", "停止", "等待", "等绿灯"),
        ("继续通行", "继续走", "直接过去", "闯红灯", "走过去"),
        ("stop and wait", "stop", "wait", "wait for green"),
        ("keep crossing", "continue crossing", "keep walking", "cross the road", "go through"),
    ),
    "judgment_smell_gas": SafetyRule(
        ("关闭火源", "关火", "关闭煤气", "关煤气", "开窗", "请人帮助", "找人帮助", "报警"),
        ("继续开火", "继续用火", "点火", "打开电器", "开灯"),
        ("turn off the flame", "turn it off", "turn off the gas", "open a window", "ask for help", "call emergency services"),
        ("keep using a flame", "light a flame", "turn on a light", "use an electrical switch"),
    ),
    "judgment_unknown_medicine": SafetyRule(
        ("询问医生", "问医生", "询问家属", "问家属", "先确认", "先核实"),
        ("随便服用", "直接服用", "直接吃", "马上吃", "自己试"),
        ("ask a doctor", "ask family", "check first", "verify first"),
        ("take it", "take the medicine", "try it", "swallow it"),
    ),
    "judgment_phone_scam": SafetyRule(
        ("联系家人", "找家人", "先核实", "核实", "报警", "联系银行"),
        ("马上转账", "立即转账", "直接转账", "给他转账", "汇款", "给钱"),
        ("contact family", "check first", "verify", "call the police", "contact the bank"),
        ("transfer money", "send money", "pay immediately", "give them money"),
    ),
    "judgment_dizzy": SafetyRule(
        ("先坐下", "坐下", "停下来", "呼叫帮助", "叫人", "找人帮助", "联系家人"),
        ("继续独自行走", "继续走", "自己走", "独自行走", "强行走"),
        ("sit down", "stop", "call for help", "ask for help", "contact family"),
        ("keep walking", "walk alone", "continue alone"),
    ),
    "judgment_hot_water": SafetyRule(
        ("先放凉", "放凉", "冷却", "试温", "小心试温", "先试温度"),
        ("马上大口喝", "马上喝", "直接喝", "立即喝", "趁热喝"),
        ("let it cool", "cool it", "test the temperature", "check the temperature"),
        ("drink immediately", "drink it straight away", "drink it now", "drink while boiling"),
    ),
    "judgment_lost": SafetyRule(
        ("安全地点", "原地等待", "留在原地", "联系家人", "打电话", "找警察", "求助"),
        ("继续乱走", "到处乱走", "继续走", "自己乱找", "四处走"),
        ("stay somewhere safe", "stay where I am", "contact family", "call family", "ask for help", "find the police"),
        ("keep wandering", "wander around", "keep walking", "continue alone"),
    ),
    "judgment_wet_floor": SafetyRule(
        ("放慢脚步", "慢慢走", "绕开", "小心走", "小心"),
        ("快速走过去", "快走", "跑过去", "直接冲过去"),
        ("slow down", "go around", "walk carefully", "be careful"),
        ("walk quickly", "run across", "rush across"),
    ),
    "judgment_door": SafetyRule(
        ("核实身份", "确认身份", "联系家人", "先确认", "报警"),
        ("直接开门", "马上开门", "让他进来", "放他进来", "开门"),
        ("verify their identity", "check their identity", "contact family", "check first", "call the police"),
        ("open the door", "let them in", "allow them inside"),
    ),
    "judgment_power": SafetyRule(
        ("远离", "保持距离", "请人处理", "找人处理", "关闭电源", "断电", "报警"),
        ("用手触碰", "直接触碰", "用手碰", "直接碰", "摸电线", "捡起来"),
        ("stay away", "keep a distance", "ask someone", "call for help", "turn off the power"),
        ("touch it", "touch the wire", "pick it up", "handle it"),
    ),
}

_ZH_NEGATION_BEFORE = re.compile(
    r"(?:不应该|不能|不要|不可|不该|不用|千万别|切勿|别|不|没)[^，,。.！!？?；;]{0,5}$"
)
_ZH_NEGATION_AFTER = re.compile(r"^(?:是不对的|不对|不好|不应该|不可以|不行)")
_EN_NEGATION_BEFORE = re.compile(
    r"(?:do not|don't|should not|shouldn't|cannot|can't|must not|never|not)\s+(?:\w+\s+){0,3}$",
    re.IGNORECASE,
)
_EN_NEGATION_AFTER = re.compile(
    r"^\s+(?:is wrong|would be wrong|is unsafe|is not right)",
    re.IGNORECASE,
)
_ZH_UNCERTAIN = ("不知道", "不清楚", "不确定", "随便", "都可以", "看情况", "想不起来")
_EN_UNCERTAIN = ("i don't know", "not sure", "either", "it depends", "no idea", "cannot remember")


@dataclass(frozen=True)
class JudgmentAnalysis:
    decision: str
    safe_evidence: tuple[str, ...]
    dangerous_evidence: tuple[str, ...]
    negated_safe_evidence: tuple[str, ...]
    negated_dangerous_evidence: tuple[str, ...]
    uncertainty_evidence: tuple[str, ...]
    reason: str
    evidence_strength: float
    rule_version: str = RULE_VERSION

    @property
    def earned(self) -> int:
        return 1 if self.decision == "correct" else 0

    def audit_record(self) -> dict:
        """返回不包含用户原话的规则审计记录。"""
        return {
            "judgment_decision": self.decision,
            "judgment_reason": self.reason,
            "safe_rule_hits": list(self.safe_evidence),
            "dangerous_rule_hits": list(self.dangerous_evidence),
            "negated_safe_rule_hits": list(self.negated_safe_evidence),
            "negated_dangerous_rule_hits": list(self.negated_dangerous_evidence),
            "uncertainty_detected": bool(self.uncertainty_evidence),
            "judgment_evidence_strength": self.evidence_strength,
            "judgment_rule_version": self.rule_version,
        }


def _normalise(text) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    return re.sub(r"\s+", " ", text).strip()


def _is_negated(text: str, start: int, end: int, english: bool) -> bool:
    before = text[max(0, start - (35 if english else 12)):start]
    after = text[end:end + (24 if english else 8)]
    if english:
        return bool(_EN_NEGATION_BEFORE.search(before) or _EN_NEGATION_AFTER.match(after))
    return bool(_ZH_NEGATION_BEFORE.search(before) or _ZH_NEGATION_AFTER.match(after))


def _classify_phrases(text: str, phrases: tuple[str, ...], english: bool):
    positive = []
    negated = []
    # 长短表达重叠时优先记录更具体的长表达。
    occupied = []
    for phrase in sorted(phrases, key=len, reverse=True):
        for match in re.finditer(re.escape(phrase.lower()), text):
            span = match.span()
            if any(span[0] >= old[0] and span[1] <= old[1] for old in occupied):
                continue
            occupied.append(span)
            target = negated if _is_negated(text, span[0], span[1], english) else positive
            target.append(phrase)
    return tuple(dict.fromkeys(positive)), tuple(dict.fromkeys(negated))


def analyze_judgment_answer(question_id: str, answer, language="zh") -> JudgmentAnalysis:
    rule = RULES.get(question_id)
    if rule is None:
        raise KeyError(f"缺少日常判断题规则: {question_id}")

    text = _normalise(answer)
    english = str(language or "").lower() in {"en", "english"}
    safe_phrases = rule.safe_en if english else rule.safe_zh
    dangerous_phrases = rule.dangerous_en if english else rule.dangerous_zh
    uncertain_phrases = _EN_UNCERTAIN if english else _ZH_UNCERTAIN

    safe, negated_safe = _classify_phrases(text, safe_phrases, english)
    dangerous, negated_dangerous = _classify_phrases(text, dangerous_phrases, english)
    uncertainty = tuple(phrase for phrase in uncertain_phrases if phrase in text)

    correct_support = bool(safe or negated_dangerous)
    incorrect_support = bool(dangerous or negated_safe)
    if correct_support and incorrect_support:
        decision, reason, strength = "conflict", "同时识别到安全和危险行为证据", 0.35
    elif correct_support:
        decision, reason, strength = "correct", "肯定安全行为或明确否定危险行为", 0.95
    elif incorrect_support:
        decision, reason, strength = "incorrect", "肯定危险行为或明确否定安全行为", 0.95
    else:
        decision = "uncertain"
        reason = "回答包含不确定表达" if uncertainty else "未识别到明确的安全或危险行为"
        strength = 0.25

    return JudgmentAnalysis(
        decision=decision,
        safe_evidence=safe,
        dangerous_evidence=dangerous,
        negated_safe_evidence=negated_safe,
        negated_dangerous_evidence=negated_dangerous,
        uncertainty_evidence=uncertainty,
        reason=reason,
        evidence_strength=strength,
    )
