"""口语回答的轻量清洗与最终答案提取。

该模块只改善规则评分前的文本质量，不生成医学结论，也不改变题库分值。
原始转写由调用方短暂持有；持久化时只应保存计数类行为指标。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


PROCESSING_VERSION = "spoken-answer-cleaner-v1"

_SEPARATORS = r"\s，,。.!！？?、；;：:"

_ZH_FILLER_ANYWHERE = re.compile(r"(?:嗯+|呃+|额+)")
_ZH_LEADING_FILLER = re.compile(
    rf"^(?:(?:啊+|这个(?:嘛)?|那个(?:嘛)?|让我想想|我想想|怎么说呢|就是)"
    rf"[{_SEPARATORS}]*)+"
)
_ZH_SEGMENT_FILLER = re.compile(
    rf"(?P<prefix>[，,。.!！？?、；;：:]\s*)(?:啊+|这个嘛|那个嘛|怎么说呢)"
    rf"(?=$|[{_SEPARATORS}])"
)
_EN_FILLER = re.compile(
    r"\b(?:um+|uh+|erm+|er+|hmm+|you know|let me think)\b",
    re.IGNORECASE,
)

_ZH_REPAIRS = (
    re.compile(r"不是[^，,。.!！？?；;]*[，,；;]\s*(?:应该)?是"),
    re.compile(r"(?:不对|我说错了|说错了|更正为|改成|我是说|应该是|其实是)"),
)
_EN_REPAIR = re.compile(
    r"\b(?:no|sorry|i mean|rather|actually|it should be)\b",
    re.IGNORECASE,
)

_ZH_ANSWER_PREFIX = re.compile(
    r"^(?:我的答案是|答案是|我觉得(?:应该)?是|我想(?:应该)?是|可能是|应该是|就是)\s*"
)
_EN_ANSWER_PREFIX = re.compile(
    r"^(?:my answer is|the answer is|i think\s+(?:it is|it's)?|it is|it's|probably)\s*",
    re.IGNORECASE,
)

_ZH_REPEATED_CHUNK = re.compile(
    r"(?P<chunk>[\u4e00-\u9fff]{1,8})(?:[\s，,、]+(?P=chunk))+"
    r"(?=$|[\s，,。.!！？?、；;：:])"
)
_EN_REPEATED_WORD = re.compile(
    r"\b(?P<word>[a-z]+)(?:[\s,]+(?P=word)\b)+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AnswerProcessingResult:
    """一次清洗结果；文本字段不应被写入长期测试记录。"""

    raw_text: str
    cleaned_text: str
    final_answer: str
    filler_count: int
    repetition_count: int
    self_correction_count: int
    version: str = PROCESSING_VERSION

    def behavior_metrics(self) -> dict:
        """返回可安全汇总的计数指标，不包含原始或清洗后的回答。"""
        return {
            "answer_processing_version": self.version,
            "filler_count": self.filler_count,
            "repetition_count": self.repetition_count,
            "self_correction_count": self.self_correction_count,
        }


def _normalise_surface(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def _remove_fillers(text: str, english: bool) -> tuple[str, int]:
    count = 0

    def remove(_match):
        nonlocal count
        count += 1
        return " "

    if english:
        return _EN_FILLER.sub(remove, text), count

    text = _ZH_FILLER_ANYWHERE.sub(remove, text)

    while True:
        match = _ZH_LEADING_FILLER.match(text)
        if not match:
            break
        count += 1
        text = text[match.end():]

    def remove_segment(match):
        nonlocal count
        count += 1
        return match.group("prefix")

    return _ZH_SEGMENT_FILLER.sub(remove_segment, text), count


def _collapse_repetitions(text: str, english: bool) -> tuple[str, int]:
    pattern = _EN_REPEATED_WORD if english else _ZH_REPEATED_CHUNK
    count = 0

    def collapse(match):
        nonlocal count
        repeated_text = match.group(0)
        separator_count = len(re.findall(r"[\s，,、]+", repeated_text))
        count += max(1, separator_count)
        return match.group("word" if english else "chunk")

    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(collapse, text)
    return text, count


def _extract_last_correction(text: str, english: bool) -> tuple[str, int]:
    matches = list(_EN_REPAIR.finditer(text)) if english else [
        match for pattern in _ZH_REPAIRS for match in pattern.finditer(text)
    ]
    if not matches:
        return text, 0
    last = max(matches, key=lambda match: match.start())
    candidate = text[last.end():]
    candidate = re.sub(rf"^[{_SEPARATORS}]+", "", candidate)
    return candidate or text, len(matches)


def _strip_answer_framing(text: str, english: bool) -> str:
    pattern = _EN_ANSWER_PREFIX if english else _ZH_ANSWER_PREFIX
    previous = None
    while previous != text:
        previous = text
        text = pattern.sub("", text).strip()
    return text


def process_spoken_answer(value, language="zh") -> AnswerProcessingResult:
    """清洗口语转写，并尽量提取说话者最后确认的答案。

    这里不做模糊语义判分。清洗后的结果仍交给项目原有的确定性规则评分器。
    """
    raw_text = _normalise_surface(value)
    english = str(language or "").lower() in {"en", "english"}

    cleaned, filler_count = _remove_fillers(raw_text, english)
    cleaned, repetition_count = _collapse_repetitions(cleaned, english)
    final_answer, correction_count = _extract_last_correction(cleaned, english)
    final_answer = _strip_answer_framing(final_answer, english)
    final_answer = re.sub(rf"^[{_SEPARATORS}]+|[{_SEPARATORS}]+$", "", final_answer)
    final_answer = re.sub(r"\s+", " ", final_answer).strip()

    # 如果清洗规则遇到意外表达导致结果为空，回退到原回答，避免丢失作答。
    if not final_answer:
        final_answer = raw_text

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return AnswerProcessingResult(
        raw_text=raw_text,
        cleaned_text=cleaned,
        final_answer=final_answer,
        filler_count=filler_count,
        repetition_count=repetition_count,
        self_correction_count=correction_count,
    )
