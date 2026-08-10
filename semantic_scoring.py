"""本地多语言语义相似度评分。

模型惰性加载：应用启动不会下载模型；只有规则无法明确判分时才会尝试加载。
若依赖、模型或网络不可用，调用方会收到 ``uncertain``，现有规则流程仍可继续。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable

from semantic_rubrics import SEMANTIC_RUBRICS, examples_for


DEFAULT_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
SEMANTIC_VERSION = "semantic-judgment-v1"
DECISIVE_LABELS = frozenset({"correct", "incorrect", "dangerous"})


@dataclass(frozen=True)
class SemanticScore:
    decision: str
    confidence: float
    margin: float
    class_scores: dict[str, float]
    matched_label: str | None
    matched_example: str | None
    available: bool
    reason: str
    model_name: str
    version: str = SEMANTIC_VERSION

    @property
    def decisive(self) -> bool:
        return self.available and self.decision in DECISIVE_LABELS

    def audit_record(self) -> dict:
        """仅保存模型元数据和静态样例，不保存用户原始回答。"""
        return {
            "semantic_decision": self.decision,
            "semantic_confidence": self.confidence,
            "semantic_margin": self.margin,
            "semantic_class_scores": dict(self.class_scores),
            "semantic_matched_label": self.matched_label,
            "semantic_matched_example": self.matched_example,
            "semantic_model_available": self.available,
            "semantic_reason": self.reason,
            "semantic_model_name": self.model_name,
            "semantic_version": self.version,
        }


def _default_model_loader(model_name: str, local_files_only: bool):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name,
        local_files_only=local_files_only,
    )


def _to_rows(values) -> list[list[float]]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        raise TypeError("模型没有返回可读取的向量。")
    if values and isinstance(values[0], (int, float)):
        values = [values]
    return [[float(value) for value in row] for row in values]


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


class LocalSemanticScorer:
    """对回答与四类审核样例计算余弦相似度。"""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        similarity_threshold: float = 0.55,
        margin_threshold: float = 0.06,
        local_files_only: bool = False,
        model_loader: Callable | None = None,
        rubrics=None,
    ):
        self.model_name = model_name
        self.similarity_threshold = float(similarity_threshold)
        self.margin_threshold = float(margin_threshold)
        self.local_files_only = bool(local_files_only)
        self.model_loader = model_loader or _default_model_loader
        self.rubrics = rubrics or SEMANTIC_RUBRICS
        self._model = None
        self._load_error = None
        self._reference_cache = {}

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        try:
            self._model = self.model_loader(
                self.model_name,
                self.local_files_only,
            )
        except Exception as error:  # 依赖缺失、离线或模型文件损坏均安全降级
            self._load_error = f"{type(error).__name__}: {error}"
            raise RuntimeError(self._load_error) from error
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        values = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return _to_rows(values)

    def _reference_rows(self, question_id: str, language: str):
        language = "en" if str(language).lower() in {"en", "english"} else "zh"
        cache_key = (question_id, language)
        if cache_key in self._reference_cache:
            return self._reference_cache[cache_key]

        rubric = self.rubrics.get(question_id)
        if rubric is None:
            raise KeyError(f"缺少语义样例库: {question_id}")
        groups = rubric[language]
        labels, texts = [], []
        for label, examples in groups.items():
            for example in examples:
                labels.append(label)
                texts.append(example)
        rows = self._encode(texts)
        if len(rows) != len(texts):
            raise ValueError("模型返回的参考向量数量不正确。")
        cached = (labels, texts, rows)
        self._reference_cache[cache_key] = cached
        return cached

    def score(self, question_id: str, answer: str, language: str = "zh") -> SemanticScore:
        if not str(answer or "").strip():
            return self._unavailable("回答为空。")
        try:
            labels, texts, rows = self._reference_rows(question_id, language)
            query = self._encode([str(answer).strip()])[0]
        except Exception as error:
            return self._unavailable(f"本地语义模型不可用：{error}")

        class_scores = {}
        best_examples = {}
        for label, text, row in zip(labels, texts, rows):
            similarity = _cosine(query, row)
            if label not in class_scores or similarity > class_scores[label]:
                class_scores[label] = similarity
                best_examples[label] = text

        ranked = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
        best_label, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0
        margin = best_score - second_score
        confidence = max(0.0, min(1.0, best_score))

        if best_score < self.similarity_threshold:
            decision = "uncertain"
            reason = "最高相似度低于自动判分阈值。"
        elif margin < self.margin_threshold:
            decision = "uncertain"
            reason = "最相近的两个类别差距过小，需要澄清或人工复核。"
        elif best_label == "partial":
            decision = "partial"
            reason = "回答方向基本安全，但行动信息不完整。"
        else:
            decision = best_label
            reason = "本地语义样例匹配达到阈值。"

        return SemanticScore(
            decision=decision,
            confidence=round(confidence, 4),
            margin=round(max(0.0, margin), 4),
            class_scores={key: round(max(0.0, min(1.0, value)), 4)
                          for key, value in class_scores.items()},
            matched_label=best_label,
            matched_example=best_examples.get(best_label),
            available=True,
            reason=reason,
            model_name=self.model_name,
        )

    def _unavailable(self, reason: str) -> SemanticScore:
        return SemanticScore(
            decision="uncertain",
            confidence=0.0,
            margin=0.0,
            class_scores={},
            matched_label=None,
            matched_example=None,
            available=False,
            reason=reason,
            model_name=self.model_name,
        )


@lru_cache(maxsize=4)
def get_semantic_scorer(
    model_name: str = DEFAULT_MODEL_NAME,
    similarity_threshold: float = 0.55,
    margin_threshold: float = 0.06,
    local_files_only: bool = False,
) -> LocalSemanticScorer:
    return LocalSemanticScorer(
        model_name,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        local_files_only=local_files_only,
    )


def score_semantic_answer(
    question_id: str,
    answer: str,
    language: str = "zh",
    *,
    enabled: bool = True,
    model_name: str = DEFAULT_MODEL_NAME,
    similarity_threshold: float = 0.55,
    margin_threshold: float = 0.06,
    local_files_only: bool = False,
) -> SemanticScore:
    if not enabled:
        return SemanticScore(
            decision="uncertain",
            confidence=0.0,
            margin=0.0,
            class_scores={},
            matched_label=None,
            matched_example=None,
            available=False,
            reason="本地语义评分已关闭。",
            model_name=model_name,
        )
    scorer = get_semantic_scorer(
        model_name,
        similarity_threshold,
        margin_threshold,
        local_files_only,
    )
    return scorer.score(question_id, answer, language)
