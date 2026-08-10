"""中英文语义样例评分与答题路由集成测试。"""

import json

import pytest

from memory_lantern.assessment.semantic_scoring import (
    LocalSemanticScorer,
    SemanticScore,
)
from tests.helpers import api_headers


RUBRICS = {
    "demo": {
        "zh": {
            "correct": ("安全做法",),
            "partial": ("部分做法",),
            "incorrect": ("没有回答",),
            "dangerous": ("危险做法",),
        },
        "en": {
            "correct": ("safe action",),
            "partial": ("partial action",),
            "incorrect": ("no answer",),
            "dangerous": ("dangerous action",),
        },
    }
}


class FakeModel:
    vectors = {
        "安全做法": [1.0, 0.0, 0.0, 0.0],
        "部分做法": [0.0, 1.0, 0.0, 0.0],
        "没有回答": [0.0, 0.0, 1.0, 0.0],
        "危险做法": [0.0, 0.0, 0.0, 1.0],
        "换一种说法的安全答案": [0.98, 0.05, 0.0, 0.0],
        "模糊答案": [0.70, 0.70, 0.0, 0.0],
        "safe action": [1.0, 0.0, 0.0, 0.0],
        "partial action": [0.0, 1.0, 0.0, 0.0],
        "no answer": [0.0, 0.0, 1.0, 0.0],
        "dangerous action": [0.0, 0.0, 0.0, 1.0],
        "an English paraphrase": [0.96, 0.04, 0.0, 0.0],
    }

    def encode(self, texts, **_kwargs):
        return [self.vectors[text] for text in texts]


def make_scorer():
    return LocalSemanticScorer(
        "fake-model",
        model_loader=lambda _name, _local_only: FakeModel(),
        rubrics=RUBRICS,
        similarity_threshold=0.55,
        margin_threshold=0.06,
    )


def test_local_semantic_model_matches_chinese_paraphrase():
    result = make_scorer().score("demo", "换一种说法的安全答案", "zh")

    assert result.decision == "correct"
    assert result.decisive is True
    assert result.matched_example == "安全做法"
    assert result.confidence > 0.9


def test_local_semantic_model_uses_independent_english_examples():
    result = make_scorer().score("demo", "an English paraphrase", "en")

    assert result.decision == "correct"
    assert result.matched_example == "safe action"


def test_close_categories_are_not_forced_into_a_score():
    result = make_scorer().score("demo", "模糊答案", "zh")

    assert result.decision == "uncertain"
    assert result.margin < 0.06


def test_missing_dependency_or_model_safely_degrades():
    scorer = LocalSemanticScorer(
        "missing-model",
        model_loader=lambda *_args: (_ for _ in ()).throw(ImportError("missing")),
        rubrics=RUBRICS,
    )

    result = scorer.score("demo", "换一种说法的安全答案")
    assert result.available is False
    assert result.decision == "uncertain"


def _replace_current_question(state, *, english=False):
    state["questions"][0] = {
        "id": "judgment_phone_scam",
        "domain": "日常判断",
        "domain_label": "Everyday judgement" if english else "日常判断",
        "prompt": "What would you do?" if english else "接到可疑转账电话怎么办？",
        "type": "judgment",
        "answers": (),
        "points": 1,
    }
    state["ui_language"] = "en" if english else "zh"


@pytest.mark.parametrize(
    "english,answer,expected_language",
    (
        (False, "我会先找可信的人一起确认这件事", "zh"),
        (True, "I would pause and check with someone I trust.", "en"),
    ),
)
def test_route_uses_semantic_model_for_unmatched_chinese_and_english_answers(
    app_module,
    client,
    user_factory,
    login_as,
    monkeypatch,
    english,
    answer,
    expected_language,
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])
    client.post("/test/start", json={"input_language": "browser"}, headers=api_headers(token))
    with client.session_transaction() as browser_session:
        state = app_module.TEST_SESSIONS[browser_session["test_id"]]
    _replace_current_question(state, english=english)

    calls = []

    def fake_semantic(question_id, text, language="zh", **_kwargs):
        calls.append((question_id, text, language))
        return SemanticScore(
            decision="correct",
            confidence=0.88,
            margin=0.21,
            class_scores={"correct": 0.88, "partial": 0.67},
            matched_label="correct",
            matched_example="static reviewed example",
            available=True,
            reason="test",
            model_name="fake-model",
        )

    monkeypatch.setitem(app_module.app.config, "SEMANTIC_SCORING_ENABLED", True)
    monkeypatch.setattr(app_module, "score_semantic_answer", fake_semantic)
    response = client.post(
        "/test/answer",
        json={"message": answer},
        headers=api_headers(token),
    )

    assert response.status_code == 200
    assert state["score"] == 1
    assert state["question_index"] == 1
    assert state["scored_answers"][0]["judgment_scoring_source"] == "semantic_model"
    assert state["scored_answers"][0]["resolved_judgment_decision"] == "correct"
    assert calls[0][2] == expected_language


def test_scoring_audit_can_be_saved_without_raw_answer(
    app_module,
    user_factory,
):
    elder = user_factory(account_type="elder")
    audit = {
        "scored_answers": [{
            "question_id": "judgment_phone_scam",
            "semantic_decision": "correct",
            "semantic_confidence": 0.88,
        }]
    }
    with app_module.app.app_context():
        assessment_id = app_module.database.save_assessment(
            test_session_id="semantic-audit-test",
            user_id=elder["id"],
            score=1,
            maximum_score=1,
            percentage=100,
            level="test",
            domain_scores={"日常判断": 1},
            input_language="browser",
            scoring_audit=audit,
        )
        row = app_module.database.get_assessment(assessment_id)

    assert json.loads(row["scoring_audit"]) == audit
