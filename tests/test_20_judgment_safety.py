"""日常判断题的否定、危险行为、矛盾和追问测试。"""

import pytest

from judgment_scoring import analyze_judgment_answer
from tests.helpers import api_headers


ZH_CASES = (
    ("judgment_red_light", "停下来等绿灯", "继续通行"),
    ("judgment_smell_gas", "关闭火源并开窗求助", "继续开火"),
    ("judgment_unknown_medicine", "先询问医生", "直接服用"),
    ("judgment_phone_scam", "联系家人先核实", "马上转账"),
    ("judgment_dizzy", "先坐下并叫人帮助", "继续独自行走"),
    ("judgment_hot_water", "先放凉再试温", "马上大口喝"),
    ("judgment_lost", "留在安全地点联系家人", "继续乱走"),
    ("judgment_wet_floor", "放慢脚步并绕开", "快速走过去"),
    ("judgment_door", "先核实身份并联系家人", "直接开门"),
    ("judgment_power", "远离并请人处理", "用手触碰"),
)


@pytest.mark.parametrize("question_id,safe_answer,dangerous_answer", ZH_CASES)
def test_each_judgment_rule_separates_safe_and_dangerous_actions(
    question_id,
    safe_answer,
    dangerous_answer,
):
    assert analyze_judgment_answer(question_id, safe_answer).decision == "correct"
    assert analyze_judgment_answer(question_id, dangerous_answer).decision == "incorrect"


def test_negation_reverses_the_meaning_of_an_action():
    safe = analyze_judgment_answer("judgment_red_light", "不能继续通行，应该等待")
    unsafe = analyze_judgment_answer("judgment_red_light", "不应该停下，要继续通行")

    assert safe.decision == "correct"
    assert "继续通行" in safe.negated_dangerous_evidence
    assert unsafe.decision == "incorrect"
    assert "停下" in unsafe.negated_safe_evidence


def test_conflict_and_uncertainty_are_not_forced_into_a_score():
    conflict = analyze_judgment_answer("judgment_red_light", "停下等待，也可以继续通行")
    uncertain = analyze_judgment_answer("judgment_red_light", "我不知道，看情况吧")

    assert conflict.decision == "conflict"
    assert uncertain.decision == "uncertain"


def test_english_negation_and_dangerous_actions_are_supported():
    safe = analyze_judgment_answer(
        "judgment_door",
        "I should not open the door; I would contact family.",
        language="en",
    )
    unsafe = analyze_judgment_answer(
        "judgment_door",
        "I would open the door and let them in.",
        language="en",
    )

    assert safe.decision == "correct"
    assert unsafe.decision == "incorrect"


def _replace_current_question_with_red_light(state):
    state["questions"][0] = {
        "id": "judgment_red_light",
        "domain": "日常判断",
        "domain_label": "日常判断",
        "prompt": "红灯亮起时应该怎么办？",
        "type": "judgment",
        "answers": ("停下等待", "停下"),
        "points": 1,
    }


def test_api_asks_once_for_a_neutral_clarification_then_scores(
    app_module,
    client,
    user_factory,
    login_as,
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])
    client.post("/test/start", json={"input_language": "browser"}, headers=api_headers(token))
    with client.session_transaction() as browser_session:
        test_id = browser_session["test_id"]
    state = app_module.TEST_SESSIONS[test_id]
    _replace_current_question_with_red_light(state)

    first = client.post(
        "/test/answer",
        json={"message": "停下等待或者继续通行都可以"},
        headers=api_headers(token),
    )
    assert first.status_code == 200
    assert first.get_json()["clarification_required"] is True
    assert state["question_index"] == 0
    assert state["score"] == 0

    second = client.post(
        "/test/answer",
        json={"message": "应该停下等待"},
        headers=api_headers(token),
    )
    assert second.status_code == 200
    assert state["question_index"] == 1
    assert state["score"] == 1
    assert state["scored_answers"][0]["judgment_decision"] == "correct"


def test_second_unclear_answer_advances_with_zero_and_audit_only(
    app_module,
    client,
    user_factory,
    login_as,
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])
    client.post("/test/start", json={"input_language": "browser"}, headers=api_headers(token))
    with client.session_transaction() as browser_session:
        test_id = browser_session["test_id"]
    state = app_module.TEST_SESSIONS[test_id]
    _replace_current_question_with_red_light(state)

    for answer in ("不知道", "还是不知道"):
        response = client.post(
            "/test/answer",
            json={"message": answer},
            headers=api_headers(token),
        )
        assert response.status_code == 200

    assert state["question_index"] == 1
    assert state["score"] == 0
    assert state["scored_answers"][0]["judgment_decision"] == "uncertain"
    assert "不知道" not in str(state["scored_answers"][0])
