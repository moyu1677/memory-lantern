"""口语清洗与规则评分接入测试。"""

from answer_processing import process_spoken_answer
from tests.helpers import api_headers


def test_chinese_fillers_are_removed_without_losing_memory_words():
    result = process_spoken_answer("嗯……苹果，呃，钥匙，还有河流", language="zh")

    assert "嗯" not in result.final_answer
    assert "呃" not in result.final_answer
    assert all(word in result.final_answer for word in ("苹果", "钥匙", "河流"))
    assert result.filler_count == 2


def test_last_self_correction_becomes_the_scored_answer():
    result = process_spoken_answer("嗯，我觉得是八，不对，应该是七。", language="zh")

    assert result.final_answer == "七"
    assert result.self_correction_count >= 1


def test_repeated_words_are_collapsed_but_negation_is_preserved():
    repeated = process_spoken_answer("苹果，苹果，钥匙", language="zh")
    judgement = process_spoken_answer("不能开门，要先联系家人", language="zh")

    assert repeated.final_answer.count("苹果") == 1
    assert repeated.repetition_count >= 1
    assert "不能开门" in judgement.final_answer


def test_english_fillers_and_correction_are_supported():
    result = process_spoken_answer(
        "Um, I think it is eight, sorry, I mean seven.",
        language="en",
    )

    assert result.final_answer.lower() == "seven"
    assert result.filler_count == 1
    assert result.self_correction_count >= 1


def test_answer_endpoint_scores_the_cleaned_final_answer(
    app_module,
    client,
    user_factory,
    login_as,
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])
    start = client.post(
        "/test/start",
        json={"input_language": "browser"},
        headers=api_headers(token),
    )
    assert start.status_code == 200

    with client.session_transaction() as browser_session:
        test_id = browser_session["test_id"]
    state = app_module.TEST_SESSIONS[test_id]
    state["questions"][0] = {
        "id": "cleaner_integration",
        "domain": "计算能力",
        "domain_label": "计算能力",
        "prompt": "八减一等于多少？",
        "type": "exact",
        "answers": ("七", "7"),
        "points": 1,
    }

    response = client.post(
        "/test/answer",
        json={"message": "嗯，我觉得是八，不对，应该是七"},
        headers=api_headers(token),
    )
    assert response.status_code == 200
    assert state["score"] == 1
    assert state["scored_answers"][0]["filler_count"] == 1
    assert state["scored_answers"][0]["self_correction_count"] >= 1
