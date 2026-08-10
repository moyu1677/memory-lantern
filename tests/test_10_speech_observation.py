"""第一阶段语音行为观察的自动化测试。"""

from array import array
from io import BytesIO
from types import SimpleNamespace

from tests.helpers import api_headers, correct_answer, drawing_answer_payload
from speech_analysis import analyze_pcm_bytes, build_speech_summary


def synthetic_pcm():
    """0.6秒静音、0.8秒发声、0.6秒停顿、0.7秒发声。"""
    sample_rate = 16_000
    values = array("h")
    values.extend([0] * round(sample_rate * 0.6))
    values.extend([5000, -5000] * round(sample_rate * 0.8 / 2))
    values.extend([0] * round(sample_rate * 0.6))
    values.extend([5000, -5000] * round(sample_rate * 0.7 / 2))
    return values.tobytes()


def test_pause_analysis_extracts_descriptive_features():
    observation = analyze_pcm_bytes(synthetic_pcm(), transcript="今天星期四")

    assert observation["valid"] is True
    assert 540 <= observation["initial_silence_ms"] <= 660
    assert observation["pause_count"] == 1
    assert 540 <= observation["pause_total_ms"] <= 660
    assert observation["speech_duration_ms"] >= 1400
    assert observation["speaking_rate_cpm"] is not None


def test_summary_never_creates_a_diagnostic_label():
    observation = analyze_pcm_bytes(synthetic_pcm(), transcript="今天星期四")
    observation.update({
        "filler_count": 2,
        "repetition_count": 1,
        "self_correction_count": 1,
    })
    summary = build_speech_summary([observation, observation])

    assert summary["voice_answer_count"] == 2
    assert summary["valid_sample_count"] == 2
    assert summary["average_pause_count"] == 1.0
    assert summary["total_filler_count"] == 4
    assert summary["total_repetition_count"] == 2
    assert summary["total_self_correction_count"] == 2
    assert "risk" not in summary
    assert "diagnosis" not in summary


def test_browser_analysis_attaches_features_to_current_question(
    app_module,
    client,
    user_factory,
    login_as,
    monkeypatch,
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])
    start = client.post(
        "/test/start",
        json={"input_language": "browser"},
        headers=api_headers(token),
    )
    assert start.status_code == 200

    def fake_conversion(_input_path, pcm_path):
        pcm_path.write_bytes(synthetic_pcm())
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(app_module, "convert_audio_to_pcm", fake_conversion)
    response = client.post(
        "/speech/analyze",
        data={
            "transcript": "今天星期四",
            "audio": (BytesIO(synthetic_pcm()), "mandarin.webm"),
        },
        headers=api_headers(token),
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.get_json()["speech_observation"]["pause_count"] == 1

    with client.session_transaction() as browser_session:
        test_id = browser_session["test_id"]
    state = app_module.TEST_SESSIONS[test_id]
    assert state["pending_speech_observation"]["question_id"] == state["questions"][0]["id"]


def test_completed_assessment_persists_summary_without_audio_or_answer(
    app_module,
    client,
    user_factory,
    login_as,
):
    elder = user_factory(account_type="elder")
    token = login_as(client, elder["id"])
    client.post(
        "/test/start",
        json={"input_language": "browser"},
        headers=api_headers(token),
    )
    with client.session_transaction() as browser_session:
        test_id = browser_session["test_id"]

    for index in range(10):
        state = app_module.TEST_SESSIONS[test_id]
        question = state["questions"][state["question_index"]]
        if index == 0:
            observation = analyze_pcm_bytes(
                synthetic_pcm(),
                transcript="不会被写入数据库的回答",
            )
            app_module.TEST_SESSIONS[test_id]["pending_speech_observation"] = {
                **observation,
                "question_id": question["id"],
                "language": "browser",
            }
        response = client.post(
            "/test/answer",
            json={"message": correct_answer(question)},
            headers=api_headers(token),
        )
        assert response.status_code == 200

    response = client.post(
        "/test/drawing-answer",
        json=drawing_answer_payload(),
        headers=api_headers(token),
    )
    assert response.status_code == 201

    with app_module.app.app_context():
        assessment = app_module.database.list_assessments(elder["id"])[0]
        metrics_text = assessment["speech_metrics"]
        assessment_id = assessment["id"]
        assert '"voice_answer_count": 1' in metrics_text
        assert "不会被写入数据库的回答" not in metrics_text
    detail = client.get(f"/history/{assessment_id}")
    report = client.get(f"/history/{assessment_id}/report")
    assert "语音表现观察" in detail.get_data(as_text=True)
    assert "语音表现观察" in report.get_data(as_text=True)
