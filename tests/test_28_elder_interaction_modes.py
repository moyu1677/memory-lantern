"""The elder portal keeps its companion flows while presenting a real test UI."""


def test_elder_portal_exposes_assessment_controls_and_two_voice_interaction_modes(
    client, login_as, user_factory
):
    elder = user_factory(account_type="elder")
    login_as(client, elder["id"])

    response = client.get("/assistant")
    page = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="test-question-stage"' in page
    assert 'id="test-narration-toggle"' in page
    assert 'id="test-narration-replay"' in page
    assert "toggleTestNarration()" in page
    assert "replayTestNarration()" in page
    assert 'id="test-answer-preview"' in page
    assert 'id="test-answer-preview-text"' in page
    assert 'id="test-assessment-status"' in page
    assert 'id="test-drawing-mount"' in page
    assert "updateTestAnswerPreview" in page
    assert "updateTestStatus" in page
    assert "'/test/start'" in page
    assert "'/test/answer'" in page
    assert "'/test/drawing-answer'" in page
    assert 'id="review-mode-btn"' in page
    assert 'id="live-mode-btn"' in page
    assert "selectInteractionMode('review')" in page
    assert "selectInteractionMode('live')" in page
    assert 'id="live-call-btn"' in page
    assert "toggleLiveConversation()" in page
    assert "handleRecognisedText" in page
    assert "autoSend: false" in page
    assert "resumeLive" in page


def test_elder_portal_english_translates_the_new_interaction_controls(
    client, login_as, user_factory
):
    elder = user_factory(account_type="elder")
    login_as(client, elder["id"])
    with client.session_transaction() as session:
        session["ui_language"] = "en"

    page = client.get("/assistant").get_data(as_text=True)

    for text in (
        "Cognitive task",
        "Pause narration",
        "Replay from the beginning",
        "I HEARD:",
        "Waiting for your answer",
        "Record to text",
        "Continuous voice conversation",
    ):
        assert text in page
