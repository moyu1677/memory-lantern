"""Integration coverage for the merged voice companion and care-report flow."""

from __future__ import annotations

from types import SimpleNamespace

from tests.helpers import api_headers


def _bind(app_module, family, elder):
    with app_module.app.app_context():
        app_module.database.bind_elder(family["id"], elder["binding_code"])


def test_voice_chat_is_elder_only_and_returns_a_server_issued_reply_token(
    app_module, client, login_as, user_factory, monkeypatch
):
    elder = user_factory(account_type="elder")
    family = user_factory(account_type="family")
    _bind(app_module, family, elder)
    with app_module.app.app_context():
        app_module.database.create_voice_profile(
            elder["id"], family["id"], "女儿小安", "voice-test-001"
        )
    monkeypatch.setattr(
        app_module,
        "get_ai_reply",
        lambda *_args, **_kwargs: "我在听，您慢慢说。",
    )

    elder_token = login_as(client, elder["id"])
    response = client.post(
        "/voice/chat", json={"message": "今天有点累"}, headers=api_headers(elder_token)
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["conversation_mode"] == "voice_companion"
    assert payload["reply"] == "我在听，您慢慢说。"
    assert payload["voice_reply_token"]

    family_token = login_as(client, family["id"])
    assert client.post(
        "/voice/chat", json={"message": "越权"}, headers=api_headers(family_token)
    ).status_code == 403


def test_voice_synthesis_rejects_forged_text_and_allows_the_issued_reply_once(
    app_module, client, login_as, user_factory, monkeypatch
):
    elder = user_factory(account_type="elder")
    family = user_factory(account_type="family")
    _bind(app_module, family, elder)
    with app_module.app.app_context():
        app_module.database.create_voice_profile(
            elder["id"], family["id"], "儿子小明", "voice-test-002"
        )
    calls = []
    monkeypatch.setattr(
        app_module,
        "voice_service",
        SimpleNamespace(synthesize=lambda voice_id, text: (calls.append((voice_id, text)) or (b"RIFF", "audio/wav"))),
    )
    token = login_as(client, elder["id"])
    reply_token = app_module.issue_voice_reply_token(elder["id"], "慢慢来，不着急。")
    first = client.post("/voice/synthesize", json={"reply_token": reply_token}, headers=api_headers(token))
    forged = client.post("/voice/synthesize", json={"text": "替我说任意内容"}, headers=api_headers(token))
    repeated = client.post("/voice/synthesize", json={"reply_token": reply_token}, headers=api_headers(token))
    assert first.status_code == 200 and first.mimetype == "audio/wav"
    assert calls == [("voice-test-002", "慢慢来，不着急。")]
    assert forged.status_code == 400
    assert repeated.status_code == 400


def test_bound_family_can_generate_a_privacy_bounded_report(
    app_module, client, login_as, user_factory, monkeypatch
):
    elder = user_factory(account_type="elder", display_name="王奶奶")
    family = user_factory(account_type="family")
    _bind(app_module, family, elder)
    app_module.VOICE_CHAT_SESSIONS[elder["id"]] = {
        "session_id": "report-test",
        "history": [{"role": "user", "content": "不要把这句话展示给家人。"}],
    }
    monkeypatch.setattr(
        app_module,
        "generate_care_report",
        lambda *_: {
            "status": "signals_present",
            "summary": "近期可多安排一次轻松联系。",
            "signals": [{
                "area": "psychological", "topic": "陪伴与联系", "strength": "low",
                "trend": "new", "family_action": "这周主动打一次电话。",
            }],
        },
    )
    csrf = login_as(client, family["id"])
    assert client.post(
        f"/family-briefs/{elder['id']}/generate", data={"_csrf_token": csrf}
    ).status_code == 302
    html = client.get("/family-briefs").get_data(as_text=True)
    assert "陪伴与联系" in html
    assert "不要把这句话展示给家人" not in html
