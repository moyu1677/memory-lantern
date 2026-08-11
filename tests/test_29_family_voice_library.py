"""Coverage for the family-only, reviewed voice-registration flow."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

from tests.helpers import api_headers


def _bind(app_module, family, elder):
    with app_module.app.app_context():
        app_module.database.bind_elder(family["id"], elder["binding_code"])


def test_family_voice_library_has_two_input_methods_and_no_elder_handoff(
    app_module, client, login_as, user_factory, monkeypatch
):
    elder = user_factory(account_type="elder", display_name="王奶奶")
    family = user_factory(account_type="family", display_name="王小兰")
    _bind(app_module, family, elder)
    monkeypatch.setattr(app_module, "voice_service", SimpleNamespace(configured=True))
    monkeypatch.setattr(app_module, "asr_service", SimpleNamespace(configured=True))

    login_as(client, family["id"])
    page = client.get("/voice-profiles").get_data(as_text=True)

    assert "直接录音" in page
    assert "上传音频文件" in page
    assert "识别这段声音并核对内容" in page
    assert "由老人账号使用" in page
    assert 'id="voice-reference-audio"' in page
    assert 'href="/elder"' not in page
    assert 'href="/assistant"' not in page


def test_bound_family_can_review_a_voice_sample_but_cannot_target_unbound_elder(
    app_module, client, login_as, user_factory, monkeypatch
):
    elder = user_factory(account_type="elder")
    unbound_elder = user_factory(account_type="elder")
    family = user_factory(account_type="family")
    _bind(app_module, family, elder)
    monkeypatch.setattr(
        app_module,
        "asr_service",
        SimpleNamespace(transcribe=lambda *_args, **_kwargs: "今天我想陪您聊聊天。"),
    )
    csrf = login_as(client, family["id"])

    response = client.post(
        "/voice-profiles/transcribe",
        data={
            "elder_id": str(elder["id"]),
            "reference_audio": (BytesIO(b"audio"), "family-voice.webm"),
        },
        content_type="multipart/form-data",
        headers=api_headers(csrf),
    )
    assert response.status_code == 200
    assert response.get_json() == {
        "text": "今天我想陪您聊聊天。",
        "language": "browser",
        "notice": "转写只用于本页核对，不会保存到本应用。",
    }

    denied = client.post(
        "/voice-profiles/transcribe",
        data={
            "elder_id": str(unbound_elder["id"]),
            "reference_audio": (BytesIO(b"audio"), "family-voice.webm"),
        },
        content_type="multipart/form-data",
        headers=api_headers(csrf),
    )
    assert denied.status_code == 403


def test_elder_chooses_the_authorised_voice_from_their_own_library(
    app_module, client, login_as, user_factory
):
    elder = user_factory(account_type="elder")
    family = user_factory(account_type="family", display_name="小兰")
    _bind(app_module, family, elder)
    with app_module.app.app_context():
        first = app_module.database.create_voice_profile(
            elder["id"], family["id"], "小兰的声音", "voice-test-003"
        )
        second = app_module.database.create_voice_profile(
            elder["id"], family["id"], "小兰的问候", "voice-test-004"
        )

    csrf = login_as(client, elder["id"])
    page = client.get("/elder/voice-library").get_data(as_text=True)
    assert "选择您想听到的家人声音" in page
    assert "小兰的声音" in page and "小兰的问候" in page
    selected = client.post(
        "/elder/voice-library",
        data={"_csrf_token": csrf, "profile_id": str(second)},
    )
    assert selected.status_code == 302
    with app_module.app.app_context():
        assert app_module.database.get_active_voice_profile(elder["id"])["id"] == second
        assert first != second
