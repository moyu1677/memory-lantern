from __future__ import annotations

from io import BytesIO

import pytest

from asr_service import AsrServiceError, MAX_AUDIO_BYTES, RemoteAsrService


def test_asr_client_rejects_oversized_audio_before_network_call(monkeypatch):
    service = RemoteAsrService("http://127.0.0.1:18000", "test-key")
    monkeypatch.setattr("asr_service.requests.post", lambda *args, **kwargs: pytest.fail("network call"))

    with pytest.raises(AsrServiceError, match="20 MB"):
        service.transcribe(BytesIO(b"x" * (MAX_AUDIO_BYTES + 1)), "recording.wav", "audio/wav")


def test_asr_client_sanitizes_filename_and_returns_text(monkeypatch):
    calls = []

    class Response:
        ok = True

        @staticmethod
        def json():
            return {"text": "识别成功"}

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr("asr_service.requests.post", fake_post)
    service = RemoteAsrService("http://127.0.0.1:18000", "test-key")

    assert service.transcribe(BytesIO(b"audio"), "../../recording.WAV", "audio/wav") == "识别成功"
    assert calls[0][1]["files"]["audio"][0] == "recording.WAV"
