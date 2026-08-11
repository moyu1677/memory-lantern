"""Hard bounds that keep IndexTTS2 dialogue emotion subtle."""

from __future__ import annotations

from remote_tts.emotion_controls import (
    MAX_EMOTION_COMPONENTS,
    MAX_EMOTION_STRENGTH,
    MAX_EMOTION_VECTOR_SUM,
    clamp_emotion_control,
    clamp_text_emotion_strength,
    deterministic_speech_seed,
)


def test_remote_tts_clamps_stale_or_overstrong_emotion_controls():
    vector, strength = clamp_emotion_control([1.0] * 8, 0.95)

    assert vector is not None
    assert sum(vector) <= MAX_EMOTION_VECTOR_SUM + 1e-9
    assert vector[0] <= MAX_EMOTION_COMPONENTS[0]
    assert vector[6] <= MAX_EMOTION_COMPONENTS[6]
    assert strength == MAX_EMOTION_STRENGTH


def test_remote_tts_preserves_already_subtle_emotion_controls():
    requested = [0.015, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30]
    vector, strength = clamp_emotion_control(requested, 0.10)

    assert vector == requested
    assert strength == 0.10


def test_text_emotion_uses_its_own_documented_natural_range():
    assert clamp_text_emotion_strength(0.55) == 0.55
    assert clamp_text_emotion_strength(0.95) == 0.60


def test_speech_seed_is_stable_for_the_same_voice_text_and_control():
    first = deterministic_speech_seed("reference.wav", "慢慢聊", [0.015] + [0.0] * 7, 0.1)
    second = deterministic_speech_seed("reference.wav", "慢慢聊", [0.015] + [0.0] * 7, 0.1)
    changed_text = deterministic_speech_seed("reference.wav", "明天再聊", [0.015] + [0.0] * 7, 0.1)

    assert first == second
    assert first != changed_text
