"""Dependency-free guardrails for restrained IndexTTS2 prosody."""

from __future__ import annotations

import hashlib

# The reference recording must remain the dominant speaker condition.  These
# bounds also protect callers that have not yet restarted an older web process.
MAX_EMOTION_STRENGTH = 0.16
# IndexTTS2 documents 0.6 or lower as the natural range for its learned text
# emotion controls.  This is separate from raw hand-written vectors: the
# service first converts the instruction into IndexTTS2's canonical vector.
MAX_TEXT_EMOTION_STRENGTH = 0.60
MAX_EMOTION_VECTOR_SUM = 0.32
# IndexTTS2 order: joy, anger, sadness, fear, disgust, melancholy, surprise,
# calm.  Joy and surprise receive the tightest bounds because they are most
# likely to produce an unnatural pitch jump in long-running voice dialogue.
MAX_EMOTION_COMPONENTS = (0.015, 0.02, 0.04, 0.02, 0.02, 0.05, 0.0, 0.30)


def clamp_emotion_control(
    emotion_vector: list[float] | None, emotion_strength: float
) -> tuple[list[float] | None, float]:
    """Bound a valid emotion control without removing its emotional direction."""

    strength = min(float(emotion_strength), MAX_EMOTION_STRENGTH)
    if emotion_vector is None:
        return None, strength

    vector = [
        min(float(value), component_limit)
        for value, component_limit in zip(emotion_vector, MAX_EMOTION_COMPONENTS)
    ]
    vector_sum = sum(vector)
    if vector_sum > MAX_EMOTION_VECTOR_SUM:
        scale = MAX_EMOTION_VECTOR_SUM / vector_sum
        vector = [value * scale for value in vector]
    return vector, strength


def clamp_text_emotion_strength(emotion_strength: float) -> float:
    """Bound learned text-emotion mixing without applying vector limits."""

    return min(float(emotion_strength), MAX_TEXT_EMOTION_STRENGTH)


def deterministic_speech_seed(
    reference_name: str,
    text: str,
    emotion_vector: list[float] | None,
    emotion_strength: float,
    emotion_text: str | None = None,
) -> int:
    """Return a stable per-request seed without retaining any voice content."""

    controls = ",".join(f"{float(value):.6f}" for value in emotion_vector or [])
    payload = "\x1f".join(
        (
            str(reference_name),
            str(text),
            controls,
            f"{float(emotion_strength):.6f}",
            str(emotion_text or ""),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "big") % (
        2**31 - 1
    )
