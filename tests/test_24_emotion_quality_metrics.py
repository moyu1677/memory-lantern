"""Unit tests for objective SER report helpers (no model download required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "remote_tts" / "evaluate_emotion_quality.py"
SPEC = importlib.util.spec_from_file_location("emotion_quality_metrics", MODULE_PATH)
assert SPEC and SPEC.loader
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def emotion_sample(predicted: str, target_probability: float = 0.8):
    probabilities = {label: 0.0 for label in METRICS.EMOTION_LABELS}
    probabilities[predicted] = target_probability
    return {"emotion": {"predicted_label": predicted, "probabilities": probabilities}}


def test_normalise_bilingual_emotion_labels():
    assert METRICS.normalise_emotion_label("开心/happy") == "happy"
    assert METRICS.normalise_emotion_label("<unk>") == "unknown"


def test_target_parser_rejects_unknown_categories():
    label, category = METRICS.parse_target_argument("bright_01=happy")
    assert (label, category) == ("bright_01", "happy")

    try:
        METRICS.parse_target_argument("comfort_01=calm")
    except Exception as exc:  # argparse.ArgumentTypeError
        assert "target must" in str(exc)
    else:
        raise AssertionError("unsupported SER target must be rejected")


def test_balanced_summary_uses_uar_and_macro_f1():
    samples = {
        "a": emotion_sample("happy"),
        "b": emotion_sample("happy"),
        "c": emotion_sample("neutral"),
        "d": emotion_sample("sad"),
    }
    summary = METRICS.classification_summary(
        samples,
        {"a": "happy", "b": "neutral", "c": "neutral", "d": "sad"},
    )

    assert summary is not None
    assert summary["count"] == 4
    assert summary["top1_accuracy"] == 0.75
    assert summary["uar"] == 0.8333
    assert summary["macro_f1"] == 0.7778
