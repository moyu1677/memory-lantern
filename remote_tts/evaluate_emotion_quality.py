"""Measure identity, prosody, and recognisable emotion for TTS regression checks.

Usage runs entirely on the inference host.  It never uploads audio or writes
copies of the authorised reference sample.

``speaker_cosine_to_reference`` (SS) measures identity only.  When an
emotion2vec+ model is supplied, this script also runs speech-emotion
recognition (SER) on every rendered file.  It reports the full probability
distribution and, for labelled test sets, UAR and macro-F1.

SER cannot decide whether a reply is *empathetic* in its dialogue context;
the classifier only recognises audible broad emotion.  In particular, its
taxonomy has no "calm" or "concerned" class.  For a companionship product,
those subtle styles should normally remain close to neutral and be guarded by
the supplied F0 / loudness statistics as well.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from pathlib import Path


EMOTION_LABELS = {
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
}
ACTIVE_EMOTIONS = {"angry", "fearful", "happy", "surprised"}
NEGATIVE_EMOTIONS = {"angry", "disgusted", "fearful", "sad"}


def parse_audio_argument(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    path = Path(raw_path if separator else label)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"audio does not exist: {path}")
    return (label if separator else path.stem), path


def parse_target_argument(value: str) -> tuple[str, str]:
    """Parse ``render_label=expected_ser_label`` for a balanced test set."""

    label, separator, target = value.partition("=")
    label = label.strip()
    target = target.strip().lower()
    if not separator or not label or target not in EMOTION_LABELS:
        choices = ", ".join(sorted(EMOTION_LABELS))
        raise argparse.ArgumentTypeError(
            f"target must be render_label=one_of({choices})"
        )
    return label, target


def load_embedder(checkpoint: Path):
    import torch
    from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus

    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model


def speaker_embedding(path: Path, model):
    import torch
    import torch.nn.functional as functional
    import torchaudio

    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    features = torchaudio.compliance.kaldi.fbank(
        waveform, num_mel_bins=80, dither=0, sample_frequency=16000
    )
    features = features - features.mean(dim=0, keepdim=True)
    with torch.inference_mode():
        embedding = model(features.unsqueeze(0))
    return functional.normalize(embedding.squeeze(0), dim=0)


def prosody_metrics(path: Path) -> dict[str, float | None]:
    import librosa
    import numpy as np

    samples, sample_rate = librosa.load(path, sr=16000, mono=True)
    if samples.size == 0:
        raise ValueError(f"empty audio: {path}")
    trimmed, _ = librosa.effects.trim(samples, top_db=40)
    if trimmed.size == 0:
        raise ValueError(f"silent audio: {path}")
    rms = librosa.feature.rms(y=trimmed)[0]
    voiced = librosa.yin(trimmed, fmin=65, fmax=400, sr=sample_rate)
    voiced = voiced[np.isfinite(voiced)]
    result: dict[str, float | None] = {
        "duration_s": round(float(trimmed.size / sample_rate), 3),
        "rms_db": round(float(20 * np.log10(max(np.mean(rms), 1e-8))), 2),
        "f0_median_hz": None,
        "f0_p10_hz": None,
        "f0_p90_hz": None,
    }
    if voiced.size:
        result.update(
            {
                "f0_median_hz": round(float(np.median(voiced)), 2),
                "f0_p10_hz": round(float(np.percentile(voiced, 10)), 2),
                "f0_p90_hz": round(float(np.percentile(voiced, 90)), 2),
            }
        )
    return result


def normalise_emotion_label(label: str) -> str:
    """Map emotion2vec+'s bilingual labels (``开心/happy``) to English keys."""

    canonical = str(label).split("/")[-1].strip().lower()
    return "unknown" if canonical == "<unk>" else canonical


def load_emotion_recogniser(model_path: Path, device: str):
    from funasr import AutoModel

    # FunASR emits progress and checkpoint notices to stdout.  Keep stdout
    # reserved for this script's final JSON report so it can be parsed in CI.
    with contextlib.redirect_stdout(sys.stderr):
        return AutoModel(
            model=str(model_path),
            device=device,
            disable_update=True,
        )


def emotion_metrics(path: Path, recogniser) -> dict[str, object]:
    """Return classifier probabilities plus transparent arousal proxies.

    The two proxies are deliberately labelled as model-derived.  They are
    useful for catching an over-excited render, but are not a replacement for
    a continuous valence/arousal model or a human listening study.
    """

    with tempfile.TemporaryDirectory(prefix="memory-lantern-ser-") as output_dir:
        with contextlib.redirect_stdout(sys.stderr):
            response = recogniser.generate(
                input=str(path),
                output_dir=output_dir,
                granularity="utterance",
                extract_embedding=False,
            )
    if not response or not isinstance(response[0], dict):
        raise ValueError("emotion recogniser returned no utterance result")

    result = response[0]
    labels = result.get("labels") or []
    scores = result.get("scores") or []
    if len(labels) != len(scores) or not labels:
        raise ValueError("emotion recogniser returned invalid label scores")
    probabilities = {
        normalise_emotion_label(label): round(float(score), 6)
        for label, score in zip(labels, scores)
    }
    predicted_label = max(probabilities, key=probabilities.get)
    return {
        "predicted_label": predicted_label,
        "predicted_probability": probabilities[predicted_label],
        "probabilities": probabilities,
        "model_derived_activation": round(
            sum(probabilities.get(label, 0.0) for label in ACTIVE_EMOTIONS), 6
        ),
        "model_derived_negative_affect": round(
            sum(probabilities.get(label, 0.0) for label in NEGATIVE_EMOTIONS), 6
        ),
        "model_derived_neutrality": probabilities.get("neutral", 0.0),
    }


def classification_summary(samples: dict[str, dict[str, object]], targets: dict[str, str]):
    """Compute standard balanced-class metrics for samples with known targets."""

    evaluated = [
        (label, target, str(samples[label]["emotion"]["predicted_label"]))
        for label, target in targets.items()
        if label in samples and "emotion" in samples[label]
    ]
    if not evaluated:
        return None

    target_labels = sorted({target for _, target, _ in evaluated})
    labels = sorted(set(target_labels) | {predicted for _, _, predicted in evaluated})
    class_metrics: dict[str, dict[str, float | int]] = {}
    for category in labels:
        true_positive = sum(
            target == category and predicted == category
            for _, target, predicted in evaluated
        )
        false_positive = sum(
            target != category and predicted == category
            for _, target, predicted in evaluated
        )
        false_negative = sum(
            target == category and predicted != category
            for _, target, predicted in evaluated
        )
        support = true_positive + false_negative
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_metrics[category] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    target_probabilities = [
        float(samples[label]["emotion"]["probabilities"].get(target, 0.0))
        for label, target, _ in evaluated
    ]
    correct = sum(target == predicted for _, target, predicted in evaluated)
    uar = sum(class_metrics[label]["recall"] for label in target_labels) / len(target_labels)
    macro_f1 = sum(class_metrics[label]["f1"] for label in target_labels) / len(target_labels)
    return {
        "count": len(evaluated),
        "top1_accuracy": round(correct / len(evaluated), 4),
        "uar": round(uar, 4),
        "macro_f1": round(macro_f1, 4),
        "mean_target_probability": round(sum(target_probabilities) / len(target_probabilities), 4),
        "per_class": {label: class_metrics[label] for label in target_labels},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--campplus-checkpoint", required=True, type=Path)
    parser.add_argument("--audio", action="append", default=[], type=parse_audio_argument)
    parser.add_argument(
        "--emotion-model",
        type=Path,
        help="Local emotion2vec+ model directory. Enables objective SER metrics.",
    )
    parser.add_argument(
        "--emotion-device",
        default="cpu",
        help="Device for SER (default: cpu, so inference GPU is not disturbed).",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        type=parse_target_argument,
        help="Known SER target, e.g. bright_01=happy. Repeat for a balanced set.",
    )
    arguments = parser.parse_args()
    if not arguments.reference.is_file():
        parser.error(f"reference does not exist: {arguments.reference}")
    if not arguments.campplus_checkpoint.is_file():
        parser.error("campplus checkpoint does not exist")
    if arguments.emotion_model is not None and not arguments.emotion_model.is_dir():
        parser.error("emotion model directory does not exist")
    targets = dict(arguments.target)
    audio_labels = {label for label, _ in arguments.audio}
    unknown_targets = sorted(set(targets) - audio_labels)
    if unknown_targets:
        parser.error(f"targets have no matching --audio: {', '.join(unknown_targets)}")
    if targets and arguments.emotion_model is None:
        parser.error("--target requires --emotion-model")

    import torch.nn.functional as functional

    embedder = load_embedder(arguments.campplus_checkpoint)
    emotion_recogniser = None
    if arguments.emotion_model is not None:
        emotion_recogniser = load_emotion_recogniser(
            arguments.emotion_model, arguments.emotion_device
        )
    reference_embedding = speaker_embedding(arguments.reference, embedder)
    report = {
        "reference": prosody_metrics(arguments.reference),
        "samples": {},
        "emotion_evaluation": {
            "model": str(arguments.emotion_model) if arguments.emotion_model else None,
            "device": arguments.emotion_device if arguments.emotion_model else None,
            "note": (
                "SER measures audible broad emotion only; it does not measure "
                "dialogue empathy or replace listening tests."
            ) if arguments.emotion_model else None,
        },
    }
    for label, path in arguments.audio:
        cosine = float(functional.cosine_similarity(
            reference_embedding, speaker_embedding(path, embedder), dim=0
        ))
        sample = {
            "speaker_cosine_to_reference": round(cosine, 4),
            **prosody_metrics(path),
        }
        if emotion_recogniser is not None:
            sample["emotion"] = emotion_metrics(path, emotion_recogniser)
        report["samples"][label] = sample
    if emotion_recogniser is not None:
        report["emotion_evaluation"]["classification_summary"] = classification_summary(
            report["samples"], targets
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
