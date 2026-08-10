"""从短音频中提取可解释的语音行为特征。

本模块只做语音活动与停顿统计，不判断认知障碍，也不保存原始录音。
输入是 ffmpeg 转换后的 16 kHz、单声道、16 位小端 PCM 数据。
"""

from __future__ import annotations

from array import array
from pathlib import Path
import math
import re
import sys


ANALYSIS_VERSION = "rms-vad-v1"
SAMPLE_RATE = 16_000
FRAME_MS = 30
MIN_PAUSE_MS = 500


def _percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return float(ordered[position])


def _round_or_none(value, digits=1):
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def analyze_pcm_bytes(pcm_bytes, transcript=""):
    """返回停顿、语音时长和近似语速等描述性特征。"""
    usable_length = len(pcm_bytes) - (len(pcm_bytes) % 2)
    samples = array("h")
    samples.frombytes(pcm_bytes[:usable_length])
    if sys.byteorder != "little":
        samples.byteswap()

    frame_samples = SAMPLE_RATE * FRAME_MS // 1000
    frame_rms = []
    for offset in range(0, len(samples), frame_samples):
        frame = samples[offset:offset + frame_samples]
        if len(frame) < frame_samples // 2:
            continue
        mean_square = sum(value * value for value in frame) / len(frame)
        frame_rms.append(math.sqrt(mean_square))

    audio_duration_ms = round(len(samples) / SAMPLE_RATE * 1000)
    if not frame_rms:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "valid": False,
            "quality": "empty",
            "audio_duration_ms": audio_duration_ms,
            "speech_duration_ms": 0,
            "silence_duration_ms": audio_duration_ms,
            "initial_silence_ms": None,
            "pause_count": 0,
            "pause_total_ms": 0,
            "pause_mean_ms": None,
            "pause_max_ms": None,
            "voiced_ratio": 0.0,
            "transcript_char_count": 0,
            "speaking_rate_cpm": None,
        }

    noise_floor = _percentile(frame_rms, 0.20)
    peak_rms = max(frame_rms)
    # 阈值同时参考环境底噪与本次音量；下限避免把压缩噪声当作说话。
    threshold = max(220.0, noise_floor * 2.5, peak_rms * 0.08)
    voiced = [value >= threshold for value in frame_rms]

    voiced_indexes = [index for index, active in enumerate(voiced) if active]
    if not voiced_indexes:
        return {
            "analysis_version": ANALYSIS_VERSION,
            "valid": False,
            "quality": "no_speech",
            "audio_duration_ms": audio_duration_ms,
            "speech_duration_ms": 0,
            "silence_duration_ms": audio_duration_ms,
            "initial_silence_ms": None,
            "pause_count": 0,
            "pause_total_ms": 0,
            "pause_mean_ms": None,
            "pause_max_ms": None,
            "voiced_ratio": 0.0,
            "transcript_char_count": 0,
            "speaking_rate_cpm": None,
        }

    first_voiced = voiced_indexes[0]
    last_voiced = voiced_indexes[-1]
    speech_frames = len(voiced_indexes)
    speech_duration_ms = speech_frames * FRAME_MS
    silence_duration_ms = max(0, audio_duration_ms - speech_duration_ms)
    initial_silence_ms = first_voiced * FRAME_MS

    pauses = []
    gap_frames = 0
    for active in voiced[first_voiced:last_voiced + 1]:
        if active:
            if gap_frames * FRAME_MS >= MIN_PAUSE_MS:
                pauses.append(gap_frames * FRAME_MS)
            gap_frames = 0
        else:
            gap_frames += 1

    transcript_chars = len(
        re.findall(r"[\u3400-\u9fffA-Za-z0-9]", str(transcript or ""))
    )
    speaking_rate = None
    if transcript_chars and speech_duration_ms >= 300:
        speaking_rate = transcript_chars / (speech_duration_ms / 60_000)

    voiced_ratio = speech_duration_ms / max(audio_duration_ms, 1)
    valid = audio_duration_ms >= 500 and speech_duration_ms >= 300
    if not valid:
        quality = "too_short"
    elif voiced_ratio < 0.08:
        quality = "limited_speech"
    else:
        quality = "usable"

    return {
        "analysis_version": ANALYSIS_VERSION,
        "valid": valid,
        "quality": quality,
        "audio_duration_ms": audio_duration_ms,
        "speech_duration_ms": speech_duration_ms,
        "silence_duration_ms": silence_duration_ms,
        "initial_silence_ms": initial_silence_ms,
        "pause_count": len(pauses),
        "pause_total_ms": sum(pauses),
        "pause_mean_ms": _round_or_none(sum(pauses) / len(pauses)) if pauses else None,
        "pause_max_ms": max(pauses) if pauses else None,
        "voiced_ratio": _round_or_none(voiced_ratio, 3),
        "transcript_char_count": transcript_chars,
        "speaking_rate_cpm": _round_or_none(speaking_rate),
    }


def analyze_pcm_file(path, transcript=""):
    return analyze_pcm_bytes(Path(path).read_bytes(), transcript=transcript)


def build_speech_summary(observations):
    """汇总一次测试中的多道语音回答，不给出疾病风险结论。"""
    samples = [dict(item) for item in observations if isinstance(item, dict)]
    valid_samples = [item for item in samples if item.get("valid")]

    def average(field):
        values = [
            float(item[field])
            for item in valid_samples
            if item.get(field) is not None
        ]
        return _round_or_none(sum(values) / len(values)) if values else None

    def total(field):
        return sum(
            int(item.get(field) or 0)
            for item in samples
            if isinstance(item.get(field, 0), (int, float))
        )

    return {
        "analysis_version": ANALYSIS_VERSION,
        "voice_answer_count": len(samples),
        "valid_sample_count": len(valid_samples),
        "average_audio_duration_ms": average("audio_duration_ms"),
        "average_speech_duration_ms": average("speech_duration_ms"),
        "average_initial_silence_ms": average("initial_silence_ms"),
        "average_pause_count": average("pause_count"),
        "average_pause_total_ms": average("pause_total_ms"),
        "average_voiced_ratio": average("voiced_ratio"),
        "average_speaking_rate_cpm": average("speaking_rate_cpm"),
        "total_filler_count": total("filler_count"),
        "total_repetition_count": total("repetition_count"),
        "total_self_correction_count": total("self_correction_count"),
        "samples": samples,
    }
