"""部署在 GPU 服务器上的 IndexTTS2 最小 HTTP 服务。

首次授权时接收并保存参考音频，后续仅使用 voice_id 合成。应用服务器只
接收合成的音频字节，不保存家人的原始声音文件。

本服务使用 IndexTTS2 环境通常已经具备的 FastAPI、Uvicorn 与 multipart，
避免为推理服务额外创建 Flask 环境或改动现有训练环境。
"""

from __future__ import annotations

import os
import random
import re
import threading
import uuid
import logging
import math
import subprocess
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from starlette.background import BackgroundTask
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from emotion_controls import (
    clamp_emotion_control,
    clamp_text_emotion_strength,
    deterministic_speech_seed,
)


APP_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("indextts2-service")


def load_env_file(path: Path) -> None:
    """加载简单的 KEY=VALUE 文件，不依赖 python-dotenv。"""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env_file(APP_ROOT / ".env")
PROFILE_DIR = Path(os.getenv("VOICE_PROFILE_DIR", APP_ROOT / "voice_profiles"))
MODEL_DIR = os.getenv("INDEXTTS2_MODEL_DIR", "checkpoints")
CONFIG_PATH = os.getenv("INDEXTTS2_CONFIG_PATH", "checkpoints/config.yaml")
API_KEY = os.getenv("VOICE_API_KEY", "")
MAX_AUDIO_BYTES = 10 * 1024 * 1024
ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
MAX_EMOTION_TEXT_LENGTH = 120
# These are deliberately *reference-audio* styles, not arbitrary emotion
# labels exposed to an elder.  Each file must be spoken by the same authorised
# family member as the base voice profile.  Keeping this small vocabulary is
# important: it prevents a dialogue turn from jumping between theatrical
# emotions while still giving the backend a natural, content-driven choice.
EMOTION_REFERENCE_STYLES = {"positive", "comfort"}
MAX_EMOTION_REFERENCE_STRENGTH = 0.45

app = FastAPI(
    title="Memory Lantern IndexTTS2 Service",
    docs_url=None,
    redoc_url=None,
)
inference_lock = threading.Lock()
_model = None


class EmotionSegment(BaseModel):
    """A bounded, server-validated prosody plan for one sentence."""

    text: str
    emotion_vector: list[float] | None = None
    emotion_text: str | None = None
    emotion_strength: float = 0.0


class SpeechRequest(BaseModel):
    voice_id: str
    text: str
    # IndexTTS2 emotion-vector order: joy, anger, sadness, fear, disgust,
    # melancholy, surprise, calm.  None preserves compatibility with clients
    # that intentionally want the reference recording's original prosody.
    emotion_vector: list[float] | None = None
    emotion_text: str | None = None
    emotion_strength: float = 0.0
    emotion_segments: list[EmotionSegment] | None = None
    # ``neutral`` keeps the base reference's natural manner.  ``positive``
    # and ``comfort`` only take effect when the matching same-speaker emotion
    # reference has been explicitly enrolled for this voice ID.
    emotion_style: str = "neutral"


def require_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="远程语音服务未设置 VOICE_API_KEY。",
        )
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="未授权的语音服务请求。")


def model_instance():
    global _model
    if _model is None:
        from indextts.infer_v2 import IndexTTS2

        _model = IndexTTS2(
            cfg_path=CONFIG_PATH,
            model_dir=MODEL_DIR,
            use_fp16=os.getenv("INDEXTTS2_USE_FP16", "true").lower() == "true",
            use_cuda_kernel=os.getenv("INDEXTTS2_USE_CUDA_KERNEL", "false").lower() == "true",
            use_deepspeed=os.getenv("INDEXTTS2_USE_DEEPSPEED", "false").lower() == "true",
        )
    return _model


@app.on_event("startup")
def preload_model() -> None:
    """Warm IndexTTS2 before the web demo reports the service as ready."""

    model_instance()


def profile_path(voice_id: str) -> Path | None:
    """Return the base timbre reference, never an emotion-reference file."""

    candidate = PROFILE_DIR / f"{voice_id}.wav"
    return candidate if candidate.is_file() else None


def emotion_reference_path(voice_id: str, style: str) -> Path | None:
    """Find a separately authorised same-speaker emotional reference."""

    if style not in EMOTION_REFERENCE_STYLES:
        return None
    candidate = PROFILE_DIR / f"{voice_id}.{style}.wav"
    return candidate if candidate.is_file() else None


def store_normalized_reference(
    content: bytes,
    suffix: str,
    voice_id: str,
    *,
    style: str | None = None,
) -> None:
    """Decode an authorised reference and retain a bounded mono WAV profile."""

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    suffix_tag = f".{style}" if style else ""
    source_path = PROFILE_DIR / f".{voice_id}{suffix_tag}{suffix}"
    normalized_path = PROFILE_DIR / f"{voice_id}{suffix_tag}.wav"
    source_path.write_bytes(content)
    try:
        subprocess.run(
            [
                FFMPEG_BIN,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source_path),
                "-t",
                "30",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s16le",
                str(normalized_path),
            ],
            check=True,
            capture_output=True,
            timeout=45,
        )
        if not normalized_path.exists() or normalized_path.stat().st_size < 1024:
            raise ValueError("normalization produced no usable audio")
    except (FileNotFoundError, subprocess.SubprocessError, ValueError) as exc:
        normalized_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="上传的音频无法解码，请重新录制或换一种音频格式。",
        ) from exc
    finally:
        source_path.unlink(missing_ok=True)


def cleanup_output(path: Path) -> None:
    path.unlink(missing_ok=True)


def validate_emotion_control(
    emotion_vector: list[float] | None, emotion_strength: float
) -> tuple[list[float] | None, float]:
    """Validate and bound one IndexTTS2 emotion control.

    The service deliberately clips rather than rejects an over-strong client
    request.  A conversation should remain audible and retain the family
    member's identity even if an older local web process sends stale values.
    """

    if emotion_vector is not None and (
        len(emotion_vector) != 8
        or any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
            or value > 1.0
            for value in emotion_vector
        )
    ):
        raise HTTPException(status_code=400, detail="情感参数无效。")
    try:
        emotion_strength = float(emotion_strength)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="情感强度无效。") from exc
    if not math.isfinite(emotion_strength) or not 0.0 <= emotion_strength <= 1.0:
        raise HTTPException(status_code=400, detail="情感强度无效。")
    return clamp_emotion_control(
        [float(value) for value in emotion_vector] if emotion_vector is not None else None,
        emotion_strength,
    )


def validate_emotion_text(emotion_text: str | None) -> str | None:
    """Validate a short, private instruction for IndexTTS2's emotion encoder."""

    value = str(emotion_text or "").strip() or None
    if value is not None and len(value) > MAX_EMOTION_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail="情感描述过长。")
    return value


def validate_text_emotion_strength(emotion_strength: float) -> float:
    """Validate learned text emotion strength in IndexTTS2's natural range."""

    try:
        value = float(emotion_strength)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="情感强度无效。") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise HTTPException(status_code=400, detail="情感强度无效。")
    return clamp_text_emotion_strength(value)


def validate_emotion_reference_style(value: str | None) -> str:
    style = str(value or "neutral").strip().lower()
    if style not in {"neutral", *EMOTION_REFERENCE_STYLES}:
        raise HTTPException(status_code=400, detail="情感参考类型无效。")
    return style


def validate_emotion_reference_strength(value: float) -> float:
    """Keep a same-speaker anchor audible but below theatrical intensity."""

    try:
        strength = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="情感强度无效。") from exc
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise HTTPException(status_code=400, detail="情感强度无效。")
    return min(strength, MAX_EMOTION_REFERENCE_STRENGTH)


def render_speech(
    model,
    reference_audio: Path,
    text: str,
    output_path: Path,
    emotion_vector: list[float] | None,
    emotion_strength: float,
    emotion_text: str | None = None,
    emotion_audio: Path | None = None,
) -> None:
    """Render one sentence while retaining the reference as the primary prompt."""

    if emotion_text is not None:
        # IndexTTS2's own Qwen emotion encoder turns our short private
        # instruction into its canonical, ordered 8-D vector.  Passing that
        # vector explicitly (rather than asking ``infer`` to encode it again)
        # lets the deterministic seed depend on the *actual* control.  Thus
        # two descriptions that map to the same calm vector cannot produce two
        # arbitrary prosodies merely because their wording differed.
        emotion_dict = model.qwen_emo.inference(emotion_text)
        emotion_vector = list(emotion_dict.values())
        emotion_text = None
    if emotion_vector is not None:
        # ``normalize_emo_vec`` caps the raw sum at 0.8.  The application sends
        # much smaller vectors, and ``emo_alpha`` reduces them again, leaving
        # the majority of the conditioning from the authorized voice sample.
        emotion_vector = model.normalize_emo_vec(emotion_vector, apply_bias=True)
    # IndexTTS2 includes stochastic diffusion after autoregressive decoding.
    # The service holds ``inference_lock`` around this call, so seeding global
    # RNGs here makes a given family voice/text/control combination repeatable.
    # Use common random numbers for every emotional rendering of the same
    # family voice and sentence.  Otherwise a change from (say) 0.16 to 0.20
    # also changes diffusion noise, making the apparent emotion comparison a
    # comparison of two unrelated samples.  The selected control still affects
    # the model, but no longer receives a random prosody bonus or penalty.
    seed = deterministic_speech_seed(reference_audio.name, text, None, 0.0)
    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inference_kwargs = {
        "spk_audio_prompt": str(reference_audio),
        "text": text,
        "output_path": str(output_path),
        "emo_vector": emotion_vector,
        "emo_alpha": emotion_strength,
        "use_emo_text": False,
        "emo_text": None,
        # Voice companionship must be repeatable.  IndexTTS2's sampling can
        # occasionally turn even a weak joy vector into an abrupt pitch peak.
        # Greedy decoding preserves the selected direction without that random
        # prosody swing.
        "do_sample": False,
        "interval_silence": 80,
        "verbose": False,
    }
    if emotion_audio is not None:
        # IndexTTS2 keeps timbre and prosody as separate conditions here: the
        # base prompt fixes *who* speaks, while this short same-speaker sample
        # supplies a restrained manner of speaking.  This is markedly more
        # stable than asking a text-vector to invent a family member's emotion.
        inference_kwargs["emo_audio_prompt"] = str(emotion_audio)
    model.infer(
        **inference_kwargs,
    )


def join_rendered_segments(segment_paths: list[Path], output_path: Path) -> None:
    """Join trusted temporary WAVs without handing arbitrary paths to ffmpeg."""

    manifest = NamedTemporaryFile(prefix="indextts2_concat_", suffix=".txt", delete=False)
    manifest_path = Path(manifest.name)
    try:
        manifest.write(
            "".join(f"file '{path.as_posix()}'\n" for path in segment_paths).encode("utf-8")
        )
        manifest.close()
        subprocess.run(
            [
                FFMPEG_BIN,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_path),
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        if not output_path.exists() or output_path.stat().st_size < 1024:
            raise ValueError("segment synthesis produced no usable audio")
    finally:
        manifest.close()
        manifest_path.unlink(missing_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/v1/voices", status_code=201)
async def enroll_voice(
    reference_audio: UploadFile = File(...),
    consent_confirmed: str = Form(...),
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)
    suffix = Path(reference_audio.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="仅支持 WAV、MP3、M4A、WEBM 或 OGG 音频。",
        )
    if consent_confirmed != "true":
        raise HTTPException(status_code=400, detail="必须确认声音所有者已授权。")

    content = await reference_audio.read(MAX_AUDIO_BYTES + 1)
    if not content or len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail="声音样本应大于 0 且不超过 10MB。")

    voice_id = uuid.uuid4().hex
    store_normalized_reference(content, suffix, voice_id)
    return {"voice_id": voice_id}


@app.put("/v1/voices/{voice_id}/emotion-references/{style}", status_code=204)
async def enroll_emotion_reference(
    voice_id: str,
    style: str,
    emotion_audio: UploadFile = File(...),
    consent_confirmed: str = Form(...),
    x_api_key: str | None = Header(default=None),
):
    """Store one optional emotion anchor for an already authorised voice.

    A base profile is intentionally required first.  The server accepts only
    two calm companionship-oriented styles and does not keep the upload's
    original bytes after normalisation.
    """

    require_api_key(x_api_key)
    if not re.fullmatch(r"[0-9a-f]{32}", voice_id):
        raise HTTPException(status_code=400, detail="声音编号无效。")
    style = validate_emotion_reference_style(style)
    if style == "neutral":
        raise HTTPException(status_code=400, detail="基础音色无需重复登记。")
    if profile_path(voice_id) is None:
        raise HTTPException(status_code=404, detail="找不到已授权的家人音色。")
    if consent_confirmed != "true":
        raise HTTPException(status_code=400, detail="必须确认声音所有者已明确授权。")

    suffix = Path(emotion_audio.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="仅支持 WAV、MP3、M4A、WEBM 或 OGG 音频。")
    content = await emotion_audio.read(MAX_AUDIO_BYTES + 1)
    if not content or len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail="声音样本应大于 0 且不超过 10MB。")
    store_normalized_reference(content, suffix, voice_id, style=style)
    return Response(status_code=204)


@app.post("/v1/speech")
def synthesize(
    payload: SpeechRequest,
    x_api_key: str | None = Header(default=None),
):
    require_api_key(x_api_key)
    voice_id = payload.voice_id.strip()
    text = payload.text.strip()
    if (
        not re.fullmatch(r"[0-9a-f]{32}", voice_id)
        or not text
        or len(text) > 500
    ):
        raise HTTPException(status_code=400, detail="请求参数无效。")

    emotion_text = validate_emotion_text(payload.emotion_text)
    emotion_style = validate_emotion_reference_style(payload.emotion_style)
    if emotion_style != "neutral" and (
        payload.emotion_vector is not None
        or emotion_text is not None
        or payload.emotion_segments is not None
    ):
        raise HTTPException(
            status_code=400,
            detail="情感参考音频不能与向量、文字或分句情感参数同时使用。",
        )
    if payload.emotion_vector is not None and emotion_text is not None:
        raise HTTPException(status_code=400, detail="不能同时使用情感向量和情感描述。")
    if emotion_style != "neutral":
        # Audio-reference prosody has a separate calibration range from the
        # old 8-D vector controls.  Do not accidentally clamp it to the tiny
        # vector ceiling, otherwise an enrolled anchor becomes inaudible.
        emotion_vector = None
        emotion_strength = validate_emotion_reference_strength(payload.emotion_strength)
    elif emotion_text is not None:
        emotion_vector = None
        emotion_strength = validate_text_emotion_strength(payload.emotion_strength)
    else:
        emotion_vector, emotion_strength = validate_emotion_control(
            payload.emotion_vector, payload.emotion_strength
        )
    emotion_segments = payload.emotion_segments
    if emotion_segments is not None:
        if (
            emotion_vector is not None
            or emotion_text is not None
            or not 2 <= len(emotion_segments) <= 3
        ):
            raise HTTPException(status_code=400, detail="分句情感参数无效。")
        normalized_segments = []
        for segment in emotion_segments:
            segment_text = segment.text.strip()
            segment_emotion_text = validate_emotion_text(segment.emotion_text)
            if segment.emotion_vector is not None and segment_emotion_text is not None:
                raise HTTPException(status_code=400, detail="不能同时使用情感向量和情感描述。")
            if segment_emotion_text is not None:
                vector = None
                strength = validate_text_emotion_strength(segment.emotion_strength)
            else:
                vector, strength = validate_emotion_control(
                    segment.emotion_vector, segment.emotion_strength
                )
            if not segment_text:
                raise HTTPException(status_code=400, detail="分句情感文本无效。")
            normalized_segments.append((segment_text, vector, strength, segment_emotion_text))
        if "".join(item[0] for item in normalized_segments) != text:
            raise HTTPException(status_code=400, detail="分句情感文本与朗读内容不一致。")
    else:
        normalized_segments = None

    reference_audio = profile_path(voice_id)
    if reference_audio is None:
        raise HTTPException(status_code=404, detail="找不到已授权的家人音色。")
    emotion_audio = emotion_reference_path(voice_id, emotion_style)
    # A stale local record must never make a live conversation fail.  If an
    # anchor was withdrawn remotely, continue with the authorised base voice.
    effective_emotion_style = emotion_style if emotion_audio is not None else "neutral"

    temporary = NamedTemporaryFile(prefix="indextts2_", suffix=".wav", delete=False)
    output_path = Path(temporary.name)
    temporary.close()
    started_at = time.perf_counter()
    queue_started_at = started_at
    segment_paths: list[Path] = []
    try:
        with inference_lock:
            queue_wait_ms = round((time.perf_counter() - queue_started_at) * 1000)
            model = model_instance()
            if normalized_segments is None:
                render_speech(
                    model,
                    reference_audio,
                    text,
                    output_path,
                    emotion_vector,
                    emotion_strength,
                    emotion_text,
                    emotion_audio,
                )
            else:
                for segment_text, vector, strength, segment_emotion_text in normalized_segments:
                    temporary_segment = NamedTemporaryFile(
                        prefix="indextts2_segment_", suffix=".wav", delete=False
                    )
                    segment_path = Path(temporary_segment.name)
                    temporary_segment.close()
                    segment_paths.append(segment_path)
                    render_speech(
                        model,
                        reference_audio,
                        segment_text,
                        segment_path,
                        vector,
                        strength,
                        segment_emotion_text,
                    )
                join_rendered_segments(segment_paths, output_path)
    except Exception as exc:
        output_path.unlink(missing_ok=True)
        logger.exception("IndexTTS2 synthesis failed for an authorized voice profile")
        raise HTTPException(
            status_code=502,
            detail="语音合成失败，请查看 GPU 服务器日志。",
        ) from exc
    finally:
        for segment_path in segment_paths:
            segment_path.unlink(missing_ok=True)

    inference_ms = round((time.perf_counter() - started_at) * 1000)
    emotion_mode = (
        "segmented-text"
        if normalized_segments is not None
        else "anchor-" + effective_emotion_style
        if effective_emotion_style != "neutral"
        else "text" if emotion_text is not None else "vector" if emotion_vector is not None else "reference"
    )
    logger.info(
        "IndexTTS2 synthesis complete: queue=%sms inference=%sms emotion=%s",
        queue_wait_ms,
        inference_ms,
        emotion_mode,
    )
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename="assistant-reply.wav",
        headers={
            "Server-Timing": f"tts-queue;dur={queue_wait_ms}, tts;dur={inference_ms}",
            "X-IndexTTS-Emotion-Control": emotion_mode,
        },
        background=BackgroundTask(cleanup_output, output_path),
    )


@app.delete("/v1/voices/{voice_id}", status_code=204)
def delete_voice(
    voice_id: str,
    x_api_key: str | None = Header(default=None),
):
    """撤回授权，并删除云端保存的唯一参考音频。"""

    require_api_key(x_api_key)
    if not re.fullmatch(r"[0-9a-f]{32}", voice_id):
        raise HTTPException(status_code=400, detail="音色编号无效。")
    reference_audio = profile_path(voice_id)
    if reference_audio is None:
        raise HTTPException(status_code=404, detail="找不到该已授权音色。")
    reference_audio.unlink(missing_ok=True)
    for style in EMOTION_REFERENCE_STYLES:
        anchor = emotion_reference_path(voice_id, style)
        if anchor is not None:
            anchor.unlink(missing_ok=True)
    return Response(status_code=204)


if __name__ == "__main__":
    uvicorn.run(
        app,
        # 默认只监听本机回环地址，通过 SSH 隧道暴露给应用服务器，避免把
        # 声纹登记和合成接口直接开放到公网。
        host=os.getenv("VOICE_SERVER_HOST", "127.0.0.1"),
        port=int(os.getenv("VOICE_SERVER_PORT", "5000")),
    )
