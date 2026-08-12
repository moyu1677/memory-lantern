"""仅监听回环地址的 Whisper large-v3 转写服务，供 SSH 隧道调用。"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, status


def load_service_env() -> None:
    """Load this service's simple ``.env`` file without executing its content.

    The server is launched by Uvicorn on the remote host, so relying on an
    interactive shell to source configuration makes restarts fragile.  An
    already-provided process environment still takes precedence.
    """

    env_path = Path(__file__).with_name(".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


load_service_env()


# Keep ASR off the training card by default. This runs before PyTorch imports.
# An explicit CUDA_VISIBLE_DEVICES still wins.
ASR_CUDA_VISIBLE_DEVICES = os.getenv("ASR_CUDA_VISIBLE_DEVICES", "1").strip()
if ASR_CUDA_VISIBLE_DEVICES:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", ASR_CUDA_VISIBLE_DEVICES)


MAX_AUDIO_BYTES = 20 * 1024 * 1024
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
app = FastAPI(docs_url=None, redoc_url=None, title="Memory Lantern Whisper ASR")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
# Whisper has a 30-second feature window.  The browser deliberately permits a
# longer turn for older adults, so longer recordings must use Whisper's
# sequential long-form path rather than silently truncating the latter half.
WHISPER_SHORTFORM_SAMPLES = 30 * 16_000
VERBATIM_TRANSCRIPTION_PROMPT = (
    "Verbatim transcription. Keep fillers, repetitions, and self-corrections: "
    "嗯、呃、啊、uh、um. Do not summarize, correct, or add words."
)


def expected_api_key() -> str:
    key = os.getenv("ASR_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ASR_API_KEY must be configured")
    return key


def require_api_key(x_api_key: str | None) -> None:
    if not x_api_key or x_api_key != expected_api_key():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


@lru_cache(maxsize=1)
def model_bundle():
    """Load the already-downloaded official Whisper large-v3 checkpoint.

    ``local_files_only`` is intentional: this service must never silently
    download weights or switch models while a family is using the demo.
    """

    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    model_path = os.getenv("ASR_MODEL_PATH", "").strip()
    if not model_path or not Path(model_path).exists():
        raise RuntimeError("ASR_MODEL_PATH must point to a local Whisper large-v3 checkpoint")
    device = os.getenv("ASR_DEVICE", "cuda:0")
    use_fp16 = os.getenv("ASR_USE_FP16", "true").lower() == "true" and device.startswith("cuda")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, processor, device


@app.on_event("startup")
def preload_model() -> None:
    """Avoid a long first-turn pause after the demo web page is opened."""

    model_bundle()


@app.get("/health")
def health() -> dict[str, bool]:
    return {"status": True, "model_loaded": bool(model_bundle.cache_info().currsize)}


def normalize_audio(input_path: Path) -> Path:
    """Decode browser uploads once to mono 16 kHz WAV for Whisper."""

    normalized = Path(
        tempfile.NamedTemporaryFile(
            prefix="memory_lantern_whisper_", suffix=".wav", delete=False
        ).name
    )
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
                str(input_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(normalized),
            ],
            check=True,
            capture_output=True,
            timeout=45,
        )
        return normalized
    except Exception:
        normalized.unlink(missing_ok=True)
        raise


def transcribe_with_whisper(audio_path: Path, language: str) -> str:
    import librosa
    import torch

    model, processor, device = model_bundle()
    samples, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    if samples.size == 0:
        return ""
    is_longform = samples.size > WHISPER_SHORTFORM_SAMPLES
    # ``truncation=False`` is essential for the long path: the previous
    # one-shot extraction retained only a short Whisper window.  Conversely,
    # a single short recording must be padded to Whisper's fixed 3000 mel
    # frames; padding it merely to its own length causes model.generate to
    # reject recordings such as a brief English sentence.
    feature_kwargs = {
        "sampling_rate": 16_000,
        "return_tensors": "pt",
        "truncation": False,
        "return_attention_mask": True,
    }
    if is_longform:
        feature_kwargs["padding"] = "longest"
    else:
        feature_kwargs.update(
            {"padding": "max_length", "max_length": WHISPER_SHORTFORM_SAMPLES}
        )
    features = processor(samples, **feature_kwargs)
    language_map = {"zh": "chinese", "en": "english"}
    selected_language = language_map.get(str(language or "").lower())
    generation_kwargs = {
        "task": "transcribe",
        "prompt_ids": processor.get_prompt_ids(
            VERBATIM_TRANSCRIPTION_PROMPT, return_tensors="pt"
        ).to(device),
    }
    if selected_language:
        generation_kwargs["language"] = selected_language

    if is_longform:
        # Transformers 4.52 implements Whisper's sequential timestamp-based
        # long-form algorithm.  It advances through the entire recording and
        # carries prior context across natural pauses instead of dropping the
        # second half of a 30–60 second elderly user's reply.
        generation_kwargs.update(
            {
                "return_timestamps": True,
                "condition_on_prev_tokens": True,
                "prompt_condition_type": "all-segments",
                "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                "compression_ratio_threshold": 1.35,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "attention_mask": features.attention_mask.to(device),
            }
        )
    with torch.inference_mode():
        generated_ids = model.generate(
            input_features=features.input_features.to(device=device, dtype=model.dtype),
            **generation_kwargs,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


@app.post("/v1/transcriptions")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("zh"),
    x_api_key: str | None = Header(default=None),
) -> dict[str, str]:
    require_api_key(x_api_key)
    suffix = Path(audio.filename or "recording.webm").suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    content = await audio.read(MAX_AUDIO_BYTES + 1)
    if not content or len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail="Audio must be between 1 byte and 20MB")

    handle = tempfile.NamedTemporaryFile(prefix="memory_lantern_asr_", suffix=suffix, delete=False)
    normalized_path = None
    try:
        handle.write(content)
        handle.close()
        normalized_path = normalize_audio(Path(handle.name))
        text = re.sub(
            r"<\|[^|]+\|>", "", transcribe_with_whisper(normalized_path, language)
        ).strip()
        if not text:
            raise HTTPException(status_code=422, detail="No clear speech detected")
        return {"text": text}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ASR failed: {exc}") from exc
    finally:
        try:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)
            if normalized_path is not None:
                normalized_path.unlink(missing_ok=True)
        except OSError:
            pass
