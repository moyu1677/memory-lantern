"""Contract tests for the GPU-hosted Whisper long-form call."""

import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import numpy as np


REMOTE_ASR_SOURCE = (
    Path(__file__).resolve().parents[1] / "remote_asr" / "remote_whisper_server.py"
)


class _FakeFastApi:
    def __init__(self, **_kwargs):
        pass

    def on_event(self, _name):
        return lambda function: function

    def get(self, _path):
        return lambda function: function

    def post(self, _path):
        return lambda function: function


class _Tensor:
    def to(self, *_args, **_kwargs):
        return self


def load_remote_asr(monkeypatch):
    fastapi = types.ModuleType("fastapi")
    fastapi.FastAPI = _FakeFastApi
    fastapi.File = lambda *args, **kwargs: None
    fastapi.Form = lambda *args, **kwargs: None
    fastapi.Header = lambda *args, **kwargs: None
    fastapi.HTTPException = RuntimeError
    fastapi.UploadFile = object
    fastapi.status = types.SimpleNamespace(HTTP_401_UNAUTHORIZED=401)
    monkeypatch.setitem(sys.modules, "fastapi", fastapi)

    spec = importlib.util.spec_from_file_location("remote_asr_test_module", REMOTE_ASR_SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def configure_whisper_fakes(monkeypatch, module, samples):
    processor = Mock()
    processor.return_value = types.SimpleNamespace(
        input_features=_Tensor(), attention_mask=_Tensor()
    )
    processor.get_prompt_ids.return_value = _Tensor()
    processor.batch_decode.return_value = ["嗯，我想说完整的一句话。"]

    model = Mock()
    model.dtype = "float16"
    model.generate.return_value = _Tensor()
    monkeypatch.setattr(module, "model_bundle", lambda: (model, processor, "cuda:0"))

    librosa = types.ModuleType("librosa")
    librosa.load = lambda *_args, **_kwargs: (samples, 16_000)
    torch = types.ModuleType("torch")
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "librosa", librosa)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return model, processor


def test_long_recording_uses_full_audio_and_whisper_sequential_decoding(monkeypatch):
    module = load_remote_asr(monkeypatch)
    model, processor = configure_whisper_fakes(
        monkeypatch, module, np.zeros(module.WHISPER_SHORTFORM_SAMPLES + 1)
    )

    assert module.transcribe_with_whisper(Path("recording.wav"), "zh")
    assert processor.call_args.kwargs["truncation"] is False
    assert processor.call_args.kwargs["padding"] == "longest"
    assert processor.call_args.kwargs["return_attention_mask"] is True
    generation_kwargs = model.generate.call_args.kwargs
    assert generation_kwargs["return_timestamps"] is True
    assert generation_kwargs["condition_on_prev_tokens"] is True
    assert generation_kwargs["prompt_condition_type"] == "all-segments"
    assert generation_kwargs["language"] == "chinese"
    assert "prompt_ids" in generation_kwargs


def test_short_recording_still_uses_verbatim_transcription_prompt(monkeypatch):
    module = load_remote_asr(monkeypatch)
    model, processor = configure_whisper_fakes(monkeypatch, module, np.zeros(16_000))

    module.transcribe_with_whisper(Path("recording.wav"), "auto")

    assert processor.call_args.kwargs["truncation"] is False
    assert processor.call_args.kwargs["padding"] == "max_length"
    assert processor.call_args.kwargs["max_length"] == module.WHISPER_SHORTFORM_SAMPLES
    assert model.generate.call_args.kwargs["task"] == "transcribe"
    assert "prompt_ids" in model.generate.call_args.kwargs
    assert "language" not in model.generate.call_args.kwargs
