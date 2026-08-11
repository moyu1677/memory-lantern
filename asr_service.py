"""受控远程 Whisper ASR 客户端。"""

from __future__ import annotations

import os
from pathlib import Path

import requests


MAX_AUDIO_BYTES = 20 * 1024 * 1024
ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}


class AsrServiceError(RuntimeError):
    """远程语音识别服务未能返回可用文本。"""


class RemoteAsrService:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: int = 90):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls) -> "RemoteAsrService":
        return cls(
            os.getenv("ASR_URL", "").strip(),
            os.getenv("ASR_API_KEY", "").strip(),
            int(os.getenv("ASR_TIMEOUT_SECONDS", "90")),
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def transcribe(self, audio_stream, filename: str, mimetype: str | None) -> str:
        if not self.configured:
            raise AsrServiceError("语音识别服务尚未连接。")
        safe_name = Path(filename or "recording.webm").name
        if Path(safe_name).suffix.lower() not in ALLOWED_AUDIO_SUFFIXES:
            raise AsrServiceError("不支持该录音格式。")
        try:
            audio_stream.seek(0, 2)
            audio_size = audio_stream.tell()
            audio_stream.seek(0)
        except (AttributeError, OSError) as exc:
            raise AsrServiceError("录音文件无法读取，请重新录制后再试。") from exc
        if audio_size > MAX_AUDIO_BYTES:
            raise AsrServiceError("录音文件超过 20 MB，无法上传识别。")

        try:
            response = requests.post(
                f"{self.base_url}/v1/transcriptions",
                headers={"X-API-Key": self.api_key},
                files={
                    "audio": (
                        safe_name,
                        audio_stream,
                        mimetype or "application/octet-stream",
                    )
                },
                data={"language": "zh"},
                timeout=self.timeout_seconds,
            )
        except (OSError, requests.RequestException) as exc:
            raise AsrServiceError("远程语音识别暂时不可用，请稍后再试。") from exc

        if not response.ok:
            try:
                message = response.json().get("detail") or response.json().get("error")
            except ValueError:
                message = None
            raise AsrServiceError(message or "语音识别失败，请靠近麦克风后重试。")

        try:
            text = str(response.json().get("text") or "").strip()
        except ValueError as exc:
            raise AsrServiceError("语音识别服务返回格式异常。") from exc
        if not text:
            raise AsrServiceError("没有识别到清晰的话语，请再说一次。")
        return text
