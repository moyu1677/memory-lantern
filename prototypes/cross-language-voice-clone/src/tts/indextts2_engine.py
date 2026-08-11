"""IndexTTS2 HTTP 客户端。

服务端保存经授权的参考音频；本项目仅保存/传递 ``voice_id``，不会把本机路径
误发给远程服务器。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests


class IndexTTS2Error(RuntimeError):
    """远程 IndexTTS2 服务不可用或返回无效数据。"""


class IndexTTS2Engine:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 120,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("INDEXTTS2_URL", "http://127.0.0.1:5000")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("INDEXTTS2_API_KEY", "")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    @property
    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise IndexTTS2Error("未配置 INDEXTTS2_API_KEY")
        return {"X-API-Key": self.api_key}

    @staticmethod
    def _error(response: requests.Response, action: str) -> IndexTTS2Error:
        detail = ""
        try:
            payload: dict[str, Any] = response.json()
            detail = str(payload.get("detail") or payload.get("error") or "")
        except (ValueError, AttributeError):
            pass
        suffix = f"：{detail}" if detail else f"（HTTP {response.status_code}）"
        return IndexTTS2Error(f"{action}失败{suffix}")

    def enroll_voice(self, reference_audio: str, consent_confirmed: bool) -> str:
        """首次上传获授权的参考音频，返回可复用的 ``voice_id``。"""

        if not consent_confirmed:
            raise IndexTTS2Error("必须确认声音所有者已明确授权")
        path = Path(reference_audio)
        if not path.is_file():
            raise FileNotFoundError(f"找不到参考音频：{path}")

        with path.open("rb") as stream:
            response = self.session.post(
                f"{self.base_url}/v1/voices",
                headers=self._headers,
                files={"reference_audio": (path.name, stream)},
                data={"consent_confirmed": "true"},
                timeout=self.timeout_seconds,
            )
        if response.status_code != 201:
            raise self._error(response, "音色登记")
        try:
            voice_id = str(response.json()["voice_id"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexTTS2Error("音色登记没有返回有效 voice_id") from exc
        if len(voice_id) != 32:
            raise IndexTTS2Error("音色登记返回的 voice_id 格式无效")
        return voice_id

    def synthesize(self, text: str, output_path: str, voice_id: str) -> str:
        """使用已经登记的音色合成，将返回的 WAV 字节写入 ``output_path``。"""

        text = text.strip()
        if not text or len(text) > 500:
            raise ValueError("合成文本不能为空且不能超过 500 个字符")

        response = self.session.post(
            f"{self.base_url}/v1/speech",
            headers={**self._headers, "Content-Type": "application/json"},
            json={"voice_id": voice_id, "text": text},
            timeout=self.timeout_seconds,
        )
        if response.status_code != 200:
            raise self._error(response, "语音合成")
        if not response.content:
            raise IndexTTS2Error("语音合成返回了空音频")

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".partial")
        temporary.write_bytes(response.content)
        temporary.replace(destination)
        return str(destination)

    def synthesize_segments(
        self, segments: list[dict[str, Any]], output_dir: str, voice_id: str
    ) -> list[str]:
        """按 ASR 原文逐段合成。"""

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        outputs: list[str] = []
        for index, segment in enumerate(segments):
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            path = output_root / f"tts_{index:04d}.wav"
            outputs.append(self.synthesize(text, str(path), voice_id))
            segment["tts_audio"] = str(path)
        return outputs
