"""远程 IndexTTS2 服务的受限客户端。

浏览器不会直接访问 GPU 服务器；所有鉴权、音色 ID 和音频转发都由
Flask 后端完成。这样远程密钥不会泄露到前端，也不会要求远程服务访问
本机磁盘路径。
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from typing import BinaryIO

import requests


ALLOWED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}
MAX_TTS_TEXT_LENGTH = 500
MAX_EMOTION_SEGMENTS = 3
VOICE_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
EMOTION_REFERENCE_STYLES = {"neutral", "positive", "comfort"}


class VoiceServiceError(RuntimeError):
    """面向用户的远程音色服务错误。"""


def validate_reference_audio(filename: str, mimetype: str | None = None) -> str:
    """只接受适合作为参考音频的常见格式，返回规范化扩展名。"""

    name = str(filename or "").strip()
    suffix = os.path.splitext(name)[1].lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise VoiceServiceError("请上传 WAV、MP3、M4A、WEBM 或 OGG 格式的声音样本。")
    if mimetype and not str(mimetype).lower().startswith(("audio/", "video/webm")):
        raise VoiceServiceError("上传的文件不是可识别的音频格式。")
    return suffix


@dataclass(frozen=True)
class RemoteVoiceService:
    """与项目内置的 IndexTTS2 Flask 服务通讯的客户端。"""

    base_url: str
    api_key: str = ""
    timeout_seconds: float = 90.0

    @classmethod
    def from_environment(cls) -> "RemoteVoiceService":
        return cls(
            base_url=os.getenv("INDEXTTS2_URL", "").rstrip("/"),
            api_key=os.getenv("INDEXTTS2_API_KEY", ""),
            timeout_seconds=float(os.getenv("INDEXTTS2_TIMEOUT_SECONDS", "90")),
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json, audio/wav, audio/mpeg"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request_error(self, response: requests.Response, action: str) -> VoiceServiceError:
        detail = ""
        try:
            payload = response.json()
            detail = str(payload.get("error") or payload.get("detail") or "").strip()
        except (ValueError, AttributeError):
            pass
        if detail:
            return VoiceServiceError(f"{action}失败：{detail}")
        return VoiceServiceError(f"{action}失败，远程服务返回 {response.status_code}。")

    def enroll(
        self,
        audio_stream: BinaryIO,
        filename: str,
        mimetype: str | None,
        label: str,
    ) -> str:
        """上传一次已授权的参考音频，返回远程保存的 voice_id。"""

        if not self.configured:
            raise VoiceServiceError("语音服务尚未配置，请联系管理员连接远程语音服务器。")
        validate_reference_audio(filename, mimetype)

        try:
            response = requests.post(
                f"{self.base_url}/v1/voices",
                headers=self._headers(),
                files={
                    "reference_audio": (
                        filename,
                        audio_stream,
                        mimetype or "application/octet-stream",
                    )
                },
                data={"label": label, "consent_confirmed": "true"},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise VoiceServiceError("无法连接远程语音服务器，请检查服务器地址和网络。") from exc

        if not response.ok:
            raise self._request_error(response, "声音授权测试")
        try:
            payload = response.json()
        except ValueError as exc:
            raise VoiceServiceError("远程语音服务器返回了无法识别的数据。") from exc

        voice_id = str(payload.get("voice_id") or "").strip()
        if not VOICE_ID_PATTERN.fullmatch(voice_id):
            raise VoiceServiceError("远程语音服务器没有返回有效的音色编号。")
        return voice_id

    def synthesize(
        self,
        voice_id: str,
        text: str,
        emotion_vector: list[float] | None = None,
        emotion_text: str | None = None,
        emotion_strength: float = 0.0,
        emotion_segments: list[dict] | None = None,
        emotion_style: str = "neutral",
    ) -> tuple[bytes, str]:
        """以远程 voice_id 合成音频；服务必须直接返回音频字节。

        ``emotion_segments`` is a bounded internal-only plan.  It lets the
        backend use a subtle acknowledgement tone for one sentence and return
        to the reference voice for the next sentence without exposing emotion
        controls to the elder-facing page.
        """

        if not self.configured:
            raise VoiceServiceError("语音服务尚未配置。")
        voice_id = str(voice_id or "").strip()
        if not VOICE_ID_PATTERN.fullmatch(voice_id):
            raise VoiceServiceError("音色编号无效，无法合成语音。")
        text = str(text or "").strip()
        if not text:
            raise VoiceServiceError("没有需要朗读的内容。")
        if len(text) > MAX_TTS_TEXT_LENGTH:
            raise VoiceServiceError("回复过长，已拒绝合成。请让助手用更短的句子回答。")
        emotion_style = str(emotion_style or "neutral").strip().lower()
        if emotion_style not in EMOTION_REFERENCE_STYLES:
            raise VoiceServiceError("语音情感参考类型无效。")
        if emotion_vector is not None and (
            len(emotion_vector) != 8
            or any(
                not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
                for value in emotion_vector
            )
        ):
            raise VoiceServiceError("语音情感参数无效。")
        try:
            emotion_strength = float(emotion_strength)
        except (TypeError, ValueError) as exc:
            raise VoiceServiceError("语音情感强度无效。") from exc
        if not 0.0 <= emotion_strength <= 1.0:
            raise VoiceServiceError("语音情感强度无效。")
        emotion_text = str(emotion_text or "").strip() or None
        if emotion_text is not None and len(emotion_text) > 120:
            raise VoiceServiceError("语音情感描述过长。")
        if emotion_style != "neutral" and (
            emotion_vector is not None
            or emotion_text is not None
            or emotion_segments is not None
        ):
            raise VoiceServiceError("情感参考音频不能与文字、向量或分句情感参数同时使用。")
        if emotion_vector is not None and emotion_text is not None:
            raise VoiceServiceError("不能同时使用情感向量和情感描述。")

        safe_segments = None
        if emotion_segments is not None:
            if emotion_vector is not None or emotion_text is not None:
                raise VoiceServiceError("不能同时使用整段和分句情感参数。")
            if not isinstance(emotion_segments, list) or not (
                2 <= len(emotion_segments) <= MAX_EMOTION_SEGMENTS
            ):
                raise VoiceServiceError("分句情感参数无效。")
            safe_segments = []
            for segment in emotion_segments:
                if not isinstance(segment, dict):
                    raise VoiceServiceError("分句情感参数无效。")
                segment_text = str(segment.get("text") or "").strip()
                vector = segment.get("emotion_vector")
                segment_emotion_text = str(segment.get("emotion_text") or "").strip() or None
                try:
                    strength = float(segment.get("emotion_strength", 0.0))
                except (TypeError, ValueError) as exc:
                    raise VoiceServiceError("分句情感强度无效。") from exc
                if not segment_text or not 0.0 <= strength <= 1.0:
                    raise VoiceServiceError("分句情感参数无效。")
                if vector is not None and (
                    not isinstance(vector, list)
                    or len(vector) != 8
                    or any(
                        not isinstance(value, (int, float))
                        or not 0.0 <= float(value) <= 1.0
                        for value in vector
                    )
                ):
                    raise VoiceServiceError("分句情感参数无效。")
                if segment_emotion_text is not None and len(segment_emotion_text) > 120:
                    raise VoiceServiceError("分句情感描述过长。")
                if vector is not None and segment_emotion_text is not None:
                    raise VoiceServiceError("不能同时使用情感向量和情感描述。")
                safe_segments.append(
                    {
                        "text": segment_text,
                        "emotion_vector": [float(value) for value in vector]
                        if vector is not None
                        else None,
                        "emotion_text": segment_emotion_text,
                        "emotion_strength": strength,
                    }
                )
            if "".join(item["text"] for item in safe_segments) != text:
                raise VoiceServiceError("分句情感文本与朗读内容不一致。")

        request_payload = {
            "voice_id": voice_id,
            "text": text,
            "emotion_style": emotion_style,
        }
        if emotion_style != "neutral":
            request_payload["emotion_strength"] = emotion_strength
        elif safe_segments is not None:
            request_payload["emotion_segments"] = safe_segments
        elif emotion_vector is not None or emotion_text is not None:
            if emotion_vector is not None:
                request_payload["emotion_vector"] = [
                    float(value) for value in emotion_vector
                ]
            if emotion_text is not None:
                request_payload["emotion_text"] = emotion_text
            request_payload["emotion_strength"] = emotion_strength

        try:
            response = requests.post(
                f"{self.base_url}/v1/speech",
                headers={**self._headers(), "Content-Type": "application/json"},
                json=request_payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise VoiceServiceError("远程语音合成暂时不可用，请稍后重试。") from exc

        if not response.ok:
            raise self._request_error(response, "语音合成")

        content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
        if content_type.startswith("audio/"):
            return response.content, content_type

        # 便于兼容已有服务：也接受 JSON 中的 base64 音频，不接受任意 URL，
        # 以免应用被诱导去请求内网地址。
        try:
            payload = response.json()
            encoded = payload.get("audio_base64")
            returned_type = str(payload.get("content_type") or "audio/wav")
            if not isinstance(encoded, str):
                raise ValueError("missing audio_base64")
            return base64.b64decode(encoded, validate=True), returned_type
        except (ValueError, TypeError) as exc:
            raise VoiceServiceError("远程语音服务器没有返回音频数据。") from exc

    def enroll_emotion_reference(
        self,
        voice_id: str,
        style: str,
        audio_stream: BinaryIO,
        filename: str,
        mimetype: str | None,
    ) -> None:
        """Attach an authorised same-speaker prosody anchor to one voice ID."""

        if not self.configured:
            raise VoiceServiceError("语音服务尚未配置。")
        voice_id = str(voice_id or "").strip()
        style = str(style or "").strip().lower()
        if not VOICE_ID_PATTERN.fullmatch(voice_id):
            raise VoiceServiceError("声音编号无效，无法登记情感样本。")
        if style not in EMOTION_REFERENCE_STYLES - {"neutral"}:
            raise VoiceServiceError("情感样本类型无效。")
        validate_reference_audio(filename, mimetype)
        try:
            response = requests.put(
                f"{self.base_url}/v1/voices/{voice_id}/emotion-references/{style}",
                headers=self._headers(),
                files={
                    "emotion_audio": (
                        filename,
                        audio_stream,
                        mimetype or "application/octet-stream",
                    )
                },
                data={"consent_confirmed": "true"},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise VoiceServiceError("无法连接远程语音服务器，情感样本暂未保存。") from exc
        if not response.ok:
            raise self._request_error(response, "情感样本登记")

    def delete_voice(self, voice_id: str) -> None:
        """撤回授权时删除远程服务器保存的参考音频。

        删除接口是幂等的：远程已不存在的音色仍视为撤回成功，避免本地
        授权记录无法清理。
        """

        if not self.configured:
            raise VoiceServiceError("语音服务尚未配置，暂时无法撤回声音授权。")
        voice_id = str(voice_id or "").strip()
        if not VOICE_ID_PATTERN.fullmatch(voice_id):
            raise VoiceServiceError("音色编号无效，无法撤回声音授权。")

        try:
            response = requests.delete(
                f"{self.base_url}/v1/voices/{voice_id}",
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise VoiceServiceError("无法连接远程语音服务器，暂时无法撤回声音授权。") from exc

        if response.status_code == 404:
            return
        if not response.ok:
            raise self._request_error(response, "撤回声音授权")
