"""单轮语音对话：ASR → AI 回复 → 已授权音色的 IndexTTS2。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .asr.asr_engine import ASREngine
from .chat.llm_engine import ChatEngine
from .tts.indextts2_engine import IndexTTS2Engine


SYSTEM_PROMPT = "你是一名耐心、简洁的中文语音陪伴助手。每次回复不超过三句话。"


class VoiceConversationPipeline:
    def __init__(
        self,
        asr: ASREngine,
        chat: ChatEngine,
        tts: IndexTTS2Engine,
        max_history_messages: int = 8,
    ) -> None:
        self.asr = asr
        self.chat = chat
        self.tts = tts
        self.max_history_messages = max_history_messages
        self._sessions: dict[str, list[dict[str, str]]] = {}

    def handle_turn(
        self,
        audio_path: str,
        voice_id: str,
        output_path: str,
        session_id: str = "default",
    ) -> dict[str, str]:
        transcript, _ = self.asr.transcribe_one(audio_path)
        transcript = transcript.strip()
        if not transcript:
            raise ValueError("未识别到可用语音")

        history = self._sessions.setdefault(
            session_id, [{"role": "system", "content": SYSTEM_PROMPT}]
        )
        history.append({"role": "user", "content": transcript})
        reply = self.chat.reply(history)
        history.append({"role": "assistant", "content": reply})
        self._sessions[session_id] = [history[0], *history[-self.max_history_messages :]]

        audio_output = self.tts.synthesize(reply, output_path, voice_id)
        return {"transcript": transcript, "reply": reply, "audio_path": audio_output}

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
