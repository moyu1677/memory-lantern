"""最小 OpenAI 兼容聊天客户端，用于 ASR 后的对话回复。"""

from __future__ import annotations

import os
from typing import Any

import requests


class ChatEngineError(RuntimeError):
    pass


class ChatEngine:
    def __init__(
        self,
        api_base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 60,
        session: requests.Session | None = None,
    ) -> None:
        self.api_base_url = (
            api_base_url or os.getenv("CHAT_API_BASE_URL", "https://api.deepseek.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("CHAT_API_KEY", "")
        self.model = model or os.getenv("CHAT_MODEL", "deepseek-chat")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def reply(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise ChatEngineError("未配置 CHAT_API_KEY，无法生成 AI 回复")
        response = self.session.post(
            f"{self.api_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=self.timeout_seconds,
        )
        if response.status_code != 200:
            raise ChatEngineError(f"聊天模型请求失败（HTTP {response.status_code}）")
        try:
            content: Any = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ChatEngineError("聊天模型没有返回有效文本") from exc
        text = str(content).strip()
        if not text:
            raise ChatEngineError("聊天模型返回了空文本")
        return text
