"""Minimal Hyperbolic API client (OpenAI-compatible chat completions)."""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

from . import config

load_dotenv()


class HyperbolicClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = config.HYPERBOLIC_BASE_URL,
        model: str = config.LLM_MODEL,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("HYPERBOLIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "HYPERBOLIC_API_KEY not found. Set it in .env or environment."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
        response_format_json: bool = False,
    ) -> str:
        """Send a chat completion request and return the assistant text."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    def chat_json(self, system: str, user: str, **kwargs: Any) -> Any:
        """Chat completion that parses the response as JSON."""
        raw = self.chat(system, user, response_format_json=True, **kwargs)
        return _extract_json(raw)


def _extract_json(text: str) -> Any:
    """Parse JSON, tolerating code fences or leading/trailing prose."""
    text = text.strip()
    if text.startswith("```"):
        # strip ```json ... ``` fencing
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # best-effort: find the first { or [ and last } or ]
        start_obj = text.find("{")
        start_arr = text.find("[")
        candidates = [i for i in (start_obj, start_arr) if i != -1]
        if not candidates:
            raise
        start = min(candidates)
        end = max(text.rfind("}"), text.rfind("]"))
        return json.loads(text[start : end + 1])
