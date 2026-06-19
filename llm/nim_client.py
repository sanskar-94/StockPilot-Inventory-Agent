"""NVIDIA NIM wrapper with retry and an OpenAI-style fallback provider.

Two things keep us inside the free-tier rate limit: send the whole plan in one
call rather than one call per SKU, and fall back to a second OpenAI-style
endpoint on a 429 or a timeout. This wrapper is reusable across the stack.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import requests

NIM_URL = os.getenv("NIM_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
NIM_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-70b-instruct")

FALLBACK_URL = os.getenv("FALLBACK_URL", "")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-4o-mini")


class LLMUnavailable(RuntimeError):
    """Raised when neither NIM nor the fallback could answer."""


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
            "Accept": "application/json"}


def _post(url: str, api_key: str, model: str, messages: list, timeout: int) -> str:
    r = requests.post(
        url, headers=_headers(api_key),
        json={"model": model, "messages": messages, "temperature": 0.2,
              "max_tokens": 2048},
        timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_fallback(messages: list, timeout: int = 60) -> str:
    """Second OpenAI-style provider (OpenAI, Groq, Together, OpenRouter, ...)."""
    api_key = os.getenv("FALLBACK_API_KEY", "")
    url = os.getenv("FALLBACK_URL", FALLBACK_URL)
    model = os.getenv("FALLBACK_MODEL", FALLBACK_MODEL)
    if not (api_key and url):
        raise LLMUnavailable("NIM failed and no fallback provider is configured")
    return _post(url, api_key, model, messages, timeout)


def call_nim(messages: list, model: Optional[str] = None, retries: int = 2,
             timeout: int = 60) -> str:
    """Call NIM with exponential backoff; on exhaustion, use the fallback.
    Returns the raw assistant message content (a string)."""
    api_key = os.getenv("NIM_API_KEY", "")
    model = model or os.getenv("NIM_MODEL", NIM_MODEL)
    url = os.getenv("NIM_URL", NIM_URL)

    if not api_key:
        # no NIM key -> go straight to the fallback if available
        return call_fallback(messages, timeout)

    for attempt in range(retries + 1):
        try:
            return _post(url, api_key, model, messages, timeout)
        except Exception:
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                return call_fallback(messages, timeout)
