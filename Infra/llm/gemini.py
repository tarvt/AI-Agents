# -*- coding: utf-8 -*-
"""
Gemini-based chat completion. Used by RAG, Shop, and other agents.
No dependency on app.*; config via env (GEMINI_API_KEY, etc.).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List

import httpx
from httpx import AsyncHTTPTransport
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SAFE_MODE = os.getenv("SAFE_MODE", "0") == "1"
PROXY_URL = os.getenv("PROXY_URL", "").strip()


def _build_transport() -> AsyncHTTPTransport:
    return AsyncHTTPTransport(proxy=PROXY_URL) if PROXY_URL else AsyncHTTPTransport()


def _fallback_from_context(messages: List[Dict[str, str]]) -> str:
    """Fallback when API key missing or SAFE_MODE: use CONTEXT from system messages."""
    ctx = ""
    for m in messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str) and m["content"].startswith("CONTEXT:"):
            ctx = m["content"][len("CONTEXT:"):].strip()
            break
    return (
        "LLM is in safe mode or API key is not configured.\n\n"
        "Based on your documents (preview):\n"
        f"{(ctx or '')[:800]}"
    )


def _to_gemini_payload(messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> Dict[str, Any]:
    """Convert role/content messages to Gemini generateContent payload."""
    system_msgs: List[str] = []
    contents: List[Dict[str, Any]] = []

    for m in messages:
        role = m.get("role")
        txt = m.get("content", "") or ""
        if not txt:
            continue
        if role == "system":
            system_msgs.append(txt)
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": txt}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": txt}]})

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
        },
    }
    if system_msgs:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_msgs)}]}
    return payload


async def _gemini_complete(
    model: str | None,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    base = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
    ver = os.getenv("GEMINI_API_VERSION", "v1").strip()
    default_model = os.getenv("GEMINI_CHAT_MODEL", "models/gemini-2.5-flash")

    if SAFE_MODE or not api_key:
        return {
            "content": _fallback_from_context(messages),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "status": "safe_mode",
            "provider": "google",
        }

    model = model or default_model
    url = f"{base}/{ver}/{model}:generateContent?key={api_key}"
    payload = _to_gemini_payload(messages, temperature, max_tokens)

    start = asyncio.get_event_loop().time()
    async with httpx.AsyncClient(timeout=60, transport=_build_transport(), trust_env=False) as client:
        resp = await client.post(url, json=payload)
    latency_ms = int((asyncio.get_event_loop().time() - start) * 1000)

    if resp.status_code != 200:
        logging.error("Gemini API Error %s: %s", resp.status_code, resp.text)
        return {
            "content": f"[Gemini {resp.status_code}] {resp.text}",
            "usage": {},
            "latency_ms": latency_ms,
            "status": "error",
            "provider": "google",
        }

    data = resp.json()
    cand = (data.get("candidates") or [])[0] if data.get("candidates") else {}
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts) or "I received an empty response."
    finish_reason = cand.get("finishReason")

    if not parts:
        if finish_reason == "SAFETY":
            logging.warning("Gemini generation blocked due to SAFETY filter.")
            text = "The response was blocked by safety filters."
        elif finish_reason:
            logging.warning("Gemini generation failed. Finish Reason: %s", finish_reason)
            text = f"The model failed to generate a response (Reason: {finish_reason})."
        else:
            logging.error("Gemini generation failed. Full JSON: %s", json.dumps(data))

    usage_src = data.get("usageMetadata") or {}
    usage = {
        "prompt_tokens": usage_src.get("promptTokenCount", 0),
        "completion_tokens": usage_src.get("candidatesTokenCount", 0),
        "total_tokens": usage_src.get("totalTokenCount", 0),
    }
    return {
        "content": text,
        "usage": usage,
        "latency_ms": latency_ms,
        "status": "ok" if finish_reason not in ("SAFETY", "RECITATION") else "error",
        "provider": "google",
        "error": f"Finish Reason: {finish_reason}" if finish_reason and finish_reason != "STOP" else None,
    }


async def chat_complete(
    *,
    model: str | None,
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """
    Public entrypoint used by all agents (matches ChatCompleteFn).
    Returns dict with at least: {"content": str, "usage": {...}}.
    """
    result = await _gemini_complete(model, messages, temperature, max_tokens)
    return {
        "content": result.get("content", ""),
        "usage": result.get("usage", {}),
    }
