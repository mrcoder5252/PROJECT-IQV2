"""Gemini call helper that survives retired models: on a 404 it asks the API which models the key can
actually use, picks the newest stable Flash model, remembers it, and retries once."""

from __future__ import annotations

import re
from typing import Any, List, Optional

from src.config import config
from src.logger import get_logger

logger = get_logger("project_iq.llm")
_resolved: dict = {}
_SKIP = ("lite", "image", "tts", "live", "audio", "embedding", "exp", "preview", "thinking", "robotics", "computer")


def make_client(api_key: str, timeout_s: int = 45):
    """Gemini client with a hard request timeout so the UI can never hang on a slow/retrying API call."""
    from google import genai
    try:
        from google.genai import types
        return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_s * 1000))
    except Exception:
        return genai.Client(api_key=api_key)


def current_model() -> str:
    return _resolved.get("model") or config.llm_model_name


def _usable_models(client) -> List[str]:
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            names.append(str(m.name).replace("models/", ""))
    return names


def _pick(names: List[str]) -> Optional[str]:
    flash = [n for n in names if "flash" in n and not any(s in n for s in _SKIP)]
    flash.sort(key=lambda n: [int(x) for x in re.findall(r"\d+", n)], reverse=True)
    return flash[0] if flash else None


def _fast_config(gen_config: Any):
    """Default to no 'thinking' tokens: they add many seconds and are not needed for RAG answers."""
    if gen_config is not None:
        return gen_config
    try:
        from google.genai import types
        return types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0))
    except Exception:
        return None


def _call(client, model: str, prompt: str, cfg: Any) -> str:
    kwargs = {"config": cfg} if cfg is not None else {}
    try:
        return client.models.generate_content(model=model, contents=prompt, **kwargs).text
    except Exception as exc:
        msg = str(exc)
        if cfg is not None and ("thinking" in msg.lower() or "INVALID_ARGUMENT" in msg):
            logger.info("Model rejected thinking config; retrying without it.")
            return client.models.generate_content(model=model, contents=prompt).text
        if "504" in msg or "DEADLINE" in msg or "503" in msg or "UNAVAILABLE" in msg:
            logger.warning("Transient Gemini error (%s); retrying once.", msg[:80])
            return client.models.generate_content(model=model, contents=prompt, **kwargs).text
        raise


def generate_text(client, prompt: str, gen_config: Any = None) -> str:
    model = current_model()
    cfg = _fast_config(gen_config)
    try:
        return _call(client, model, prompt, cfg)
    except Exception as exc:
        if "404" not in str(exc) and "NOT_FOUND" not in str(exc):
            raise
        alt = _pick(_usable_models(client))
        if not alt or alt == model:
            raise
        logger.warning("Model '%s' unavailable; switching to '%s'. Set PROJECTIQ_LLM_MODEL=%s to make this permanent.", model, alt, alt)
        _resolved["model"] = alt
        return _call(client, alt, prompt, cfg)
