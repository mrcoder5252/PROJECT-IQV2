"""Gemini call helper that survives retired models: on error (404, 503, 504, 429) it falls back across
fast, verified Flash models, remembers the working one, and avoids hanging the UI."""

from __future__ import annotations

import re
from typing import Any, List, Optional

from src.config import config
from src.logger import get_logger

logger = get_logger("project_iq.llm")
_resolved: dict = {}

# Fast, verified working models in priority order
_FALLBACK_CANDIDATES = [
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]


def make_client(api_key: str, timeout_s: int = 25):
    """Gemini client with a hard request timeout so the UI can never hang on a slow API call."""
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
    try:
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" in actions:
                names.append(str(m.name).replace("models/", ""))
    except Exception:
        pass
    return names


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
        raise


def generate_text(client, prompt: str, gen_config: Any = None) -> str:
    primary = current_model()
    cfg = _fast_config(gen_config)

    # Order models: primary first, followed by fallback candidates
    models_to_try = [primary] + [m for m in _FALLBACK_CANDIDATES if m != primary]

    last_exc = None
    for model in models_to_try:
        try:
            res = _call(client, model, prompt, cfg)
            if model != primary and _resolved.get("model") != model:
                logger.info("Switched to working model '%s'.", model)
                _resolved["model"] = model
            return res
        except Exception as exc:
            msg = str(exc)
            logger.warning("Gemini model '%s' failed (%s). Trying fallback candidate...", model, msg[:80])
            last_exc = exc
            # If thinking config was rejected, retry without config once for the same model
            if cfg is not None:
                try:
                    res = _call(client, model, prompt, None)
                    if model != primary:
                        _resolved["model"] = model
                    return res
                except Exception:
                    pass

    # If all candidate fallbacks failed, attempt auto-discovery from client.models.list()
    if last_exc:
        discovered = _usable_models(client)
        flash_discovered = [m for m in discovered if "flash" in m and m not in models_to_try]
        for alt in flash_discovered[:2]:
            try:
                res = _call(client, alt, prompt, None)
                _resolved["model"] = alt
                return res
            except Exception:
                pass
        raise last_exc

    raise RuntimeError("No Gemini models available.")
