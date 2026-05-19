from __future__ import annotations

import json
import time
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.services.chat_store import load_app_state_value, save_app_state_value

LLM_SETTINGS_KEY = "llm_settings"
LLMProvider = Literal["openrouter", "ollama"]

MODEL_OPTIONS: dict[LLMProvider, list[str]] = {
    "ollama": ["gemma4:e4b", "qwen3.5:9b"],
    "openrouter": ["openrouter/free"],
}

DEFAULT_PROVIDER_SETTINGS: dict[LLMProvider, dict[str, str]] = {
    "ollama": {
        "model": "gemma4:e4b",
        "api_key": "",
        "base_url": "http://127.0.0.1:11434",
    },
    "openrouter": {
        "model": "openrouter/free",
        "api_key": "",
        "base_url": "https://openrouter.ai/api/v1",
    },
}


def _normalize_provider(value: Any) -> LLMProvider:
    provider = str(value or "ollama").strip().lower()
    if provider not in {"openrouter", "ollama"}:
        return "ollama"
    return provider  # type: ignore[return-value]


def _normalize_provider_settings(provider: LLMProvider, payload: dict[str, Any]) -> dict[str, str]:
    defaults = DEFAULT_PROVIDER_SETTINGS[provider]
    model = str(payload.get("model") or "").strip() or defaults["model"]
    if model not in MODEL_OPTIONS[provider]:
        model = defaults["model"]
    return {
        "model": model,
        "api_key": str(payload.get("api_key") or "").strip(),
        "base_url": (str(payload.get("base_url") or "").strip() or defaults["base_url"]).rstrip("/"),
    }


def normalize_llm_settings(payload: dict[str, Any]) -> dict[str, Any]:
    if "providers" in payload:
        active_provider = _normalize_provider(payload.get("active_provider"))
        providers_payload = payload.get("providers") if isinstance(payload.get("providers"), dict) else {}
    else:
        # Backward compatibility for the previous flat shape.
        active_provider = _normalize_provider(payload.get("provider"))
        providers_payload = {
            active_provider: payload,
        }

    providers = {
        provider: _normalize_provider_settings(
            provider,
            providers_payload.get(provider) if isinstance(providers_payload.get(provider), dict) else {},
        )
        for provider in ("ollama", "openrouter")
    }

    return {
        "active_provider": active_provider,
        "providers": providers,
        "model_options": MODEL_OPTIONS,
    }


def active_llm_settings(settings: dict[str, Any]) -> dict[str, str]:
    normalized = normalize_llm_settings(settings)
    provider = _normalize_provider(normalized.get("active_provider"))
    profile = normalized["providers"][provider]
    return {
        "provider": provider,
        "model": profile["model"],
        "api_key": profile["api_key"],
        "base_url": profile["base_url"],
    }


def load_llm_settings() -> dict[str, Any]:
    return normalize_llm_settings(load_app_state_value(LLM_SETTINGS_KEY))


def save_llm_settings(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_llm_settings(settings)
    save_app_state_value(LLM_SETTINGS_KEY, normalized)
    return normalized


def validate_llm_settings(settings: dict[str, Any]) -> dict[str, str]:
    active = active_llm_settings(settings)
    provider = active["provider"]
    missing_fields = [
        label
        for label, value in (
            ("Provider", provider),
            ("Model", active["model"]),
            ("Base URL", active["base_url"]),
            ("API Key", active["api_key"] if provider == "openrouter" else "not-required"),
        )
        if not value
    ]
    if missing_fields:
        raise ValueError(f"当前模型配置未填写完整: {', '.join(missing_fields)}")
    return active


def _test_ollama(settings: dict[str, str]) -> dict[str, Any]:
    model = settings["model"]
    base_url = settings["base_url"].rstrip("/")
    started_at = time.perf_counter()
    try:
        with urlopen(f"{base_url}/api/tags", timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ValueError(f"Ollama联通测试失败: {exc}") from exc

    models = payload.get("models")
    names: list[str] = []
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                names.append(item["name"])
    if model not in names:
        available = ", ".join(names[:8]) if names else "无可用模型"
        raise ValueError(f"Ollama可访问，但未找到模型 {model}。当前模型: {available}")

    latency_ms = round((time.perf_counter() - started_at) * 1000)
    return {"ok": True, "message": f"Ollama联通正常，响应 {latency_ms} ms", "latency_ms": latency_ms}


def _test_openrouter(settings: dict[str, str]) -> dict[str, Any]:
    model = settings["model"]
    base_url = settings["base_url"].rstrip("/")
    started_at = time.perf_counter()
    request = Request(
        f"{base_url}/models",
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ValueError(f"OpenRouter联通测试失败: HTTP {exc.code}") from exc
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ValueError(f"OpenRouter联通测试失败: {exc}") from exc

    latency_ms = round((time.perf_counter() - started_at) * 1000)
    return {
        "ok": True,
        "message": f"OpenRouter联通正常，模型 {model} 将用于后续对话，响应 {latency_ms} ms",
        "latency_ms": latency_ms,
    }


def test_llm_settings_connection(settings: dict[str, Any]) -> dict[str, Any]:
    active = validate_llm_settings(settings)
    if active["provider"] == "openrouter":
        return _test_openrouter(active)
    return _test_ollama(active)
