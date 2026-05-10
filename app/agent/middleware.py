from __future__ import annotations

import os
from typing import Any

from langchain.agents.middleware import (
    ContextEditingMiddleware,
    LLMToolSelectorMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.context_editing import ClearToolUsesEdit


DEFAULT_MIDDLEWARE = "model_retry"


def _enabled_names() -> list[str]:
    raw = os.getenv("LANGCG_AGENT_MIDDLEWARE", DEFAULT_MIDDLEWARE)
    if raw.strip().lower() in {"", "none", "off", "false", "0"}:
        return []
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def middleware_names() -> list[str]:
    return _enabled_names()


def build_agent_middleware(
    *,
    summary_model: Any | None = None,
    tool_selector_model: Any | None = None,
) -> list[Any]:
    middleware: list[Any] = []

    for name in _enabled_names():
        if name == "model_retry":
            middleware.append(
                ModelRetryMiddleware(
                    max_retries=_env_int("LANGCG_MODEL_RETRY_MAX_RETRIES", 2) or 0,
                    initial_delay=_env_float("LANGCG_MODEL_RETRY_INITIAL_DELAY", 1.0),
                    backoff_factor=_env_float("LANGCG_MODEL_RETRY_BACKOFF_FACTOR", 2.0),
                    max_delay=_env_float("LANGCG_MODEL_RETRY_MAX_DELAY", 60.0),
                    jitter=_env_bool("LANGCG_MODEL_RETRY_JITTER", True),
                )
            )
            continue

        if name == "tool_retry":
            middleware.append(
                ToolRetryMiddleware(
                    max_retries=_env_int("LANGCG_TOOL_RETRY_MAX_RETRIES", 2) or 0,
                    initial_delay=_env_float("LANGCG_TOOL_RETRY_INITIAL_DELAY", 1.0),
                    backoff_factor=_env_float("LANGCG_TOOL_RETRY_BACKOFF_FACTOR", 2.0),
                    max_delay=_env_float("LANGCG_TOOL_RETRY_MAX_DELAY", 60.0),
                    jitter=_env_bool("LANGCG_TOOL_RETRY_JITTER", True),
                )
            )
            continue

        if name == "model_call_limit":
            middleware.append(
                ModelCallLimitMiddleware(
                    thread_limit=_env_int("LANGCG_MODEL_CALL_THREAD_LIMIT"),
                    run_limit=_env_int("LANGCG_MODEL_CALL_RUN_LIMIT", 10),
                    exit_behavior=os.getenv("LANGCG_MODEL_CALL_LIMIT_EXIT_BEHAVIOR", "end"),
                )
            )
            continue

        if name == "tool_call_limit":
            middleware.append(
                ToolCallLimitMiddleware(
                    thread_limit=_env_int("LANGCG_TOOL_CALL_THREAD_LIMIT"),
                    run_limit=_env_int("LANGCG_TOOL_CALL_RUN_LIMIT", 20),
                    exit_behavior=os.getenv("LANGCG_TOOL_CALL_LIMIT_EXIT_BEHAVIOR", "continue"),
                )
            )
            continue

        if name == "context_editing":
            middleware.append(
                ContextEditingMiddleware(
                    edits=[
                        ClearToolUsesEdit(
                            trigger=_env_int("LANGCG_CONTEXT_EDITING_TRIGGER", 24000)
                            or 24000,
                            clear_at_least=_env_int(
                                "LANGCG_CONTEXT_EDITING_CLEAR_AT_LEAST", 4000
                            )
                            or 0,
                            keep=_env_int("LANGCG_CONTEXT_EDITING_KEEP", 3) or 0,
                            clear_tool_inputs=_env_bool(
                                "LANGCG_CONTEXT_EDITING_CLEAR_TOOL_INPUTS", False
                            ),
                            placeholder=os.getenv(
                                "LANGCG_CONTEXT_EDITING_PLACEHOLDER", "[cleared]"
                            ),
                        )
                    ],
                    token_count_method=os.getenv(
                        "LANGCG_CONTEXT_EDITING_TOKEN_COUNT_METHOD", "approximate"
                    ),
                )
            )
            continue

        if name == "summarization":
            summary_llm = os.getenv("LANGCG_SUMMARIZATION_MODEL") or summary_model
            if summary_llm is None:
                raise RuntimeError(
                    "启用 summarization middleware 时需要 LANGCG_SUMMARIZATION_MODEL，"
                    "或由 agent 传入摘要模型。"
                )
            middleware.append(
                SummarizationMiddleware(
                    model=summary_llm,
                    trigger=(
                        os.getenv("LANGCG_SUMMARIZATION_TRIGGER_TYPE", "tokens"),
                        _env_int("LANGCG_SUMMARIZATION_TRIGGER_VALUE", 24000) or 24000,
                    ),
                    keep=(
                        os.getenv("LANGCG_SUMMARIZATION_KEEP_TYPE", "messages"),
                        _env_int("LANGCG_SUMMARIZATION_KEEP_VALUE", 20) or 20,
                    ),
                )
            )
            continue

        if name in {"tool_selector", "llm_tool_selector"}:
            always_include = [
                item.strip()
                for item in os.getenv("LANGCG_TOOL_SELECTOR_ALWAYS_INCLUDE", "").split(",")
                if item.strip()
            ]
            middleware.append(
                LLMToolSelectorMiddleware(
                    model=os.getenv("LANGCG_TOOL_SELECTOR_MODEL") or tool_selector_model,
                    max_tools=_env_int("LANGCG_TOOL_SELECTOR_MAX_TOOLS"),
                    always_include=always_include or None,
                )
            )
            continue

        if name in {"openai_moderation", "openai_content_moderation"}:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY 未配置，无法启用 OpenAI content moderation middleware。"
                )
            from langchain_openai.middleware import OpenAIModerationMiddleware

            middleware.append(
                OpenAIModerationMiddleware(
                    model=os.getenv("LANGCG_OPENAI_MODERATION_MODEL", "omni-moderation-latest"),
                    check_input=_env_bool("LANGCG_OPENAI_MODERATION_CHECK_INPUT", True),
                    check_output=_env_bool("LANGCG_OPENAI_MODERATION_CHECK_OUTPUT", True),
                    check_tool_results=_env_bool(
                        "LANGCG_OPENAI_MODERATION_CHECK_TOOL_RESULTS", False
                    ),
                    exit_behavior=os.getenv("LANGCG_OPENAI_MODERATION_EXIT_BEHAVIOR", "end"),
                    violation_message=os.getenv("LANGCG_OPENAI_MODERATION_VIOLATION_MESSAGE"),
                )
            )
            continue

        raise RuntimeError(f"不支持的 LangChain middleware: {name}")

    return middleware
