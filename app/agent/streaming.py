from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def message_key(message: BaseMessage) -> tuple[Any, ...]:
    tool_calls = getattr(message, "tool_calls", None)
    return (
        type(message).__name__,
        getattr(message, "id", None),
        message_text(message.content),
        repr(tool_calls),
    )


def print_tool_calls(message: AIMessage) -> None:
    for tool_call in message.tool_calls:
        print(f"[tool] {tool_call['name']}({tool_call['args']})", flush=True)


def print_tool_result(message: ToolMessage) -> None:
    print(f"[tool-result] {message_text(message.content)}", flush=True)
