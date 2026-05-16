from __future__ import annotations

import json
import os
import re
from typing import Any


KEY_LINE_PATTERN = re.compile(
    r"(error|exception|traceback|failed|failure|warn|warning|success|completed|done|"
    r"saved|created|exported|deleted|moved|copied|processed|total|path|output|"
    r"错误|失败|异常|警告|成功|完成|保存|创建|导出|删除|移动|复制|处理|总计|路径)",
    re.IGNORECASE,
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _stringify_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(result)


def _compact_line(line: str, *, max_chars: int = 240) -> str:
    compact = " ".join(line.strip().split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def _select_key_lines(lines: list[str], limit: int) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    for line in lines:
        compact = _compact_line(line)
        if not compact or compact in seen:
            continue
        if KEY_LINE_PATTERN.search(compact):
            selected.append(compact)
            seen.add(compact)
        if len(selected) >= limit:
            return selected

    for line in lines:
        compact = _compact_line(line)
        if compact and compact not in seen:
            selected.append(compact)
            seen.add(compact)
        if len(selected) >= limit:
            break

    return selected


def format_tool_result(result: Any, *, success: bool = True) -> str:
    """Return a compact, structured tool result for model and UI consumption."""

    text = _stringify_result(result).strip()
    max_chars = _env_int("LANGCG_TOOL_RESULT_MAX_CHARS", 4000)
    max_lines = _env_int("LANGCG_TOOL_RESULT_MAX_LINES", 80)
    key_line_limit = _env_int("LANGCG_TOOL_RESULT_KEY_LINES", 8)

    lines = text.splitlines()
    truncated = len(text) > max_chars or len(lines) > max_lines
    key_lines = _select_key_lines(lines, key_line_limit)

    if not text:
        summary = "工具执行完成，无文本输出。"
    elif success:
        summary = "工具执行完成。"
    else:
        summary = "工具执行失败。"

    if truncated:
        preview_lines = lines[:max_lines]
        preview = "\n".join(preview_lines)
        if len(preview) > max_chars:
            preview = preview[:max_chars]
        preview = f"{preview.rstrip()}\n...[truncated]"
        summary = f"{summary} 原始输出较长，已保留摘要和关键片段。"
    else:
        preview = text

    payload = {
        "success": success,
        "summary": summary,
        "key_lines": key_lines,
        "output_preview": preview,
        "truncated": truncated,
        "original_chars": len(text),
        "original_lines": len(lines),
    }
    return json.dumps(payload, ensure_ascii=False)
