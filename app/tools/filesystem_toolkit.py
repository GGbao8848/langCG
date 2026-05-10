from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_community.agent_toolkits import FileManagementToolkit

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FILESYSTEM_ROOT = ROOT_DIR / "data" / "filesystem_workspace"
DEFAULT_FILESYSTEM_TOOLS = [
    "read_file",
    "write_file",
    "list_directory",
    "file_search",
    "copy_file",
    "move_file",
]
PATH_ARGUMENTS = {
    "file_path",
    "dir_path",
    "source_path",
    "destination_path",
}


def _split_items(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_tools(value: str | None) -> list[str]:
    return _split_items(value) or DEFAULT_FILESYSTEM_TOOLS


def _parse_root_item(item: str, index: int) -> tuple[str, Path]:
    alias = f"root{index + 1}"
    path_text = item

    if "=" in item:
        alias_text, path_text = item.split("=", 1)
        alias_text = alias_text.strip()
        if alias_text:
            alias = alias_text

    return alias, Path(path_text).expanduser().resolve()


def _configured_roots() -> list[tuple[str, Path]]:
    raw_roots = _split_items(os.getenv("LANGCG_FILESYSTEM_ROOTS"))
    if raw_roots:
        roots = [_parse_root_item(item, index) for index, item in enumerate(raw_roots)]
    else:
        root = Path(os.getenv("LANGCG_FILESYSTEM_ROOT", str(DEFAULT_FILESYSTEM_ROOT))).expanduser().resolve()
        roots = [("default", root)]

    seen_aliases: set[str] = set()
    resolved_roots: list[tuple[str, Path]] = []
    for alias, root in roots:
        if alias in seen_aliases:
            raise ValueError(f"重复的 filesystem root alias: {alias}")
        root.mkdir(parents=True, exist_ok=True)
        seen_aliases.add(alias)
        resolved_roots.append((alias, root))
    return resolved_roots


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class MultiRootFileSystemTool:
    def __init__(self, tool_name: str, tools_by_alias: dict[str, BaseTool]) -> None:
        self.tool_name = tool_name
        self.tools_by_alias = tools_by_alias
        self.default_alias = next(iter(tools_by_alias))
        self.default_tool = tools_by_alias[self.default_alias]
        self.roots = {
            alias: Path(getattr(tool, "root_dir")).expanduser().resolve()
            for alias, tool in tools_by_alias.items()
        }

    def invoke(self, **kwargs: Any) -> Any:
        alias = self._select_alias(kwargs)
        tool = self.tools_by_alias[alias]
        translated = {
            key: self._translate_path(value, alias) if key in PATH_ARGUMENTS and isinstance(value, str) else value
            for key, value in kwargs.items()
        }
        return tool.invoke(translated)

    def _select_alias(self, kwargs: dict[str, Any]) -> str:
        selected_alias: str | None = None
        for key in PATH_ARGUMENTS:
            value = kwargs.get(key)
            if not isinstance(value, str):
                continue
            alias = self._alias_for_path(value)
            if selected_alias is None:
                selected_alias = alias
            elif selected_alias != alias:
                raise ValueError("同一次文件操作不能跨 filesystem root。请先复制到同一 root 内再操作。")
        return selected_alias or self.default_alias

    def _alias_for_path(self, raw_path: str) -> str:
        alias, path_text = self._split_alias(raw_path)
        if alias:
            if alias not in self.tools_by_alias:
                raise ValueError(f"未知 filesystem root alias: {alias}")
            return alias

        path = Path(path_text).expanduser()
        if path.is_absolute():
            resolved = path.resolve()
            for candidate_alias, root in self.roots.items():
                if _is_relative_to(resolved, root):
                    return candidate_alias
            raise ValueError(f"路径不在允许的 filesystem roots 中: {raw_path}")

        return self.default_alias

    def _translate_path(self, raw_path: str, alias: str) -> str:
        path_alias, path_text = self._split_alias(raw_path)
        if path_alias:
            path_text = path_text.lstrip("/")

        path = Path(path_text).expanduser()
        if path.is_absolute():
            root = self.roots[alias]
            return str(path.resolve().relative_to(root))
        return path_text

    @staticmethod
    def _split_alias(raw_path: str) -> tuple[str | None, str]:
        prefix, separator, remainder = raw_path.partition(":")
        if separator and prefix and "/" not in prefix and "\\" not in prefix:
            return prefix, remainder or "."
        return None, raw_path


def _tool_description(tool: BaseTool, roots: list[tuple[str, Path]]) -> str:
    root_description = ", ".join(f"{alias}={root}" for alias, root in roots)
    first_alias = roots[0][0]
    return (
        f"{tool.description}\n"
        f"Allowed filesystem roots: {root_description}. "
        "Paths may be relative to the first root, absolute inside an allowed root, "
        f"or use alias:path such as {first_alias}:/folder/file.txt."
    )


def get_filesystem_tools() -> list[BaseTool]:
    roots = _configured_roots()
    selected_tools = _split_tools(os.getenv("LANGCG_FILESYSTEM_TOOLS"))
    toolkit_by_alias = {
        alias: {
            tool.name: tool
            for tool in FileManagementToolkit(root_dir=str(root), selected_tools=selected_tools).get_tools()
        }
        for alias, root in roots
    }

    tools: list[BaseTool] = []
    for tool_name in selected_tools:
        tools_by_alias = {
            alias: tools_for_root[tool_name]
            for alias, tools_for_root in toolkit_by_alias.items()
            if tool_name in tools_for_root
        }
        if not tools_by_alias:
            continue

        multi_root_tool = MultiRootFileSystemTool(tool_name, tools_by_alias)
        tools.append(
            StructuredTool.from_function(
                func=multi_root_tool.invoke,
                name=tool_name,
                description=_tool_description(multi_root_tool.default_tool, roots),
                args_schema=multi_root_tool.default_tool.args_schema,
                return_direct=multi_root_tool.default_tool.return_direct,
                response_format=multi_root_tool.default_tool.response_format,
            )
        )
    return tools
