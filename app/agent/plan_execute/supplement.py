from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agent.plan_execute.config import PATH_PATTERN
from app.agent.plan_execute.models import MissingField


@dataclass
class ParsedSupplement:
    values: dict[str, str] = field(default_factory=dict)
    confirmed_actions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _extract_paths(text: str) -> list[str]:
    return [match.group(0).rstrip("，,。；;") for match in PATH_PATTERN.finditer(text)]


def parse_supplement(text: str, missing_fields: list[MissingField]) -> ParsedSupplement:
    normalized = text.strip()
    lowered = normalized.lower()
    paths = _extract_paths(normalized)
    values: dict[str, str] = {}

    if "确认" in normalized or "允许" in normalized:
        if "训练" in normalized:
            values["confirm_training"] = "true"
        if "导出" in normalized:
            values["confirm_export"] = "true"
        if "删除" in normalized or "清理" in normalized:
            values["confirm_delete"] = "true"

    oldyaml_match = re.search(r"(?:oldyaml|历史\s*yaml|yaml)\s*[=:：]?\s*([^\s，,。；;]+)", normalized, flags=re.I)
    if oldyaml_match:
        values["oldyaml"] = oldyaml_match.group(1).strip()
    elif paths:
        yaml_path = next((path for path in paths if Path(path).suffix.lower() in {".yaml", ".yml"}), None)
        if yaml_path and any(item.field == "oldyaml" for item in missing_fields):
            values["oldyaml"] = yaml_path

    if "detector_path" in lowered:
        match = re.search(r"detector_path\s*[=:：]?\s*([^\s，,。；;]+)", normalized, flags=re.I)
        if match:
            values["detector_path"] = match.group(1).strip()
    if "检测器目录" in normalized or "发布到" in normalized or "使用" in normalized or "用" in normalized:
        if paths and "oldyaml" not in values:
            values["detector_path"] = paths[-1]
    elif paths and "oldyaml" not in values and any(item.field == "detector_path" for item in missing_fields):
        values["detector_path"] = paths[-1]

    parsed = ParsedSupplement(values=values)
    parsed.errors.extend(validate_supplement_values(parsed, missing_fields))
    return parsed


def validate_supplement_values(parsed: ParsedSupplement, missing_fields: list[MissingField]) -> list[str]:
    field_types = {item.field: item.field_type for item in missing_fields}
    errors: list[str] = []

    for field_name, value in parsed.values.items():
        if field_name not in field_types:
            continue
        field_type = field_types[field_name]
        if field_type in {"absolute_directory_path", "yaml_path"} and not value.startswith("/"):
            errors.append(f"{field_name} 必须是绝对路径: {value}")
        if field_type == "yaml_path" and value and Path(value).suffix.lower() not in {".yaml", ".yml"}:
            errors.append(f"oldyaml 应该是 .yaml 或 .yml 文件路径: {value}")

    required_groups: set[tuple[str, ...]] = {
        tuple(item.required_one_of)
        for item in missing_fields
        if item.required_one_of
    }
    for group in required_groups:
        if not any(field_name in parsed.values for field_name in group):
            errors.append(f"需要补充其中一个字段: {', '.join(group)}")
    return errors
