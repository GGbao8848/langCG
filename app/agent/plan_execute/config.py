from __future__ import annotations

import os
import re
from typing import Any

from app.agent.tool_registry import TOOLS


PLAN_EXECUTE_TOOL_NAMES = [
    "annotate_visualize",
    "publish_yolo_dataset",
    "augment_yolo_dataset",
    "clean_irregular_dataset",
    "convert_xml_to_yolo",
    "reindex_yolo_labels",
    "split_yolo_dataset",
    "prune_yolo_by_visualized",
    "collect_wubao_images",
    "yolo_sliding_window_crop",
    "export_yolo_torchscript",
    "launch_yolo_training",
]

TOOL_TASK_KEYWORDS = {
    "xml",
    "yolo",
    "数据集",
    "标注",
    "清洗",
    "整理",
    "转换",
    "转yolo",
    "划分",
    "滑窗",
    "裁剪",
    "增强",
    "发布",
    "训练",
    "导出",
    "可视化",
    "误报",
    "wubao",
    "background",
    "detector",
    "oldyaml",
}

PATH_PATTERN = re.compile(r"/[^\s，,。；;]+")
OUTPUT_DIR_PATTERN = re.compile(r"output_dir=([^，\n]+)")
PUBLISHED_DIR_PATTERN = re.compile(r"published_dataset_dir=([^，\n]+)")
YAML_PATH_PATTERN = re.compile(r"yaml_path=([^，\n]+)")
STEP_REF_PATTERN = re.compile(r"^\$(?:steps\.)?([A-Za-z][A-Za-z0-9_-]*)\.(output_dir|yaml_path|published_dataset_dir)$")
STEP_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def should_attempt_plan_execute(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    has_path = bool(PATH_PATTERN.search(text))
    has_keyword = any(keyword in normalized for keyword in TOOL_TASK_KEYWORDS)
    return has_path and has_keyword


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def allowed_tool_names() -> list[str]:
    configured = os.getenv("LANGCG_PLAN_EXECUTE_TOOLS")
    if configured:
        names = [name.strip() for name in configured.split(",") if name.strip()]
        return [name for name in names if name in TOOLS]
    return [name for name in PLAN_EXECUTE_TOOL_NAMES if name in TOOLS]


def tool_schema_summary() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for name in allowed_tool_names():
        tool = TOOLS[name]
        schema = tool.args_schema.model_json_schema() if tool.args_schema is not None else {}
        summaries.append(
            {
                "name": tool.name,
                "description": tool.description,
                "required": schema.get("required", []),
                "properties": schema.get("properties", {}),
            }
        )
    return summaries


def tool_properties(tool_name: str) -> set[str]:
    tool = TOOLS[tool_name]
    if tool.args_schema is None:
        return set()
    return set(tool.args_schema.model_json_schema().get("properties", {}))


def required_tool_args(tool_name: str) -> set[str]:
    tool = TOOLS[tool_name]
    if tool.args_schema is None:
        return set()
    return set(tool.args_schema.model_json_schema().get("required", []))
