from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from app.agent.tool_results import format_tool_result
from app.tools.annotate_visualize_tool import annotate_visualize
from app.tools.collect_wubao_images_tool import collect_wubao_images
from app.tools.dataset_clean_tool import clean_irregular_dataset
from app.tools.filesystem_toolkit import get_filesystem_tools
from app.tools.prune_yolo_by_visualized_tool import prune_yolo_by_visualized
from app.tools.publish_yolo_dataset_tool import publish_yolo_dataset
from app.tools.split_yolo_dataset_tool import split_yolo_dataset
from app.tools.xml_to_yolo_tool import convert_xml_to_yolo
from app.tools.yolo_augment_tool import augment_yolo_dataset
from app.tools.yolo_export_tool import export_yolo_torchscript
from app.tools.yolo_reindex_tool import reindex_yolo_labels
from app.tools.yolo_sliding_window_tool import yolo_sliding_window_crop
from app.tools.yolo_train_launcher_tool import launch_yolo_training


def _handle_tool_error(error: Exception) -> str:
    return f"tool执行失败: {format_tool_result(str(error), success=False)}"


def _handle_tool_validation_error(error: Exception) -> str:
    return f"tool执行失败: {format_tool_result(str(error), success=False)}"


def _make_safe_tool(tool: BaseTool) -> StructuredTool:
    def _safe_func(**kwargs: Any) -> str:
        try:
            result = tool.invoke(kwargs)
        except Exception as error:
            return _handle_tool_error(error)
        return format_tool_result(result, success=True)

    return StructuredTool.from_function(
        func=_safe_func,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        return_direct=tool.return_direct,
        response_format=tool.response_format,
        handle_validation_error=_handle_tool_validation_error,
    )


RAW_TOOLS = {
    annotate_visualize.name: annotate_visualize,
    publish_yolo_dataset.name: publish_yolo_dataset,
    augment_yolo_dataset.name: augment_yolo_dataset,
    clean_irregular_dataset.name: clean_irregular_dataset,
    convert_xml_to_yolo.name: convert_xml_to_yolo,
    reindex_yolo_labels.name: reindex_yolo_labels,
    split_yolo_dataset.name: split_yolo_dataset,
    prune_yolo_by_visualized.name: prune_yolo_by_visualized,
    collect_wubao_images.name: collect_wubao_images,
    yolo_sliding_window_crop.name: yolo_sliding_window_crop,
    export_yolo_torchscript.name: export_yolo_torchscript,
    launch_yolo_training.name: launch_yolo_training,
}

for filesystem_tool in get_filesystem_tools():
    RAW_TOOLS[filesystem_tool.name] = filesystem_tool

TOOLS = {name: _make_safe_tool(tool) for name, tool in RAW_TOOLS.items()}

# Read-only metadata for clients. Exposing this does not grant execution access.
VISIBLE_TOOLS = TOOLS

# Tools available to the generic LangChain chat agent. Keep empty so tool tasks
# must pass through the LangGraph planner/executor gate.
CHAT_TOOLS: list[StructuredTool] = []
