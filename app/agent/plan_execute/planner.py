from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.chat import _build_llm
from app.agent.plan_execute.config import tool_schema_summary
from app.agent.plan_execute.models import AgentPlan


def _planner_system_prompt() -> str:
    return (
        "你是数据处理与模型训练任务的 Planner。你的职责是把用户自然语言请求翻译成 JSON 计划，"
        "不要执行工具，不要编造工具结果。\n"
        "只允许使用提供的工具名。输出必须是一个 JSON object，不能包含 markdown。\n"
        "必须覆盖用户明确要求的所有步骤；用户说发布就必须包含 publish_yolo_dataset，"
        "说增强就必须包含 augment_yolo_dataset，说滑窗/裁剪就必须包含 yolo_sliding_window_crop，"
        "说划分就必须包含 split_yolo_dataset，说转 yolo/XML 转换就必须包含 convert_xml_to_yolo。\n"
        "JSON schema:\n"
        "{\n"
        '  "should_execute": true,\n'
        '  "summary": "一句话任务摘要",\n'
        '  "answer": "",\n'
        '  "steps": [\n'
        '    {"id": "convert", "tool": "convert_xml_to_yolo", "args": {"input_dir": "/path"}, "reason": "..."}\n'
        "  ]\n"
        "}\n"
        "当后续步骤需要使用前序步骤输出目录时，用占位符字符串："
        "$steps.<step_id>.output_dir。例如 split 的 input_dir 可以是 $steps.convert.output_dir。\n"
        "只能引用已经完成的前序步骤，禁止引用当前步骤或未来步骤。"
        "不要为 output_dir 填写 $steps.<当前步骤>.output_dir；除非用户明确给出具体输出路径，否则省略 output_dir。\n"
        "常见顺序: clean -> convert_xml_to_yolo -> reindex_yolo_labels -> split_yolo_dataset -> "
        "yolo_sliding_window_crop -> augment_yolo_dataset -> publish_yolo_dataset -> "
        "launch_yolo_training -> export_yolo_torchscript。\n"
        "如果用户要求索引 0,1,2 都转为 0，使用 reindex_yolo_labels，"
        "args={\"source_indices\":\"0,1,2\",\"target_index\":\"0\"}。\n"
        "发布裁剪结果与增强结果时，publish_yolo_dataset 使用 "
        "input_dir=$steps.<crop>.output_dir, input_dirs=[$steps.<augment>.output_dir]。\n"
        "训练和导出默认 execute=false，除非系统另有确认机制；不要设置 execute=true。"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.I).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("planner output must be a JSON object")
    return value


def plan_with_llm(request: str, provider: str, model: str) -> AgentPlan:
    llm = _build_llm(provider, model)
    response = llm.invoke(
        [
            SystemMessage(content=_planner_system_prompt()),
            HumanMessage(
                content=(
                    "可用工具 JSON:\n"
                    f"{json.dumps(tool_schema_summary(), ensure_ascii=False)}\n\n"
                    f"用户请求:\n{request}"
                )
            ),
        ]
    )
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    payload = _extract_json_object(str(content))
    return AgentPlan.model_validate(payload)
