from __future__ import annotations

from typing import Any

from app.agent.plan_execute.config import (
    STEP_ID_PATTERN,
    STEP_REF_PATTERN,
    allowed_tool_names,
    env_int,
    required_tool_args,
    tool_properties,
)
from app.agent.plan_execute.models import AgentPlan, PlanValidationError


def _validate_step_id(step_id: str) -> None:
    if not STEP_ID_PATTERN.match(step_id):
        raise PlanValidationError(f"非法 step id: {step_id}")


def _required_tools_from_request(request: str | None) -> set[str]:
    if not request:
        return set()
    normalized = request.lower().replace(" ", "")
    required: set[str] = set()
    if "发布" in normalized or "detector" in normalized:
        required.add("publish_yolo_dataset")
    if "增强" in normalized:
        required.add("augment_yolo_dataset")
    if "滑窗" in normalized or "裁剪" in normalized:
        required.add("yolo_sliding_window_crop")
    if "划分" in normalized or "trainval" in normalized or "train_val" in normalized:
        required.add("split_yolo_dataset")
    if "转yolo" in normalized or "xml" in normalized:
        required.add("convert_xml_to_yolo")
    if (
        "索引" in normalized
        or "重映射" in normalized
        or "映射" in normalized
        or "转为0" in normalized
        or "转成0" in normalized
    ):
        required.add("reindex_yolo_labels")
    if "训练" in normalized:
        required.add("launch_yolo_training")
    if "导出" in normalized or "torchscript" in normalized:
        required.add("export_yolo_torchscript")
    return required


def _referenced_step_ids(value: Any) -> set[str]:
    if isinstance(value, str):
        match = STEP_REF_PATTERN.match(value.strip())
        return {match.group(1)} if match else set()
    if isinstance(value, list):
        refs: set[str] = set()
        for item in value:
            refs.update(_referenced_step_ids(item))
        return refs
    if isinstance(value, dict):
        refs: set[str] = set()
        for item in value.values():
            refs.update(_referenced_step_ids(item))
        return refs
    return set()


def validate_plan(plan: AgentPlan, request: str | None = None) -> AgentPlan:
    if not plan.should_execute:
        raise PlanValidationError("planner chose not to execute")
    if not plan.steps:
        raise PlanValidationError("planner returned no executable steps")
    max_steps = env_int("LANGCG_PLAN_EXECUTE_MAX_STEPS", 10)
    if len(plan.steps) > max_steps:
        raise PlanValidationError(f"plan has too many steps: {len(plan.steps)} > {max_steps}")

    planned_tools = {step.tool for step in plan.steps}
    missing_required_tools = sorted(_required_tools_from_request(request) - planned_tools)
    if missing_required_tools:
        raise PlanValidationError(f"plan omitted tools explicitly requested by the user: {missing_required_tools}")

    allowed_tools = set(allowed_tool_names())
    seen_step_ids: set[str] = set()
    for step in plan.steps:
        _validate_step_id(step.id)
        if step.id in seen_step_ids:
            raise PlanValidationError(f"duplicate step id: {step.id}")
        if step.tool not in allowed_tools:
            raise PlanValidationError(f"tool is not allowed in plan/execute mode: {step.tool}")
        if not isinstance(step.args, dict):
            raise PlanValidationError(f"args must be an object for step {step.id}")

        allowed_args = tool_properties(step.tool)
        if allowed_args:
            unexpected = sorted(set(step.args) - allowed_args)
            if unexpected:
                raise PlanValidationError(f"unexpected args for {step.tool}: {unexpected}")

        required_args = required_tool_args(step.tool)
        for arg_name, arg_value in list(step.args.items()):
            invalid_refs = _referenced_step_ids(arg_value) - seen_step_ids
            if not invalid_refs:
                continue
            if arg_name not in required_args:
                step.args.pop(arg_name)
                continue
            raise PlanValidationError(
                f"{step.id}.{arg_name} references current, future, or unknown steps: {sorted(invalid_refs)}"
            )

        if step.tool in {"launch_yolo_training", "export_yolo_torchscript"} and step.args.get("execute") is True:
            raise PlanValidationError(f"{step.tool} execute=true requires an explicit confirmation flow")
        if step.tool == "prune_yolo_by_visualized" and step.args.get("confirm_delete") is True:
            raise PlanValidationError("prune_yolo_by_visualized confirm_delete=true requires an explicit confirmation flow")

        seen_step_ids.add(step.id)

    return plan
