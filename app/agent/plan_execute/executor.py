from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml

from app.agent.tool_registry import TOOLS
from app.agent.plan_execute.config import (
    OUTPUT_DIR_PATTERN,
    PUBLISHED_DIR_PATTERN,
    STEP_REF_PATTERN,
    YAML_PATH_PATTERN,
)
from app.agent.plan_execute.models import AgentPlan, PlannedStep, PlanExecuteToolCall, PlanValidationError, StepOutput


def _preview_from_tool_result(result: str) -> str:
    text = result.removeprefix("tool执行失败:").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return result
    if isinstance(payload, dict) and isinstance(payload.get("output_preview"), str):
        return payload["output_preview"]
    return result


def _extract_field(pattern, text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip().rstrip("，,。；;")


def _step_output_from_result(result: str) -> StepOutput:
    preview = _preview_from_tool_result(result)
    return StepOutput(
        output_dir=_extract_field(OUTPUT_DIR_PATTERN, preview),
        yaml_path=_extract_field(YAML_PATH_PATTERN, preview),
        published_dataset_dir=_extract_field(PUBLISHED_DIR_PATTERN, preview),
        raw_result=result,
    )


def _resolve_arg_value(value: Any, outputs: dict[str, StepOutput]) -> Any:
    if isinstance(value, str):
        match = STEP_REF_PATTERN.match(value.strip())
        if not match:
            return value
        step_id, field_name = match.groups()
        output = outputs.get(step_id)
        if output is None:
            raise PlanValidationError(f"step reference points to unknown or unfinished step: {value}")
        resolved = getattr(output, field_name)
        if not resolved:
            raise PlanValidationError(f"step reference has no value: {value}")
        return resolved
    if isinstance(value, list):
        return [_resolve_arg_value(item, outputs) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_arg_value(item, outputs) for key, item in value.items()}
    return value


def _resolve_args(args: dict[str, Any], outputs: dict[str, StepOutput]) -> dict[str, Any]:
    return {key: _resolve_arg_value(value, outputs) for key, value in args.items()}


def _validate_resolved_args(tool_name: str, args: dict[str, Any]) -> None:
    tool = TOOLS[tool_name]
    if tool.args_schema is not None:
        tool.args_schema.model_validate(args)


def _is_local_path(value: str | None) -> bool:
    if not value:
        return False
    return value.startswith("/") and not value.startswith(("sftp://", "ssh://"))


def _verify_step_output(step: PlannedStep, output: StepOutput) -> list[str]:
    checks: list[str] = []
    if _is_local_path(output.output_dir):
        output_path = Path(output.output_dir or "")
        if not output_path.exists():
            raise PlanValidationError(f"{step.id} output_dir does not exist: {output_path}")
        checks.append(f"{step.id} output_dir exists")

    if step.tool == "publish_yolo_dataset" and _is_local_path(output.yaml_path):
        yaml_path = Path(output.yaml_path or "")
        if not yaml_path.is_file():
            raise PlanValidationError(f"publish yaml does not exist: {yaml_path}")
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for split in ("train", "val", "test"):
                value = data.get(split)
                if not value:
                    continue
                paths = value if isinstance(value, list) else [value]
                path_texts = [str(item) for item in paths]
                if len(path_texts) != len(set(path_texts)):
                    raise PlanValidationError(f"publish yaml has duplicate {split} paths")
                if any("/augment/" in item for item in path_texts):
                    raise PlanValidationError(f"publish yaml has nested augment path in {split}")
        checks.append("publish yaml exists and has no duplicated split paths")
    return checks


def execute_plan(plan) -> tuple[list[PlanExecuteToolCall], dict[str, StepOutput], str | None]:
    outputs: dict[str, StepOutput] = {}
    tool_calls: list[PlanExecuteToolCall] = []

    for step in plan.steps:
        try:
            args = _resolve_args(step.args, outputs)
            _validate_resolved_args(step.tool, args)
            result = str(TOOLS[step.tool].invoke(args))
            status: Literal["done", "error"] = "error" if result.startswith("tool执行失败") else "done"
            call = PlanExecuteToolCall(
                id=f"plan-{step.id}",
                name=step.tool,
                args=args,
                status=status,
                result=result,
            )
            tool_calls.append(call)
            output = _step_output_from_result(result)
            output.checks = _verify_step_output(step, output) if status == "done" else []
            outputs[step.id] = output
            if status == "error":
                return tool_calls, outputs, f"{step.id} failed"
        except Exception as error:
            tool_calls.append(
                PlanExecuteToolCall(
                    id=f"plan-{step.id}",
                    name=step.tool,
                    args=step.args,
                    status="error",
                    result=f"tool执行失败: {error}",
                )
            )
            return tool_calls, outputs, str(error)

    return tool_calls, outputs, None


def execute_plan_steps(plan: AgentPlan):
    outputs: dict[str, StepOutput] = {}
    tool_calls: list[PlanExecuteToolCall] = []

    for step in plan.steps:
        try:
            args = _resolve_args(step.args, outputs)
            _validate_resolved_args(step.tool, args)
            yield "start", PlanExecuteToolCall(id=f"plan-{step.id}", name=step.tool, args=args), outputs, None

            result = str(TOOLS[step.tool].invoke(args))
            status: Literal["done", "error"] = "error" if result.startswith("tool执行失败") else "done"
            call = PlanExecuteToolCall(
                id=f"plan-{step.id}",
                name=step.tool,
                args=args,
                status=status,
                result=result,
            )
            tool_calls.append(call)
            output = _step_output_from_result(result)
            output.checks = _verify_step_output(step, output) if status == "done" else []
            outputs[step.id] = output
            yield "end", call, outputs, f"{step.id} failed" if status == "error" else None
            if status == "error":
                return
        except Exception as error:
            call = PlanExecuteToolCall(
                id=f"plan-{step.id}",
                name=step.tool,
                args=step.args,
                status="error",
                result=f"tool执行失败: {error}",
            )
            tool_calls.append(call)
            yield "end", call, outputs, str(error)
            return
