from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class PlannedStep(BaseModel):
    id: str = Field(description="Stable step id, for example convert, reindex, split.")
    tool: str = Field(description="Tool name from the allowed tool list.")
    args: dict[str, Any] = Field(default_factory=dict, description="Tool arguments.")
    reason: str = Field(default="", description="Short reason for this step.")


class AgentPlan(BaseModel):
    should_execute: bool = Field(description="Whether this request should be handled by the plan/execute graph.")
    summary: str = Field(default="", description="One sentence task summary.")
    answer: str = Field(default="", description="Direct answer when should_execute is false.")
    steps: list[PlannedStep] = Field(default_factory=list, description="Ordered executable tool steps.")


class PlanExecuteState(TypedDict, total=False):
    request: str
    provider: str
    model: str
    plan: dict[str, Any]
    validation_error: str
    fallback_reason: str
    tool_calls: list[dict[str, Any]]
    final_text: str
    status: str


@dataclass
class PlanExecuteToolCall:
    id: str
    name: str
    args: dict[str, Any]
    status: Literal["done", "error"] = "done"
    result: str | None = None


@dataclass
class PlanExecuteResult:
    text: str
    tool_calls: list[PlanExecuteToolCall]
    fallback_reason: str | None = None


@dataclass
class StepOutput:
    output_dir: str | None = None
    yaml_path: str | None = None
    published_dataset_dir: str | None = None
    raw_result: str = ""
    checks: list[str] = field(default_factory=list)


class PlanValidationError(ValueError):
    pass
