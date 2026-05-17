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
    clarification: dict[str, Any]
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
    needs_clarification: bool = False


@dataclass
class StepOutput:
    output_dir: str | None = None
    yaml_path: str | None = None
    published_dataset_dir: str | None = None
    raw_result: str = ""
    checks: list[str] = field(default_factory=list)


class PlanValidationError(ValueError):
    pass


@dataclass
class MissingField:
    step_id: str
    tool: str
    field: str
    field_type: str
    description: str
    example: str
    required_one_of: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "field": self.field,
            "field_type": self.field_type,
            "description": self.description,
            "example": self.example,
            "required_one_of": self.required_one_of,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MissingField":
        return cls(
            step_id=str(value.get("step_id") or ""),
            tool=str(value.get("tool") or ""),
            field=str(value.get("field") or ""),
            field_type=str(value.get("field_type") or ""),
            description=str(value.get("description") or ""),
            example=str(value.get("example") or ""),
            required_one_of=[str(item) for item in value.get("required_one_of") or []],
        )


@dataclass
class PlanClarification:
    questions: list[str]
    blocking_reasons: list[str] = field(default_factory=list)
    missing_fields: list[MissingField] = field(default_factory=list)

    def render(self, plan: AgentPlan | None = None) -> str:
        lines = ["执行前需要补充信息，已暂停工具执行。"]
        if plan is not None:
            lines.extend(["", "当前计划:"])
            for index, step in enumerate(plan.steps, start=1):
                lines.append(f"{index}. {step.id} -> `{step.tool}`")
        if self.blocking_reasons:
            lines.extend(["", "发现的问题:"])
            lines.extend(f"- {reason}" for reason in self.blocking_reasons)
        if self.questions:
            lines.extend(["", "请补充:"])
            lines.extend(f"- {question}" for question in self.questions)
        if self.missing_fields:
            lines.extend(["", "字段要求:"])
            for item in self.missing_fields:
                one_of = f"，可选字段: {', '.join(item.required_one_of)}" if item.required_one_of else ""
                lines.append(
                    f"- {item.tool}.{item.field}: {item.description}；类型: {item.field_type}；"
                    f"示例: `{item.example}`{one_of}"
                )
        return "\n".join(lines)
