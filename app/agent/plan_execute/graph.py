from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.agent.plan_execute.config import should_attempt_plan_execute
from app.agent.plan_execute.executor import execute_plan, execute_plan_steps
from app.agent.plan_execute.models import (
    AgentPlan,
    PlanExecuteResult,
    PlanExecuteState,
    PlanExecuteToolCall,
    PlanValidationError,
)
from app.agent.plan_execute.planner import plan_with_llm
from app.agent.plan_execute.reporter import render_report
from app.agent.plan_execute.synthesizer import synthesize_plan_from_request
from app.agent.plan_execute.validator import validate_plan


def _build_graph(plan_override: AgentPlan | None = None):
    def plan_node(state: PlanExecuteState) -> dict[str, Any]:
        try:
            plan = plan_override or plan_with_llm(
                state["request"],
                state["provider"],
                state["model"],
            )
        except (ValidationError, json.JSONDecodeError, ValueError) as error:
            return {"fallback_reason": f"planner failed: {error}", "status": "fallback"}
        return {"plan": plan.model_dump(mode="json"), "status": "planned"}

    def validate_node(state: PlanExecuteState) -> dict[str, Any]:
        try:
            plan = AgentPlan.model_validate(state["plan"])
            validate_plan(plan, state.get("request"))
        except (ValidationError, PlanValidationError) as error:
            synthesized_plan = synthesize_plan_from_request(state.get("request", ""))
            if synthesized_plan is not None:
                try:
                    validate_plan(synthesized_plan, state.get("request"))
                except PlanValidationError:
                    pass
                else:
                    return {
                        "plan": synthesized_plan.model_dump(mode="json"),
                        "status": "validated",
                    }
            return {
                "validation_error": str(error),
                "fallback_reason": f"plan validation failed: {error}",
                "status": "fallback",
            }
        return {"status": "validated"}

    def execute_node(state: PlanExecuteState) -> dict[str, Any]:
        plan = AgentPlan.model_validate(state["plan"])
        tool_calls, outputs, error = execute_plan(plan)
        return {
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "args": call.args,
                    "status": call.status,
                    "result": call.result,
                }
                for call in tool_calls
            ],
            "final_text": render_report(plan, tool_calls, outputs, error),
            "status": "error" if error else "done",
        }

    def report_node(state: PlanExecuteState) -> dict[str, Any]:
        if state.get("final_text"):
            return {}
        return {
            "fallback_reason": state.get("fallback_reason") or "plan/execute graph did not produce a final report",
            "status": "fallback",
        }

    def route_after_plan(state: PlanExecuteState) -> str:
        return "report" if state.get("fallback_reason") else "validate"

    def route_after_validate(state: PlanExecuteState) -> str:
        return "report" if state.get("fallback_reason") else "execute"

    graph = StateGraph(PlanExecuteState)
    graph.add_node("plan", plan_node)
    graph.add_node("validate", validate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("report", report_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_after_plan, {"validate": "validate", "report": "report"})
    graph.add_conditional_edges("validate", route_after_validate, {"execute": "execute", "report": "report"})
    graph.add_edge("execute", "report")
    graph.add_edge("report", END)
    return graph.compile()


def run_plan_execute_agent(
    *,
    request: str,
    provider: str,
    model: str,
    plan_override: AgentPlan | None = None,
) -> PlanExecuteResult | None:
    if plan_override is None and not should_attempt_plan_execute(request):
        return None

    graph = _build_graph(plan_override=plan_override)
    state = graph.invoke(
        {
            "request": request,
            "provider": provider,
            "model": model,
        }
    )
    fallback_reason = state.get("fallback_reason")
    if fallback_reason:
        return PlanExecuteResult(text="", tool_calls=[], fallback_reason=fallback_reason)

    tool_calls = [
        PlanExecuteToolCall(
            id=str(item["id"]),
            name=str(item["name"]),
            args=dict(item.get("args") or {}),
            status=item.get("status", "done"),
            result=item.get("result"),
        )
        for item in state.get("tool_calls", [])
    ]
    return PlanExecuteResult(
        text=state.get("final_text", ""),
        tool_calls=tool_calls,
    )


def prepare_plan_execute_agent(
    *,
    request: str,
    provider: str,
    model: str,
    plan_override: AgentPlan | None = None,
) -> tuple[AgentPlan | None, str | None]:
    if plan_override is None and not should_attempt_plan_execute(request):
        return None, "request did not match plan/execute routing"

    try:
        plan = plan_override or plan_with_llm(request, provider, model)
    except (ValidationError, json.JSONDecodeError, ValueError) as error:
        return None, f"planner failed: {error}"

    try:
        validate_plan(plan, request)
    except PlanValidationError as error:
        synthesized_plan = synthesize_plan_from_request(request)
        if synthesized_plan is None:
            return None, f"plan validation failed: {error}"
        try:
            validate_plan(synthesized_plan, request)
        except PlanValidationError as synthesized_error:
            return None, f"plan validation failed: {synthesized_error}"
        plan = synthesized_plan

    return plan, None


def stream_plan_execute_agent(
    *,
    request: str,
    provider: str,
    model: str,
    plan_override: AgentPlan | None = None,
):
    plan, error = prepare_plan_execute_agent(
        request=request,
        provider=provider,
        model=model,
        plan_override=plan_override,
    )
    if error or plan is None:
        yield {"type": "fallback", "reason": error or "unknown plan/execute error"}
        return

    yield {"type": "plan", "plan": plan}
    tool_calls: list[PlanExecuteToolCall] = []
    outputs = {}
    final_error: str | None = None

    for event_type, call, outputs, error in execute_plan_steps(plan):
        if event_type == "start":
            yield {"type": "tool_start", "call": call}
            continue
        tool_calls.append(call)
        final_error = error
        yield {"type": "tool_end", "call": call}
        if error:
            break

    yield {
        "type": "done",
        "text": render_report(plan, tool_calls, outputs, final_error),
    }
