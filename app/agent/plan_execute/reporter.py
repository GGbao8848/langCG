from __future__ import annotations

from app.agent.plan_execute.models import AgentPlan, PlanExecuteToolCall, StepOutput


def render_report(
    plan: AgentPlan,
    tool_calls: list[PlanExecuteToolCall],
    outputs: dict[str, StepOutput],
    error: str | None,
) -> str:
    lines = [
        "已使用 LangGraph plan/execute 智能体处理。",
        f"计划: {plan.summary or '未提供摘要'}",
        "",
        "执行结果:",
    ]
    for step in plan.steps:
        call = next((item for item in tool_calls if item.id == f"plan-{step.id}"), None)
        if call is None:
            lines.append(f"- 未执行: {step.id} -> `{step.tool}`")
            continue
        status_text = "成功" if call.status == "done" else "失败"
        output = outputs.get(step.id)
        evidence = output.yaml_path if output and output.yaml_path else output.output_dir if output else None
        lines.append(f"- {status_text}: {step.id} -> `{step.tool}`")
        if evidence:
            lines.append(f"  证据: `{evidence}`")

    checks = [check for output in outputs.values() for check in output.checks]
    if checks:
        lines.append("")
        lines.append("验收:")
        for check in checks[:8]:
            lines.append(f"- {check}")

    if error:
        lines.append("")
        lines.append(f"失败原因: `{error}`")
        lines.append("后续步骤已停止，避免在错误数据上继续处理。")
    else:
        publish_output = next((output for output in outputs.values() if output.yaml_path), None)
        if publish_output and publish_output.yaml_path:
            lines.append("")
            lines.append(f"交付物: `{publish_output.yaml_path}`")

    return "\n".join(lines)
