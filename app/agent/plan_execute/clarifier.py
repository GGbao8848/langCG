from __future__ import annotations

from app.agent.plan_execute.models import AgentPlan, MissingField, PlanClarification


def _has_value(args: dict, key: str) -> bool:
    value = args.get(key)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(bool(str(item).strip()) for item in value)
    return True


def clarify_plan(plan: AgentPlan, request: str | None = None) -> PlanClarification | None:
    questions: list[str] = []
    reasons: list[str] = []
    missing_fields: list[MissingField] = []

    for step in plan.steps:
        args = step.args or {}
        if step.tool == "publish_yolo_dataset":
            if not _has_value(args, "detector_path") and not _has_value(args, "oldyaml"):
                reasons.append("发布数据集需要明确发布上下文，但计划中缺少 detector_path 或 oldyaml。")
                questions.append("请提供 detector_path（检测器目录）或 oldyaml（历史 yaml 路径）。")
                missing_fields.extend(
                    [
                        MissingField(
                            step_id=step.id,
                            tool=step.tool,
                            field="detector_path",
                            field_type="absolute_directory_path",
                            description="检测器目录路径，用于推断数据集发布目标。",
                            example="/Users/songkui/mycode/langCG/publish_workspace/nzxj_louyou",
                            required_one_of=["detector_path", "oldyaml"],
                        ),
                        MissingField(
                            step_id=step.id,
                            tool=step.tool,
                            field="oldyaml",
                            field_type="yaml_path",
                            description="历史数据集 yaml 路径，用于增量发布并继承类别信息。",
                            example="/Users/songkui/mycode/langCG/publish_workspace/nzxj_louyou/datasets/v1/v1.yaml",
                            required_one_of=["detector_path", "oldyaml"],
                        ),
                    ]
                )

        if step.tool == "launch_yolo_training" and args.get("execute") is True:
            reasons.append("训练任务设置了 execute=true，属于会启动长任务的高风险操作。")
            questions.append("请确认是否立即执行训练；若确认，请明确回复允许执行训练。")

        if step.tool == "export_yolo_torchscript" and args.get("execute") is True:
            reasons.append("模型导出设置了 execute=true，属于会执行本地/远程命令的操作。")
            questions.append("请确认是否立即执行模型导出。")

        if step.tool == "prune_yolo_by_visualized" and args.get("confirm_delete") is True:
            reasons.append("误报清理计划包含 confirm_delete=true，会删除数据。")
            questions.append("请确认是否允许删除误报样本。")

    if not questions:
        return None
    return PlanClarification(questions=questions, blocking_reasons=reasons, missing_fields=missing_fields)
