from __future__ import annotations

import re

from app.agent.plan_execute.config import PATH_PATTERN
from app.agent.plan_execute.models import AgentPlan, PlannedStep


def _paths_from_request(request: str) -> list[str]:
    return [match.group(0).rstrip("，,。；;") for match in PATH_PATTERN.finditer(request)]


def _split_ratio_from_request(request: str) -> str | None:
    match = re.search(r"(\d+)\s*[:：]\s*(\d+)", request)
    if not match:
        return None
    return f"{match.group(1)}:{match.group(2)}"


def _reindex_args_from_request(request: str) -> dict[str, str] | None:
    compact = request.replace(" ", "")
    if not ("索引" in compact or "重映射" in compact or "映射" in compact):
        return None
    if not ("转为0" in compact or "转成0" in compact or "到0" in compact):
        return None
    match = re.search(r"索引([0-9,，、\s]+).*?(?:转为|转成|到)0", request)
    source_indices = "0,1,2"
    if match:
        source_indices = ",".join(part for part in re.split(r"[,，、\s]+", match.group(1).strip()) if part)
    return {"source_indices": source_indices, "target_index": "0"}


def _publish_detector_path(request: str, paths: list[str]) -> str | None:
    if "发布到" not in request and "detector" not in request.lower() and "检测器目录" not in request:
        return None
    return paths[-1] if len(paths) >= 2 else None


def synthesize_plan_from_request(request: str) -> AgentPlan | None:
    paths = _paths_from_request(request)
    if not paths:
        return None

    compact = request.lower().replace(" ", "")
    steps: list[PlannedStep] = []
    current_ref = paths[0]

    def add_step(step_id: str, tool: str, args: dict, reason: str) -> None:
        nonlocal current_ref
        steps.append(PlannedStep(id=step_id, tool=tool, args=args, reason=reason))
        current_ref = f"$steps.{step_id}.output_dir"

    if "转yolo" in compact or "xml" in compact:
        add_step(
            "convert",
            "convert_xml_to_yolo",
            {"input_dir": current_ref},
            "用户要求转换为 YOLO 数据集。",
        )

    reindex_args = _reindex_args_from_request(request)
    if reindex_args:
        add_step(
            "reindex",
            "reindex_yolo_labels",
            {"input_dir": current_ref, **reindex_args},
            "用户要求将指定类别索引重映射为 0。",
        )

    if "划分" in compact or "trainval" in compact or "train_val" in compact:
        split_args = {"input_dir": current_ref, "mode": "train_val"}
        split_ratio = _split_ratio_from_request(request)
        if split_ratio:
            split_args["split_ratio"] = split_ratio
        add_step("split", "split_yolo_dataset", split_args, "用户要求划分 train/val 数据集。")

    if "滑窗" in compact or "裁剪" in compact:
        add_step("crop", "yolo_sliding_window_crop", {"input_dir": current_ref}, "用户要求滑窗裁剪。")

    crop_ref = current_ref
    if "增强" in compact:
        add_step("augment", "augment_yolo_dataset", {"input_dir": current_ref}, "用户要求对裁剪结果增强。")

    if "发布" in compact or "detector" in compact:
        publish_args: dict[str, object] = {"input_dir": crop_ref}
        if any(step.id == "augment" for step in steps):
            publish_args["input_dirs"] = ["$steps.augment.output_dir"]
        detector_path = _publish_detector_path(request, paths)
        if detector_path:
            publish_args["detector_path"] = detector_path
        steps.append(
            PlannedStep(
                id="publish",
                tool="publish_yolo_dataset",
                args=publish_args,
                reason="用户要求发布数据集，数据来源为裁剪结果及其增强结果。",
            )
        )

    if not steps:
        return None
    return AgentPlan(
        should_execute=True,
        summary="按用户请求执行数据集转换、重映射、划分、裁剪、增强和发布流程。",
        steps=steps,
    )
