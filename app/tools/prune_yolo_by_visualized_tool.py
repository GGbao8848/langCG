from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from langchain.tools import tool

from app.tools.dataset_clean_tool import IMAGE_SUFFIXES


def _find_visualized_datasets(input_dir: Path) -> list[tuple[Path, Path, Path]]:
    datasets: list[tuple[Path, Path, Path]] = []
    for images_dir in sorted(path for path in input_dir.rglob("images") if path.is_dir()):
        dataset_root = images_dir.parent
        labels_dir = dataset_root / "labels"
        visualized_dir = dataset_root / "visualized"
        if labels_dir.is_dir() and visualized_dir.is_dir():
            datasets.append((images_dir, labels_dir, visualized_dir))
    return datasets


def _has_visualized_image(visualized_dir: Path, rel_image_path: Path) -> bool:
    exact_path = visualized_dir / rel_image_path
    if exact_path.is_file():
        return True

    # Allow manual review tools that normalize output extensions while keeping stems.
    stem_path = (visualized_dir / rel_image_path).with_suffix("")
    return any(stem_path.with_suffix(suffix).is_file() for suffix in IMAGE_SUFFIXES)


def _cleanup_empty_parents(path: Path, stop_dir: Path) -> None:
    current = path.parent
    stop_dir = stop_dir.resolve()
    while current.resolve() != stop_dir and stop_dir in current.resolve().parents:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _remove_or_move(path: Path, dataset_root: Path, trash_dir: Optional[Path]) -> Path:
    if trash_dir is None:
        path.unlink()
        return path

    target = trash_dir / path.relative_to(dataset_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return target


@tool
def prune_yolo_by_visualized(
    input_dir: str,
    dry_run: bool = True,
    confirm_delete: bool = False,
    max_delete_ratio: float = 0.5,
    trash_dir: Optional[str] = None,
) -> str:
    """按可视化审核结果清理 YOLO 数据集：visualized 中已删除的图，对应删除原图和 labels 标注。

    Args:
        input_dir (str): 数据集根目录，要求每组数据包含同级 images、labels、visualized。
        dry_run (bool): 默认 True，只统计将要删除的文件，不实际删除。
        confirm_delete (bool): 真正删除或移动时必须显式传 True，防止误删。
        max_delete_ratio (float): 单个数据集允许删除的最大图像比例，默认 0.5。
        trash_dir (Optional[str]): 如果提供，则把文件移动到该目录而不是直接删除。
    """
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise ValueError(f"input_dir不存在: {input_path}")
    if not 0 <= max_delete_ratio <= 1:
        raise ValueError("max_delete_ratio必须在[0, 1]范围内")

    datasets = _find_visualized_datasets(input_path)
    if not datasets:
        raise ValueError("未找到有效数据集。需要同级 images、labels、visualized 目录。")

    trash_path = Path(trash_dir).expanduser().resolve() if trash_dir else None
    if not dry_run and not confirm_delete:
        raise ValueError("真正删除或移动前必须传 confirm_delete=True。建议先 dry_run=True 预览。")

    total_images = 0
    total_prune_images = 0
    total_prune_labels = 0
    total_missing_labels = 0
    examples: list[str] = []

    for images_dir, labels_dir, visualized_dir in datasets:
        dataset_root = images_dir.parent
        image_paths = [
            path
            for path in sorted(images_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not image_paths:
            continue

        prune_pairs: list[tuple[Path, Optional[Path]]] = []
        for image_path in image_paths:
            rel_image_path = image_path.relative_to(images_dir)
            if _has_visualized_image(visualized_dir, rel_image_path):
                continue

            label_path = (labels_dir / rel_image_path).with_suffix(".txt")
            prune_pairs.append((image_path, label_path if label_path.is_file() else None))

        delete_ratio = len(prune_pairs) / len(image_paths)
        if delete_ratio > max_delete_ratio:
            raise ValueError(
                f"{dataset_root} 将删除 {len(prune_pairs)}/{len(image_paths)} "
                f"({delete_ratio:.2%})，超过 max_delete_ratio={max_delete_ratio:.2%}。"
            )

        total_images += len(image_paths)
        total_prune_images += len(prune_pairs)
        total_prune_labels += sum(1 for _image_path, label_path in prune_pairs if label_path is not None)
        total_missing_labels += sum(1 for _image_path, label_path in prune_pairs if label_path is None)

        for image_path, label_path in prune_pairs[:10]:
            examples.append(str(image_path))
            if len(examples) >= 10:
                break

        if dry_run:
            continue

        for image_path, label_path in prune_pairs:
            _remove_or_move(image_path, dataset_root, trash_path)
            _cleanup_empty_parents(image_path, images_dir)
            if label_path is not None:
                _remove_or_move(label_path, dataset_root, trash_path)
                _cleanup_empty_parents(label_path, labels_dir)

    action = "预览" if dry_run else ("已移动" if trash_path is not None else "已删除")
    example_text = "；examples=" + " | ".join(examples) if examples else ""
    trash_text = f"，trash_dir={trash_path}" if trash_path is not None else ""
    return (
        f"{action}。datasets={len(datasets)}，images={total_images}，"
        f"prune_images={total_prune_images}，prune_labels={total_prune_labels}，"
        f"missing_labels={total_missing_labels}{trash_text}{example_text}"
    )
