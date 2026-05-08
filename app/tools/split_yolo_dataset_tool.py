from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Optional

from langchain.tools import tool

from app.tools.dataset_clean_tool import IMAGE_SUFFIXES


def _parse_ratio_pair(split_ratio: Optional[str]) -> tuple[int, int]:
    if not split_ratio:
        return 8, 2

    text = split_ratio.strip().replace("：", ":").replace("，", ",")
    if ":" in text:
        parts = [part.strip() for part in text.split(":")]
    else:
        parts = [part.strip() for part in text.split(",")]

    if len(parts) != 2:
        raise ValueError("split_ratio格式必须类似'8:2'或'8,2'")

    train_ratio = int(parts[0])
    val_ratio = int(parts[1])
    if train_ratio < 0 or val_ratio < 0:
        raise ValueError("split_ratio中的值必须为非负数")
    if train_ratio == 0 and val_ratio == 0:
        raise ValueError("split_ratio values cannot both be 0")
    return train_ratio, val_ratio


def _find_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path, Path]]:
    pairs: list[tuple[Path, Path, Path]] = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel_path = image_path.relative_to(images_dir)
        label_path = (labels_dir / rel_path).with_suffix(".txt")
        if not label_path.is_file():
            continue
        pairs.append((image_path, label_path, rel_path))
    return pairs


def _copy_pair(
    image_path: Path,
    label_path: Path,
    rel_path: Path,
    split_name: str,
    output_path: Path,
) -> None:
    target_image = output_path / split_name / "images" / rel_path
    target_label = output_path / split_name / "labels" / rel_path.with_suffix(".txt")
    target_image.parent.mkdir(parents=True, exist_ok=True)
    target_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, target_image)
    shutil.copy2(label_path, target_label)


@tool
def split_yolo_dataset(
    input_dir: str,
    mode: str = "train_val",
    output_dir: Optional[str] = None,
    split_ratio: Optional[str] = None,
    shuffle: bool = True,
    seed: int = 42,
) -> str:
    """划分 YOLO 数据集，只支持 train_only 和 train_val。

    Args:
        input_dir: 输入数据集根目录，要求包含 images/labels。
        mode: 划分模式，train_only 或 train_val。
        output_dir: 输出目录，默认生成到同级目录下的 <input>_split。
        split_ratio: train_val 模式的比例，默认 8:2；可写成 8:2 或 8,2。
        shuffle: 是否打乱，默认 True。
        seed: 打乱随机种子，默认 42。
    """
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise ValueError(f"input_dir不存在: {input_path}")

    images_dir = input_path / "images"
    labels_dir = input_path / "labels"
    if not images_dir.is_dir():
        raise ValueError(f"images目录不存在: {images_dir}")
    if not labels_dir.is_dir():
        raise ValueError(f"labels目录不存在: {labels_dir}")

    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"train_only", "train_val"}:
        raise ValueError("mode必须为'train_only'或'train_val'")

    if output_dir:
        output_path = Path(output_dir).expanduser().resolve()
    else:
        output_path = input_path.parent / f"{input_path.name}_split"

    pairs = _find_pairs(images_dir, labels_dir)
    if not pairs:
        raise ValueError(f"在目录下未找到图像与标签配对数据: {input_path}")

    working_pairs = list(pairs)
    if shuffle:
        random.Random(seed).shuffle(working_pairs)

    train_ratio, val_ratio = _parse_ratio_pair(split_ratio)
    if normalized_mode == "train_only":
        train_pairs = working_pairs
        val_pairs: list[tuple[Path, Path, Path]] = []
    else:
        total = len(working_pairs)
        train_count = round(total * train_ratio / (train_ratio + val_ratio))
        train_count = max(0, min(train_count, total))
        train_pairs = working_pairs[:train_count]
        val_pairs = working_pairs[train_count:]

    for image_path, label_path, rel_path in train_pairs:
        _copy_pair(image_path, label_path, rel_path, "train", output_path)
    for image_path, label_path, rel_path in val_pairs:
        _copy_pair(image_path, label_path, rel_path, "val", output_path)

    classes_src = input_path / "classes.txt"
    if classes_src.is_file():
        output_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(classes_src, output_path / "classes.txt")

    return (
        f"完成。mode={normalized_mode}，total_pairs={len(pairs)}，"
        f"train={len(train_pairs)}，val={len(val_pairs)}，"
        f"ratio={'train_only' if normalized_mode == 'train_only' else f'{train_ratio}:{val_ratio}'}，"
        f"output_dir={output_path}"
    )
