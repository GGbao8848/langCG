from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.tools import tool
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# 滑窗标注策略说明：
# 1. 目标在当前滑窗中的可见比例 >= min_vis_ratio 时，保留该目标并写出新标注。
# 2. 目标在当前滑窗中的可见比例 <= ignore_vis_ratio 时，认为只露出极小一部分，直接忽略。
# 3. 目标在当前滑窗中的可见比例落在 (ignore_vis_ratio, min_vis_ratio) 之间时，
#    认为这是“危险区域”，直接丢弃整个滑窗，而不是只删除这个目标。
#
# 这样做的原因是：如果一个目标在图里仍然明显可见，但我们只删掉它的标注、继续保留该滑窗，
# 就会制造“图里有目标但标签缺失”的漏标样本。对于一个滑窗里的多个目标，当前策略不是逐个做
# 激进删减，而是只要存在这种半截目标，就整窗放弃，避免产生含糊或错误的训练数据。

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _parse_ratio(aspect_ratio: str) -> Tuple[int, int]:
    text = aspect_ratio.replace("：", ":").strip()
    parts = [part.strip() for part in text.split(":")]
    if len(parts) != 2:
        raise ValueError("aspect_ratio格式必须类似'1:1'或'2:1'")
    width_ratio = int(parts[0])
    height_ratio = int(parts[1])
    if width_ratio <= 0 or height_ratio <= 0:
        raise ValueError("aspect_ratio中的值必须为正整数")
    return width_ratio, height_ratio


def _find_yolo_datasets(input_dir: Path) -> List[Tuple[Path, Path]]:
    datasets: List[Tuple[Path, Path]] = []
    for path in input_dir.rglob("images"):
        if not path.is_dir():
            continue
        labels_dir = path.parent / "labels"
        if labels_dir.is_dir():
            datasets.append((path, labels_dir))
    return sorted(datasets)


def _iter_windows(image_width: int, image_height: int, window_width: int, stride_x: int) -> List[Tuple[int, int, int, int]]:
    if window_width >= image_width:
        return [(0, 0, image_width, image_height)]

    windows: List[Tuple[int, int, int, int]] = []
    left = 0
    while left + window_width < image_width:
        windows.append((left, 0, left + window_width, image_height))
        left += stride_x
    windows.append((image_width - window_width, 0, image_width, image_height))
    return windows


def _yolo_to_xyxy(
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    x1 = (x_center - box_width / 2.0) * image_width
    y1 = (y_center - box_height / 2.0) * image_height
    x2 = (x_center + box_width / 2.0) * image_width
    y2 = (y_center + box_height / 2.0) * image_height
    return x1, y1, x2, y2


def _clip_box_xyxy(
    box: Tuple[float, float, float, float],
    crop_box: Tuple[int, int, int, int],
) -> Optional[Tuple[float, float, float, float]]:
    x1, y1, x2, y2 = box
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    clipped_x1 = max(x1, crop_left)
    clipped_y1 = max(y1, crop_top)
    clipped_x2 = min(x2, crop_right)
    clipped_y2 = min(y2, crop_bottom)

    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        return None
    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


def _area_xyxy(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _xyxy_to_crop_yolo(
    clipped_box: Tuple[float, float, float, float],
    crop_box: Tuple[int, int, int, int],
) -> Tuple[float, float, float, float]:
    clipped_x1, clipped_y1, clipped_x2, clipped_y2 = clipped_box
    crop_left, crop_top, crop_right, crop_bottom = crop_box
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top
    new_x_center = ((clipped_x1 + clipped_x2) / 2.0 - crop_left) / crop_width
    new_y_center = ((clipped_y1 + clipped_y2) / 2.0 - crop_top) / crop_height
    new_width = (clipped_x2 - clipped_x1) / crop_width
    new_height = (clipped_y2 - clipped_y1) / crop_height
    return new_x_center, new_y_center, new_width, new_height


def _load_yolo_labels(label_path: Path) -> List[Tuple[str, float, float, float, float]]:
    if not label_path.exists():
        return []

    labels: List[Tuple[str, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x_center, y_center, width, height = parts
        labels.append((cls, float(x_center), float(y_center), float(width), float(height)))
    return labels


def _build_crop_labels(
    labels: List[Tuple[str, float, float, float, float]],
    image_width: int,
    image_height: int,
    crop_box: Tuple[int, int, int, int],
    min_vis_ratio: float,
    ignore_vis_ratio: float,
) -> List[str]:
    written_lines: List[str] = []
    for cls, x_center, y_center, width, height in labels:
        box = _yolo_to_xyxy(
            x_center=x_center,
            y_center=y_center,
            box_width=width,
            box_height=height,
            image_width=image_width,
            image_height=image_height,
        )
        area = _area_xyxy(box)
        if area <= 0:
            continue

        clipped = _clip_box_xyxy(box, crop_box)
        if clipped is None:
            continue
        visible_ratio = _area_xyxy(clipped) / area
        if ignore_vis_ratio < visible_ratio < min_vis_ratio:
            # 命中危险区域时，整窗丢弃。
            return []
        if visible_ratio < min_vis_ratio:
            continue

        new_x_center, new_y_center, new_width, new_height = _xyxy_to_crop_yolo(clipped, crop_box)
        written_lines.append(
            f"{cls} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}"
        )
    return written_lines


@tool(parse_docstring=True)
def yolo_sliding_window_crop(
    input_dir: str,
    output_dir: Optional[str] = None,
    aspect_ratio: str = "1:1",
    overlap_ratio: float = 0.5,
    min_vis_ratio: float = 0.5,
    ignore_vis_ratio: float = 0.05,
) -> str:
    """对 YOLO 数据集执行滑窗裁剪，递归支持多层级的 images/labels 同级目录结构。

    Args:
        input_dir (str): 输入目录，例如 /dataset。
        output_dir (Optional[str]): 输出目录，默认自动生成为 /dataset_yolo_sliding_window。
        aspect_ratio (str): 窗口宽高比，默认 1:1；例如 2:1 表示宽度 = 当前图像高度 * 2。
        overlap_ratio (float): 相邻窗口重叠比例，默认 0.5。
        min_vis_ratio (float): 标签在窗口内的最小可见比例；低于该值不保留，默认 0.5。
        ignore_vis_ratio (float): 标签在窗口内的极小可见比例阈值；介于该值和 min_vis_ratio 之间视为危险区域并丢弃整窗，默认 0.05。
    """
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise ValueError(f"input_dir不存在: {input_path}")

    if output_dir:
        output_path = Path(output_dir).expanduser().resolve()
    else:
        output_path = input_path.parent / f"{input_path.name}_yolo_sliding_window"

    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio必须在[0, 1)范围内")
    if not 0 <= min_vis_ratio <= 1:
        raise ValueError("min_vis_ratio必须在[0, 1]范围内")
    if not 0 <= ignore_vis_ratio <= 1:
        raise ValueError("ignore_vis_ratio必须在[0, 1]范围内")
    if ignore_vis_ratio > min_vis_ratio:
        raise ValueError("ignore_vis_ratio必须小于等于min_vis_ratio")

    width_ratio, height_ratio = _parse_ratio(aspect_ratio)
    datasets = _find_yolo_datasets(input_path)
    if not datasets:
        raise ValueError("No valid YOLO dataset found. Expect sibling images/labels folders.")

    classes_src = input_path / "classes.txt"
    if classes_src.is_file():
        output_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(classes_src, output_path / "classes.txt")

    total_datasets = 0
    total_images = 0
    total_crops = 0
    total_boxes = 0

    for images_dir, labels_dir in datasets:
        total_datasets += 1
        relative_root = images_dir.parent.relative_to(input_path)
        output_images_dir = output_path / relative_root / "images"
        output_labels_dir = output_path / relative_root / "labels"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
                continue

            total_images += 1
            label_path = labels_dir / f"{image_path.stem}.txt"
            labels = _load_yolo_labels(label_path)
            if not labels:
                continue

            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                image_width, image_height = rgb_image.size
                window_height = image_height
                window_width = max(1, round(window_height * width_ratio / height_ratio))
                stride_x = max(1, round(window_width * (1 - overlap_ratio)))
                windows = _iter_windows(image_width, image_height, window_width, stride_x)

                for index, crop_box in enumerate(windows):
                    label_lines = _build_crop_labels(
                        labels=labels,
                        image_width=image_width,
                        image_height=image_height,
                        crop_box=crop_box,
                        min_vis_ratio=min_vis_ratio,
                        ignore_vis_ratio=ignore_vis_ratio,
                    )
                    if not label_lines:
                        continue

                    crop = rgb_image.crop(crop_box)
                    crop_stem = f"{image_path.stem}__x{crop_box[0]}_y{crop_box[1]}_w{crop.width}_h{crop.height}_{index:03d}"
                    output_image_path = output_images_dir / f"{crop_stem}{image_path.suffix.lower()}"
                    output_label_path = output_labels_dir / f"{crop_stem}.txt"

                    crop.save(output_image_path)
                    output_label_path.write_text("\n".join(label_lines), encoding="utf-8")
                    total_crops += 1
                    total_boxes += len(label_lines)

    return (
        f"完成。datasets={total_datasets}，images={total_images}，crops={total_crops}，"
        f"boxes={total_boxes}，output_dir={output_path}"
    )
