from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from langchain.tools import tool
from PIL import Image, ImageEnhance

from app.tools.dataset_clean_tool import IMAGE_SUFFIXES

Image.MAX_IMAGE_PIXELS = None

_DEFAULT_FACTORS: dict[str, float] = {
    "brightness_up": 1.4,
    "brightness_down": 0.6,
    "contrast_up": 1.4,
    "contrast_down": 0.6,
}


def _discover_dataset_units(input_dir: Path) -> list[tuple[Path, Path, Path]]:
    dataset_units: list[tuple[Path, Path, Path]] = []
    seen_images_dirs: set[Path] = set()

    direct_images_dir = input_dir / "images"
    direct_labels_dir = input_dir / "labels"
    if direct_images_dir.is_dir() and direct_labels_dir.is_dir():
        dataset_units.append((Path("."), direct_images_dir, direct_labels_dir))
        seen_images_dirs.add(direct_images_dir.resolve())

    for images_dir in sorted(p for p in input_dir.rglob("images") if p.is_dir()):
        resolved = images_dir.resolve()
        if resolved in seen_images_dirs:
            continue
        labels_dir = images_dir.parent / "labels"
        if not labels_dir.is_dir():
            continue
        dataset_units.append((images_dir.parent.relative_to(input_dir), images_dir, labels_dir))
        seen_images_dirs.add(resolved)

    return dataset_units


def _load_yolo_labels(label_path: Path) -> list[tuple[str, float, float, float, float]]:
    labels: list[tuple[str, float, float, float, float]] = []
    for idx, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"YOLO标签行无效: {label_path}:{idx}: {raw_line!r}")
        try:
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError as exc:
            raise ValueError(f"YOLO标签行无效: {label_path}:{idx}: {raw_line!r}") from exc
        labels.append((parts[0], x_center, y_center, width, height))
    return labels


def _save_yolo_labels(
    label_path: Path,
    labels: list[tuple[str, float, float, float, float]],
) -> None:
    lines = [
        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        for class_id, x_center, y_center, width, height in labels
    ]
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _apply_image_transform(image: Image.Image, augmentation_name: str) -> Image.Image:
    if augmentation_name == "hflip":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if augmentation_name == "vflip":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if augmentation_name == "brightness_up":
        return ImageEnhance.Brightness(image).enhance(_DEFAULT_FACTORS["brightness_up"])
    if augmentation_name == "brightness_down":
        return ImageEnhance.Brightness(image).enhance(_DEFAULT_FACTORS["brightness_down"])
    if augmentation_name == "contrast_up":
        return ImageEnhance.Contrast(image).enhance(_DEFAULT_FACTORS["contrast_up"])
    if augmentation_name == "contrast_down":
        return ImageEnhance.Contrast(image).enhance(_DEFAULT_FACTORS["contrast_down"])
    if augmentation_name == "rotate_left_90":
        return image.transpose(Image.Transpose.ROTATE_90)
    if augmentation_name == "rotate_right_90":
        return image.transpose(Image.Transpose.ROTATE_270)
    raise ValueError(f"不支持的增强方式: {augmentation_name}")


def _apply_label_transform(
    labels: list[tuple[str, float, float, float, float]],
    augmentation_name: str,
) -> list[tuple[str, float, float, float, float]]:
    transformed: list[tuple[str, float, float, float, float]] = []
    for class_id, x_center, y_center, width, height in labels:
        if augmentation_name == "hflip":
            transformed.append((class_id, 1.0 - x_center, y_center, width, height))
        elif augmentation_name == "vflip":
            transformed.append((class_id, x_center, 1.0 - y_center, width, height))
        elif augmentation_name == "rotate_left_90":
            transformed.append((class_id, y_center, 1.0 - x_center, height, width))
        elif augmentation_name == "rotate_right_90":
            transformed.append((class_id, 1.0 - y_center, x_center, height, width))
        else:
            transformed.append((class_id, x_center, y_center, width, height))
    return transformed


def _selected_augmentations(
    horizontal_flip: bool,
    vertical_flip: bool,
    brightness_up: bool,
    brightness_down: bool,
    contrast_up: bool,
    contrast_down: bool,
    rotate_left_90: bool,
    rotate_right_90: bool,
) -> list[str]:
    selected: list[str] = []
    if horizontal_flip:
        selected.append("hflip")
    if vertical_flip:
        selected.append("vflip")
    if brightness_up:
        selected.append("brightness_up")
    if brightness_down:
        selected.append("brightness_down")
    if contrast_up:
        selected.append("contrast_up")
    if contrast_down:
        selected.append("contrast_down")
    if rotate_left_90:
        selected.append("rotate_left_90")
    if rotate_right_90:
        selected.append("rotate_right_90")
    return selected


@tool
def augment_yolo_dataset(
    input_dir: str,
    output_dir: Optional[str] = None,
    horizontal_flip: bool = True,
    vertical_flip: bool = True,
    brightness_up: bool = True,
    brightness_down: bool = True,
    contrast_up: bool = True,
    contrast_down: bool = True,
    rotate_left_90: bool = False,
    rotate_right_90: bool = False,
    rotate_left90: Optional[bool] = None,
    rotate_right90: Optional[bool] = None,
) -> str:
    """增强 YOLO 数据集，只保留翻转、亮度、对比度和可选 90 度旋转。"""
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise ValueError(f"input_dir不存在: {input_path}")

    if output_dir:
        output_path = Path(output_dir).expanduser().resolve()
    else:
        output_path = input_path / "augment"

    if output_path == input_path:
        raise ValueError("output_dir不能与input_dir相同")

    dataset_units = _discover_dataset_units(input_path)
    if not dataset_units:
        raise ValueError(f"在目录下未找到成对的images/labels目录: {input_path}")

    if rotate_left90 is not None:
        rotate_left_90 = rotate_left90
    if rotate_right90 is not None:
        rotate_right_90 = rotate_right90

    selected = _selected_augmentations(
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        brightness_up=brightness_up,
        brightness_down=brightness_down,
        contrast_up=contrast_up,
        contrast_down=contrast_down,
        rotate_left_90=rotate_left_90,
        rotate_right_90=rotate_right_90,
    )
    if not selected:
        raise ValueError("至少要启用一种增强选项")

    processed_images = 0
    skipped_images = 0
    generated_images = 0

    classes_src = input_path / "classes.txt"
    if classes_src.is_file():
        output_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(classes_src, output_path / "classes.txt")

    for relative_root, images_dir, labels_dir in dataset_units:
        output_base = output_path if relative_root == Path(".") else output_path / relative_root
        output_images_dir = output_base / "images"
        output_labels_dir = output_base / "labels"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(images_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            rel_path = image_path.relative_to(images_dir)
            label_path = (labels_dir / rel_path).with_suffix(".txt")
            if not label_path.is_file():
                skipped_images += 1
                continue

            labels = _load_yolo_labels(label_path)
            with Image.open(image_path) as img:
                source_image = img.convert("RGB")
                for augmentation_name in selected:
                    augmented_image = _apply_image_transform(source_image, augmentation_name)
                    augmented_labels = _apply_label_transform(labels, augmentation_name)
                    target_rel = rel_path.with_name(
                        f"{rel_path.stem}_{augmentation_name}{rel_path.suffix.lower()}"
                    )
                    target_image = output_images_dir / target_rel
                    target_label = output_labels_dir / target_rel.with_suffix(".txt")
                    target_image.parent.mkdir(parents=True, exist_ok=True)
                    augmented_image.save(target_image)
                    _save_yolo_labels(target_label, augmented_labels)
                    generated_images += 1

            processed_images += 1

    return (
        f"完成。processed_images={processed_images}，skipped_images={skipped_images}，"
        f"generated_images={generated_images}，generated_labels={generated_images}，"
        f"augmentations={selected}，output_dir={output_path}"
    )
