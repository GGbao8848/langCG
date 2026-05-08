from __future__ import annotations

from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

from langchain.tools import tool
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from dataset_clean_tool import IMAGE_SUFFIXES, _extract_valid_xml_object

Image.MAX_IMAGE_PIXELS = None

_COLOR_PALETTE: list[tuple[int, int, int]] = [
    (255, 64, 64),
    (64, 200, 64),
    (64, 128, 255),
    (255, 180, 64),
    (200, 64, 200),
    (64, 200, 200),
]


def _color_for_label(label: str) -> tuple[int, int, int]:
    return _COLOR_PALETTE[sum(ord(ch) for ch in label) % len(_COLOR_PALETTE)]


def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()


def _discover_dataset_units(input_dir: Path) -> list[Path]:
    units: list[Path] = []
    direct_images_dir = input_dir / "images"
    if direct_images_dir.is_dir():
        units.append(direct_images_dir)
    for images_dir in sorted(p for p in input_dir.rglob("images") if p.is_dir()):
        if images_dir not in units:
            units.append(images_dir)
    return units


def _resolve_xml_dir(dataset_root: Path) -> Optional[Path]:
    for dirname in ("xmls", "xml", "annotations", "annotation", "XML", "XMLS"):
        candidate = dataset_root / dirname
        if candidate.is_dir():
            return candidate
    return None


def _detect_annotation_mode(dataset_root: Path, annotation_format: Optional[str]) -> tuple[str, Path]:
    labels_dir = dataset_root / "labels"
    xml_dir = _resolve_xml_dir(dataset_root)

    has_yolo = labels_dir.is_dir()
    has_xml = xml_dir is not None

    if annotation_format:
        normalized = annotation_format.strip().lower()
        if normalized == "yolo":
            if not has_yolo:
                raise ValueError(f"在数据集目录下未找到labels目录: {dataset_root}")
            return "yolo", labels_dir
        if normalized == "xml":
            if not has_xml or xml_dir is None:
                raise ValueError(f"在数据集目录下未找到xml/xmls目录: {dataset_root}")
            return "xml", xml_dir
        raise ValueError("annotation_format必须为'yolo'或'xml'")

    if has_yolo and has_xml and xml_dir is not None:
        raise ValueError(
            f"在{dataset_root}下同时检测到了YOLO和XML标注。"
            "请明确指定annotation_format='yolo'或annotation_format='xml'。"
        )
    if has_yolo:
        return "yolo", labels_dir
    if has_xml and xml_dir is not None:
        return "xml", xml_dir
    raise ValueError(f"在数据集目录下未找到labels或xml/xmls目录: {dataset_root}")


def _load_class_names(input_path: Path) -> list[str] | None:
    classes_path = input_path / "classes.txt"
    if not classes_path.is_file():
        return None
    lines = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def _parse_yolo_boxes(
    label_path: Path,
    image_width: int,
    image_height: int,
    class_names: list[str] | None,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue

        x1 = int(max(0, min((x_center - width / 2.0) * image_width, image_width - 1)))
        y1 = int(max(0, min((y_center - height / 2.0) * image_height, image_height - 1)))
        x2 = int(max(0, min((x_center + width / 2.0) * image_width, image_width - 1)))
        y2 = int(max(0, min((y_center + height / 2.0) * image_height, image_height - 1)))
        if x2 <= x1 or y2 <= y1:
            continue

        label = str(cls_id)
        if class_names and 0 <= cls_id < len(class_names):
            label = class_names[cls_id]
        boxes.append((label, (x1, y1, x2, y2)))
    return boxes


def _parse_xml_boxes(xml_path: Path) -> list[tuple[str, tuple[int, int, int, int]]]:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []

    boxes: list[tuple[str, tuple[int, int, int, int]]] = []
    for obj in tree.getroot().findall("object"):
        item = _extract_valid_xml_object(obj)
        if item is None:
            continue
        boxes.append(
            (
                item["name"],
                (item["xmin"], item["ymin"], item["xmax"], item["ymax"]),
            )
        )
    return boxes


def _draw_boxes(
    image: Image.Image,
    boxes: list[tuple[str, tuple[int, int, int, int]]],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    line_width: int,
) -> None:
    draw = ImageDraw.Draw(image)
    for label, (x1, y1, x2, y2) in boxes:
        color = _color_for_label(label)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        draw.text((x1 + 2, max(0, y1 - 16)), label, fill=color, font=font)


@tool
def annotate_visualize(
    input_dir: str,
    annotation_format: Optional[str] = None,
    line_width: int = 2,
) -> str:
    """可视化 YOLO 或 Pascal VOC XML 标注，输出到与 images 同级的 visualized。"""
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise ValueError(f"input_dir不存在: {input_path}")

    images_dirs = _discover_dataset_units(input_path)
    if not images_dirs:
        raise ValueError(f"在input_dir下未找到images目录: {input_path}")

    class_names = _load_class_names(input_path)
    font = _load_font()

    total_images = 0
    written_images = 0
    skipped_images = 0
    modes_used: set[str] = set()

    for images_dir in images_dirs:
        dataset_root = images_dir.parent
        mode, annotation_dir = _detect_annotation_mode(dataset_root, annotation_format)
        modes_used.add(mode)
        output_dir = dataset_root / "visualized"

        for image_path in sorted(images_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            total_images += 1
            rel_path = image_path.relative_to(images_dir)
            output_path = output_dir / rel_path

            if mode == "yolo":
                annotation_path = (annotation_dir / rel_path).with_suffix(".txt")
            else:
                annotation_path = (annotation_dir / rel_path).with_suffix(".xml")

            if not annotation_path.is_file():
                skipped_images += 1
                continue

            try:
                with Image.open(image_path) as image:
                    rgb_image = image.convert("RGB")
                    image_width, image_height = rgb_image.size
                    if mode == "yolo":
                        boxes = _parse_yolo_boxes(
                            annotation_path,
                            image_width=image_width,
                            image_height=image_height,
                            class_names=class_names,
                        )
                    else:
                        boxes = _parse_xml_boxes(annotation_path)

                    _draw_boxes(rgb_image, boxes, font=font, line_width=line_width)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    rgb_image.save(output_path)
                    written_images += 1
            except (UnidentifiedImageError, OSError, ValueError):
                skipped_images += 1

    return (
        f"完成。modes={sorted(modes_used)}，total_images={total_images}，"
        f"written_images={written_images}，skipped_images={skipped_images}，"
        f"output_dir=每个images同级的visualized"
    )
