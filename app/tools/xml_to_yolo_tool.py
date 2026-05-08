from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

from langchain.tools import tool

from app.tools.dataset_clean_tool import IMAGE_SUFFIXES, _extract_valid_xml_object, _list_files


def _find_image_for_xml(xml_path: Path) -> Optional[Path]:
    same_dir = xml_path.parent
    for suffix in sorted(IMAGE_SUFFIXES):
        candidate = same_dir / f"{xml_path.stem}{suffix}"
        if candidate.is_file():
            return candidate

    parent_dir = xml_path.parent.parent
    for dirname in ("images", "Images", "JPEGImages"):
        for suffix in sorted(IMAGE_SUFFIXES):
            candidate = parent_dir / dirname / f"{xml_path.stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def _safe_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return None


def _xml_size(root: ET.Element) -> Optional[tuple[int, int]]:
    size = root.find("size")
    if size is None:
        return None
    width = _safe_float(size.findtext("width"))
    height = _safe_float(size.findtext("height"))
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    return int(round(width)), int(round(height))


def _clip_box(xmin: int, ymin: int, xmax: int, ymax: int, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    xmin = max(0, min(xmin, width))
    xmax = max(0, min(xmax, width))
    ymin = max(0, min(ymin, height))
    ymax = max(0, min(ymax, height))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def _xyxy_to_yolo(xmin: int, ymin: int, xmax: int, ymax: int, width: int, height: int) -> tuple[float, float, float, float]:
    box_width = xmax - xmin
    box_height = ymax - ymin
    x_center = xmin + box_width / 2.0
    y_center = ymin + box_height / 2.0
    return x_center / width, y_center / height, box_width / width, box_height / height


def _parse_classes(classes: Optional[str]) -> List[str]:
    if not classes:
        return []
    items = [item.strip() for item in classes.split(",")]
    return [item for item in items if item]


def _default_output_root(input_path: Path) -> Path:
    marker_dirs = {"xml", "xmls", "annotation", "annotations", "xml_labels"}
    image_dirnames = {"images", "Images", "JPEGImages"}

    if input_path.name.lower() in marker_dirs:
        parent = input_path.parent
        if any((parent / dirname).is_dir() for dirname in image_dirnames):
            return parent
    return input_path


def _label_relative_path(xml_path: Path, input_path: Path) -> Path:
    relative = xml_path.relative_to(input_path).with_suffix(".txt")
    parts = list(relative.parts)
    marker_dirs = {"xml", "xmls", "annotation", "annotations", "xml_labels"}

    for index, part in enumerate(parts[:-1]):
        if part.lower() in marker_dirs:
            parts[index] = "labels"
            return Path(*parts)

    return Path("labels") / relative


@tool
def convert_xml_to_yolo(
    input_dir: str,
    output_dir: Optional[str] = None,
    classes: Optional[str] = None,
) -> str:
    """递归将 Pascal VOC XML 标注转换为 YOLO TXT 标注，并保留相对目录结构。"""
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise ValueError(f"input_dir不存在: {input_path}")

    if output_dir:
        output_path = Path(output_dir).expanduser().resolve()
    else:
        output_path = _default_output_root(input_path)

    xml_paths = _list_files(input_path, {".xml"})
    if not xml_paths:
        raise ValueError(f"在目录下未找到xml文件: {input_path}")

    class_names = _parse_classes(classes)
    class_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}

    converted = 0
    skipped = 0
    total_boxes = 0

    for xml_path in xml_paths:
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            skipped += 1
            continue

        root = tree.getroot()
        size = _xml_size(root)
        if size is None:
            image_path = _find_image_for_xml(xml_path)
            if image_path is None:
                skipped += 1
                continue
            from PIL import Image  # local import to keep tool load light
            with Image.open(image_path) as image:
                width, height = image.size
        else:
            width, height = size

        lines: List[str] = []
        for obj in root.findall("object"):
            item = _extract_valid_xml_object(obj)
            if item is None:
                continue

            class_name = item["name"]
            if class_name not in class_to_id:
                class_to_id[class_name] = len(class_to_id)
                class_names.append(class_name)

            clipped = _clip_box(item["xmin"], item["ymin"], item["xmax"], item["ymax"], width, height)
            if clipped is None:
                continue

            x_center, y_center, box_width, box_height = _xyxy_to_yolo(*clipped, width, height)
            lines.append(
                f"{class_to_id[class_name]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )

        if not lines:
            skipped += 1
            continue

        output_label_path = output_path / _label_relative_path(xml_path, input_path)
        output_label_path.parent.mkdir(parents=True, exist_ok=True)
        output_label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        converted += 1
        total_boxes += len(lines)

    if class_names:
        classes_path = output_path / "classes.txt"
        classes_path.parent.mkdir(parents=True, exist_ok=True)
        classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    return (
        f"完成。xml_found={len(xml_paths)}，converted={converted}，skipped={skipped}，"
        f"boxes={total_boxes}，classes={len(class_names)}，output_dir={output_path}"
    )
