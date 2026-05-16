from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

from langchain_core.tools import tool

from app.tools.dataset_clean_tool import IMAGE_SUFFIXES, _extract_valid_xml_object, _list_files


MARKER_DIRS = {"xml", "xmls", "annotation", "annotations", "xml_labels"}
IMAGE_DIRNAMES = ("images", "Images", "JPEGImages")


def _find_image_for_xml(xml_path: Path) -> Optional[Path]:
    same_dir = xml_path.parent
    for suffix in sorted(IMAGE_SUFFIXES):
        candidate = same_dir / f"{xml_path.stem}{suffix}"
        if candidate.is_file():
            return candidate

    parent_dir = xml_path.parent.parent
    for dirname in IMAGE_DIRNAMES:
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
    if input_path.name.lower() in MARKER_DIRS:
        return input_path.parent
    if any(path.is_file() and path.suffix.lower() == ".xml" for path in input_path.iterdir()):
        return input_path.parent
    return input_path


def _is_standard_xml_dataset_root(input_path: Path) -> bool:
    has_xml_dir = any((input_path / dirname).is_dir() for dirname in MARKER_DIRS)
    has_image_dir = any((input_path / dirname).is_dir() for dirname in IMAGE_DIRNAMES)
    return has_xml_dir and has_image_dir


def _normalize_output_root(input_path: Path, output_dir: Optional[str]) -> tuple[Path, bool]:
    if not output_dir:
        return _default_output_root(input_path), False

    output_path = Path(output_dir).expanduser().absolute()
    if _is_standard_xml_dataset_root(input_path):
        if output_path == input_path / "labels":
            return input_path, True
        if output_path.parent == input_path.parent and output_path.name == f"{input_path.name}_labels":
            return input_path, True

    return output_path, True


def _label_relative_path(xml_path: Path, input_path: Path) -> Path:
    relative = xml_path.relative_to(input_path).with_suffix(".txt")
    parts = list(relative.parts)

    for index, part in enumerate(parts[:-1]):
        if part.lower() in MARKER_DIRS:
            parts[index] = "labels"
            return Path(*parts)

    if len(parts) == 1:
        return Path("labels") / parts[0]
    return Path(*parts[:-1]) / "labels" / parts[-1]


def _source_label_path(xml_path: Path, input_path: Path) -> Path:
    relative = xml_path.relative_to(input_path).with_suffix(".txt")
    parts = list(relative.parts)

    for index, part in enumerate(parts[:-1]):
        if part.lower() not in MARKER_DIRS:
            continue
        marker_relative = Path(*parts[:index]) if index > 0 else Path()
        tail = Path(*parts[index + 1 :]) if index + 1 < len(parts) else Path(parts[-1])
        return input_path / marker_relative / "labels" / tail

    return xml_path.parent.parent / "labels" / f"{xml_path.stem}.txt"


def _label_output_path(
    xml_path: Path,
    input_path: Path,
    output_path: Path,
    has_explicit_output_dir: bool,
) -> Path:
    if has_explicit_output_dir:
        return output_path / _label_relative_path(xml_path, input_path)
    return _source_label_path(xml_path, input_path)


def _image_relative_path(image_path: Path, xml_path: Path, input_path: Path) -> Path:
    try:
        relative = image_path.relative_to(input_path)
    except ValueError:
        return Path("images") / image_path.name

    parts = list(relative.parts)
    if any(part in IMAGE_DIRNAMES for part in parts[:-1]):
        return relative

    try:
        xml_relative = xml_path.relative_to(input_path)
    except ValueError:
        return Path("images") / image_path.name

    xml_parts = list(xml_relative.parts)
    for index, part in enumerate(xml_parts[:-1]):
        if part.lower() in MARKER_DIRS:
            xml_parts[index] = "images"
            xml_parts[-1] = image_path.name
            return Path(*xml_parts)
    return Path("images") / image_path.name


@tool(parse_docstring=True)
def convert_xml_to_yolo(
    input_dir: str,
    output_dir: Optional[str] = None,
    classes: Optional[str] = None,
) -> str:
    """递归将 Pascal VOC XML 标注转换为 YOLO TXT 标注。

    未提供 output_dir 时，会根据 XML 标注目录在其同级生成 labels 目录：
    dataset/xmls/a.xml -> dataset/labels/a.txt。
    提供 output_dir 时，会在 output_dir 内按相对结构生成 labels 目录。
    对于包含 images/xmls 的标准数据集根目录，如果调用方误传 output_dir=<input>_labels
    或 output_dir=<input>/labels，工具会自动归一为 input_dir，避免生成兄弟 _labels
    目录或 labels/labels 嵌套目录。

    Args:
        input_dir: 输入目录，递归查找 Pascal VOC XML 文件。
        output_dir: 输出根目录；为空时在 XML 标注目录同级生成 labels 目录。
        classes: 类别名称列表，使用逗号分隔；为空时按扫描到的类别名称动态追加。
    """
    input_path = Path(input_dir).expanduser().absolute()
    if not input_path.is_dir():
        raise ValueError(f"input_dir不存在: {input_path}")

    output_path, has_explicit_output_dir = _normalize_output_root(input_path, output_dir)

    xml_paths = _list_files(input_path, {".xml"})
    if not xml_paths:
        raise ValueError(f"在目录下未找到xml文件: {input_path}")

    class_names = _parse_classes(classes)
    class_to_id: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}

    converted = 0
    skipped = 0
    total_boxes = 0
    copied_images: set[Path] = set()

    for xml_path in xml_paths:
        image_path = _find_image_for_xml(xml_path)
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            skipped += 1
            continue

        root = tree.getroot()
        size = _xml_size(root)
        if size is None:
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

        output_label_path = _label_output_path(
            xml_path,
            input_path,
            output_path,
            has_explicit_output_dir,
        )
        output_label_path.parent.mkdir(parents=True, exist_ok=True)
        output_label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if has_explicit_output_dir and image_path is not None:
            output_image_path = output_path / _image_relative_path(image_path, xml_path, input_path)
            if output_image_path not in copied_images:
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, output_image_path)
                copied_images.add(output_image_path)
        converted += 1
        total_boxes += len(lines)

    if class_names:
        classes_path = output_path / "classes.txt"
        classes_path.parent.mkdir(parents=True, exist_ok=True)
        classes_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")

    return (
        f"完成。xml_found={len(xml_paths)}，converted={converted}，skipped={skipped}，"
        f"boxes={total_boxes}，classes={len(class_names)}，images_copied={len(copied_images)}，"
        f"output_dir={output_path}"
    )
