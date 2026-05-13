from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import shutil
import xml.etree.ElementTree as ET

from langchain_core.tools import tool
from PIL import Image, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = None

# 数据集清洗规则：
# 1. 递归扫描输入目录，兼容图像与 xml 同目录、images/labels/xml/xmls 混放、以及多层嵌套的不规则结构。
# 2. 以图像为主键，按同 stem 自动寻找最接近的 xml/txt 标注；优先同目录，其次相邻目录，再退化到整树最近路径。
# 3. 正常样本统一整理到 output_dir/images、output_dir/xmls、output_dir/labels。
# 4. 没有任何标注的正常图像整理到 output_dir/backgrounds。
# 5. 坏图整理到 output_dir/badimages；若同时存在对应 xml/txt，也一并整理到 badxmls/badlabels。
# 6. 空标注单独视为异常：空 xml -> badxmls，空 txt -> badlabels；对应图像同时归入 badimages。
# 7. 所有输出均为复制，不修改原始数据；没有实际内容的目录不会创建。

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
XML_SUFFIXES = {".xml"}
TXT_SUFFIXES = {".txt"}


def _list_files(root: Path, suffixes: set[str]) -> List[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _common_prefix_len(parts_a: Sequence[str], parts_b: Sequence[str]) -> int:
    count = 0
    for part_a, part_b in zip(parts_a, parts_b):
        if part_a != part_b:
            break
        count += 1
    return count


def _path_distance(path_a: Path, path_b: Path) -> int:
    parts_a = path_a.resolve().parts
    parts_b = path_b.resolve().parts
    common = _common_prefix_len(parts_a, parts_b)
    return (len(parts_a) - common) + (len(parts_b) - common)


def _candidate_priority(image_path: Path, annotation_path: Path) -> tuple[int, int, str]:
    image_parent = image_path.parent.resolve()
    annotation_parent = annotation_path.parent.resolve()

    if annotation_parent == image_parent:
        level = 0
    elif annotation_parent.parent == image_parent.parent:
        level = 1
    else:
        level = 2
    return level, _path_distance(image_parent, annotation_parent), str(annotation_path)


def _pick_best_annotation(image_path: Path, candidates: List[Path]) -> Optional[Path]:
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: _candidate_priority(image_path, candidate))


def _is_bad_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image.load()
        return False
    except (UnidentifiedImageError, OSError):
        return True


def _is_empty_xml(xml_path: Path) -> bool:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return True

    root = tree.getroot()
    objects = [node for node in root.findall("object") if _extract_valid_xml_object(node) is not None]
    return len(objects) == 0


def _is_empty_txt(txt_path: Path) -> bool:
    valid_lines = 0
    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        if len(text.split()) >= 5:
            valid_lines += 1
    return valid_lines == 0


def _safe_output_stem(root: Path, file_path: Path) -> str:
    relative = file_path.relative_to(root)
    parts = list(relative.parts)
    parts[-1] = file_path.stem
    return "__".join(parts)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _get_image_size(image_path: Path) -> tuple[int, int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
        bands = image.getbands()
        depth = len(bands) if bands else 3
    return width, height, depth


def _safe_int(text: str | None) -> Optional[int]:
    if text is None:
        return None
    try:
        return int(round(float(text.strip())))
    except (TypeError, ValueError):
        return None


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def _extract_valid_xml_object(obj: ET.Element) -> Optional[dict]:
    name = (obj.findtext("name") or "").strip()
    bndbox = obj.find("bndbox")
    if bndbox is None:
        return None

    xmin = _safe_int(bndbox.findtext("xmin"))
    ymin = _safe_int(bndbox.findtext("ymin"))
    xmax = _safe_int(bndbox.findtext("xmax"))
    ymax = _safe_int(bndbox.findtext("ymax"))
    if None in {xmin, ymin, xmax, ymax}:
        return None

    return {
        "name": name or "0",
        "pose": (obj.findtext("pose") or "Unspecified").strip() or "Unspecified",
        "truncated": (obj.findtext("truncated") or "0").strip() or "0",
        "occluded": (obj.findtext("occluded") or "0").strip() or "0",
        "difficult": (obj.findtext("difficult") or "0").strip() or "0",
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }


def _normalize_xml_object(raw_obj: dict, image_width: int, image_height: int) -> Optional[dict]:
    xmin = min(raw_obj["xmin"], raw_obj["xmax"])
    ymin = min(raw_obj["ymin"], raw_obj["ymax"])
    xmax = max(raw_obj["xmin"], raw_obj["xmax"])
    ymax = max(raw_obj["ymin"], raw_obj["ymax"])

    xmin = _clamp(xmin, 0, image_width)
    xmax = _clamp(xmax, 0, image_width)
    ymin = _clamp(ymin, 0, image_height)
    ymax = _clamp(ymax, 0, image_height)

    if xmax <= xmin or ymax <= ymin:
        return None

    return {
        "name": raw_obj["name"],
        "pose": raw_obj["pose"],
        "truncated": raw_obj["truncated"],
        "occluded": raw_obj["occluded"],
        "difficult": raw_obj["difficult"],
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
    }


def _normalize_xml_content(
    xml_path: Path,
    image_output_name: str,
    image_width: int,
    image_height: int,
    image_depth: int,
    output_xml_dir: Path,
) -> Optional[str]:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None

    normalized_objects: list[dict] = []
    for obj in tree.getroot().findall("object"):
        raw_obj = _extract_valid_xml_object(obj)
        if raw_obj is None:
            continue
        normalized_obj = _normalize_xml_object(raw_obj, image_width, image_height)
        if normalized_obj is not None:
            normalized_objects.append(normalized_obj)

    if not normalized_objects:
        return None

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = str(output_xml_dir)
    ET.SubElement(annotation, "filename").text = image_output_name

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_width)
    ET.SubElement(size, "height").text = str(image_height)
    ET.SubElement(size, "depth").text = str(image_depth)

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "https://github.com/CVHub520/X-AnyLabeling"

    for item in normalized_objects:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = item["name"]
        ET.SubElement(obj, "pose").text = item["pose"]
        ET.SubElement(obj, "truncated").text = item["truncated"]
        ET.SubElement(obj, "occluded").text = item["occluded"]
        ET.SubElement(obj, "difficult").text = item["difficult"]
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(item["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(item["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(item["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(item["ymax"])

    ET.indent(annotation, space="  ")
    return ET.tostring(annotation, encoding="unicode")


@tool(parse_docstring=True)
def clean_irregular_dataset(
    input_dir: str,
    output_dir: Optional[str] = None,
) -> str:
    """清洗嵌套且不规则的数据集，统一整理为 images/xmls/labels/backgrounds 结构。

    兼容以下情况：
    - 图像与 xml 标注在同一个目录
    - images / labels / xml / xmls 命名不统一
    - 存在多层嵌套的不规则目录
    - 坏图、空 xml、空 txt

    Args:
        input_dir: 输入的不规则数据集根目录。
        output_dir: 输出目录；为空时生成到同级 <input>_cleaned。
    """
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise ValueError(f"input_dir不存在: {input_path}")

    if output_dir:
        output_path = Path(output_dir).expanduser().resolve()
    else:
        output_path = input_path.parent / f"{input_path.name}_cleaned"

    image_paths = _list_files(input_path, IMAGE_SUFFIXES)
    xml_paths = _list_files(input_path, XML_SUFFIXES)
    txt_paths = _list_files(input_path, TXT_SUFFIXES)

    if not image_paths:
        raise ValueError(f"在目录下未找到图像文件: {input_path}")

    xml_by_stem: Dict[str, List[Path]] = {}
    txt_by_stem: Dict[str, List[Path]] = {}
    for xml_path in xml_paths:
        xml_by_stem.setdefault(xml_path.stem, []).append(xml_path)
    for txt_path in txt_paths:
        txt_by_stem.setdefault(txt_path.stem, []).append(txt_path)

    counts = {
        "images": 0,
        "xmls": 0,
        "labels": 0,
        "backgrounds": 0,
        "badimages": 0,
        "badxmls": 0,
        "badlabels": 0,
    }

    for image_path in image_paths:
        stem = _safe_output_stem(input_path, image_path)
        xml_path = _pick_best_annotation(image_path, xml_by_stem.get(image_path.stem, []))
        txt_path = _pick_best_annotation(image_path, txt_by_stem.get(image_path.stem, []))

        is_bad_image = _is_bad_image(image_path)
        normalized_xml_content: str | None = None
        image_width = 0
        image_height = 0
        image_depth = 0

        if not is_bad_image:
            image_width, image_height, image_depth = _get_image_size(image_path)
            if xml_path is not None:
                normalized_xml_content = _normalize_xml_content(
                    xml_path=xml_path,
                    image_output_name=f"{stem}{image_path.suffix.lower()}",
                    image_width=image_width,
                    image_height=image_height,
                    image_depth=image_depth,
                    output_xml_dir=output_path / "xmls",
                )

        is_empty_xml = xml_path is not None and normalized_xml_content is None
        is_empty_txt = txt_path is not None and _is_empty_txt(txt_path)

        if is_bad_image or is_empty_xml or is_empty_txt:
            bad_image_path = output_path / "badimages" / f"{stem}{image_path.suffix.lower()}"
            _copy_file(image_path, bad_image_path)
            counts["badimages"] += 1

            if xml_path is not None:
                bad_xml_path = output_path / "badxmls" / f"{stem}.xml"
                _copy_file(xml_path, bad_xml_path)
                counts["badxmls"] += 1

            if txt_path is not None:
                bad_txt_path = output_path / "badlabels" / f"{stem}.txt"
                _copy_file(txt_path, bad_txt_path)
                counts["badlabels"] += 1
            continue

        if xml_path is None and txt_path is None:
            background_path = output_path / "backgrounds" / f"{stem}{image_path.suffix.lower()}"
            _copy_file(image_path, background_path)
            counts["backgrounds"] += 1
            continue

        clean_image_path = output_path / "images" / f"{stem}{image_path.suffix.lower()}"
        _copy_file(image_path, clean_image_path)
        counts["images"] += 1

        if normalized_xml_content is not None:
            clean_xml_path = output_path / "xmls" / f"{stem}.xml"
            clean_xml_path.parent.mkdir(parents=True, exist_ok=True)
            clean_xml_path.write_text('<?xml version="1.0" ?>\n' + normalized_xml_content + "\n", encoding="utf-8")
            counts["xmls"] += 1

        if txt_path is not None:
            clean_txt_path = output_path / "labels" / f"{stem}.txt"
            _copy_file(txt_path, clean_txt_path)
            counts["labels"] += 1

    return (
        f"完成。images_scanned={len(image_paths)}，xml_found={len(xml_paths)}，txt_found={len(txt_paths)}，"
        f"clean_images={counts['images']}，clean_xmls={counts['xmls']}，clean_labels={counts['labels']}，"
        f"backgrounds={counts['backgrounds']}，badimages={counts['badimages']}，"
        f"badxmls={counts['badxmls']}，badlabels={counts['badlabels']}，output_dir={output_path}"
    )
