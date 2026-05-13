from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


def _parse_index_list(indices: str) -> list[str]:
    items = [item.strip() for item in indices.replace("，", ",").split(",")]
    parsed = [item for item in items if item]
    if not parsed:
        raise ValueError("source_indices不能为空")
    return parsed


def _normalize_target_index(target_index: str | int) -> str:
    text = str(target_index).strip()
    if not text:
        raise ValueError("target_index不能为空")
    return text


def _parse_mapping(mapping: str) -> dict[str, str]:
    items = [item.strip() for item in mapping.replace("，", ",").split(",")]
    parsed: dict[str, str] = {}
    for item in items:
        if not item:
            continue
        if "->" not in item:
            raise ValueError("mapping格式必须类似'1->0,2->0'")
        source, target = item.split("->", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError("mapping格式必须类似'1->0,2->0'")
        parsed[source] = target
    if not parsed:
        raise ValueError("mapping不能为空")
    return parsed


def _build_mapping(
    source_indices: Optional[str],
    target_index: Optional[str],
    mapping: Optional[str],
) -> dict[str, str]:
    if mapping:
        return _parse_mapping(mapping)
    if source_indices is None or target_index is None:
        raise ValueError("必须提供mapping，或同时提供source_indices和target_index")
    target = _normalize_target_index(target_index)
    return {source: target for source in _parse_index_list(source_indices)}


def _find_classes_file(input_path: Path) -> Optional[Path]:
    direct = input_path / "classes.txt"
    if direct.is_file():
        return direct

    parent = input_path.parent / "classes.txt"
    if parent.is_file():
        return parent
    return None


def _rewrite_classes_file(classes_path: Path, output_path: Path, mapping: dict[str, str]) -> bool:
    original = classes_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not original:
        return False

    rewritten: dict[int, str] = {}
    old_names = {str(index): name for index, name in enumerate(original)}

    for old_index, class_name in old_names.items():
        new_index = mapping.get(old_index, old_index)
        try:
            new_index_int = int(new_index)
        except ValueError:
            continue

        if new_index_int in rewritten:
            continue
        rewritten[new_index_int] = old_names.get(new_index, class_name)

    if not rewritten:
        return False

    max_index = max(rewritten)
    output_lines = [rewritten.get(index, "") for index in range(max_index + 1)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return True


def _output_classes_path(input_path: Path, output_path: Path, classes_path: Path) -> Path:
    if output_path == input_path:
        return classes_path
    return output_path / "classes.txt"


def _rewrite_label_file(label_path: Path, output_path: Path, mapping: dict[str, str]) -> tuple[int, int]:
    changed = 0
    total = 0
    output_lines: list[str] = []

    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            output_lines.append(raw_line)
            continue

        total += 1
        if parts[0] in mapping:
            parts[0] = mapping[parts[0]]
            changed += 1
        output_lines.append(" ".join(parts))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = "\n" if output_lines else ""
    output_path.write_text("\n".join(output_lines) + suffix, encoding="utf-8")
    return total, changed


@tool(parse_docstring=True)
def reindex_yolo_labels(
    input_dir: str,
    source_indices: Optional[str] = None,
    target_index: Optional[str] = None,
    mapping: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """按需批量重置 YOLO 标签类别索引。

    示例：
    - source_indices='1,2', target_index='0'
    - mapping='0->0,1->0,2->3,5->1'

    Args:
        input_dir: 输入 YOLO labels 根目录或数据集根目录，递归处理其中的 txt 标签。
        source_indices: 需要改写的原始类别索引列表，逗号分隔；与 target_index 配合使用。
        target_index: source_indices 中所有类别要改成的目标索引。
        mapping: 显式索引映射，例如 0->0,1->0,2->3；提供后优先于 source_indices/target_index。
        output_dir: 输出目录；为空时原地改写。
    """
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise ValueError(f"input_dir不存在: {input_path}")

    if output_dir:
        output_path = Path(output_dir).expanduser().resolve()
    else:
        output_path = input_path

    mapping_dict = _build_mapping(source_indices, target_index, mapping)

    label_files = sorted(input_path.rglob("*.txt"))
    if not label_files:
        raise ValueError(f"在目录下未找到txt标签文件: {input_path}")

    classes_rewritten = False
    files_changed = 0
    total_files = 0
    total_labels = 0
    changed_labels = 0

    for label_path in label_files:
        relative = label_path.relative_to(input_path)
        output_label_path = output_path / relative
        before = changed_labels
        file_total, file_changed = _rewrite_label_file(label_path, output_label_path, mapping_dict)
        total_files += 1
        total_labels += file_total
        changed_labels += file_changed
        if changed_labels > before:
            files_changed += 1

    classes_path = _find_classes_file(input_path)
    if classes_path is not None:
        classes_rewritten = _rewrite_classes_file(
            classes_path,
            _output_classes_path(input_path, output_path, classes_path),
            mapping_dict,
        )

    return (
        f"完成。files={total_files}，files_changed={files_changed}，labels={total_labels}，"
        f"labels_changed={changed_labels}，classes_updated={classes_rewritten}，"
        f"mapping={dict(sorted(mapping_dict.items()))}，"
        f"output_dir={output_path}"
    )
