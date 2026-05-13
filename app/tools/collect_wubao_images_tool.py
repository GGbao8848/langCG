from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool


def _normalize_root(root: str) -> Path:
    text = root.strip()
    if not text:
        raise ValueError("root不能为空")
    path = Path(text).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"结果根目录不存在: {path}")
    return path


def _is_generated_dir(path: Path, prefix: str) -> bool:
    return path.name == "wubao" or path.name.startswith(f"{prefix}_")


def _train_name(root: Path, file_path: Path) -> str:
    try:
        return file_path.relative_to(root).parts[0]
    except (IndexError, ValueError):
        return file_path.parent.parent.name


def _dedupe_target(output_dir: Path, filename: str) -> Path:
    target = output_dir / filename
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    counter = 2
    while target.exists():
        target = output_dir / f"{stem}__{counter}{suffix}"
        counter += 1
    return target


@tool(parse_docstring=True)
def collect_wubao_images(
    root: str,
    prefix: str = "wubao",
    output_dir: Optional[str] = None,
    max_missing_examples: int = 20,
) -> str:
    """根据各级 wubao 文件夹中的文件名，从同级 images 文件夹复制同名图像并汇总。

    Args:
        root: 结果根目录。工具会递归查找 root 下各列车目录中的 wubao 文件夹。
        prefix: 输出文件夹前缀，默认 wubao；未传 output_dir 时输出到 root/prefix_时间戳。
        output_dir: 可选的指定输出目录。未传时自动创建带时间戳的目录。
        max_missing_examples: 返回的缺失文件示例数量，默认20。
    """
    root_path = _normalize_root(root)
    if not prefix.strip():
        raise ValueError("prefix不能为空")
    if max_missing_examples < 0:
        raise ValueError("max_missing_examples不能为负数")

    if output_dir:
        resolved_output_dir = Path(output_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_output_dir = root_path / f"{prefix}_{timestamp}"

    resolved_output_dir.mkdir(parents=True, exist_ok=False)

    copied = 0
    missing: list[Path] = []
    skipped_generated_dirs = 0
    scanned_wubao_dirs = 0

    for wubao_dir in sorted(root_path.rglob("wubao")):
        if not wubao_dir.is_dir():
            continue
        if wubao_dir.parent == root_path:
            skipped_generated_dirs += 1
            continue
        if any(_is_generated_dir(parent, prefix) for parent in wubao_dir.parents if parent != root_path):
            skipped_generated_dirs += 1
            continue

        scanned_wubao_dirs += 1
        images_dir = wubao_dir.parent / "images"
        for wubao_file in sorted(wubao_dir.iterdir()):
            if not wubao_file.is_file():
                continue

            source = images_dir / wubao_file.name
            if not source.is_file():
                missing.append(source)
                continue

            target_name = f"{_train_name(root_path, wubao_file)}__{wubao_file.name}"
            target = _dedupe_target(resolved_output_dir, target_name)
            shutil.copy2(source, target)
            copied += 1

    examples = [str(path) for path in missing[:max_missing_examples]]
    examples_text = "\nmissing_examples=" + "\n".join(examples) if examples else ""
    return (
        "wubao图像汇总完成。\n"
        f"root={root_path}\n"
        f"output_dir={resolved_output_dir}\n"
        f"scanned_wubao_dirs={scanned_wubao_dirs}\n"
        f"copied={copied}\n"
        f"missing={len(missing)}\n"
        f"skipped_generated_dirs={skipped_generated_dirs}"
        f"{examples_text}"
    )
