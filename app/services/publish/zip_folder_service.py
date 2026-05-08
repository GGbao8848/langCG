from __future__ import annotations

import zipfile
from pathlib import Path


def run_zip_folder_service(
    input_dir: str,
    output_zip_path: str | None = None,
    *,
    include_root_dir: bool = True,
    overwrite: bool = False,
) -> dict:
    input_path = Path(input_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise ValueError(f"input_dir不存在: {input_path}")

    if output_zip_path:
        output_path = Path(output_zip_path).expanduser().resolve()
    else:
        output_path = input_path.parent / f"{input_path.name}.zip"

    if output_path.suffix.lower() != ".zip":
        output_path = output_path.with_suffix(".zip")

    if output_path.exists() and not overwrite:
        raise ValueError(f"输出zip已存在: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    packed_files = 0
    total_bytes = 0
    mode = "w" if overwrite else "x"
    with zipfile.ZipFile(output_path, mode=mode, compression=zipfile.ZIP_DEFLATED) as zipf:
        for path in sorted(input_path.rglob("*")):
            if not path.is_file():
                continue
            arcname = (
                path.relative_to(input_path.parent)
                if include_root_dir
                else path.relative_to(input_path)
            )
            zipf.write(path, arcname=arcname)
            packed_files += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass

    return {
        "input_dir": str(input_path),
        "output_zip_path": str(output_path),
        "packed_files": packed_files,
        "total_bytes": total_bytes,
    }
