from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Optional

import yaml
from langchain.tools import tool

from app.services.user_settings import load_user_settings


def _normalize_run_dir(run_dir: str) -> Path:
    text = run_dir.strip()
    if not text:
        raise ValueError("run_dir不能为空")
    path = Path(os.path.expanduser(text)).resolve()
    if not path.is_dir():
        raise ValueError(f"训练目录不存在: {path}")
    return path


def _resolve_local_yolo_executable(venv_path: str) -> Path:
    path = Path(venv_path).expanduser()
    if not str(path).strip():
        raise ValueError("需要先在左侧设置local_yolo_train_venv_path，或调用工具时传入local_venv_path")

    if path.name == "yolo" and path.is_file():
        return path

    yolo_path = path / "bin" / "yolo"
    if not yolo_path.is_file():
        raise ValueError(f"本地YOLO虚拟环境无效，未找到: {yolo_path}")
    return yolo_path


def _load_args_yaml(args_path: Path) -> dict[str, Any]:
    if not args_path.is_file():
        raise ValueError(f"未找到训练参数文件: {args_path}")
    try:
        data = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"args.yaml解析失败: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"args.yaml内容不是对象: {args_path}")
    return data


def _normalize_imgsz(value: Any) -> str:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("args.yaml中的imgsz必须为正数")
        return str(value)
    if isinstance(value, (list, tuple)) and value:
        sizes: list[str] = []
        for item in value:
            if not isinstance(item, int) or item <= 0:
                raise ValueError("args.yaml中的imgsz列表必须只包含正整数")
            sizes.append(str(item))
        return ",".join(sizes)
    raise ValueError("args.yaml中缺少有效的imgsz参数")


def _quote_command(args: list[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in args)


@tool
def export_yolo_torchscript(
    run_dir: str,
    execute: bool = False,
    local_venv_path: Optional[str] = None,
    timeout_seconds: int = 600,
) -> str:
    """从YOLO训练run目录导出weights/best.pt为TorchScript，imgsz来自该目录args.yaml，固定half=True且device=0。

    Args:
        run_dir: YOLO训练输出目录，例如.../runs/train/name。工具会读取run_dir/args.yaml和run_dir/weights/best.pt。
        execute: 是否立即执行导出；默认False，只返回将执行的命令和预期产物。
        local_venv_path: 本地YOLO虚拟环境目录或yolo可执行文件路径；默认读取左侧保存的local_yolo_train_venv_path。
        timeout_seconds: 执行导出时的超时时间，默认600秒。
    """
    resolved_run_dir = _normalize_run_dir(run_dir)
    args_yaml_path = resolved_run_dir / "args.yaml"
    weights_path = resolved_run_dir / "weights" / "best.pt"
    if not weights_path.is_file():
        raise ValueError(f"未找到训练权重: {weights_path}")

    args_yaml = _load_args_yaml(args_yaml_path)
    imgsz = _normalize_imgsz(args_yaml.get("imgsz"))

    user_settings = load_user_settings()
    resolved_local_venv_path = (
        (local_venv_path or "").strip()
        or str(user_settings.get("local_yolo_train_venv_path") or "").strip()
    )
    yolo_path = _resolve_local_yolo_executable(resolved_local_venv_path)

    command_args = [
        str(yolo_path),
        "export",
        f"model={weights_path}",
        "format=torchscript",
        f"imgsz={imgsz}",
        "half=True",
        "device=0",
    ]
    expected_output = weights_path.with_suffix(".torchscript")
    command = _quote_command(command_args)

    if not execute:
        return (
            "已生成TorchScript导出命令，默认未执行。\n"
            f"run_dir={resolved_run_dir}\n"
            f"args_yaml={args_yaml_path}\n"
            f"model={weights_path}\n"
            f"imgsz={imgsz}\n"
            "format=torchscript\n"
            "half=True\n"
            "device=0\n"
            f"expected_output={expected_output}\n"
            f"local_venv_path={resolved_local_venv_path}\n"
            f"command={command}"
        )

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds必须为正数")

    try:
        result = subprocess.run(
            command_args,
            cwd=str(resolved_run_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"TorchScript导出超时，超过{timeout_seconds}秒") from error
    except OSError as error:
        raise ValueError(f"无法运行YOLO导出命令: {error}") from error

    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if result.returncode != 0:
        raise ValueError(f"TorchScript导出失败，exit={result.returncode}\n{output[-2000:]}")

    exists_text = "存在" if expected_output.is_file() else "未在预期路径找到"
    return (
        "TorchScript导出完成。\n"
        f"run_dir={resolved_run_dir}\n"
        f"imgsz={imgsz}\n"
        "format=torchscript\n"
        "half=True\n"
        "device=0\n"
        f"output={expected_output}\n"
        f"output_status={exists_text}\n"
        f"command={command}\n"
        f"log={output[-2000:]}"
    )
