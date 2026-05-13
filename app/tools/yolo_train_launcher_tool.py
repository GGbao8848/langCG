from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Optional
from urllib.parse import unquote, urlparse

import paramiko
from langchain.tools import tool

from app.services.user_settings import load_user_settings
from app.services.publish.remote_transfer_service import _load_private_key


def _is_remote_yaml_path(yaml_path: str) -> bool:
    return urlparse(yaml_path.strip()).scheme == "sftp"


def _parse_remote_yaml(yaml_path: str) -> tuple[str | None, str | None, int | None]:
    parsed = urlparse(yaml_path.strip())
    if parsed.scheme != "sftp":
        return None, None, None
    return parsed.hostname, parsed.username, parsed.port


def _normalize_yaml_path(yaml_path: str) -> str:
    text = yaml_path.strip()
    if not text:
        raise ValueError("yaml_path不能为空")

    parsed = urlparse(text)
    if parsed.scheme == "sftp":
        if not parsed.path:
            raise ValueError("sftp路径缺少远端文件路径")
        return unquote(parsed.path)
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError("yaml_path只支持本地路径、file://路径或sftp://host/path.yaml")
    if parsed.scheme == "file":
        return unquote(parsed.path)
    return os.path.expanduser(text)


def _assert_yaml_file(path_text: str, require_exists: bool) -> None:
    suffix = PurePosixPath(path_text).suffix.lower()
    if suffix not in {".yaml", ".yml"}:
        raise ValueError(f"yaml_path必须指向.yaml或.yml文件: {path_text}")
    if require_exists and not Path(path_text).expanduser().is_file():
        raise ValueError(f"yaml文件不存在: {path_text}")


def _derive_train_dirs(path_text: str) -> tuple[str, str]:
    path = PurePosixPath(path_text)
    parts = path.parts
    dataset_index: int | None = None
    for index, part in enumerate(parts):
        if part in {"dataset", "datasets"}:
            dataset_index = index

    if dataset_index is None:
        raise ValueError("无法从yaml_path中识别dataset或datasets目录，需手动指定project_dir")
    if dataset_index < 1:
        raise ValueError("无法从yaml_path中识别detector name，需手动指定project_dir")

    work_dir = PurePosixPath(*parts[: dataset_index - 1])
    detector_dir = PurePosixPath(*parts[:dataset_index])
    return str(work_dir), str(detector_dir / "runs" / "train")


def _shell_assignment(key: str, value: str | int) -> str:
    if isinstance(value, int):
        return f"{key}={value}"
    if key not in {"data", "project", "name"} and not any(char.isspace() for char in value):
        return f"{key}={value}"
    return f'{key}="{value}"'


def _build_train_command(
    data_path: str,
    project_dir: str,
    name: str,
    command_prefix: str,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    cache: str,
    device: str | int | None,
) -> list[str]:
    command = [
        *shlex.split(command_prefix),
        "detect",
        "train",
        _shell_assignment("data", data_path),
        _shell_assignment("model", model),
        _shell_assignment("epochs", epochs),
        _shell_assignment("imgsz", imgsz),
        _shell_assignment("batch", batch),
        _shell_assignment("workers", workers),
        _shell_assignment("cache", cache),
        _shell_assignment("project", project_dir),
        _shell_assignment("name", name),
    ]
    if device is not None and str(device).strip():
        command.append(_shell_assignment("device", str(device).strip()))
    return command


def _resolve_local_yolo_executable(venv_path: str) -> Path:
    path = Path(venv_path).expanduser()
    if not str(path).strip():
        raise ValueError("本地训练需要先在前端环境变量中配置local_yolo_train_venv_path，或调用工具时传入local_venv_path")

    if path.name == "yolo" and path.is_file():
        return path

    yolo_path = path / "bin" / "yolo"
    if not yolo_path.is_file():
        raise ValueError(f"本地YOLO训练虚拟环境无效，未找到: {yolo_path}")
    return yolo_path


def _local_python_for_yolo(yolo_path: Path) -> Path | None:
    bin_dir = yolo_path.parent
    python_path = bin_dir / "python"
    if python_path.is_file():
        return python_path
    python3_path = bin_dir / "python3"
    if python3_path.is_file():
        return python3_path
    return None


def _detect_local_device(yolo_path: Path) -> str:
    python_path = _local_python_for_yolo(yolo_path)
    if python_path is None:
        return "cpu"

    probe_code = (
        "import torch\n"
        "device='cpu'\n"
        "if torch.cuda.is_available():\n"
        "    device='0'\n"
        "elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n"
        "    device='mps'\n"
        "print(device)\n"
    )
    try:
        result = subprocess.run(
            [str(python_path), "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "cpu"

    if result.returncode != 0:
        return "cpu"
    detected = result.stdout.strip().splitlines()[-1:] or ["cpu"]
    return detected[0] if detected[0] in {"0", "mps", "cpu"} else "cpu"


def _remote_train_command(
    *,
    work_dir: str,
    command: str,
) -> str:
    inner_command = f"cd {shlex.quote(work_dir)} && {command}"
    return f"bash -l -i -c {shlex.quote(inner_command)}"


def _run_remote_training(
    *,
    host: str,
    port: int,
    username: str,
    private_key_path: str | None,
    password: str | None,
    command: str,
) -> tuple[int, str, str]:
    if password and private_key_path:
        raise ValueError("remote_password和remote_private_key_path不能同时提供")
    if not password and not private_key_path:
        raise ValueError("远程训练需要remote_private_key_path或remote_password")

    pkey = _load_private_key(private_key_path) if private_key_path else None
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=port,
            username=username,
            password=password,
            pkey=pkey,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
        )
        _, stdout, stderr = client.exec_command(command, get_pty=True)
        channel = stdout.channel
        while not channel.exit_status_ready():
            time.sleep(0.2)
        exit_code = channel.recv_exit_status()
        stdout_text = stdout.read().decode("utf-8", errors="replace").strip()
        stderr_text = stderr.read().decode("utf-8", errors="replace").strip()
        return exit_code, stdout_text, stderr_text
    except paramiko.AuthenticationException as exc:
        raise ValueError(f"SSH认证失败: {exc}") from exc
    except paramiko.SSHException as exc:
        raise ValueError(f"SSH连接失败: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"SSH执行失败: {exc}") from exc
    finally:
        client.close()


@tool
def launch_yolo_training(
    yaml_path: str,
    model: str = "yolo11m.pt",
    epochs: int = 200,
    imgsz: int = 800,
    batch: Optional[int] = None,
    workers: int = 4,
    cache: str = "disk",
    device: Optional[str] = None,
    project_dir: Optional[str] = None,
    name: Optional[str] = None,
    command_prefix: str = "subyolo",
    execute: bool = False,
    require_exists: bool = False,
    local_venv_path: Optional[str] = None,
    remote_username: Optional[str] = None,
    remote_private_key_path: Optional[str] = None,
    remote_password: Optional[str] = None,
    remote_port: Optional[int] = None,
) -> str:
    """生成或启动YOLO训练命令，支持本地yaml路径和sftp://host/path.yaml路径。

    Args:
        yaml_path: 数据集yaml路径，可为本地路径、file://路径或sftp://host/path.yaml。
        model: 模型权重，默认yolo11m.pt；常用值包括yolo11n/s/m/l/x.pt和yolov8n/s/m/l/x.pt，也允许其他Ultralytics支持的权重路径或名称。
        epochs: 训练轮数，默认200。
        imgsz: 图像尺寸，默认800。
        batch: batch大小；远程训练默认24，本地训练默认8。
        workers: dataloader workers，默认4。
        cache: cache参数，默认disk。
        device: 训练设备；远程默认不指定，本地默认自适应检测为0、mps或cpu。需要固定设备时可传0、cpu、mps、-1等。
        project_dir: 训练输出目录；默认使用<detector目录>/runs/train。命令会先cd到detector name前的目录。
        name: 训练任务名称；默认使用yaml文件名stem。
        command_prefix: 训练命令前缀，默认subyolo。
        execute: 是否真正启动训练；默认False，只返回cd目录和命令。
        require_exists: 是否要求yaml文件在当前机器存在；远端sftp路径默认不强制检查。
        local_venv_path: 本地YOLO训练虚拟环境目录或python路径；默认读取前端保存的local_yolo_train_venv_path。
        remote_username: 远程训练用户名；默认读取左侧保存配置或sftp路径中的用户名。
        remote_private_key_path: 远程训练私钥路径；默认读取左侧保存配置。
        remote_password: 远程训练密码。
        remote_port: 远程训练SSH端口；默认读取左侧保存配置或sftp路径端口。
    """
    is_remote = _is_remote_yaml_path(yaml_path)
    yaml_host, yaml_username, yaml_port = _parse_remote_yaml(yaml_path)
    data_path = _normalize_yaml_path(yaml_path)
    _assert_yaml_file(data_path, require_exists=require_exists or not is_remote)
    if not model.strip():
        raise ValueError("model不能为空")

    resolved_work_dir, default_project_dir = _derive_train_dirs(data_path)
    resolved_project_dir = project_dir or default_project_dir
    resolved_name = name or PurePosixPath(data_path).stem
    resolved_batch = batch if batch is not None else (24 if is_remote else 8)
    resolved_device = device

    user_settings = load_user_settings()
    configured_host = str(user_settings.get("remote_sftp_host") or "").strip()
    configured_username = str(user_settings.get("remote_sftp_username") or "").strip()
    configured_key = str(user_settings.get("remote_sftp_private_key_path") or "").strip()
    configured_port = int(user_settings.get("remote_sftp_port") or 22)
    resolved_local_venv_path = (
        (local_venv_path or "").strip()
        or str(user_settings.get("local_yolo_train_venv_path") or "").strip()
    )
    local_yolo_path: Path | None = None
    if not is_remote:
        local_yolo_path = _resolve_local_yolo_executable(resolved_local_venv_path)
        if resolved_device is None:
            resolved_device = _detect_local_device(local_yolo_path)

    resolved_command_prefix = command_prefix.strip() if is_remote else str(local_yolo_path)
    if not resolved_command_prefix:
        raise ValueError("command_prefix不能为空")
    if epochs <= 0 or imgsz <= 0 or resolved_batch <= 0 or workers < 0:
        raise ValueError("epochs/imgsz/batch必须为正数，workers不能为负数")
    if not cache.strip():
        raise ValueError("cache不能为空")

    command_parts = _build_train_command(
        data_path=data_path,
        project_dir=resolved_project_dir,
        name=resolved_name,
        command_prefix=resolved_command_prefix,
        model=model,
        epochs=epochs,
        imgsz=imgsz,
        batch=resolved_batch,
        workers=workers,
        cache=cache,
        device=resolved_device,
    )
    command = " ".join(command_parts)
    cd_command = f'cd "{resolved_work_dir}" && {command}'
    remote_command = _remote_train_command(
        work_dir=resolved_work_dir,
        command=command,
    )

    if not execute:
        return (
            "已生成训练启动命令，默认未执行。\n"
            f"mode={'remote_script' if is_remote else 'local_venv'}\n"
            f"data={data_path}\n"
            f"work_dir={resolved_work_dir}\n"
            f"project={resolved_project_dir}\n"
            f"name={resolved_name}\n"
            f"batch={resolved_batch}\n"
            f"device={resolved_device or ''}\n"
            f"local_venv_path={resolved_local_venv_path if not is_remote else ''}\n"
            f"command={remote_command if is_remote else cd_command}"
        )

    if is_remote:
        resolved_remote_host = yaml_host or configured_host
        resolved_remote_username = (
            (remote_username or "").strip()
            or yaml_username
            or configured_username
        )
        resolved_remote_key = (remote_private_key_path or "").strip() or configured_key or None
        resolved_remote_port = remote_port or yaml_port or configured_port or 22
        if not resolved_remote_host:
            raise ValueError("远程训练无法确定host，请使用sftp://host/path.yaml或配置remote_sftp_host")
        if not resolved_remote_username:
            raise ValueError("远程训练需要remote_username或左侧保存的remote_sftp_username")

        exit_code, stdout_text, stderr_text = _run_remote_training(
            host=resolved_remote_host,
            port=resolved_remote_port,
            username=resolved_remote_username,
            private_key_path=resolved_remote_key,
            password=remote_password,
            command=remote_command,
        )
        stdout_suffix = f"\nstdout={stdout_text[-2000:]}" if stdout_text else ""
        stderr_suffix = f"\nstderr={stderr_text[-1000:]}" if stderr_text else ""
        if exit_code != 0:
            raise ValueError(
                f"远程训练执行失败，exit={exit_code}{stdout_suffix}{stderr_suffix}"
            )
        return (
            "远程训练命令执行完成。\n"
            "mode=remote_ssh\n"
            f"host={resolved_remote_host}\n"
            f"port={resolved_remote_port}\n"
            f"cwd={resolved_work_dir}\n"
            f"project={resolved_project_dir}\n"
            f"command={remote_command}"
            f"{stdout_suffix}"
            f"{stderr_suffix}"
        )

    Path(resolved_project_dir).expanduser().mkdir(parents=True, exist_ok=True)
    log_path = Path(resolved_project_dir).expanduser() / f"{resolved_name}.log"
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            command,
            cwd=Path(resolved_work_dir).expanduser(),
            shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    return (
        "已启动训练，后台进程已创建，工具执行成功。\n"
        f"mode={'remote_script' if is_remote else 'local_venv'}\n"
        f"pid={process.pid}\n"
        f"cwd={resolved_work_dir}\n"
        f"project={resolved_project_dir}\n"
        f"device={resolved_device or ''}\n"
        f"log={log_path}\n"
        f"command={command}"
    )
