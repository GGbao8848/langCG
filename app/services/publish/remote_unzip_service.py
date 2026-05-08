from __future__ import annotations

import re
import shlex
import time
from pathlib import Path
from urllib.parse import urlparse

import paramiko


def _parse_remote_path(target: str) -> tuple[str, int, str]:
    target = target.strip()
    if not target:
        raise ValueError("路径不能为空")

    if target.startswith(("sftp://", "ssh://")):
        parsed = urlparse(target)
        if not parsed.hostname:
            raise ValueError(f"远程路径格式无效，缺少host: {target}")
        host = parsed.hostname
        port = parsed.port or 22
        path = parsed.path or "/"
        path = path if path.startswith("/") else f"/{path}"
        return host, port, path

    scp_match = re.match(r"^([^@]+)@([^:]+):(.+)$", target)
    if scp_match:
        return scp_match.group(2), 22, scp_match.group(3)

    raise ValueError(f"远程路径格式无效: {target}。应为sftp://host/path或user@host:path")


def _extract_username_from_target(target: str) -> str | None:
    target = target.strip()
    if target.startswith(("sftp://", "ssh://")):
        parsed = urlparse(target)
        if parsed.username:
            return parsed.username
    if "@" in target and ":" in target:
        match = re.match(r"^([^@]+)@[^:]+:", target)
        if match:
            return match.group(1)
    return None


def _load_private_key(private_key_path: str) -> paramiko.PKey:
    key_path = Path(private_key_path).expanduser().resolve()
    if not key_path.exists():
        raise ValueError(f"private_key_path不存在: {key_path}")
    try:
        return paramiko.Ed25519Key.from_private_key_file(str(key_path))
    except paramiko.ssh_exception.SSHException:
        try:
            return paramiko.RSAKey.from_private_key_file(str(key_path))
        except paramiko.ssh_exception.SSHException as exc:
            raise ValueError(f"加载私钥失败: {exc}") from exc


def run_remote_unzip_service(
    archive_path: str,
    output_dir: str | None = None,
    *,
    username: str | None = None,
    private_key_path: str | None = None,
    password: str | None = None,
    port: int | None = None,
    overwrite: bool = False,
) -> dict:
    archive_host, archive_port, archive_remote_path = _parse_remote_path(archive_path)
    if output_dir:
        output_host, output_port, output_remote_path = _parse_remote_path(output_dir)
        if output_host != archive_host or output_port != archive_port:
            raise ValueError("archive_path和output_dir必须指向同一个远程host/port")
    else:
        output_host, output_port = archive_host, archive_port
        archive_obj = Path(archive_remote_path)
        output_remote_path = str(archive_obj.parent / archive_obj.stem).replace("\\", "/")

    auth_user = username or _extract_username_from_target(archive_path)
    if not auth_user:
        raise ValueError("必须提供username，或使用user@host:path格式")

    if password and private_key_path:
        raise ValueError("use either password or private_key_path, not both")
    if not password and not private_key_path:
        raise ValueError("必须提供password或private_key_path中的一个")

    pkey = _load_private_key(private_key_path) if private_key_path else None
    use_port = port if port is not None else archive_port

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    overwrite_flag = "-o" if overwrite else "-n"
    command = (
        f"mkdir -p {shlex.quote(output_remote_path)} && "
        f"unzip {overwrite_flag} {shlex.quote(archive_remote_path)} -d {shlex.quote(output_remote_path)}"
    )
    try:
        client.connect(
            hostname=archive_host,
            port=use_port,
            username=auth_user,
            password=password,
            pkey=pkey,
            timeout=30,
        )
        _, stdout, stderr = client.exec_command(command)
        channel = stdout.channel
        while not channel.exit_status_ready():
            time.sleep(0.5)
        exit_code = channel.recv_exit_status()
        err_text = stderr.read().decode("utf-8", errors="replace").strip()
        if exit_code != 0:
            raise ValueError(
                f"remote unzip failed (exit={exit_code}) on {archive_host}:{use_port}: "
                f"{err_text or 'unknown error'}"
            )
        return {
            "archive_path": archive_remote_path,
            "output_dir": output_remote_path,
            "target_host": output_host,
            "target_port": output_port,
            "command": command,
        }
    except paramiko.AuthenticationException as exc:
        raise ValueError(f"SSH authentication failed: {exc}") from exc
    except paramiko.SSHException as exc:
        raise ValueError(f"SSH connection failed: {exc}") from exc
    finally:
        client.close()
