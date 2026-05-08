from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import paramiko


def _parse_target(target: str) -> tuple[str, int, str]:
    target = target.strip()
    if not target:
        raise ValueError("target不能为空")

    if target.startswith(("sftp://", "ssh://")):
        parsed = urlparse(target)
        if not parsed.hostname:
            raise ValueError(f"target格式无效，缺少host: {target}")
        host = parsed.hostname
        port = parsed.port or 22
        path = parsed.path or "/"
        path = path if path.startswith("/") else f"/{path}"
        return host, port, path

    scp_match = re.match(r"^([^@]+)@([^:]+):(.+)$", target)
    if scp_match:
        return scp_match.group(2), 22, scp_match.group(3)

    colon_match = re.match(r"^([^@:]+):(.+)$", target)
    if colon_match:
        return colon_match.group(1), 22, colon_match.group(2)

    raise ValueError(f"target格式无效: {target}")


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


def run_remote_transfer_service(
    source_path: str,
    target: str,
    *,
    username: str | None = None,
    private_key_path: str | None = None,
    password: str | None = None,
    port: int | None = None,
    overwrite: bool = False,
) -> dict:
    local_path = Path(source_path).expanduser().resolve()
    if not local_path.exists():
        raise ValueError(f"source_path不存在: {local_path}")

    host, default_port, remote_path = _parse_target(target)
    auth_user = username or _extract_username_from_target(target)
    if not auth_user:
        raise ValueError("必须提供username，或在target中使用user@host:path格式")

    if password and private_key_path:
        raise ValueError("password和private_key_path不能同时提供")
    if not password and not private_key_path:
        raise ValueError("必须提供password或private_key_path中的一个")

    pkey = _load_private_key(private_key_path) if private_key_path else None
    use_port = port if port is not None else default_port

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=use_port,
            username=auth_user,
            password=password,
            pkey=pkey,
            timeout=30,
        )
        sftp = client.open_sftp()

        def mkdir_p(path: str) -> None:
            current = ""
            for part in path.split("/"):
                if not part:
                    continue
                current = f"{current}/{part}" if current else f"/{part}"
                try:
                    sftp.stat(current)
                except FileNotFoundError:
                    sftp.mkdir(current)

        transferred_files = 0
        total_bytes = 0

        if local_path.is_file():
            target_dir = remote_path.rstrip("/")
            remote_file = (
                f"/{local_path.name}"
                if target_dir in {"", "/"}
                else f"{target_dir}/{local_path.name}"
            )
            try:
                sftp.stat(remote_file)
                if not overwrite:
                    raise ValueError(f"远程文件已存在且overwrite=false: {remote_file}")
            except FileNotFoundError:
                pass

            remote_parent = str(Path(remote_file).parent).replace("\\", "/")
            if remote_parent not in {"", ".", "/"}:
                mkdir_p(remote_parent)
            sftp.put(str(local_path), remote_file)
            transferred_files = 1
            total_bytes = local_path.stat().st_size
            transferred_type = "file"
            final_remote = remote_file
        else:
            base_name = local_path.name
            remote_dir = remote_path.rstrip("/")
            remote_dir = (
                f"/{base_name}" if remote_dir in {"", "/"} else f"{remote_dir}/{base_name}"
            )
            mkdir_p(remote_dir)

            def upload_dir(local_dir: Path, remote_dir_path: str) -> None:
                nonlocal transferred_files, total_bytes
                for item in sorted(local_dir.iterdir()):
                    remote_item = f"{remote_dir_path}/{item.name}"
                    if item.is_file():
                        if not overwrite:
                            try:
                                sftp.stat(remote_item)
                                continue
                            except FileNotFoundError:
                                pass
                        sftp.put(str(item), remote_item)
                        transferred_files += 1
                        total_bytes += item.stat().st_size
                    else:
                        mkdir_p(remote_item)
                        upload_dir(item, remote_item)

            upload_dir(local_path, remote_dir)
            transferred_type = "directory"
            final_remote = remote_dir

        return {
            "source_path": str(local_path),
            "target": target,
            "target_host": host,
            "target_port": use_port,
            "target_path": final_remote,
            "transferred_type": transferred_type,
            "transferred_files": transferred_files,
            "total_bytes": total_bytes,
        }
    except paramiko.AuthenticationException as exc:
        raise ValueError(f"SSH认证失败: {exc}") from exc
    except paramiko.SSHException as exc:
        raise ValueError(f"SSH连接失败: {exc}") from exc
    finally:
        client.close()


def read_sftp_file_text(
    target: str,
    *,
    username: str | None = None,
    private_key_path: str | None = None,
    password: str | None = None,
    port: int | None = None,
) -> str:
    host, default_port, remote_path = _parse_target(target)
    auth_user = username or _extract_username_from_target(target)
    if not auth_user:
        raise ValueError("必须提供username，或在target中使用user@host:path格式")

    if password and private_key_path:
        raise ValueError("password和private_key_path不能同时提供")
    if not password and not private_key_path:
        raise ValueError("必须提供password或private_key_path中的一个")

    pkey = _load_private_key(private_key_path) if private_key_path else None
    use_port = port if port is not None else default_port

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=use_port,
            username=auth_user,
            password=password,
            pkey=pkey,
            timeout=30,
        )
        sftp = client.open_sftp()
        with sftp.open(remote_path, "r") as remote_file:
            return remote_file.read().decode("utf-8")
    except paramiko.AuthenticationException as exc:
        raise ValueError(f"SSH认证失败: {exc}") from exc
    except paramiko.SSHException as exc:
        raise ValueError(f"SSH连接失败: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"远程文件不可读: {remote_path}: {exc}") from exc
    finally:
        client.close()
