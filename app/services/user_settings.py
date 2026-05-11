from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import paramiko

from app.services.chat_store import load_app_state_value, save_app_state_value
from app.services.publish.remote_transfer_service import _load_private_key

USER_SETTINGS_KEY = "user_settings"

DEFAULT_USER_SETTINGS: dict[str, Any] = {
    "remote_sftp_host": "172.31.1.42",
    "remote_sftp_username": "",
    "remote_sftp_private_key_path": "/home/qzq/.ssh/id_ed25519",
    "remote_sftp_port": 22,
}


def _normalize_user_settings(payload: dict[str, Any]) -> dict[str, Any]:
    settings = {**DEFAULT_USER_SETTINGS, **payload}
    try:
        port = int(settings.get("remote_sftp_port") or DEFAULT_USER_SETTINGS["remote_sftp_port"])
    except (TypeError, ValueError):
        port = int(DEFAULT_USER_SETTINGS["remote_sftp_port"])

    return {
        "remote_sftp_host": str(settings.get("remote_sftp_host") or "").strip(),
        "remote_sftp_username": str(settings.get("remote_sftp_username") or "").strip(),
        "remote_sftp_private_key_path": str(settings.get("remote_sftp_private_key_path") or "").strip(),
        "remote_sftp_port": port,
    }


def load_user_settings() -> dict[str, Any]:
    return _normalize_user_settings(load_app_state_value(USER_SETTINGS_KEY))


def save_user_settings(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_user_settings(settings)
    save_app_state_value(USER_SETTINGS_KEY, normalized)
    return normalized


def test_user_settings_connection(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_user_settings(settings)
    host = normalized["remote_sftp_host"]
    username = normalized["remote_sftp_username"]
    private_key_path = normalized["remote_sftp_private_key_path"]
    port = normalized["remote_sftp_port"]

    missing_fields = [
        label
        for label, value in (
            ("Host", host),
            ("Username", username),
            ("Private key", private_key_path),
            ("Port", port),
        )
        if not value
    ]
    if missing_fields:
        raise ValueError(f"用户信息未填写完整: {', '.join(missing_fields)}")

    key_path = Path(private_key_path).expanduser()
    if not key_path.exists():
        raise ValueError(f"Private key不存在: {key_path}")

    pkey = _load_private_key(private_key_path)
    started_at = time.perf_counter()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=port,
            username=username,
            pkey=pkey,
            timeout=8,
            banner_timeout=8,
            auth_timeout=8,
        )
        sftp = client.open_sftp()
        sftp.listdir(".")
        latency_ms = round((time.perf_counter() - started_at) * 1000)
        return {
            "ok": True,
            "message": f"联通正常，SFTP响应 {latency_ms} ms",
            "latency_ms": latency_ms,
        }
    except paramiko.AuthenticationException as exc:
        raise ValueError(f"SSH认证失败: {exc}") from exc
    except paramiko.SSHException as exc:
        raise ValueError(f"SSH连接失败: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"SFTP联通失败: {exc}") from exc
    finally:
        client.close()
