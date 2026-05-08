from __future__ import annotations

import os
import re
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import yaml
from langchain.tools import tool

from app.services.publish import (
    read_sftp_file_text,
    run_remote_transfer_service,
    run_remote_unzip_service,
    run_zip_folder_service,
)

_DEFAULT_SPLITS = ["train", "val", "test"]


def _report_stage(current: int, total: int, message: str) -> None:
    print(f"[阶段 {current}/{total}] {message}", flush=True)


def _is_remote_text(value: str | None) -> bool:
    if not value:
        return False
    text = value.strip()
    return text.startswith(("sftp://", "ssh://")) or bool(re.match(r"^[^@:\s]+@[^:\s]+:.+$", text))


def _parse_remote_like(value: str) -> tuple[str | None, int | None, PurePosixPath]:
    text = value.strip()
    if text.startswith(("sftp://", "ssh://")):
        parsed = urlparse(text)
        return parsed.hostname, parsed.port, PurePosixPath(parsed.path or "/")
    scp_match = re.match(r"^([^@]+)@([^:]+):(.+)$", text)
    if scp_match:
        return scp_match.group(2), 22, PurePosixPath(scp_match.group(3))
    colon_match = re.match(r"^([^@:]+):(.+)$", text)
    if colon_match:
        return colon_match.group(1), 22, PurePosixPath(colon_match.group(2))
    return None, None, PurePosixPath(text)


def _extract_remote_username(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if text.startswith(("sftp://", "ssh://")):
        parsed = urlparse(text)
        return parsed.username
    match = re.match(r"^([^@]+)@[^:]+:.+$", text)
    if match:
        return match.group(1)
    return None


def _infer_context_from_oldyaml(oldyaml: str) -> dict[str, str | int | None]:
    host, port, path = _parse_remote_like(oldyaml)
    parts = path.parts
    bucket_idx = next((i for i, part in enumerate(parts) if part in {"dataset", "datasets"}), None)
    if bucket_idx is None or bucket_idx < 2 or len(parts) <= bucket_idx + 2:
        raise ValueError(
            "oldyaml格式无效。应为 .../<detector_name>/datasets/<version>/<version>.yaml"
        )

    detector_name = parts[bucket_idx - 1]
    dataset_bucket = parts[bucket_idx]
    dataset_version = parts[bucket_idx + 1]
    if Path(path.name).stem != dataset_version:
        raise ValueError(
            "oldyaml格式无效。要求yaml文件名去掉后缀后与dataset_version一致。"
        )

    root_dir = PurePosixPath(*parts[: bucket_idx - 1]).as_posix() or "/"
    payload: dict[str, str | int | None] = {
        "is_remote": bool(host),
        "detector_name": detector_name,
        "dataset_bucket": dataset_bucket,
        "last_dataset_version": dataset_version,
    }
    if host:
        payload["remote_host"] = host
        payload["remote_port"] = port or 22
        payload["remote_project_root_dir"] = root_dir
    else:
        payload["project_root_dir"] = root_dir
    return payload


def _infer_context_from_detector_path(detector_path: str) -> dict[str, str | int | None]:
    if _is_remote_text(detector_path):
        host, port, path = _parse_remote_like(detector_path)
        detector_name = path.name.strip()
        if not detector_name:
            raise ValueError("detector_path必须以detector_name结尾")
        root_dir = path.parent.as_posix() or "/"
        return {
            "is_remote": True,
            "remote_host": host,
            "remote_port": port or 22,
            "remote_project_root_dir": root_dir,
            "detector_name": detector_name,
            "dataset_bucket": "datasets",
        }

    path = Path(detector_path).expanduser().resolve()
    detector_name = path.name.strip()
    if not detector_name:
        raise ValueError("detector_path必须以detector_name结尾")
    return {
        "is_remote": False,
        "project_root_dir": str(path.parent),
        "detector_name": detector_name,
        "dataset_bucket": "datasets",
    }


def _default_dataset_version(detector_name: str) -> str:
    from datetime import datetime

    return f"{detector_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"


def _normalize_input_dirs(input_dir: str, input_dirs: list[str] | None) -> list[Path]:
    raw = [input_dir, *((input_dirs or []))]
    paths: list[Path] = []
    seen: set[Path] = set()
    for item in raw:
        text = (item or "").strip()
        if not text:
            continue
        path = Path(text).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"输入数据集目录不存在: {path}")
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)

    if len(paths) == 1:
        only = paths[0]
        only_text = str(only)
        if only_text.endswith("_aug"):
            base_candidate = Path(only_text[: -len("_aug")]).resolve()
            if base_candidate.is_dir() and base_candidate not in seen:
                paths.insert(0, base_candidate)
        else:
            aug_candidate = Path(f"{only_text}_aug").resolve()
            if aug_candidate.is_dir() and aug_candidate not in seen:
                paths.append(aug_candidate)

    if not paths:
        raise ValueError("input_dir或input_dirs至少要提供一个数据集路径")
    return paths


def _read_classes_optional(dataset_root: Path) -> list[str]:
    classes_path = dataset_root / "classes.txt"
    if not classes_path.is_file():
        return []
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _collect_classes(input_dirs: list[Path]) -> list[str]:
    class_names: list[str] = []
    for root in input_dirs:
        current = _read_classes_optional(root)
        if not current:
            continue
        if not class_names:
            class_names = current
        elif class_names != current:
            raise ValueError("所有输入数据集的classes.txt内容必须完全一致")
    return class_names


def _collect_split_dirs(dataset_root: Path) -> dict[str, list[Path]]:
    split_dirs: dict[str, list[Path]] = {}
    direct_images = dataset_root / "images"
    if direct_images.is_dir():
        split_dirs["train"] = [direct_images.resolve()]
        return split_dirs

    for split in _DEFAULT_SPLITS:
        images_dir = dataset_root / split / "images"
        if images_dir.is_dir():
            split_dirs.setdefault(split, []).append(images_dir.resolve())
    if split_dirs:
        return split_dirs

    for images_dir in sorted(p for p in dataset_root.rglob("images") if p.is_dir()):
        parent_name = images_dir.parent.name.lower()
        if parent_name in _DEFAULT_SPLITS:
            split_dirs.setdefault(parent_name, []).append(images_dir.resolve())
    if not split_dirs:
        raise ValueError(f"在数据集根目录下未找到images目录: {dataset_root}")
    return split_dirs


def _merge_split_paths(last_paths: dict[str, list[str]], new_paths: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for split in _DEFAULT_SPLITS:
        seen: set[str] = set()
        combined: list[str] = []
        for path in last_paths.get(split, []) + new_paths.get(split, []):
            if path not in seen:
                seen.add(path)
                combined.append(path)
        if combined:
            merged[split] = combined
    return merged


def _load_last_yaml_text(oldyaml: str, *, username: str | None, private_key_path: str | None, port: int | None) -> tuple[str, str]:
    if _is_remote_text(oldyaml):
        if not username or not private_key_path:
            raise ValueError(
                "远程oldyaml需要提供remote_username和remote_private_key_path"
                "（或者在环境变量中提供对应默认值）"
            )
        return read_sftp_file_text(
            oldyaml,
            username=username,
            private_key_path=private_key_path,
            port=port,
        ), "sftp"

    yaml_path = Path(oldyaml).expanduser().resolve()
    if not yaml_path.is_file():
        raise ValueError(f"oldyaml不存在: {yaml_path}")
    return yaml_path.read_text(encoding="utf-8"), "local"


def _extract_split_paths_from_yaml_text(yaml_text: str) -> dict[str, list[str]]:
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        return {}
    result: dict[str, list[str]] = {}
    for split in _DEFAULT_SPLITS:
        value = data.get(split)
        if not value:
            continue
        if isinstance(value, list):
            result[split] = [str(item) for item in value if str(item).strip()]
        else:
            result[split] = [str(value)]
    return result


def _class_names_from_yaml_text(yaml_text: str) -> list[str]:
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        return []
    names = data.get("names")
    if isinstance(names, list):
        return [str(item).strip() for item in names if str(item).strip()]
    if isinstance(names, dict):
        ordered = []
        for key, value in sorted(names.items(), key=lambda item: int(item[0])):
            ordered.append(str(value).strip())
        return [item for item in ordered if item]
    return []


def _build_yaml_text(split_paths: dict[str, list[str]], class_names: list[str]) -> str:
    payload: dict[str, object] = {}
    for split in _DEFAULT_SPLITS:
        paths = split_paths.get(split)
        if not paths:
            continue
        payload[split] = paths[0] if len(paths) == 1 else paths
    payload["names"] = {idx: name for idx, name in enumerate(class_names)}
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)


def _stage_merged_dataset(
    input_dirs: list[Path],
    project_root_dir: Path,
    detector_name: str,
    dataset_version: str,
) -> tuple[Path, dict[str, list[Path]]]:
    dataset_dir = (project_root_dir / detector_name / "datasets" / dataset_version).resolve()
    if dataset_dir.exists():
        raise ValueError(f"staging数据集目录已存在: {dataset_dir}")

    dataset_dir.mkdir(parents=True, exist_ok=False)
    split_dirs: dict[str, list[Path]] = {}
    flatten_single = len(input_dirs) == 1
    used_names: set[str] = set()

    for source_root in input_dirs:
        target_root = dataset_dir
        if not flatten_single:
            source_name = source_root.name
            candidate = source_name
            idx = 2
            while candidate in used_names:
                candidate = f"{source_name}_{idx}"
                idx += 1
            used_names.add(candidate)
            target_root = dataset_dir / candidate
            shutil.copytree(source_root, target_root)
        else:
            for child in source_root.iterdir():
                destination = target_root / child.name
                if child.is_dir():
                    shutil.copytree(child, destination)
                else:
                    shutil.copy2(child, destination)

        current_splits = _collect_split_dirs(target_root)
        for split, paths in current_splits.items():
            split_dirs.setdefault(split, []).extend(path.resolve() for path in paths)

    return dataset_dir, split_dirs


def _remote_yaml_path(host: str, port: int, remote_yaml: PurePosixPath, username: str | None = None) -> str:
    prefix = f"{username}@" if username else ""
    if port == 22:
        return f"sftp://{prefix}{host}{remote_yaml.as_posix()}"
    return f"sftp://{prefix}{host}:{port}{remote_yaml.as_posix()}"


@tool
def publish_yolo_dataset(
    input_dir: str,
    input_dirs: list[str] | None = None,
    oldyaml: str | None = None,
    detector_path: str | None = None,
    dataset_version: str | None = None,
    remote_username: str | None = None,
    remote_private_key_path: str | None = None,
    remote_password: str | None = None,
    remote_port: int | None = None,
) -> str:
    """发布 YOLO 数据集。提供 oldyaml 表示增量发布，否则必须提供 detector_path。"""
    oldyaml = (oldyaml or "").strip() or None
    detector_path = (detector_path or "").strip() or None
    if not oldyaml and not detector_path:
        raise ValueError("必须提供oldyaml或detector_path中的一个")

    total_stages = 7 if (oldyaml and _is_remote_text(oldyaml)) or (detector_path and _is_remote_text(detector_path)) else 4
    _report_stage(1, total_stages, "收集输入数据集与发布上下文")

    input_paths = _normalize_input_dirs(input_dir, input_dirs)
    class_names = _collect_classes(input_paths)

    env_username = os.getenv("REMOTE_SFTP_USERNAME")
    env_key = os.getenv("REMOTE_SFTP_PRIVATE_KEY_PATH")
    env_port = os.getenv("REMOTE_SFTP_PORT")
    remote_username = (remote_username or "").strip() or env_username or None
    remote_private_key_path = (remote_private_key_path or "").strip() or env_key or None
    resolved_remote_port = remote_port or (int(env_port) if env_port else 22)

    if oldyaml:
        _report_stage(2, total_stages, "加载oldyaml并提取历史发布信息")
        context = _infer_context_from_oldyaml(oldyaml)
        last_yaml_text, last_yaml_source = _load_last_yaml_text(
            oldyaml,
            username=remote_username or _extract_remote_username(oldyaml),
            private_key_path=remote_private_key_path,
            port=resolved_remote_port,
        )
    else:
        _report_stage(2, total_stages, "根据detector_path推断发布目标")
        context = _infer_context_from_detector_path(detector_path or "")
        last_yaml_text = None
        last_yaml_source = None

    detector_name = str(context["detector_name"])
    is_remote = bool(context["is_remote"])
    final_dataset_version = dataset_version or _default_dataset_version(detector_name)

    if not class_names and last_yaml_text:
        class_names = _class_names_from_yaml_text(last_yaml_text)
    if not class_names:
        raise ValueError("未找到classes.txt，且oldyaml中也没有names字段")

    if is_remote:
        local_publish_root = (Path.cwd() / "publish_workspace").resolve()
    else:
        local_publish_root = Path(str(context["project_root_dir"])).expanduser().resolve()

    _report_stage(3, total_stages, "构建本地staging数据集目录")
    staging_dataset_dir, staging_split_dirs = _stage_merged_dataset(
        input_paths,
        local_publish_root,
        detector_name,
        final_dataset_version,
    )

    (staging_dataset_dir / "classes.txt").write_text(
        "".join(f"{name}\n" for name in class_names),
        encoding="utf-8",
    )

    local_split_paths = {
        split: [path.as_posix() for path in paths]
        for split, paths in staging_split_dirs.items()
        if paths
    }
    if last_yaml_text:
        local_split_paths = _merge_split_paths(
            _extract_split_paths_from_yaml_text(last_yaml_text),
            local_split_paths,
        )

    _report_stage(4, total_stages, "生成新的yaml与classes.txt")
    yaml_text = _build_yaml_text(local_split_paths, class_names)
    staging_yaml_path = staging_dataset_dir / f"{final_dataset_version}.yaml"
    staging_yaml_path.write_text(yaml_text, encoding="utf-8")

    if not is_remote:
        return (
            f"完成。mode=local，detector_name={detector_name}，dataset_version={final_dataset_version}，"
            f"published_dataset_dir={staging_dataset_dir}，yaml_path={staging_yaml_path}，"
            f"last_yaml_source={last_yaml_source}"
        )

    remote_host = str(context["remote_host"] or "").strip()
    remote_project_root = PurePosixPath(str(context["remote_project_root_dir"]))
    dataset_bucket = str(context.get("dataset_bucket") or "datasets")
    if not remote_host:
        raise ValueError("无法从oldyaml或detector_path中推断remote host")
    if not remote_username:
        raise ValueError("远程发布需要提供remote_username")
    if not remote_password and not remote_private_key_path:
        raise ValueError("远程发布需要提供remote_private_key_path或remote_password")

    remote_datasets_parent = remote_project_root / detector_name / dataset_bucket
    remote_dataset_dir = remote_datasets_parent / final_dataset_version
    remote_yaml_file = remote_dataset_dir / f"{final_dataset_version}.yaml"
    remote_yaml_uri = _remote_yaml_path(remote_host, resolved_remote_port, remote_yaml_file, remote_username)

    remote_split_paths = {
        split: [
            (remote_dataset_dir / path.relative_to(staging_dataset_dir)).as_posix()
            for path in paths
        ]
        for split, paths in staging_split_dirs.items()
        if paths
    }
    if last_yaml_text:
        remote_split_paths = _merge_split_paths(
            _extract_split_paths_from_yaml_text(last_yaml_text),
            remote_split_paths,
        )
    staging_yaml_path.write_text(_build_yaml_text(remote_split_paths, class_names), encoding="utf-8")

    _report_stage(5, total_stages, "将staging数据集打包为zip")
    archive_resp = run_zip_folder_service(
        str(staging_dataset_dir),
        str(staging_dataset_dir.parent / f"{final_dataset_version}.zip"),
        include_root_dir=True,
        overwrite=False,
    )
    _report_stage(6, total_stages, "上传zip到远程服务器")
    transfer_resp = run_remote_transfer_service(
        archive_resp["output_zip_path"],
        _remote_yaml_path(remote_host, resolved_remote_port, remote_datasets_parent, remote_username).rsplit("/", 1)[0],
        username=remote_username,
        private_key_path=remote_private_key_path,
        password=remote_password,
        port=resolved_remote_port,
        overwrite=False,
    )
    _report_stage(7, total_stages, "在远程服务器解压并完成发布")
    unzip_resp = run_remote_unzip_service(
        transfer_resp["target_path"] if transfer_resp["target_path"].startswith(("sftp://", "ssh://")) else f"{remote_username}@{remote_host}:{transfer_resp['target_path']}",
        f"{remote_username}@{remote_host}:{remote_datasets_parent.as_posix()}",
        username=remote_username,
        private_key_path=remote_private_key_path,
        password=remote_password,
        port=resolved_remote_port,
        overwrite=False,
    )

    return (
        f"完成。mode=remote_sftp，detector_name={detector_name}，dataset_version={final_dataset_version}，"
        f"staging_dataset_dir={staging_dataset_dir}，local_archive_path={archive_resp['output_zip_path']}，"
        f"remote_archive_path={transfer_resp['target_path']}，remote_dataset_dir={remote_dataset_dir.as_posix()}，"
        f"yaml_path={remote_yaml_uri}，unzip_output_dir={unzip_resp['output_dir']}，"
        f"last_yaml_source={last_yaml_source}"
    )
