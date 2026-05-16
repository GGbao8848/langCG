from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.chat import RAW_TOOLS, TOOLS
from app.agent.prompts import SYSTEM_PROMPT


BUSINESS_TOOL_NAMES = [
    "annotate_visualize",
    "publish_yolo_dataset",
    "augment_yolo_dataset",
    "clean_irregular_dataset",
    "convert_xml_to_yolo",
    "reindex_yolo_labels",
    "split_yolo_dataset",
    "prune_yolo_by_visualized",
    "collect_wubao_images",
    "yolo_sliding_window_crop",
    "export_yolo_torchscript",
    "launch_yolo_training",
]


def _schema(tool_name: str) -> dict[str, Any]:
    args_schema = RAW_TOOLS[tool_name].args_schema
    if args_schema is None:
        raise AssertionError(f"{tool_name} missing args_schema")
    return args_schema.model_json_schema()


def _properties(tool_name: str) -> dict[str, Any]:
    return _schema(tool_name).get("properties", {})


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_tool_schema_descriptions() -> None:
    for tool_name in BUSINESS_TOOL_NAMES:
        tool = RAW_TOOLS[tool_name]
        _assert(bool(tool.description and tool.description.strip()), f"{tool_name} missing description")

        properties = _properties(tool_name)
        _assert(properties, f"{tool_name} missing schema properties")
        missing = [name for name, spec in properties.items() if not spec.get("description")]
        _assert(not missing, f"{tool_name} properties missing description: {missing}")


def verify_defaults() -> None:
    split_props = _properties("split_yolo_dataset")
    _assert(_schema("split_yolo_dataset").get("required") == ["input_dir"], "split_yolo_dataset required args drifted")
    _assert(split_props["mode"].get("default") == "train_val", "split_yolo_dataset mode default must stay train_val")
    _assert(split_props["split_ratio"].get("default") is None, "split_yolo_dataset split_ratio should default to None")

    train_props = _properties("launch_yolo_training")
    expected_train_defaults = {
        "model": None,
        "epochs": None,
        "imgsz": None,
        "workers": 4,
        "cache": "disk",
        "command_prefix": "subyolo",
        "execute": False,
        "require_exists": False,
    }
    for name, expected in expected_train_defaults.items():
        _assert(
            train_props[name].get("default") == expected,
            f"launch_yolo_training {name} default drifted: {train_props[name].get('default')!r}",
        )

    export_props = _properties("export_yolo_torchscript")
    _assert(export_props["execute"].get("default") is False, "export_yolo_torchscript execute default must stay False")
    _assert(export_props["timeout_seconds"].get("default") == 600, "export timeout default drifted")

    prune_props = _properties("prune_yolo_by_visualized")
    _assert(prune_props["dry_run"].get("default") is True, "prune dry_run default must stay True")
    _assert(prune_props["confirm_delete"].get("default") is False, "prune confirm_delete default must stay False")


def verify_error_wrapping() -> None:
    result = TOOLS["publish_yolo_dataset"].invoke({})
    _assert(isinstance(result, str), "safe publish result must be a string")
    _assert(result.startswith("tool执行失败"), "publish validation error should be wrapped as tool执行失败")
    _assert("oldyaml" in result or "detector_path" in result, "publish validation error should mention context fields")


def verify_prompt_contracts() -> None:
    required_fragments = [
        "publish_yolo_dataset",
        "oldyaml",
        "oldyaml 是历史版本 yaml 文件路径，必须原样传给工具",
        "增量发布时类别以 oldyaml 中的 names 为基准",
        "detector_path",
        "background_dir",
        "annotate_visualize",
        "prune_yolo_by_visualized",
        "collect_wubao_images",
        "augment_yolo_dataset",
        "yolo_sliding_window_crop",
        "launch_yolo_training",
        "本地训练 model=yolo11s.pt、epochs=120、imgsz=640",
        "远程训练 model=yolo11m.pt、epochs=200、imgsz=800",
        "cache=disk",
        "execute=False",
        "export_yolo_torchscript",
        "clean_irregular_dataset",
        "split_yolo_dataset",
        "convert_xml_to_yolo",
        "reindex_yolo_labels",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in SYSTEM_PROMPT]
    _assert(not missing, f"SYSTEM_PROMPT missing routing/default fragments: {missing}")


def verify_train_default_resolution() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        local_yaml = root / "louyou" / "datasets" / "louyou_20260513" / "louyou_20260513.yaml"
        local_yaml.parent.mkdir(parents=True)
        local_yaml.write_text("path: .\nnames: [target]\n", encoding="utf-8")
        fake_venv = root / "venv"
        (fake_venv / "bin").mkdir(parents=True)
        fake_yolo = fake_venv / "bin" / "yolo"
        fake_yolo.write_text("#!/bin/sh\n", encoding="utf-8")
        fake_yolo.chmod(0o755)

        local_result = TOOLS["launch_yolo_training"].invoke(
            {
                "yaml_path": str(local_yaml),
                "local_venv_path": str(fake_venv),
            }
        )
        _assert("mode=local_venv" in local_result, f"unexpected local training preview: {local_result}")
        _assert("model=yolo11s.pt" in local_result, f"local model default drifted: {local_result}")
        _assert("epochs=120" in local_result, f"local epochs default drifted: {local_result}")
        _assert("imgsz=640" in local_result, f"local imgsz default drifted: {local_result}")

        remote_result = TOOLS["launch_yolo_training"].invoke(
            {
                "yaml_path": "sftp://train.example.com/data/louyou/datasets/louyou_20260513/louyou_20260513.yaml",
            }
        )
        _assert("mode=remote_script" in remote_result, f"unexpected remote training preview: {remote_result}")
        _assert("model=yolo11m.pt" in remote_result, f"remote model default drifted: {remote_result}")
        _assert("epochs=200" in remote_result, f"remote epochs default drifted: {remote_result}")
        _assert("imgsz=800" in remote_result, f"remote imgsz default drifted: {remote_result}")


def verify_xml_to_yolo_output_root_guard() -> None:
    xml_text = """<annotation>
  <size><width>100</width><height>50</height></size>
  <object>
    <name>target</name>
    <bndbox><xmin>10</xmin><ymin>5</ymin><xmax>60</xmax><ymax>30</ymax></bndbox>
  </object>
</annotation>
"""
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "sample_cleaned"
        (root / "images").mkdir(parents=True)
        (root / "xmls").mkdir(parents=True)
        (root / "xmls" / "a.xml").write_text(xml_text, encoding="utf-8")

        sibling_output = root.parent / "sample_cleaned_labels"
        result = TOOLS["convert_xml_to_yolo"].invoke(
            {
                "input_dir": str(root),
                "output_dir": str(sibling_output),
            }
        )
        _assert(f"output_dir={root}" in result, f"xml output root guard did not normalize sibling path: {result}")
        _assert((root / "labels" / "a.txt").is_file(), "xml labels should be written under input labels")
        _assert((root / "classes.txt").is_file(), "xml classes should be written under input root")
        _assert(not sibling_output.exists(), "xml conversion must not create sibling *_labels output")

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "sample_cleaned"
        (root / "images").mkdir(parents=True)
        (root / "xmls").mkdir(parents=True)
        (root / "xmls" / "a.xml").write_text(xml_text, encoding="utf-8")

        nested_output = root / "labels"
        result = TOOLS["convert_xml_to_yolo"].invoke(
            {
                "input_dir": str(root),
                "output_dir": str(nested_output),
            }
        )
        _assert(f"output_dir={root}" in result, f"xml output root guard did not normalize labels path: {result}")
        _assert((root / "labels" / "a.txt").is_file(), "xml labels should not be nested under labels/labels")
        _assert(not (root / "labels" / "labels").exists(), "xml conversion must not create labels/labels")


def verify_incremental_publish_uses_oldyaml_classes() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        detector_root = root / "detector"
        old_version = "detector_20260501_1200"
        old_dataset_dir = detector_root / "datasets" / old_version
        old_dataset_dir.mkdir(parents=True)
        old_yaml = old_dataset_dir / f"{old_version}.yaml"
        agent_mangled_old_yaml = old_dataset_dir.parent / detector_root.name / old_version / f"{old_version}.yaml"
        old_yaml.write_text(
            "\n".join(
                [
                    "train:",
                    f"- {old_dataset_dir / 'train' / 'images'}",
                    "names:",
                    "  0: old_zero",
                    "  1: old_one",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        input_dir = root / "input_dataset"
        (input_dir / "train" / "images").mkdir(parents=True)
        (input_dir / "train" / "labels").mkdir(parents=True)
        (input_dir / "train" / "images" / "a.jpg").write_bytes(b"")
        (input_dir / "train" / "labels" / "a.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n",
            encoding="utf-8",
        )
        (input_dir / "classes.txt").write_text("new_wrong_class\n", encoding="utf-8")

        dataset_version = "detector_20260513_1200"
        result = TOOLS["publish_yolo_dataset"].invoke(
            {
                "input_dir": str(input_dir),
                "oldyaml": str(agent_mangled_old_yaml),
                "dataset_version": dataset_version,
            }
        )
        output_dir = detector_root / "datasets" / dataset_version
        output_yaml = output_dir / f"{dataset_version}.yaml"
        output_classes = output_dir / "classes.txt"

        _assert("mode=local" in result, f"unexpected incremental publish result: {result}")
        _assert(output_yaml.is_file(), "incremental publish output yaml missing")
        _assert(output_classes.is_file(), "incremental publish classes.txt missing")
        yaml_text = output_yaml.read_text(encoding="utf-8")
        classes_text = output_classes.read_text(encoding="utf-8")
        _assert("0: old_zero" in yaml_text, f"oldyaml class 0 missing from output yaml: {yaml_text}")
        _assert("1: old_one" in yaml_text, f"oldyaml class 1 missing from output yaml: {yaml_text}")
        _assert("new_wrong_class" not in yaml_text, f"input classes leaked into output yaml: {yaml_text}")
        _assert(classes_text == "old_zero\nold_one\n", f"classes.txt should come from oldyaml: {classes_text!r}")


def verify_augment_default_output_is_sibling() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "crop_dataset"
        (root / "train" / "images").mkdir(parents=True)
        (root / "train" / "labels").mkdir(parents=True)
        Image.new("RGB", (16, 16), color=(255, 255, 255)).save(root / "train" / "images" / "a.jpg")
        (root / "train" / "labels" / "a.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n",
            encoding="utf-8",
        )

        result = TOOLS["augment_yolo_dataset"].invoke(
            {
                "input_dir": str(root),
                "horizontal_flip": True,
                "vertical_flip": False,
                "brightness_up": False,
                "brightness_down": False,
                "contrast_up": False,
                "contrast_down": False,
            }
        )
        sibling_output = root.parent / "crop_dataset_augment"
        nested_output = root / "augment"
        _assert(f"output_dir={sibling_output}" in result, f"augment output should be sibling: {result}")
        _assert((sibling_output / "train" / "images" / "a_hflip.jpg").is_file(), "sibling augment image missing")
        _assert(not nested_output.exists(), "augment must not default to input_dir/augment")


def verify_publish_ignores_nested_augment_children() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        detector_root = root / "workspace" / "detector"
        detector_root.mkdir(parents=True)
        input_dir = root / "crop_dataset"
        (input_dir / "train" / "images").mkdir(parents=True)
        (input_dir / "train" / "labels").mkdir(parents=True)
        (input_dir / "augment" / "train" / "images").mkdir(parents=True)
        (input_dir / "augment" / "train" / "labels").mkdir(parents=True)
        (input_dir / "train" / "images" / "a.jpg").write_bytes(b"")
        (input_dir / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (input_dir / "augment" / "train" / "images" / "a_hflip.jpg").write_bytes(b"")
        (input_dir / "augment" / "train" / "labels" / "a_hflip.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n",
            encoding="utf-8",
        )
        (input_dir / "classes.txt").write_text("target\n", encoding="utf-8")

        dataset_version = "detector_20260516_1200"
        result = TOOLS["publish_yolo_dataset"].invoke(
            {
                "input_dir": str(input_dir),
                "detector_path": str(detector_root),
                "dataset_version": dataset_version,
            }
        )
        output_dir = detector_root / "datasets" / dataset_version
        yaml_text = (output_dir / f"{dataset_version}.yaml").read_text(encoding="utf-8")

        _assert("mode=local" in result, f"unexpected publish result: {result}")
        _assert("/augment/" not in yaml_text, f"nested augment path leaked into yaml: {yaml_text}")
        _assert(not (output_dir / "augment").exists(), "nested input augment dir should not be copied")


def verify_publish_rejects_nested_input_dirs() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        detector_root = root / "workspace" / "detector"
        detector_root.mkdir(parents=True)
        input_dir = root / "crop_dataset"
        nested_augment = input_dir / "augment"
        (input_dir / "train" / "images").mkdir(parents=True)
        (input_dir / "train" / "labels").mkdir(parents=True)
        (nested_augment / "train" / "images").mkdir(parents=True)
        (nested_augment / "train" / "labels").mkdir(parents=True)
        (input_dir / "train" / "images" / "a.jpg").write_bytes(b"")
        (input_dir / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (nested_augment / "train" / "images" / "a_hflip.jpg").write_bytes(b"")
        (nested_augment / "train" / "labels" / "a_hflip.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n",
            encoding="utf-8",
        )
        (input_dir / "classes.txt").write_text("target\n", encoding="utf-8")

        result = TOOLS["publish_yolo_dataset"].invoke(
            {
                "input_dir": str(input_dir),
                "input_dirs": [str(nested_augment)],
                "detector_path": str(detector_root),
                "dataset_version": "detector_20260516_1300",
            }
        )
        _assert(result.startswith("tool执行失败"), f"nested input dirs should fail: {result}")
        _assert("不能同时包含父目录和其子目录" in result, f"nested input dirs error unclear: {result}")


def verify_split_smoke() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "dataset"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir(parents=True)
        for index in range(3):
            (root / "images" / f"{index}.jpg").write_bytes(b"")
            (root / "labels" / f"{index}.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n",
                encoding="utf-8",
            )

        result = TOOLS["split_yolo_dataset"].invoke(
            {
                "input_dir": str(root),
                "mode": "train_val",
                "split_ratio": "2:1",
                "shuffle": False,
            }
        )
        output_dir = root.parent / "dataset_split"
        train_images = list((output_dir / "train" / "images").glob("*.jpg"))
        val_images = list((output_dir / "val" / "images").glob("*.jpg"))
        _assert("完成。" in result, f"split tool returned unexpected result: {result}")
        _assert(len(train_images) == 2, f"expected 2 train images, got {len(train_images)}")
        _assert(len(val_images) == 1, f"expected 1 val image, got {len(val_images)}")


def main() -> None:
    checks = [
        verify_tool_schema_descriptions,
        verify_defaults,
        verify_error_wrapping,
        verify_prompt_contracts,
        verify_train_default_resolution,
        verify_xml_to_yolo_output_root_guard,
        verify_incremental_publish_uses_oldyaml_classes,
        verify_augment_default_output_is_sibling,
        verify_publish_ignores_nested_augment_children,
        verify_publish_rejects_nested_input_dirs,
        verify_split_smoke,
    ]
    for check in checks:
        check()
        print(f"ok {check.__name__}")
    print("tool contract verification passed")


if __name__ == "__main__":
    main()
