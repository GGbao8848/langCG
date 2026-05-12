# langCG

langCG 是一个面向视觉模型数据集维护和训练的本地 Agent 工具。支持数据集预处理、数据集发布、增量补充数据集、误报 background 追加、远程训练、本地训练、模型导出，以及一长串连续任务的自动执行。

示例任务 1：新模型训练，将 `...` 中的 XML 转为 YOLO，索引都转为 `0`，按 `train:val=8:2` 划分数据，进行滑窗裁剪和数据增强，将裁剪数据与增强数据发布到 `/media/qzq/16T/TVDS/louyou_zhouxiang`，并使用默认训练参数进行训练。

示例任务 2：老模型迭代，将 `...` 整理为 YOLO 数据，索引都转为 `0`，划分数据使用 `train_only` 形式，`oldyaml` 为 `/media/qzq/16T/TVDS/louyou_zhouxiang/datasets/louyou_zhouxiang_20260512_1511/louyou_zhouxiang_20260512_1511.yaml`，增量发布后使用默认训练参数进行训练。

## 数据集目录规范

```text
detector_name/
  datasets/
    dataset_version/
      dataset_version.yaml
      classes.txt
      train/images
      train/labels
      val/images
      val/labels
      background/images
      background/labels
  runs/train/
```

示例：

```text
/media/qzq/16T/TVDS/louyou_zhouxiang/
  datasets/louyou_zhouxiang_20260512_1511/louyou_zhouxiang_20260512_1511.yaml
  runs/train/
```

## 第一次启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env
cd frontend && npm install && npm run build && cd ..
make run
```

启动后打开前端：

```text
http://127.0.0.1:8765
```
