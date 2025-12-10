# MapTR - OpenMMLab 2.0

MapTR 已成功迁移至 OpenMMLab 2.0 (MMEngine + MMDetection3D v1.4.0)。

## 安装

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- CUDA 10.2+
- MMEngine >= 0.8.0
- MMDetection3D == 1.4.0
- MMCV >= 2.0.0

### 安装步骤

1. **安装 MMEngine 和 MMCV**
```bash
pip install openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

2. **安装 MMDetection3D v1.4.0**
```bash
cd /path/to/mmdetection3d
pip install -v -e .
```

3. **安装 MapTR 插件**

MapTR 作为 MMDetection3D 的插件，位于 `projects/mmdet3d_plugin/`：

```bash
cd projects/mmdet3d_plugin
# 所有模块会在训练时自动注册
```

## 快速开始

### 准备数据

**Argoverse 2:**
```bash
# 下载 Argoverse 2 数据集
# 组织数据结构如下：
data/av2/
├── train/
├── val/
├── test/
└── av2_map_ann.json
```

**NuScenes:**
```bash
# 下载 NuScenes 数据集
data/nuscenes/
├── maps/
├── samples/
├── v1.0-trainval/
└── nuscenes_map_ann.json
```

### 训练

**单GPU训练:**
```bash
cd /path/to/mmdetection3d
python projects/mmdet3d_plugin/tools/train.py \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py
```

**多GPU训练:**
```bash
cd /path/to/mmdetection3d
bash projects/mmdet3d_plugin/tools/dist_train.sh \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py 8
```

**使用AMP训练:**
```bash
python projects/mmdet3d_plugin/tools/train.py \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py \
    --amp
```

**从检查点恢复:**
```bash
python projects/mmdet3d_plugin/tools/train.py \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py \
    --resume work_dirs/maptr_av2_example/latest.pth
```

### 测试

**单GPU测试:**
```bash
python projects/mmdet3d_plugin/tools/test.py \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py \
    work_dirs/maptr_av2_example/latest.pth
```

**多GPU测试:**
```bash
bash projects/mmdet3d_plugin/tools/dist_test.sh \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py \
    work_dirs/maptr_av2_example/latest.pth 8
```

**可视化结果:**
```bash
python projects/mmdet3d_plugin/tools/test.py \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py \
    work_dirs/maptr_av2_example/latest.pth \
    --show \
    --show-dir work_dirs/vis_results
```

## 配置文件

### 配置文件结构

```
configs/
├── _base_/
│   └── default_runtime.py          # 基础运行时配置
├── maptr_av2_example.py            # AV2 示例配置
└── maptr_nuscenes_example.py       # NuScenes 示例配置
```

### 自定义配置

你可以通过命令行覆盖配置：

```bash
python tools/train.py config.py \
    --cfg-options \
    model.pts_bbox_head.num_queries=200 \
    optim_wrapper.optimizer.lr=1e-4 \
    train_dataloader.batch_size=2
```

## 模型架构

### 主要组件

- **Backbone**: ResNet-50/101, VoVNet, EfficientNet
- **Neck**: FPN
- **Head**: MapTRHead (Transformer-based)
- **Loss**: Focal Loss, L1 Loss, GIoU Loss

### 注册器

所有组件使用 MMEngine 注册器系统：

```python
from mmdet3d.registry import MODELS

@MODELS.register_module()
class MapTR(Base3DDetector):
    ...
```

## 评估指标

MapTR 使用自定义评估指标：

- **Chamfer Distance**: 0.5m, 1.0m, 1.5m 阈值
- **IoU**: 0.5 到 0.95，步长 0.05

配置示例：

```python
val_evaluator = dict(
    type='MapMetricWithGT',
    ann_file='data/av2/av2_map_ann_val.json',
    map_classes=('divider', 'ped_crossing', 'boundary'),
    metric=['chamfer', 'iou'],
    prefix='AV2Map')
```

## 迁移说明

### 主要变化

1. **注册器**: 
   - 旧: `DETECTORS`, `HEADS` 等
   - 新: 统一使用 `MODELS`

2. **训练架构**:
   - 旧: `EpochBasedRunner`
   - 新: `mmengine.runner.Runner`

3. **评估**:
   - 旧: `DistEvalHook`
   - 新: `Evaluator` + `Metric`

4. **数据容器**:
   - 旧: `DataContainer`
   - 新: 原生 tensor/dict

5. **配置格式**:
   - 添加 `default_scope = 'mmdet3d'`
   - `train_cfg`, `val_cfg`, `test_cfg` 格式更新
   - `optimizer` → `optim_wrapper`

### 详细文档

- `MIGRATION_STATUS.md` - 迁移状态追踪
- `MIGRATION_SUMMARY.md` - 完整迁移总结
- `REFACTOR_TODO.md` - 待办事项
- `QUICKSTART.md` - 快速开始指南

## 开发

### 添加新模块

1. 在相应目录创建模块
2. 使用注册器装饰器注册
3. 在 `__init__.py` 中导入

示例：

```python
from mmdet3d.registry import MODELS

@MODELS.register_module()
class MyNewHead(BaseModule):
    def __init__(self, ...):
        super().__init__()
        ...
```

### 自定义数据集

```python
from mmdet3d.registry import DATASETS

@DATASETS.register_module()
class MyDataset(CustomNuScenesDataset):
    def __init__(self, ...):
        super().__init__(...)
        ...
```

### 自定义评估指标

```python
from mmdet3d.registry import METRICS
from mmengine.evaluator import BaseMetric

@METRICS.register_module()
class MyMetric(BaseMetric):
    def process(self, data_batch, data_samples):
        ...
    
    def compute_metrics(self, results):
        ...
```

## 故障排除

### 常见问题

**Q: ImportError: cannot import name 'DETECTORS'**

A: 使用新的注册器：
```python
# 旧
from mmdet.models import DETECTORS
# 新
from mmdet3d.registry import MODELS
```

**Q: 配置文件加载失败**

A: 确保配置文件包含 `default_scope = 'mmdet3d'`

**Q: DataContainer 相关错误**

A: 所有 DataContainer 已移除，直接使用 tensor

**Q: Runner 构建失败**

A: 检查 `train_cfg`, `val_cfg`, `test_cfg` 格式

### 调试技巧

```bash
# 启用详细日志
export MMENGINE_LOG_LEVEL=DEBUG

# 测试配置加载
python -c "
from mmengine.config import Config
cfg = Config.fromfile('configs/maptr_av2_example.py')
print(cfg.pretty_text)
"

# 测试数据加载
python -c "
from mmengine.config import Config
from mmdet3d.registry import DATASETS
cfg = Config.fromfile('configs/maptr_av2_example.py')
dataset = DATASETS.build(cfg.train_dataloader.dataset)
print(len(dataset))
"
```

## 性能

预期性能（参考）：

| Dataset | Backbone | mAP@0.5 | mAP@1.0 | mAP@1.5 |
|---------|----------|---------|---------|---------|
| AV2     | ResNet-50| TBD     | TBD     | TBD     |
| NuScenes| ResNet-50| TBD     | TBD     | TBD     |

## 引用

```bibtex
@article{maptr,
  title={MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction},
  author={},
  journal={arXiv preprint},
  year={2023}
}
```

## 许可证

本项目遵循 Apache 2.0 许可证。

## 致谢

- OpenMMLab 团队提供的优秀框架
- MMDetection3D v1.4.0
- MMEngine
