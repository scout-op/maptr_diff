# MapTR OpenMMLab 2.0 快速参考

## 🚀 常用命令

### 安装

```bash
# 安装依赖
pip install openmim
mim install mmengine "mmcv>=2.0.0"

# 安装 MMDetection3D
cd /path/to/mmdetection3d
pip install -v -e .
```

### 验证安装

```bash
# 运行验证脚本
cd /path/to/mmdetection3d
python projects/mmdet3d_plugin/tools/verify_installation.py
```

### 训练

```bash
# 单GPU
python projects/mmdet3d_plugin/tools/train.py \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py

# 多GPU (8卡)
bash projects/mmdet3d_plugin/tools/dist_train.sh \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py 8

# 使用AMP
python projects/mmdet3d_plugin/tools/train.py \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py \
    --amp

# 断点续训
python projects/mmdet3d_plugin/tools/train.py \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py \
    --resume work_dirs/maptr_av2_example/latest.pth
```

### 测试

```bash
# 单GPU测试
python projects/mmdet3d_plugin/tools/test.py \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py \
    work_dirs/maptr_av2_example/epoch_24.pth

# 多GPU测试
bash projects/mmdet3d_plugin/tools/dist_test.sh \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py \
    work_dirs/maptr_av2_example/epoch_24.pth 8

# 可视化
python projects/mmdet3d_plugin/tools/test.py \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py \
    work_dirs/maptr_av2_example/epoch_24.pth \
    --show --show-dir work_dirs/vis
```

### 配置覆盖

```bash
# 修改学习率
python tools/train.py config.py \
    --cfg-options optim_wrapper.optimizer.lr=1e-4

# 修改batch size
python tools/train.py config.py \
    --cfg-options train_dataloader.batch_size=2

# 修改训练epoch
python tools/train.py config.py \
    --cfg-options train_cfg.max_epochs=48

# 多个参数
python tools/train.py config.py \
    --cfg-options \
    optim_wrapper.optimizer.lr=1e-4 \
    train_dataloader.batch_size=2 \
    train_cfg.max_epochs=48
```

---

## 📝 配置文件模板

### 最小配置

```python
_base_ = ['./_base_/default_runtime.py']
default_scope = 'mmdet3d'

model = dict(type='MapTR', ...)
train_dataloader = dict(...)
val_dataloader = dict(...)
test_dataloader = dict(...)
val_evaluator = dict(type='MapMetric', ...)
optim_wrapper = dict(...)
param_scheduler = [...]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

### 评估器配置

```python
# 基础评估（不需要GT）
val_evaluator = dict(
    type='MapMetric',
    metric='chamfer')

# 完整评估（需要GT文件）
val_evaluator = dict(
    type='MapMetricWithGT',
    ann_file='path/to/gt.json',
    map_classes=('divider', 'ped_crossing', 'boundary'),
    metric=['chamfer', 'iou'],
    prefix='AV2Map')
```

### 优化器配置

```python
# AdamW + AMP
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

# 分层学习率
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'img_neck': dict(lr_mult=0.5),
        }))
```

---

## 🔧 调试技巧

### 检查配置

```python
from mmengine.config import Config

cfg = Config.fromfile('config.py')
print(cfg.pretty_text)
```

### 检查模型

```python
from mmdet3d.registry import MODELS
from mmengine.config import Config

cfg = Config.fromfile('config.py')
model = MODELS.build(cfg.model)
print(model)
```

### 检查数据集

```python
from mmdet3d.registry import DATASETS
from mmengine.config import Config

cfg = Config.fromfile('config.py')
dataset = DATASETS.build(cfg.train_dataloader.dataset)
print(f'Dataset size: {len(dataset)}')
print(f'First sample keys: {dataset[0].keys()}')
```

### 启用调试日志

```bash
export MMENGINE_LOG_LEVEL=DEBUG
python tools/train.py config.py
```

### 单步调试数据加载

```python
from mmengine.config import Config
from mmdet3d.registry import DATASETS

cfg = Config.fromfile('config.py')
dataset = DATASETS.build(cfg.train_dataloader.dataset)

# 加载单个样本
data = dataset[0]
print(data.keys())
print(data['img'].shape if 'img' in data else 'No img')
```

---

## 📦 注册新组件

### 注册模型

```python
from mmdet3d.registry import MODELS
from mmdet3d.models import Base3DDetector

@MODELS.register_module()
class MyDetector(Base3DDetector):
    def __init__(self, ...):
        super().__init__()
```

### 注册数据集

```python
from mmdet3d.registry import DATASETS
from mmdet3d.datasets import Det3DDataset

@DATASETS.register_module()
class MyDataset(Det3DDataset):
    def __init__(self, ...):
        super().__init__(...)
```

### 注册Transform

```python
from mmdet3d.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

@TRANSFORMS.register_module()
class MyTransform(BaseTransform):
    def transform(self, results):
        ...
        return results
```

### 注册评估指标

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

### 注册Hook

```python
from mmengine.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class MyHook(Hook):
    def before_train_epoch(self, runner):
        ...
```

---

## 🐛 常见错误

### ImportError: cannot import name 'DETECTORS'

**原因**: 使用了旧的注册器名称  
**解决**: 
```python
# ❌ 旧
from mmdet.models import DETECTORS

# ✅ 新
from mmdet3d.registry import MODELS
```

### DataContainer is deprecated

**原因**: 仍在使用 DataContainer  
**解决**: 直接使用 tensor 或 dict

### Config file error: 'default_scope' not found

**原因**: 配置文件缺少 default_scope  
**解决**: 在配置文件顶部添加：
```python
default_scope = 'mmdet3d'
```

### Runner initialization failed

**原因**: train_cfg/val_cfg/test_cfg 格式错误  
**解决**: 使用新格式：
```python
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
```

---

## 📊 性能优化

### 混合精度训练

```python
optim_wrapper = dict(
    type='AmpOptimWrapper',  # 启用AMP
    optimizer=dict(type='AdamW', lr=2e-4))
```

### 梯度累积

```python
optim_wrapper = dict(
    type='AmpOptimWrapper',
    accumulative_counts=4,  # 每4步更新一次
    optimizer=dict(type='AdamW', lr=2e-4))
```

### DataLoader优化

```python
train_dataloader = dict(
    batch_size=2,
    num_workers=8,           # 增加worker
    persistent_workers=True,  # 保持worker存活
    prefetch_factor=2,       # 预取数据
)
```

### 分布式训练

```bash
# 使用 torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29500 \
    tools/train.py config.py --launcher pytorch

# 或使用提供的脚本
bash tools/dist_train.sh config.py 8
```

---

## 📚 文档索引

| 文档 | 内容 |
|------|------|
| `README.md` | 完整使用指南 |
| `UPGRADE_COMPLETE.md` | 升级完成报告 |
| `MIGRATION_SUMMARY.md` | 迁移详细总结 |
| `QUICKSTART.md` | 快速开始 |
| `REFACTOR_TODO.md` | 待办事项（可选） |

---

## 🔗 有用的链接

- [MMEngine 文档](https://mmengine.readthedocs.io/)
- [MMDetection3D 文档](https://mmdetection3d.readthedocs.io/)
- [MMCV 文档](https://mmcv.readthedocs.io/)
- [OpenMMLab GitHub](https://github.com/open-mmlab)

---

**最后更新**: 2024-12-09
