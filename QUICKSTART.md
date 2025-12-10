# MapTR OpenMMLab 2.0 快速开始指南

## 当前状态

✅ **代码迁移完成度: 76%**

### 已完成 ✅
- 所有核心模型模块 (MapTR, BEVFormer)
- 所有数据集和Pipeline
- 所有Backbone网络
- 基础导入和注册器迁移
- DataContainer 移除

### 待完成 ⏳
- 训练/评估架构重构
- 配置文件更新
- 训练脚本编写

---

## 快速测试迁移结果

### 1. 测试导入
```python
# 测试是否能成功导入核心模块
python -c "
from projects.mmdet3d_plugin.maptr.detectors.maptr import MapTR
from projects.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer
from projects.mmdet3d_plugin.datasets.av2_map_dataset import CustomAV2LocalMapDataset
print('✅ All imports successful!')
"
```

### 2. 测试模型初始化
```python
# 测试基本模型构建
python -c "
from mmengine.config import Config
from mmdet3d.registry import MODELS

# 简单配置
cfg = dict(
    type='MapTR',
    img_backbone=dict(type='ResNet', depth=50),
    # ... 其他配置
)

# model = MODELS.build(cfg)
# print('✅ Model built successfully!')
print('⚠️ 需要完整配置文件')
"
```

---

## 下一步：最小可运行版本

### 步骤1: 创建简单配置文件

创建 `configs/maptr_test.py`:

```python
_base_ = [
    '../../../configs/_base_/default_runtime.py'
]

default_scope = 'mmdet3d'

# 模型配置
model = dict(
    type='MapTR',
    # ... 从旧配置迁移
)

# 数据配置
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        type='CustomAV2LocalMapDataset',
        # ...
    )
)

# 优化器配置
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
)

# 训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=6,
    val_interval=1
)

# 验证配置
val_cfg = dict(type='ValLoop')
val_dataloader = dict(...)
val_evaluator = dict(type='MapMetric')

# 测试配置  
test_cfg = dict(type='TestLoop')
test_dataloader = dict(...)
test_evaluator = dict(type='MapMetric')
```

### 步骤2: 实现视频序列逻辑

在 `maptr/detectors/maptr.py` 中添加：

```python
class MapTR(MVXTwoStageDetector):
    def __init__(self, ..., video_test_mode=False):
        super().__init__(...)
        self.video_test_mode = video_test_mode
        self.prev_bev = None
    
    def train_step(self, data, optim_wrapper):
        """自定义训练步骤，处理视频序列"""
        # 如果 data 包含序列
        if 'sequence' in data:
            # 处理序列前面的帧（仅推理）
            with torch.no_grad():
                for frame in data['sequence'][:-1]:
                    self.prev_bev = self.extract_feat(frame)
            
            # 训练最后一帧
            data_last = data['sequence'][-1]
            losses = self(data_last, prev_bev=self.prev_bev)
        else:
            # 普通训练
            losses = self(**data)
        
        # 解析损失并更新参数
        parsed_losses, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        
        return log_vars
```

### 步骤3: 实现评估指标

创建 `datasets/map_metric.py`:

```python
from mmengine.evaluator import BaseMetric
from mmdet3d.registry import METRICS

@METRICS.register_module()
class MapMetric(BaseMetric):
    def __init__(self, 
                 collect_device='cpu',
                 metric=['chamfer', 'mAP']):
        super().__init__(collect_device=collect_device)
        self.metrics = metric
    
    def process(self, data_batch, data_samples):
        """处理一个batch"""
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            gt = data_sample['gt_instances']
            
            result = {
                'pred': pred,
                'gt': gt,
            }
            self.results.append(result)
    
    def compute_metrics(self, results):
        """计算最终指标"""
        # 实现 chamfer distance, mAP 等计算
        metrics = {}
        
        if 'chamfer' in self.metrics:
            metrics['chamfer'] = self._compute_chamfer(results)
        
        if 'mAP' in self.metrics:
            metrics['mAP'] = self._compute_map(results)
        
        return metrics
    
    def _compute_chamfer(self, results):
        # 实现 chamfer distance 计算
        pass
    
    def _compute_map(self, results):
        # 实现 mAP 计算
        pass
```

### 步骤4: 创建训练脚本

创建 `tools/train.py`:

```python
import argparse
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='work dir')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 加载配置
    cfg = Config.fromfile(args.config)
    
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    
    # 构建 runner
    runner = Runner.from_cfg(cfg)
    
    # 开始训练
    runner.train()

if __name__ == '__main__':
    main()
```

### 步骤5: 测试运行

```bash
cd /path/to/mmdetection3d

# 测试配置加载
python tools/train.py projects/configs/maptr_test.py --work-dir work_dirs/test

# 如果遇到问题，先测试数据加载
python -c "
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmdet3d.registry import DATASETS

cfg = Config.fromfile('projects/configs/maptr_test.py')
dataset = DATASETS.build(cfg.train_dataloader.dataset)
print(f'Dataset size: {len(dataset)}')
data = dataset[0]
print(f'Data keys: {data.keys()}')
"
```

---

## 常见问题排查

### Q1: 导入错误
```python
ImportError: cannot import name 'DETECTORS' from 'mmdet.models'
```
**解决**: 确保使用 `mmdet3d.registry.MODELS` 而不是旧的 `DETECTORS`

### Q2: DataContainer 错误
```python
AttributeError: 'Tensor' object has no attribute 'data'
```
**解决**: 检查是否还有代码尝试访问 `.data` 属性，应直接使用 tensor

### Q3: 配置加载失败
```python
KeyError: 'DETECTORS is not in the xxx registry'
```
**解决**: 配置文件中的 `type='MapTR'` 前需要确保已经导入并注册该模块

### Q4: Runner 构建失败
**解决**: 检查配置文件是否有 `train_cfg`, `optim_wrapper` 等必需字段

---

## 获取帮助

1. **查看文档**:
   - `MIGRATION_STATUS.md` - 详细迁移状态
   - `REFACTOR_TODO.md` - 待办事项
   - `MIGRATION_SUMMARY.md` - 完整总结

2. **参考示例**:
   - MMDetection3D v1.4.0 的 `projects/example_project/`
   - MMDetection3D 的训练脚本

3. **调试技巧**:
   ```python
   # 启用详细日志
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # 测试单个组件
   from mmdet3d.registry import MODELS
   print(MODELS.module_dict.keys())  # 查看已注册的模块
   ```

---

## 成功指标

当你能够成功运行以下命令时，说明基础迁移已完成：

```bash
# ✅ 单GPU训练
python tools/train.py configs/maptr_test.py --work-dir work_dirs/test

# ✅ 验证
python tools/test.py configs/maptr_test.py checkpoints/latest.pth

# ✅ 多GPU训练
bash tools/dist_train.sh configs/maptr_test.py 8
```

**祝你好运！** 🚀
