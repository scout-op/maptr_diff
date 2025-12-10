# ⚠️ DEPRECATED - 需要完全重写

## 文件状态
- `eval_hooks.py` - **已废弃，需要完全重写**

## 原因
此文件使用 mmcv 1.x 的 `DistEvalHook`，与 mmengine 架构不兼容。

## mmengine 迁移方案

### mmengine 评估架构
在 mmengine 中，评估通过以下组件实现：
1. `ValLoop` - 验证循环
2. `Evaluator` - 评估器
3. `Metric` - 评估指标

### 迁移步骤

#### 1. 删除 CustomDistEvalHook
不再需要自定义 EvalHook。

#### 2. 实现自定义 Metric
```python
from mmengine.evaluator import BaseMetric

@METRICS.register_module()
class MapMetric(BaseMetric):
    def process(self, data_batch, predictions):
        # 处理每个batch的预测结果
        self.results.append(predictions)
    
    def compute_metrics(self, results):
        # 计算最终指标
        return {
            'mAP': map_score,
            'AP_50': ap_50,
            # ...
        }
```

#### 3. 在配置文件中设置评估
```python
val_cfg = dict(type='ValLoop')

val_evaluator = dict(
    type='MapMetric',
    metric=['mAP', 'AP'],
    # 其他参数
)

# 或使用多个评估器
val_evaluator = [
    dict(type='MapMetric', ...),
    dict(type='ChamferMetric', ...),
]
```

#### 4. 动态评估间隔
如果需要动态改变评估间隔（类似原 CustomDistEvalHook 的功能）：

**方案A: 使用自定义 Hook**
```python
from mmengine.hooks import Hook

@HOOKS.register_module()
class DynamicEvalIntervalHook(Hook):
    def __init__(self, dynamic_intervals):
        self.dynamic_intervals = dynamic_intervals
    
    def before_train_epoch(self, runner):
        # 根据epoch动态调整 val_interval
        epoch = runner.epoch
        for milestone, interval in self.dynamic_intervals:
            if epoch >= milestone:
                runner.val_loop.interval = interval
```

**方案B: 使用 ParamScheduler**
```python
param_scheduler = dict(
    type='StepParamScheduler',
    param_name='val_interval',
    # ...
)
```

### 评估数据集准备
确保数据集类实现了必要的评估接口：
```python
class CustomAV2LocalMapDataset(Dataset):
    def evaluate(self, results, **kwargs):
        # 实现评估逻辑
        # 或者移至 Metric 类
        pass
```

### 下一步
1. 删除 `eval_hooks.py`
2. 实现自定义 `MapMetric` 类
3. 更新配置文件使用新的评估配置
4. 测试评估流程

### 参考
- [MMEngine Evaluator 文档](https://mmengine.readthedocs.io/en/latest/design/evaluation.html)
- [MMEngine Metric 文档](https://mmengine.readthedocs.io/en/latest/tutorials/evaluation.html)
- MMDetection3D 中的评估实现示例
