# ⚠️ DEPRECATED - 需要完全重写

## 文件状态
- `epoch_based_runner.py` - **已废弃，需要完全重写**

## 原因
此文件使用 mmcv 1.x 的 `EpochBasedRunner` 架构，与 mmengine 不兼容。

## mmengine 迁移方案

### mmengine 不再使用独立的 Runner 类
在 mmengine 中：
- 训练循环逻辑移至 `mmengine.runner.Runner` + `Loop`
- 自定义逻辑通过 `Hook` 实现
- 不需要继承 `EpochBasedRunner`

### 视频序列训练的实现方式

**旧方式** (mmcv 1.x):
```python
class EpochBasedRunner_video(EpochBasedRunner):
    def run_iter(self, data_batch, train_mode):
        # 自定义视频序列处理逻辑
        prev_bev = None
        for sample in sequence[:-1]:
            prev_bev = eval_model(sample, prev_bev)
        loss = model(sequence[-1], prev_bev)
```

**新方式** (mmengine):
有几种选择：

#### 方案1: 自定义 Hook
```python
from mmengine.hooks import Hook

@HOOKS.register_module()
class VideoSequenceHook(Hook):
    def before_train_iter(self, runner, batch_idx, data_batch):
        # 在每次迭代前处理视频序列
        # 修改 data_batch 以包含序列信息
        pass
```

#### 方案2: 自定义模型的 train_step
```python
class MapTR(BaseDetector):
    def train_step(self, data, optim_wrapper):
        # 在模型内部处理视频序列逻辑
        # 生成 prev_bev
        # 计算损失
        parsed_losses = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_losses)
        return parsed_losses
```

#### 方案3: 自定义数据加载
```python
# 在 dataset/dataloader 中处理序列
# 让每个 batch 包含完整的视频序列
```

### 推荐方案
**方案2** - 在模型的 `train_step` 中实现视频序列逻辑，这样：
- 逻辑集中在模型中，更清晰
- 不需要修改 Runner
- 更容易测试和维护

### 下一步
1. 查看当前 `MapTR` 模型实现
2. 在 `train_step` 方法中添加视频序列处理逻辑
3. 删除此目录下的 `epoch_based_runner.py`
4. 使用标准 mmengine `Runner` 进行训练

### 参考
- [MMEngine Runner 文档](https://mmengine.readthedocs.io/en/latest/tutorials/runner.html)
- [MMEngine Hook 文档](https://mmengine.readthedocs.io/en/latest/tutorials/hook.html)
- MMDetection3D v1.4.0 的训练脚本
