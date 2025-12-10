# MapTR OpenMMLab 2.0 深度重构待办清单

## 🔴 高优先级 - 核心训练/评估架构

### 1. Runner架构迁移
**文件**: `bevformer/runner/epoch_based_runner.py`

**问题**: 
- mmcv 1.x使用 `EpochBasedRunner`
- mmengine使用全新的 `Runner` + `Loop` 架构

**需要做的**:
1. 移除旧的 `EpochBasedRunner_video` 类
2. 实现基于 mmengine.runner.Runner 的新训练循环
3. 如果需要视频特定逻辑，通过自定义 Hook 或 Loop 实现
4. 参考 mmdetection3d v1.4.0 中的训练脚本

**参考**:
- `mmengine.runner.Runner`
- `mmengine.runner.EpochBasedTrainLoop`

---

### 2. 评估Hook迁移
**文件**: `core/evaluation/eval_hooks.py`

**问题**:
- mmcv 1.x使用 `DistEvalHook`
- mmengine使用 `ValLoop` 进行验证

**需要做的**:
1. 移除 `CustomDistEvalHook` 类
2. 实现自定义 `ValLoop` 或使用标准 `ValLoop` + 自定义评估指标
3. 将评估逻辑移至 dataset 的 `evaluate()` 方法或独立的 evaluator

**参考**:
- `mmengine.runner.ValLoop`
- `mmengine.evaluator.Evaluator`

---

### 3. 训练API重构
**文件**: `bevformer/apis/mmdet_train.py`

**问题**:
- 导入已更新为 mmengine，但主逻辑仍使用旧API
- `custom_train_detector()` 函数需要完全重写

**需要做的**:
1. 使用 mmengine.runner.Runner 替代 build_runner
2. 配置optim_wrapper替代optimizer + fp16_hook
3. 更新hook注册机制
4. 参考 mmdetection3d v1.4.0 的 `train.py`

---

## 🟡 中优先级 - 配置文件

### 4. 配置文件格式迁移
**目录**: `projects/configs/`

**需要做的**:
1. 添加 `default_scope = 'mmdet3d'`
2. 更新 `train_cfg` / `val_cfg` / `test_cfg` 为新格式
3. 更新 `env_cfg` 配置
4. 更新 `optim_wrapper` 配置（替代 optimizer + fp16）
5. 更新 hook 配置
6. 更新 registry 名称（DETECTORS→MODELS等）

**示例结构**:
```python
default_scope = 'mmdet3d'

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=24,
    val_interval=1
)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4),
)
```

---

## 🟢 低优先级 - 优化

### 5. 移除DataContainer残留检查
**文件**: `datasets/av2_map_dataset.py`, `datasets/nuscenes_map_dataset.py`

**需要做的**:
1. 确认所有pipeline已移除 `DC()` 包装
2. 移除兼容性检查代码（hasattr data container checks）
3. 统一使用纯tensor输出

---

### 6. 语法警告修复
**文件**: 多个数据集文件

**需要做的**:
将所有 `if x.geom_type is 'Polygon':` 改为 `if x.geom_type == 'Polygon':`

---

## 📋 测试清单

完成重构后需要测试：

- [ ] 配置文件加载
- [ ] 数据pipeline运行
- [ ] 模型初始化
- [ ] 单GPU训练
- [ ] 多GPU训练
- [ ] 验证/评估
- [ ] 推理/测试

---

## 📚 参考资源

- MMEngine官方文档: https://mmengine.readthedocs.io/
- MMDetection3D v1.4.0代码: 
  - `mmdetection3d/projects/example_project/` (插件示例)
  - `tools/train.py` (新训练脚本)
- Migration Guide: https://mmengine.readthedocs.io/en/latest/migration/runner.html
