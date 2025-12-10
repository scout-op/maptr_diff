# MapTR OpenMMLab 2.0 - 文件索引

**项目地址**: `mmdetection3d/projects/mmdet3d_plugin/`  
**版本**: OpenMMLab 2.0 (MMEngine + MMDetection3D v1.4.0)  
**状态**: ✅ 升级完成 (96%)

---

## 📁 目录结构

```
projects/mmdet3d_plugin/
├── README.md                          # 主文档 - 从这里开始
├── FINAL_SUMMARY.md                   # 最终总结
├── UPGRADE_COMPLETE.md                # 升级完成报告
├── MIGRATION_SUMMARY.md               # 详细迁移总结
├── MIGRATION_STATUS.md                # 迁移状态追踪
├── QUICKSTART.md                      # 快速开始指南
├── REFACTOR_TODO.md                   # 可选重构指南
├── CHEATSHEET.md                      # 快速参考手册
├── INDEX.md                           # 本文件 - 文件索引
│
├── configs/                           # 配置文件
│   ├── _base_/
│   │   └── default_runtime.py         # 基础运行时配置
│   ├── maptr_av2_example.py           # Argoverse2 示例
│   └── maptr_nuscenes_example.py      # NuScenes 示例
│
├── tools/                             # 训练/测试工具
│   ├── train.py                       # 训练脚本
│   ├── test.py                        # 测试脚本
│   ├── dist_train.sh                  # 分布式训练
│   ├── dist_test.sh                   # 分布式测试
│   └── verify_installation.py         # 安装验证
│
├── datasets/                          # 数据集
│   ├── av2_map_dataset.py             # Argoverse2 数据集
│   ├── nuscenes_map_dataset.py        # NuScenes 数据集
│   ├── map_metric.py                  # ✨ 新增：评估指标
│   ├── pipelines/                     # 数据处理pipeline
│   └── ...
│
├── maptr/                             # MapTR 核心
│   ├── detectors/
│   │   └── maptr.py                   # MapTR 检测器
│   ├── dense_heads/
│   │   └── maptr_head.py              # MapTR 检测头
│   ├── modules/
│   │   ├── diffusion_head.py          # 扩散头
│   │   └── geometry_kernel_attention.py
│   └── ...
│
├── models/                            # 模型组件
│   ├── backbones/                     # 骨干网络
│   │   ├── swin.py
│   │   ├── efficientnet.py
│   │   └── vovnet.py
│   ├── utils/                         # 工具模块
│   └── ...
│
├── bevformer/                         # BEVFormer 相关
│   ├── detectors/
│   ├── modules/
│   ├── runner/                        # ⚠️ 已废弃
│   │   └── DEPRECATED_README.md
│   └── apis/
│
└── core/                              # 核心工具
    ├── evaluation/                    # ⚠️ 已废弃
    │   └── DEPRECATED_README.md
    └── bbox/
```

---

## 📚 文档导航

### 🚀 快速开始

1. **首次使用** → `README.md`
2. **快速参考** → `CHEATSHEET.md`
3. **验证安装** → 运行 `tools/verify_installation.py`

### 📖 详细文档

| 文档 | 用途 | 适合人群 |
|------|------|----------|
| `README.md` | 完整使用指南、安装、训练、测试 | 所有用户 |
| `CHEATSHEET.md` | 常用命令、配置模板、调试技巧 | 日常使用 |
| `QUICKSTART.md` | 快速开始、示例代码 | 新手入门 |
| `FINAL_SUMMARY.md` | 升级完成总结、验证清单 | 验证升级 |
| `UPGRADE_COMPLETE.md` | 完整升级报告、新增功能 | 了解改进 |
| `MIGRATION_SUMMARY.md` | 详细迁移说明、测试步骤 | 技术细节 |
| `MIGRATION_STATUS.md` | 模块级迁移状态 | 开发参考 |
| `REFACTOR_TODO.md` | 可选重构任务 | 深度定制 |

### 📋 按使用场景

**我想训练模型**:
1. 阅读 `README.md` → 准备数据 → 选择配置
2. 运行 `python tools/train.py configs/maptr_av2_example.py`
3. 参考 `CHEATSHEET.md` 调整参数

**我想测试模型**:
1. 阅读 `README.md` → 测试部分
2. 运行 `python tools/test.py config.py checkpoint.pth`

**我想了解迁移**:
1. `UPGRADE_COMPLETE.md` → 总体了解
2. `MIGRATION_SUMMARY.md` → 技术细节
3. `MIGRATION_STATUS.md` → 具体模块

**我遇到问题**:
1. `CHEATSHEET.md` → 常见错误
2. `README.md` → 故障排除
3. `verify_installation.py` → 验证环境

---

## 🆕 新增文件

### 评估系统
- ✨ `datasets/map_metric.py` (13.4KB)
  - MapMetric: 基础评估器
  - MapMetricWithGT: 完整评估
  - 支持 Chamfer Distance 和 IoU

### 训练/测试工具
- ✨ `tools/train.py` (3.7KB) - MMEngine 训练脚本
- ✨ `tools/test.py` (3.0KB) - MMEngine 测试脚本
- ✨ `tools/dist_train.sh` - 分布式训练启动脚本
- ✨ `tools/dist_test.sh` - 分布式测试启动脚本
- ✨ `tools/verify_installation.py` (8.8KB) - 安装验证

### 配置文件
- ✨ `configs/_base_/default_runtime.py` (1.2KB) - 基础配置
- ✨ `configs/maptr_av2_example.py` (6.3KB) - AV2 示例
- ✨ `configs/maptr_nuscenes_example.py` (6.5KB) - NuScenes 示例

### 文档
- ✨ `README.md` (7.0KB) - 主文档
- ✨ `CHEATSHEET.md` (8.2KB) - 快速参考
- ✨ `FINAL_SUMMARY.md` (5.2KB) - 最终总结
- ✨ `UPGRADE_COMPLETE.md` (8.2KB) - 升级报告
- ✨ `MIGRATION_SUMMARY.md` (10.5KB) - 迁移总结
- ✨ `QUICKSTART.md` (7.8KB) - 快速开始
- ✨ `INDEX.md` - 本文件

---

## 🔧 已更新文件 (52个)

### 核心模型 (21)
- `maptr/detectors/maptr.py`
- `maptr/dense_heads/maptr_head.py`
- `maptr/modules/diffusion_head.py`
- `maptr/modules/geometry_kernel_attention.py`
- `maptr/assigners/maptr_assigner.py`
- `bevformer/detectors/bevformer.py`
- `bevformer/detectors/bevformer_fp16.py`
- ... (查看 MIGRATION_STATUS.md 获取完整列表)

### 数据集/Pipeline (4)
- `datasets/av2_map_dataset.py` - ✅ 语法错误已修复
- `datasets/nuscenes_map_dataset.py` - ✅ 语法错误已修复
- `datasets/pipelines/loading.py`
- `datasets/pipelines/formating.py`

### Backbone (3)
- `models/backbones/swin.py`
- `models/backbones/efficientnet.py`
- `models/backbones/vovnet.py`

### 工具模块 (9)
- `models/utils/embed.py`
- `models/utils/grid_mask.py`
- `models/utils/inverted_residual.py`
- `models/utils/se_layer.py`
- `models/opt/adamw.py`
- ... (更多详见 MIGRATION_STATUS.md)

---

## ⚠️ 废弃文件

以下文件已不再使用（已被新系统替代）:

- `bevformer/runner/epoch_based_runner.py` 
  - 替代方案: MMEngine Runner + `MapTR.train_step()`
  - 说明: `bevformer/runner/DEPRECATED_README.md`

- `core/evaluation/eval_hooks.py`
  - 替代方案: `datasets/map_metric.py`
  - 说明: `core/evaluation/DEPRECATED_README.md`

- `bevformer/apis/mmdet_train.py`
  - 替代方案: `tools/train.py`

这些文件保留作为参考，但不影响当前功能。

---

## 📊 迁移完成度

| 类别 | 已完成 | 总计 | 完成度 |
|------|--------|------|--------|
| 核心模型 | 21 | 21 | 100% |
| 数据集/Pipeline | 4 | 4 | 100% |
| Backbone | 3 | 3 | 100% |
| 工具模块 | 9 | 9 | 100% |
| 评估系统 | 2 | 2 | 100% |
| 训练/测试脚本 | 5 | 5 | 100% |
| 配置文件 | 3 | 3 | 100% |
| 文档系统 | 8 | 8 | 100% |
| APIs (可选) | 1 | 3 | 33% |
| **总计** | **56** | **58** | **96%** |

---

## 🎯 使用流程

### 第一次使用

```bash
# 1. 验证环境
cd /path/to/mmdetection3d
python projects/mmdet3d_plugin/tools/verify_installation.py

# 2. 准备数据
# 按照 README.md 组织数据结构

# 3. 选择配置
# - Argoverse2: configs/maptr_av2_example.py
# - NuScenes: configs/maptr_nuscenes_example.py

# 4. 开始训练
python projects/mmdet3d_plugin/tools/train.py \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py

# 5. 测试模型
python projects/mmdet3d_plugin/tools/test.py \
    projects/mmdet3d_plugin/configs/maptr_av2_example.py \
    work_dirs/maptr_av2_example/latest.pth
```

### 日常使用

参考 `CHEATSHEET.md` 获取:
- 常用命令
- 配置模板
- 调试技巧
- 错误解决

---

## 🔗 相关链接

### 官方文档
- [MMEngine](https://mmengine.readthedocs.io/)
- [MMDetection3D](https://mmdetection3d.readthedocs.io/)
- [MMCV](https://mmcv.readthedocs.io/)

### GitHub
- [OpenMMLab](https://github.com/open-mmlab)
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)

---

## 💡 快速链接

**我要...**

- 🚀 **开始训练** → `README.md` + `configs/maptr_av2_example.py`
- 📖 **查看命令** → `CHEATSHEET.md`
- 🔍 **验证安装** → `tools/verify_installation.py`
- 🐛 **解决问题** → `CHEATSHEET.md` → 常见错误
- 📊 **了解迁移** → `UPGRADE_COMPLETE.md`
- 🎯 **快速上手** → `QUICKSTART.md`
- 🔧 **调整配置** → `configs/` + `CHEATSHEET.md`

---

**最后更新**: 2024-12-09  
**维护者**: MapTR Team
