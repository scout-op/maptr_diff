#!/usr/bin/env python
"""验证 MapTR OpenMMLab 2.0 迁移是否成功

这个脚本检查：
1. 依赖是否正确安装
2. 模块是否正确注册
3. 配置文件是否可以加载
4. 模型是否可以构建
"""

import sys
import os
from pathlib import Path


def check_dependencies():
    """检查依赖包"""
    print("="*60)
    print("1. 检查依赖包...")
    print("="*60)
    
    required = {
        'torch': 'PyTorch',
        'mmengine': 'MMEngine',
        'mmcv': 'MMCV',
        'mmdet': 'MMDetection',
        'mmdet3d': 'MMDetection3D',
    }
    
    all_ok = True
    for module, name in required.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✅ {name:20s} {version}")
        except ImportError:
            print(f"  ❌ {name:20s} NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_registries():
    """检查模块注册"""
    print("\n" + "="*60)
    print("2. 检查模块注册...")
    print("="*60)
    
    try:
        # 导入 MapTR 插件
        import projects.mmdet3d_plugin
        print("  ✅ MapTR 插件导入成功")
    except Exception as e:
        print(f"  ❌ MapTR 插件导入失败: {e}")
        return False
    
    # 检查各个注册器
    checks = []
    
    # 模型
    try:
        from mmdet3d.registry import MODELS
        models = ['MapTR', 'MapTRHead', 'BEVFormer']
        for model in models:
            if model in MODELS.module_dict:
                print(f"  ✅ {model:30s} 已注册到 MODELS")
                checks.append(True)
            else:
                print(f"  ⚠️  {model:30s} 未找到")
                checks.append(False)
    except Exception as e:
        print(f"  ❌ 检查 MODELS 失败: {e}")
        checks.append(False)
    
    # 数据集
    try:
        from mmdet3d.registry import DATASETS
        datasets = ['CustomNuScenesDataset', 'CustomNuScenesLocalMapDataset', 
                   'CustomAV2LocalMapDataset']
        for dataset in datasets:
            if dataset in DATASETS.module_dict:
                print(f"  ✅ {dataset:30s} 已注册到 DATASETS")
                checks.append(True)
            else:
                print(f"  ⚠️  {dataset:30s} 未找到")
                checks.append(False)
    except Exception as e:
        print(f"  ❌ 检查 DATASETS 失败: {e}")
        checks.append(False)
    
    # 评估指标
    try:
        from mmdet3d.registry import METRICS
        metrics = ['MapMetric', 'MapMetricWithGT']
        for metric in metrics:
            if metric in METRICS.module_dict:
                print(f"  ✅ {metric:30s} 已注册到 METRICS")
                checks.append(True)
            else:
                print(f"  ⚠️  {metric:30s} 未找到")
                checks.append(False)
    except Exception as e:
        print(f"  ❌ 检查 METRICS 失败: {e}")
        checks.append(False)
    
    # Transforms
    try:
        from mmdet3d.registry import TRANSFORMS
        transforms = ['LoadMultiViewImageFromFiles', 'CustomFormatBundle3D']
        for transform in transforms:
            if transform in TRANSFORMS.module_dict:
                print(f"  ✅ {transform:30s} 已注册到 TRANSFORMS")
                checks.append(True)
            else:
                print(f"  ⚠️  {transform:30s} 未找到")
    except Exception as e:
        print(f"  ⚠️  检查 TRANSFORMS 时出错: {e}")
    
    return all(checks) if checks else False


def check_config():
    """检查配置文件"""
    print("\n" + "="*60)
    print("3. 检查配置文件...")
    print("="*60)
    
    config_path = Path(__file__).parent.parent / 'configs' / 'maptr_av2_example.py'
    
    if not config_path.exists():
        print(f"  ❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        from mmengine.config import Config
        cfg = Config.fromfile(str(config_path))
        print(f"  ✅ 配置文件加载成功: {config_path.name}")
        print(f"     - 模型类型: {cfg.model.type}")
        print(f"     - 数据集类型: {cfg.train_dataloader.dataset.type}")
        
        # 检查必需字段
        required_fields = [
            'default_scope',
            'model',
            'train_dataloader',
            'val_dataloader',
            'test_dataloader',
            'val_evaluator',
            'optim_wrapper',
            'param_scheduler',
            'train_cfg',
            'val_cfg',
            'test_cfg',
        ]
        
        missing = []
        for field in required_fields:
            if not hasattr(cfg, field):
                missing.append(field)
        
        if missing:
            print(f"  ⚠️  缺少字段: {', '.join(missing)}")
            return False
        else:
            print(f"  ✅ 所有必需字段都存在")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_build():
    """检查模型构建"""
    print("\n" + "="*60)
    print("4. 检查模型构建...")
    print("="*60)
    
    try:
        from mmengine.config import Config
        from mmdet3d.registry import MODELS
        
        config_path = Path(__file__).parent.parent / 'configs' / 'maptr_av2_example.py'
        cfg = Config.fromfile(str(config_path))
        
        # 尝试构建模型
        print("  🔨 正在构建模型...")
        model = MODELS.build(cfg.model)
        print(f"  ✅ 模型构建成功: {type(model).__name__}")
        
        # 检查模型组件
        if hasattr(model, 'pts_bbox_head'):
            print(f"  ✅ 检测头: {type(model.pts_bbox_head).__name__}")
        if hasattr(model, 'img_backbone'):
            print(f"  ✅ 图像骨干网络: {type(model.img_backbone).__name__}")
        if hasattr(model, 'img_neck'):
            print(f"  ✅ 图像颈部网络: {type(model.img_neck).__name__}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_files():
    """检查关键文件"""
    print("\n" + "="*60)
    print("5. 检查关键文件...")
    print("="*60)
    
    base_path = Path(__file__).parent.parent
    
    files_to_check = [
        ('configs/_base_/default_runtime.py', '基础配置'),
        ('configs/maptr_av2_example.py', 'AV2示例配置'),
        ('datasets/map_metric.py', 'MapMetric评估器'),
        ('tools/train.py', '训练脚本'),
        ('tools/test.py', '测试脚本'),
        ('README.md', '使用文档'),
        ('UPGRADE_COMPLETE.md', '升级报告'),
    ]
    
    all_exist = True
    for filepath, desc in files_to_check:
        full_path = base_path / filepath
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  ✅ {desc:20s} ({size:>6d} bytes) - {filepath}")
        else:
            print(f"  ❌ {desc:20s} 不存在 - {filepath}")
            all_exist = False
    
    return all_exist


def main():
    """主函数"""
    print("\n" + "🔍 MapTR OpenMMLab 2.0 迁移验证".center(60, "="))
    print()
    
    results = {}
    
    # 运行所有检查
    results['dependencies'] = check_dependencies()
    results['registries'] = check_registries()
    results['config'] = check_config()
    results['model'] = check_model_build()
    results['files'] = check_files()
    
    # 总结
    print("\n" + "="*60)
    print("📊 验证总结")
    print("="*60)
    
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有检查通过！MapTR 已成功迁移到 OpenMMLab 2.0")
        print("="*60)
        print("\n下一步:")
        print("  1. 准备数据集")
        print("  2. 运行训练:")
        print("     python projects/mmdet3d_plugin/tools/train.py \\")
        print("         projects/mmdet3d_plugin/configs/maptr_av2_example.py")
        return 0
    else:
        print("⚠️  部分检查未通过，请查看上面的详细信息")
        print("="*60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
