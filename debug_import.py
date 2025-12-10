#!/usr/bin/env python
"""详细的导入调试脚本"""
import sys
import traceback

print("=" * 70)
print("MapTR 插件导入调试")
print("=" * 70)

# 逐步导入，找出问题
steps = [
    ("transformer_utils", "from projects.mmdet3d_plugin.models.utils.transformer_utils import inverse_sigmoid"),
    ("core.bbox", "from projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D"),
    ("core.bbox.coders", "from projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder import NMSFreeCoder"),
    ("core.bbox.match_costs", "from projects.mmdet3d_plugin.core.bbox.match_costs import BBox3DL1Cost"),
    ("datasets.pipelines", "from projects.mmdet3d_plugin.datasets.pipelines import PhotoMetricDistortionMultiViewImage"),
    ("models.backbones.vovnet", "from projects.mmdet3d_plugin.models.backbones.vovnet import VoVNet"),
    ("models.utils", "from projects.mmdet3d_plugin.models.utils import GridMask"),
    ("models.opt.adamw", "from projects.mmdet3d_plugin.models.opt.adamw import AdamW2"),
    ("bevformer", "from projects.mmdet3d_plugin import bevformer"),
    ("maptr", "from projects.mmdet3d_plugin import maptr"),
    ("models.backbones.efficientnet", "from projects.mmdet3d_plugin.models.backbones.efficientnet import EfficientNet"),
    ("完整插件", "import projects.mmdet3d_plugin"),
]

failed = []
for i, (name, import_stmt) in enumerate(steps, 1):
    try:
        print(f"\n[{i}/{len(steps)}] 测试: {name}")
        print(f"     语句: {import_stmt}")
        exec(import_stmt)
        print(f"     ✅ 成功")
    except Exception as e:
        print(f"     ❌ 失败: {e}")
        failed.append((name, import_stmt, str(e)))
        print("\n详细错误:")
        traceback.print_exc()
        print("-" * 70)

print("\n" + "=" * 70)
if not failed:
    print("✅ 所有导入测试通过！")
    print("\n测试模块注册...")
    try:
        from mmdet3d.registry import MODELS, DATASETS, METRICS
        
        modules_to_check = [
            ('MODELS', 'MapTR', MODELS),
            ('MODELS', 'MapTRHead', MODELS),
            ('DATASETS', 'CustomAV2LocalMapDataset', DATASETS),
            ('DATASETS', 'CustomNuScenesLocalMapDataset', DATASETS),
            ('METRICS', 'MapMetric', METRICS),
            ('METRICS', 'MapMetricWithGT', METRICS),
        ]
        
        for registry_name, module_name, registry in modules_to_check:
            if module_name in registry.module_dict:
                print(f"   ✅ {registry_name}.{module_name}")
            else:
                print(f"   ❌ {registry_name}.{module_name} 未注册")
                
    except Exception as e:
        print(f"注册检查失败: {e}")
else:
    print(f"❌ {len(failed)} 个导入失败:")
    for name, stmt, error in failed:
        print(f"   - {name}: {error}")

print("=" * 70)
