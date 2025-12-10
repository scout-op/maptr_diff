#!/usr/bin/env python
"""快速测试导入是否正常"""

print("=" * 60)
print("测试 MapTR 插件导入")
print("=" * 60)

try:
    print("\n1. 测试 transformer_utils 导入...")
    from projects.mmdet3d_plugin.models.utils.transformer_utils import inverse_sigmoid
    print("   ✅ inverse_sigmoid 导入成功")
    
    print("\n2. 测试 MapTR 插件导入...")
    import projects.mmdet3d_plugin
    print("   ✅ 插件导入成功")
    
    print("\n3. 测试模块注册...")
    from mmdet3d.registry import MODELS, DATASETS, METRICS
    
    if 'MapTR' in MODELS.module_dict:
        print("   ✅ MapTR 已注册")
    else:
        print("   ❌ MapTR 未注册")
        
    if 'CustomAV2LocalMapDataset' in DATASETS.module_dict:
        print("   ✅ CustomAV2LocalMapDataset 已注册")
    else:
        print("   ❌ CustomAV2LocalMapDataset 未注册")
        
    if 'MapMetric' in METRICS.module_dict:
        print("   ✅ MapMetric 已注册")
    else:
        print("   ❌ MapMetric 未注册")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    print("=" * 60)
