# MapTR OpenMMLab 2.0 Migration Status

## ✅ Completed Migrations

### Core Models
| File | Changes |
|------|---------|
| `maptr/detectors/maptr.py` | MODELS registry, mmdet3d.structures |
| `maptr/dense_heads/maptr_head.py` | MODELS registry, mmdet.utils, mmdet.structures.bbox |
| `maptr/losses/map_loss.py` | TASK_UTILS registry, removed mmcv.jit |
| `maptr/modules/encoder.py` | mmengine.model.BaseModule |
| `maptr/modules/transformer.py` | MODELS registry, mmengine.model |
| `maptr/modules/diffusion_head.py` | mmengine.model.BaseModule |
| `maptr/modules/geometry_kernel_attention.py` | mmengine.model.BaseModule |
| `maptr/assigners/maptr_assigner.py` | TASK_UTILS registry, mmdet.structures.bbox |

### BEVFormer Modules
| File | Changes |
|------|---------|
| `bevformer/detectors/bevformer.py` | MODELS registry |
| `bevformer/detectors/bevformer_fp16.py` | MODELS registry, removed auto_fp16 |
| `bevformer/dense_heads/bevformer_head.py` | MODELS registry, mmdet.utils imports |
| `bevformer/modules/transformer.py` | MODELS registry, mmengine.model |
| `bevformer/modules/encoder.py` | Removed force_fp32/auto_fp16 |
| `bevformer/modules/decoder.py` | mmengine.model, mmengine.config |
| `bevformer/modules/spatial_cross_attention.py` | mmengine.model |
| `bevformer/modules/temporal_self_attention.py` | mmengine.model, mmengine.config |
| `bevformer/modules/custom_base_transformer_layer.py` | mmengine.config, mmengine.model |
| `bevformer/hooks/custom_hooks.py` | mmengine.hooks, mmengine.registry |

### Datasets & Pipelines
| File | Changes |
|------|---------|
| `datasets/av2_map_dataset.py` | DATASETS registry, removed DataContainer |
| `datasets/pipelines/loading.py` | TRANSFORMS registry |
| `datasets/pipelines/formating.py` | Pack3DDetInputs, TRANSFORMS registry |
| `datasets/samplers/group_sampler.py` | mmengine.dist |

### Core Components
| File | Changes |
|------|---------|
| `core/bbox/assigners/hungarian_assigner_3d.py` | TASK_UTILS registry |
| `core/bbox/match_costs/match_cost.py` | TASK_UTILS registry |
| `core/bbox/coders/nms_free_coder.py` | TASK_UTILS registry |

### Models Utils & Backbones
| File | Changes |
|------|---------|
| `models/backbones/swin.py` | mmengine.model, MODELS registry |
| `models/backbones/efficientnet.py` | mmengine.model, MODELS registry |
| `models/backbones/vovnet.py` | mmengine.model, MODELS registry |
| `models/utils/embed.py` | mmengine.model.BaseModule |
| `models/utils/grid_mask.py` | Removed force_fp32/auto_fp16 |
| `models/utils/inverted_residual.py` | mmengine.model.BaseModule |
| `models/utils/se_layer.py` | mmengine.model.BaseModule |
| `models/hooks/hooks.py` | mmengine.hooks, mmengine.registry |
| `models/opt/adamw.py` | mmengine.registry.OPTIMIZERS |

### APIs & Utilities
| File | Changes |
|------|---------|
| `datasets/map_utils/tpfp.py` | mmdet.structures.bbox |
| `bevformer/apis/test.py` | mmengine.dist, removed mmdet.core imports |
| `datasets/nuscenes_mono_dataset.py` | DATASETS registry, mmdet3d.structures, mmdet3d.visualization |

## ⚠️ Requires Deep Refactoring

| File | Issue | Status |
|------|-------|--------|
| `bevformer/runner/epoch_based_runner.py` | mmengine uses Loop architecture instead of Runner | **Needs rewrite** |
| `core/evaluation/eval_hooks.py` | mmengine uses ValLoop instead of EvalHook | **Needs rewrite** |
| `bevformer/apis/mmdet_train.py` | Training API changed to mmengine.runner.Runner | **Imports updated, logic needs rewrite** |

## 📝 Notes

1. **SyntaxWarnings**: Some files have `is` vs `==` warnings for string comparisons. These are cosmetic issues.

2. **force_fp32/auto_fp16**: These decorators are removed in mmengine. The AMP (Automatic Mixed Precision) is handled differently.

3. **DataContainer**: Removed in favor of plain tensors/dicts. Pipeline outputs should be compatible with mmengine DataSample.

4. **Registry Changes**:
   - `DETECTORS` → `MODELS`
   - `HEADS` → `MODELS`
   - `BACKBONES` → `MODELS`
   - `BBOX_ASSIGNERS` → `TASK_UTILS`
   - `BBOX_CODERS` → `TASK_UTILS`
   - `MATCH_COST` → `TASK_UTILS`
   - `PIPELINES` → `TRANSFORMS`
   - `HOOKS` → `mmengine.registry.HOOKS`

5. **Config Migration**: Config files need to be updated to mmengine format with `default_scope`, `env_cfg`, `train_cfg`, etc.

## Next Steps

1. Update configuration files to mmengine format
2. Refactor runner/loop architecture for video training
3. Test full training pipeline
