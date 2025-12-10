from .nuscenes_dataset import CustomNuScenesDataset
from .nuscenes_map_dataset import CustomNuScenesLocalMapDataset
from .builder import custom_build_dataset
from .map_metric import MapMetric, MapMetricWithGT

# Optional: AV2 dataset (requires compatible av2 package)
# Apply NumPy compatibility patch for av2
try:
    import numpy as np
    # Add backward compatibility for deprecated NumPy aliases
    if not hasattr(np, 'bool'):
        np.bool = np.bool_
    if not hasattr(np, 'int'):
        np.int = np.int_
    if not hasattr(np, 'float'):
        np.float = np.float_
    if not hasattr(np, 'complex'):
        np.complex = np.complex_
    if not hasattr(np, 'object'):
        np.object = np.object_
    if not hasattr(np, 'str'):
        np.str = np.str_
    
    from .av2_map_dataset import CustomAV2LocalMapDataset
    _av2_available = True
except ImportError as e:
    import warnings
    warnings.warn(f"AV2 dataset not available: {e}. Install compatible av2 package if needed.")
    CustomAV2LocalMapDataset = None
    _av2_available = False
except AttributeError as e:
    import warnings
    warnings.warn(f"AV2 dataset not available due to NumPy compatibility issue: {e}")
    CustomAV2LocalMapDataset = None
    _av2_available = False

__all__ = [
    'CustomNuScenesDataset', 'CustomNuScenesLocalMapDataset', 
    'MapMetric', 'MapMetricWithGT'
]

if _av2_available:
    __all__.append('CustomAV2LocalMapDataset')
