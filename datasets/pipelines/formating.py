# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np

from mmdet3d.registry import TRANSFORMS
from mmdet3d.datasets.transforms import Pack3DDetInputs


@TRANSFORMS.register_module()
@TRANSFORMS.register_module(name='DefaultFormatBundle3D')
class CustomDefaultFormatBundle3D(Pack3DDetInputs):
    """Format bundle for map masks without DataContainer.

    This keeps backward compatibility with the original pipeline but replaces
    mmcv.DataContainer by plain tensors and Pack3DDetInputs.
    """
    
    def __init__(self, class_names=None, with_label=True, keys=None, **kwargs):
        """Initialize the transform.
        
        Args:
            class_names: Class names for backward compatibility (not used).
            with_label: Whether to include labels.
            keys: Keys to pack (for Pack3DDetInputs).
            **kwargs: Other arguments passed to Pack3DDetInputs.
        """
        # class_names is kept for backward compatibility but not used
        # If keys not provided, use default keys based on with_label
        if keys is None:
            if with_label:
                keys = ['img', 'gt_bboxes_3d', 'gt_labels_3d']
            else:
                keys = ['img']
        
        # Pack3DDetInputs requires meta_keys, provide defaults if not in kwargs
        if 'meta_keys' not in kwargs:
            kwargs['meta_keys'] = ('img_path', 'ori_shape', 'img_shape', 'pad_shape',
                                   'scale_factor', 'flip', 'flip_direction')
        
        # Pass both keys and meta_keys to parent
        super().__init__(keys=keys, meta_keys=kwargs['meta_keys'])
        self.with_label = with_label

    def transform(self, results):
        """Convert and pack inputs."""
        packed = super().transform(results)
        if 'gt_map_masks' in results:
            packed['data_samples'].gt_map_masks = torch.as_tensor(
                results['gt_map_masks'])
        return packed