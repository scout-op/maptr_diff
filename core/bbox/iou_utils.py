"""
IoU utilities for 3D bounding boxes
Compatibility layer for MMDetection3D 1.4.0
"""
import torch
import numpy as np


def axis_aligned_iou_3d(boxes1, boxes2):
    """
    Calculate axis-aligned IoU between two sets of 3D bounding boxes.
    
    Args:
        boxes1 (torch.Tensor or np.ndarray): (N, 7) [x, y, z, w, l, h, rot]
        boxes2 (torch.Tensor or np.ndarray): (M, 7) [x, y, z, w, l, h, rot]
        
    Returns:
        torch.Tensor or np.ndarray: (N, M) IoU matrix
        
    Note:
        This function assumes axis-aligned boxes (rotation is ignored)
        For full 3D IoU with rotation, use mmdet3d.ops functions if available
    """
    is_numpy = isinstance(boxes1, np.ndarray)
    
    if is_numpy:
        boxes1 = torch.from_numpy(boxes1).float()
        boxes2 = torch.from_numpy(boxes2).float()
    
    # Extract coordinates
    # boxes format: [x, y, z, w, l, h, ...]
    x1, y1, z1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2]
    w1, l1, h1 = boxes1[:, 3], boxes1[:, 4], boxes1[:, 5]
    
    x2, y2, z2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2]
    w2, l2, h2 = boxes2[:, 3], boxes2[:, 4], boxes2[:, 5]
    
    # Calculate box corners (min and max points)
    x1_min, x1_max = x1 - w1 / 2, x1 + w1 / 2
    y1_min, y1_max = y1 - l1 / 2, y1 + l1 / 2
    z1_min, z1_max = z1 - h1 / 2, z1 + h1 / 2
    
    x2_min, x2_max = x2 - w2 / 2, x2 + w2 / 2
    y2_min, y2_max = y2 - l2 / 2, y2 + l2 / 2
    z2_min, z2_max = z2 - h2 / 2, z2 + h2 / 2
    
    # Broadcast for pairwise comparison
    # boxes1: (N, 1), boxes2: (1, M)
    x1_min = x1_min[:, None]
    x1_max = x1_max[:, None]
    y1_min = y1_min[:, None]
    y1_max = y1_max[:, None]
    z1_min = z1_min[:, None]
    z1_max = z1_max[:, None]
    
    x2_min = x2_min[None, :]
    x2_max = x2_max[None, :]
    y2_min = y2_min[None, :]
    y2_max = y2_max[None, :]
    z2_min = z2_min[None, :]
    z2_max = z2_max[None, :]
    
    # Calculate intersection
    inter_x_min = torch.maximum(x1_min, x2_min)
    inter_x_max = torch.minimum(x1_max, x2_max)
    inter_y_min = torch.maximum(y1_min, y2_min)
    inter_y_max = torch.minimum(y1_max, y2_max)
    inter_z_min = torch.maximum(z1_min, z2_min)
    inter_z_max = torch.minimum(z1_max, z2_max)
    
    # Intersection volume
    inter_w = torch.clamp(inter_x_max - inter_x_min, min=0)
    inter_l = torch.clamp(inter_y_max - inter_y_min, min=0)
    inter_h = torch.clamp(inter_z_max - inter_z_min, min=0)
    inter_vol = inter_w * inter_l * inter_h
    
    # Union volume
    vol1 = (w1 * l1 * h1)[:, None]
    vol2 = (w2 * l2 * h2)[None, :]
    union_vol = vol1 + vol2 - inter_vol
    
    # IoU
    iou = inter_vol / torch.clamp(union_vol, min=1e-8)
    
    if is_numpy:
        iou = iou.numpy()
    
    return iou


def try_import_axis_aligned_iou_3d():
    """
    Try to import axis_aligned_iou_3d from MMDetection3D.
    If not available, use the compatibility implementation.
    
    Returns:
        function: The axis_aligned_iou_3d function
    """
    try:
        # Try MMDetection3D 1.4.0 locations
        try:
            from mmdet3d.structures.ops import box_iou_3d
            return box_iou_3d
        except (ImportError, AttributeError):
            pass
        
        try:
            from mmdet3d.ops import box_iou_3d
            return box_iou_3d
        except (ImportError, AttributeError):
            pass
            
        try:
            from mmdet3d.core.bbox import axis_aligned_iou_3d as iou_func
            return iou_func
        except (ImportError, AttributeError):
            pass
        
        # If all imports fail, use our implementation
        return axis_aligned_iou_3d
        
    except Exception:
        return axis_aligned_iou_3d
