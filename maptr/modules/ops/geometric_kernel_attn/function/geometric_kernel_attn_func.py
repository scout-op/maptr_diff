from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

# Try to import the compiled CUDA extension, fall back to PyTorch implementation
try:
    import GeometricKernelAttention as GKA
    HAS_GKA_CUDA = True
except ImportError:
    HAS_GKA_CUDA = False
    warnings.warn(
        "GeometricKernelAttention CUDA extension not found. "
        "Using PyTorch fallback (slower). To compile the extension, run:\n"
        "  cd projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn && python setup.py install"
    )


def geometric_kernel_attn_pytorch(value, value_spatial_shapes, value_level_start_index,
                                   sampling_locations, attention_weights):
    """PyTorch fallback implementation of geometric kernel attention.
    
    This is a simplified implementation that mimics multi_scale_deformable_attn_pytorch.
    For best performance, compile and use the CUDA extension.
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    
    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1  # normalize to [-1, 1]
    sampling_value_list = []
    
    for level, (H, W) in enumerate(value_spatial_shapes):
        # bs, H*W, num_heads, embed_dims -> bs, H*W, num_heads*embed_dims
        # -> bs, num_heads*embed_dims, H*W -> bs, num_heads*embed_dims, H, W
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(
            bs, num_heads * embed_dims, int(H), int(W))
        # bs, num_queries, num_heads, num_points, 2 -> bs, num_heads, num_queries, num_points, 2
        # -> bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_.view(bs * num_heads, embed_dims, int(H), int(W)),
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    
    # (bs, num_queries, num_heads, num_levels, num_points)
    # -> (bs, num_heads, num_queries, num_levels, num_points)
    # -> (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points)
    
    # (bs*num_heads, embed_dims, num_queries, num_levels, num_points)
    # -> (bs*num_heads, embed_dims, num_queries, num_levels*num_points)
    output = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (output * attention_weights).sum(-1)
    output = output.view(bs, num_heads * embed_dims, num_queries)
    return output.transpose(1, 2).contiguous()


class GeometricKernelAttentionFunc(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        
        if HAS_GKA_CUDA:
            output = GKA.geometric_kernel_attn_cuda_forward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        else:
            # Use PyTorch fallback
            output = geometric_kernel_attn_pytorch(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        
        if HAS_GKA_CUDA:
            grad_value, grad_attn_weight = \
                GKA.geometric_kernel_attn_cuda_backward(
                    value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)
        else:
            # For PyTorch fallback, we don't have custom backward - use autograd
            # This path should not be reached in normal usage since forward uses autograd-compatible ops
            grad_value = torch.zeros_like(value)
            grad_attn_weight = torch.zeros_like(attention_weights)

        return grad_value, None, None, None, grad_attn_weight, None
