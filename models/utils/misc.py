import torch
import torch.nn as nn
import torch.nn.functional as F


class MLN(nn.Module):
    """Multi-Layer Normalization / simple MLP + LayerNorm block.

    This is a lightweight stand-in for the original implementation used in
    BEVFormer/MapTR. It applies a linear projection followed by LayerNorm and
    activation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def topk_gather(x: torch.Tensor, topk_idx: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Gather top-k elements from x according to indices.

    Args:
        x: [..., N, C]
        topk_idx: [..., K] indices along ``dim``.
    """
    # Expand indices to match the last dimension
    expand_shape = list(topk_idx.shape) + [x.size(-1)]
    gather_idx = topk_idx.unsqueeze(-1).expand(expand_shape)
    return torch.gather(x, dim, gather_idx)


def transform_reference_points(reference_points: torch.Tensor, bev_h: int, bev_w: int) -> torch.Tensor:
    """Normalize reference points from BEV grid to [0, 1] range.

    The original implementation supports more modes; here we keep a simple and
    commonly-used variant.
    """
    # reference_points: (..., 2) in [0, bev_w) x [0, bev_h)
    ref_x = reference_points[..., 0] / max(bev_w, 1)
    ref_y = reference_points[..., 1] / max(bev_h, 1)
    return torch.stack([ref_x, ref_y], dim=-1)


def memory_refresh(memory: torch.Tensor, new_memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Refresh memory with new_memory according to mask.

    Args:
        memory: (..., C)
        new_memory: (..., C)
        mask: (...,) bool or 0/1 tensor, True/1 means update.
    """
    mask = mask.to(dtype=memory.dtype).unsqueeze(-1)
    return memory * (1.0 - mask) + new_memory * mask


class SELayer_Linear(nn.Module):
    """SE layer working on linear features (B, N, C).

    A simplified squeeze-and-excitation for token features.
    """

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channel, channel // reduction, bias=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        if x.ndim != 3:
            raise ValueError(f"SELayer_Linear expects (B, N, C), got {x.shape}")
        # Global average over tokens N
        y = x.mean(dim=1)  # (B, C)
        y = self.fc1(y)
        y = F.relu(y, inplace=True)
        y = self.fc2(y)
        y = torch.sigmoid(y).unsqueeze(1)  # (B, 1, C)
        return x * y
