import math
from typing import Optional

import torch
import torch.nn as nn


def pos2posemb3d(
    pos: torch.Tensor,
    num_pos_feats: int = 64,
    temperature: int = 10000,
    normalize: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """3D sinusoidal positional embedding used by BEVFormer/MapTR.

    Args:
        pos: (..., 3) tensor of xyz coordinates.
    Returns:
        (..., num_pos_feats * 3) embedding.
    """
    if scale is None:
        scale = 2 * math.pi

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    if normalize:
        eps = 1e-6
        pos = pos / (pos[..., -1:, :] + eps) * scale

    # pos[..., 0/1/2] -> (..., 1)
    pos_x = pos[..., 0] / dim_t
    pos_y = pos[..., 1] / dim_t
    pos_z = pos[..., 2] / dim_t

    def _embed(coord: torch.Tensor) -> torch.Tensor:
        # (..., num_pos_feats)
        coord = coord.unsqueeze(-1)
        sin = coord[..., 0::2].sin()
        cos = coord[..., 1::2].cos()
        out = torch.stack((sin, cos), dim=-1).flatten(-2)
        return out

    emb_x = _embed(pos_x)
    emb_y = _embed(pos_y)
    emb_z = _embed(pos_z)

    return torch.cat((emb_y, emb_x, emb_z), dim=-1)


def pos2posemb1d(
    pos: torch.Tensor,
    num_pos_feats: int = 64,
    temperature: int = 10000,
) -> torch.Tensor:
    """1D sinusoidal positional embedding.

    Args:
        pos: (...,) tensor of positions.
    Returns:
        (..., num_pos_feats * 2) embedding.
    """
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos = pos / dim_t
    pos = pos.unsqueeze(-1)
    sin = pos[..., 0::2].sin()
    cos = pos[..., 1::2].cos()
    out = torch.stack((sin, cos), dim=-1).flatten(-2)
    return out


def nerf_positional_encoding(
    x: torch.Tensor,
    num_encoding_functions: int = 6,
    include_input: bool = True,
) -> torch.Tensor:
    """NeRF-style positional encoding of coordinates.

    Args:
        x: (..., C) coordinates in [-1, 1] or world units.
    Returns:
        (..., C * (include_input + 2 * num_encoding_functions))
    """
    encodings = []
    if include_input:
        encodings.append(x)

    for i in range(num_encoding_functions):
        freq = 2.0 ** i
        encodings.append(torch.sin(freq * x))
        encodings.append(torch.cos(freq * x))

    return torch.cat(encodings, dim=-1)
