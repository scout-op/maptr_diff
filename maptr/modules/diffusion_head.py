# projects/mmdet3d_plugin/maptr/modules/diffusion_head.py
"""
Diffusion Head for Centerline Point Set Generation
Based on DDPM with AdaLN conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmengine.model import BaseModule


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal Time Embedding (from Attention is All You Need)
    Used for encoding diffusion timesteps
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: (batch_size,) tensor of timestep indices
        Returns:
            embeddings: (batch_size, dim) tensor
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AdaLNBlock(nn.Module):
    """
    Adaptive Layer Normalization Block with Residual Connection
    Modulates features using condition (time + query features)
    """
    def __init__(self, embed_dims, cond_dims):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dims)
        
        # Main processing MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dims * 2, embed_dims)
        )
        
        # AdaLN modulation: condition -> [scale, shift]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dims, embed_dims * 2)
        )
    
    def forward(self, x, cond):
        """
        Args:
            x: (bs, num_vec, num_pts, embed_dims) - point features
            cond: (bs, num_vec, num_pts, cond_dims) - condition features
        Returns:
            out: (bs, num_vec, num_pts, embed_dims)
        """
        # Generate scale and shift from condition
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        
        # Normalize
        x_norm = self.norm(x)
        
        # Modulate: element-wise affine transformation
        x_mod = x_norm * (1 + scale) + shift
        
        # Apply MLP with residual
        return x + self.mlp(x_mod)


class PointDiffusionHead(BaseModule):
    """
    Diffusion model for centerline point set denoising
    Predicts noise given noisy points and conditioning features
    """
    def __init__(self, 
                 embed_dims=256, 
                 num_pts=20, 
                 num_layers=4,
                 use_point_interaction=False,
                 init_cfg=None):
        """
        Args:
            embed_dims: dimension of hidden features
            num_pts: number of points per centerline
            num_layers: number of AdaLN blocks
            use_point_interaction: whether to use Conv1d for point-wise interaction
        """
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_pts = num_pts
        self.use_point_interaction = use_point_interaction
        
        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dims),
            nn.Linear(embed_dims, embed_dims),
            nn.SiLU(),
            nn.Linear(embed_dims, embed_dims),
        )

        # 2. Point coordinate to feature
        # Input: (x, y) normalized in [0, 1]
        self.input_proj = nn.Linear(2, embed_dims)
        
        # 3. Optional: Point interaction layer (1D Conv along point sequence)
        if use_point_interaction:
            self.point_interaction = nn.Conv1d(
                embed_dims, embed_dims, 
                kernel_size=3, padding=1, groups=embed_dims  # depthwise conv
            )
        
        # 4. Condition dimension: query_feat (embed_dims) + time_emb (embed_dims)
        cond_dims = embed_dims * 2
        
        # 5. Main denoising layers (AdaLN blocks)
        self.layers = nn.ModuleList([
            AdaLNBlock(embed_dims, cond_dims) 
            for _ in range(num_layers)
        ])

        # 6. Output: predict noise (x, y)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims // 2),
            nn.SiLU(),
            nn.Linear(embed_dims // 2, 2)
        )

    def forward(self, noisy_pts, query_feat, t):
        """
        Forward pass for noise prediction
        
        Args:
            noisy_pts: (bs, num_vec, num_pts, 2) - noisy point coordinates
            query_feat: (bs, num_vec, embed_dims) - query features from decoder
            t: (bs,) - timestep indices
        
        Returns:
            noise_pred: (bs, num_vec, num_pts, 2) - predicted noise
        """
        # ========== PRE-FLIGHT CHECK 1: Shape Validation ==========
        assert noisy_pts.ndim == 4, \
            f"[DIFFUSION ERROR] noisy_pts should be 4D (bs, num_vec, num_pts, 2), got shape: {noisy_pts.shape}"
        assert query_feat.ndim == 3, \
            f"[DIFFUSION ERROR] query_feat should be 3D (bs, num_vec, embed_dims), got shape: {query_feat.shape}"
        assert noisy_pts.shape[-1] == 2, \
            f"[DIFFUSION ERROR] Last dim of noisy_pts should be 2 (x,y), got: {noisy_pts.shape[-1]}"
        
        bs, num_vec, num_pts, _ = noisy_pts.shape
        
        # 1. Time Embedding: (bs,) -> (bs, embed_dims) -> (bs, 1, 1, embed_dims)
        t_emb = self.time_mlp(t).view(bs, 1, 1, -1)
        t_emb = t_emb.expand(bs, num_vec, num_pts, -1)  # broadcast to all points
        
        # 2. Query Feature: (bs, num_vec, embed_dims) -> (bs, num_vec, num_pts, embed_dims)
        query_emb = query_feat.unsqueeze(2).expand(bs, num_vec, num_pts, -1)
        
        # 3. Condition = [query_feat, time_emb]
        cond = torch.cat([query_emb, t_emb], dim=-1)  # (bs, num_vec, num_pts, 2*embed_dims)
        
        # 4. Point Embedding: (bs, num_vec, num_pts, 2) -> (bs, num_vec, num_pts, embed_dims)
        x = self.input_proj(noisy_pts)
        
        # 5. Optional point interaction
        if self.use_point_interaction:
            # Reshape: (bs*num_vec, embed_dims, num_pts)
            x_flat = x.view(bs * num_vec, num_pts, self.embed_dims).permute(0, 2, 1)
            x_flat = self.point_interaction(x_flat)
            x = x_flat.permute(0, 2, 1).view(bs, num_vec, num_pts, self.embed_dims)
        
        # 6. Main denoising blocks
        for layer in self.layers:
            x = layer(x, cond)
        
        # 7. Predict noise
        noise_pred = self.output_proj(x)
        
        return noise_pred


def extract(a, t, x_shape):
    """
    Extract coefficients at specified timesteps t and reshape to broadcast
    
    Args:
        a: (T,) tensor of coefficients
        t: (bs,) tensor of timestep indices
        x_shape: shape of the target tensor (bs, num_vec, num_pts, 2)
    
    Returns:
        out: (bs, 1, 1, 1) tensor for broadcasting
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.view(batch_size, 1, 1, 1)


class DDIMSampler:
    """
    DDIM sampling for fast inference
    Reduces sampling steps from 1000 to ~50 without quality loss
    """
    def __init__(self, diffusion_head, alphas_cumprod, num_inference_steps=50):
        """
        Args:
            diffusion_head: PointDiffusionHead instance
            alphas_cumprod: (T,) tensor of cumulative alphas
            num_inference_steps: number of steps for DDIM sampling
        """
        self.model = diffusion_head
        self.alphas_cumprod = alphas_cumprod
        self.num_inference_steps = num_inference_steps
        
        # Create timestep schedule (uniformly spaced)
        total_steps = len(alphas_cumprod)
        step_ratio = total_steps // num_inference_steps
        self.timesteps = torch.arange(0, total_steps, step_ratio).flip(0)
        
    @torch.no_grad()
    def sample(self, query_feat, num_vec, num_pts, eta=0.0):
        """
        DDIM sampling process
        
        Args:
            query_feat: (bs, num_vec, embed_dims) - conditioning features
            num_vec: number of centerlines
            num_pts: number of points per centerline
            eta: stochasticity parameter (0 = deterministic DDIM)
        
        Returns:
            x_0: (bs, num_vec, num_pts, 2) - generated point coordinates
        """
        bs = query_feat.shape[0]
        device = query_feat.device
        
        # Start from pure noise
        x_t = torch.randn(bs, num_vec, num_pts, 2, device=device)
        
        # Reverse diffusion
        for i, t in enumerate(self.timesteps):
            # Current timestep
            t_batch = torch.full((bs,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(x_t, query_feat, t_batch)
            
            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[self.timesteps[i + 1]] if i < len(self.timesteps) - 1 else torch.tensor(1.0)
            
            # Predict x_0
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            x_0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            
            # Clamp to valid range [0, 1] for stability
            x_0_pred = torch.clamp(x_0_pred, 0.0, 1.0)
            
            # Compute direction pointing to x_t
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)
            
            # DDIM formula
            dir_xt = sqrt_one_minus_alpha_t_prev * noise_pred
            x_t = sqrt_alpha_t_prev * x_0_pred + dir_xt
            
            # Optional: add stochasticity (eta > 0)
            if eta > 0 and i < len(self.timesteps) - 1:
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
                )
                noise = torch.randn_like(x_t)
                x_t = x_t + sigma_t * noise
        
        return x_t
