# Disclaimer: This code was influenced by
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.codegen.cfunctions import isnan
from torch.autograd import Variable
import math
import torchvision
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from utils import OsJoin
from einops import rearrange
from functools import partial
from opts import parse_opts
# from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from collections import OrderedDict
from torch.nn.functional import interpolate
from einops.layers.torch import Rearrange
from torch.autograd import grad as torch_grad

class MoELayer_time(nn.Module):
    """Mixture of Experts layer to replace standard FFN in Transformer

    Key Features:
    - Dynamic routing via gating network
    - Sparse activation (only top-k experts per token)
    - Load balancing through expert diversity

    Args:
        d_model: Hidden dimension size (input/output dimension)
        d_ff: Expert's internal feed-forward dimension
        num_experts: Total number of experts in the pool
        k: Number of experts to activate per token (k < num_experts)
    """

    def __init__(self, d_model, d_ff, num_experts=200, k=200):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        # Expert pool: Each expert is an independent FFN
        self.experts_check = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(2)
        ])
        self.experts_times= nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(200)
        ])

        # Gating network to determine expert weights
        self.gate_check = nn.Linear(d_model, 2)
        self.gate_time = nn.Linear(d_model, 200)
    def forward(self, x):
        """Forward pass with sparse expert activation

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Processed tensor of same shape as input
        """
        batch_size, seq_len, _ = x.shape

        # Flatten batch and sequence dimensions
        flat_x = x.view(-1, x.size(-1))  # [batch*seq, d_model]

        # Compute gating scores (logits)
        gate_logits_check = self.gate_check(flat_x)  # [batch*seq, num_experts]
        gates_times = self.gate_time(flat_x)
        # Convert to probabilities via softmax
        gate_probs_check = F.softmax(gate_logits_check, dim=-1)  # [batch*seq, num_experts]
        gate_probs_times = F.softmax(gates_times, dim=-1)
        # Select top-k experts for each token
        topk_probs_check, topk_indices_check = torch.topk(gate_probs_check, 1, dim=-1)  # both [batch*seq, k]
        topk_probs_times, topk_indices_times= torch.topk(gate_probs_times, 1, dim=-1)  # both [batch*seq, k]
        # Normalize top-k probabilities
        topk_probs_check = topk_probs_check / topk_probs_check.sum(dim=-1, keepdim=True)  # [batch*seq, k]
        topk_probs_times= topk_probs_times/ topk_probs_times.sum(dim=-1, keepdim=True)  # [batch*seq, k]
        # Initialize output tensor
        out_check = torch.zeros_like(flat_x)  # [batch*seq, d_model]
        out_times = torch.zeros_like(flat_x)  # [batch*seq, d_model]
        # Sparse computation: Only process activated experts
        for expert_id, expert in enumerate(self.experts_check):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices_check == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs_check[expert_mask,
                (topk_indices_check[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(flat_x[expert_mask])
                out_check[expert_mask] += expert_out * prob.unsqueeze(-1)
        for expert_id, expert in enumerate(self.experts_times):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices_times == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs_times[expert_mask,
                (topk_indices_times[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(flat_x[expert_mask])
                out_times[expert_mask] += expert_out * prob.unsqueeze(-1)
        out = out_times + out_check
        # sim =

        # Restore original shape
        return out.view(batch_size, seq_len, -1)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        # self.moe = MoELayer(dim, dim_head, heads)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout),
                MoELayer_time(dim, mlp_dim)
            ]))

    def forward(self, x):
        for attn, ff, moe in self.layers:
            x = attn(x) + x
            x = moe(x) + x

        return self.norm(x)

class Conn_gen_raw(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, channels=1,
                  dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)
        patch_size = 12
        patch_size_fMRI = 8
        image_height_dMRI, image_width_dMRI = 96,96
        image_height_fMRI, image_width_fMRI = 64,64
        image_depth_dMRI = 60
        image_depth_fMRI = 40
        num_patches = ((image_height_dMRI // patch_size) * (image_width_dMRI // patch_size)
                        * (image_depth_dMRI // patch_size))
        num_patches_fMRI = ((image_height_fMRI // patch_size_fMRI)
                            * (image_width_fMRI // patch_size_fMRI)) * (image_depth_fMRI // patch_size_fMRI)
        patch_dim = channels * patch_size * patch_size * patch_size
        patch_dim_fMRI = channels * patch_size_fMRI * patch_size_fMRI * patch_size_fMRI
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)',
                      p1 = patch_size, p2 = patch_size, p3 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_embedding_fMRI = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)',
                      p1 = patch_size_fMRI, p2 = patch_size_fMRI, p3 = patch_size_fMRI),
            nn.LayerNorm(patch_dim_fMRI),
            nn.Linear(patch_dim_fMRI, dim),
            nn.LayerNorm(dim),
        )
        self.inverse_to_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange(' b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)',
                      h=image_height_dMRI // patch_size,
                      w=image_width_dMRI // patch_size,
                      d= image_depth_dMRI // patch_size,
                      p1=patch_size, p2=patch_size, p3=patch_size),
        )
        self.inverse_to_patch_embedding_fMRI = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim_fMRI),
            nn.LayerNorm(patch_dim_fMRI),
            Rearrange(' b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)',
                      h=image_height_fMRI // patch_size_fMRI,
                      w=image_width_fMRI // patch_size_fMRI,
                      d= image_depth_fMRI // patch_size_fMRI,
                      p1=patch_size_fMRI, p2=patch_size_fMRI, p3=patch_size_fMRI),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.pos_embedding_fMRI = nn.Parameter(torch.randn(1, num_patches_fMRI, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer_fMRI = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # self.pool = pool
        # self.to_latent = nn.Identity()

    def batch_corrcoef(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Pearson Correlation Coefficient matrices for each batch.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N), where
                              B is batch size and N is the number of features.

        Returns:
            torch.Tensor: A tensor of shape (B, N, N), containing Pearson
                          correlation matrices for each sample in the batch.
        """
        B, N = x.shape

        # Step 1: Subtract mean along feature dimension (dim=1)
        mean = x.mean(dim=1, keepdim=True)  # Shape: (B, 1)
        x_centered = x - mean  # Shape: (B, N)

        # Step 2: Normalize by standard deviation to get z-scores
        std = x_centered.std(dim=1, keepdim=True, unbiased=True)  # Shape: (B, 1)
        eps = 1e-8  # Small epsilon to avoid division by zero
        x_normalized = x_centered / (std + eps)  # Shape: (B, N)

        # Step 3: Compute correlation matrix using batch matrix multiplication
        # Expand dimensions to (B, N, 1) and (B, 1, N) for outer product
        corr = torch.bmm(x_normalized.unsqueeze(-1), x_normalized.unsqueeze(-2))

        # Resulting shape: (B, N, N)
        return corr

    def coff(self, arr, mask):
           mask_flatten = mask.view(-1).cuda()
           arr = arr.view(arr.shape[0],-1)
           # print(arr.shape[1])
           # print(len(mask_flatten))
           # if len(mask_flatten) != arr.shape[1]:
           #     print('do not match')
           mean_region = torch.zeros(arr.shape[0],1).cuda()
           for cur_region in range(1,161,1):
               region_indices = (mask_flatten == cur_region)
               nonzero_idx = region_indices.nonzero(as_tuple=True)[0]
               # print(nonzero_idx)
               # region_values =  arr*(region_indices.unsqueeze(0))
               # if max(region_indices) != True:
               #     print('Error')
               if nonzero_idx.numel() > 0:  # 检查是否有元素
                   region_values = arr[:, nonzero_idx]
               else:
                   index = torch.where(mask_flatten== cur_region)
                   range_ = 0.001
                   while (len(index) <= 8):
                       index = torch.where((mask_flatten > (cur_region- range_)) & (mask_flatten< (cur_region + range_)))[0]
                       range_ +=0.0001
                   region_values = arr[:,index]  # 返回一个 dummy 张量

               # print(max(nonzero_idx))
               # region_values_compressed = arr[:, nonzero_idx]
               # print(region_values_compressed.shape)
               # print(region_values.shape)
               # print(region_values)
               cur_region_value = torch.mean(region_values, dim=1, keepdim=True)
               # print(cur_region_value)
               # cur_region_value = cur_region_value.unsqueeze(0).unsqueeze(0)
               # if region_values.size(1) > 0:
               #     cur_region_value = torch.mean(region_values, dim=1, keepdim=True)
               # else:
               #     # print('Nan value')
               #
               #     cur_region_value = torch.zeros_like(region_values[:, :1])  # dummy mean
               # print(cur_region_value)
               mean_region = torch.cat((mean_region, cur_region_value), dim=1)
               # print(mean_region)
               # current_region_index = (mask_flatten == cur_region).float()
               # cur_region_value = torch.mean(arr[current_region_index],dim=2,keepdim=True)
               # mean_region= torch.cat((mean_region,cur_region_value),dim=2)
           mean_region= (mean_region[:,1:])
           # mean_region = mean_region.transpose(1,0)
           # print(mean_region.shape) [48,160]
           # print(mean_region.shape)
           # corr = torch.corrcoef(mean_region)
           batch_corr = self.batch_corrcoef(mean_region)
           # if isnan(batch_corr):
           #     print('True')
           # print(batch_corr.shape)
           return batch_corr

    def forward(self, img_f, img_d, atlas_f, atlas_d):
        x_d = self.to_patch_embedding(img_d)
        b_d, n_d, _ = x_d.shape
        x_d += self.pos_embedding[:, :(n_d)]
        x_d = self.dropout(x_d)

        x_d = self.transformer(x_d)
        x_d = self.inverse_to_patch_embedding(x_d)
        x_f_re = interpolate(x_d, [img_f.shape[2], img_f.shape[3], img_f.shape[4]])
        f_flatten = x_f_re.view(b_d,-1)
        fc = self.coff(f_flatten, atlas_f)
        x_f = self.to_patch_embedding_fMRI(img_f)
        b_f, n_f, _ = x_f.shape
        x_f += self.pos_embedding_fMRI[:, :(n_f)]
        x_f = self.dropout(x_f)

        x_f = self.transformer_fMRI(x_f)
        x_f = self.inverse_to_patch_embedding_fMRI(x_f)
        x_d_re = interpolate(x_f, [img_d.shape[2], img_d.shape[3], img_d.shape[4]])
        d_flatten =  x_d_re.view(b_f, -1)
        sc = self.coff(d_flatten, atlas_d)

        return sc,fc,x_d_re,x_f_re
