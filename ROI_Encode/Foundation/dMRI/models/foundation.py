# Disclaimer: This code was influenced by
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py

import torch
import torch.nn as nn
import torch.nn.functional as F
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
# from muse_maskgit_pytorch import VQGanVAETrainer, MaskGitTransformer,MaskGit
# from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from collections import OrderedDict
from einops.layers.torch import Rearrange
from torch.autograd import grad as torch_grad
from torchvision.utils import make_grid, save_image
opt = parse_opts()
# class MoELayer_Network(nn.Module):
#     """Mixture of Experts layer to replace standard FFN in Transformer
#
#     Key Features:
#     - Dynamic routing via gating network
#     - Sparse activation (only top-k experts per token)
#     - Load balancing through expert diversity
#
#     Args:
#         d_model: Hidden dimension size (input/output dimension)
#         d_ff: Expert's internal feed-forward dimension
#         num_experts: Total number of experts in the pool
#         k: Number of experts to activate per token (k < num_experts)
#     """
#
#     def __init__(self, d_model, d_ff, num_experts=7, k=1):
#         super().__init__()
#         self.num_experts = num_experts
#         self.k = k
#
#         # Expert pool: Each expert is an independent FFN
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(d_model, d_ff),  # Expansion
#                 nn.GELU(),  # Activation
#                 nn.Linear(d_ff, d_model)  # Compression
#             ) for _ in range(num_experts)
#         ])
#
#         # Gating network to determine expert weights
#         self.gate = nn.Linear(d_model, num_experts)
#
#     def forward(self, x):
#         """Forward pass with sparse expert activation
#
#         Args:
#             x: Input tensor of shape [batch_size, seq_len, d_model]
#
#         Returns:
#             Processed tensor of same shape as input
#         """
#         '''
#         MoE setting
#         '''
#         batch_size, seq_len, _ = x.shape
#
#         # Flatten batch and sequence dimensions
#         flat_x = x.view(-1, x.size(-1))  # [batch*seq, d_model]
#
#         # Compute gating scores (logits)
#         gate_logits = self.gate(flat_x)  # [batch*seq, num_experts]
#
#         # Convert to probabilities via softmax
#         gate_probs = F.softmax(gate_logits, dim=-1)  # [batch*seq, num_experts]
#
#         # Select top-k experts for each token
#         topk_probs, topk_indices = torch.topk(gate_probs, self.k, dim=-1)  # both [batch*seq, k]
#
#         # Normalize top-k probabilities
#         topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # [batch*seq, k]
#
#         # Initialize output tensor
#         out = torch.zeros_like(flat_x)  # [batch*seq, d_model]
#
#         # Sparse computation: Only process activated experts
#         for expert_id, expert in enumerate(self.experts):
#             # Create mask for tokens selecting current expert
#             expert_mask = (topk_indices == expert_id).any(dim=-1)  # [batch*seq]
#
#             if expert_mask.any():
#                 # Get probability weights for current expert
#                 prob = topk_probs[expert_mask,
#                 (topk_indices[expert_mask] == expert_id).nonzero()[:, 1]]
#
#                 # Compute and weight expert outputs
#                 expert_out = expert(flat_x[expert_mask])
#                 out[expert_mask] += expert_out * prob.unsqueeze(-1)
#
#         # Restore original shape
#         return out.view(batch_size, seq_len, -1)

class MoELayer_hirachical(nn.Module):
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

    def __init__(self, d_model, d_ff, num_network=7):
        super().__init__()
        # self.num_experts = 7
        # self.k_network = 1
        # self.regions = 160
        self.experts_all_brain = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(2)
        ])
        # Expert pool: Each expert is an independent FFN
        # self.experts_network = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(d_model, d_ff),  # Expansion
        #         nn.GELU(),  # Activation
        #         nn.Linear(d_ff, d_model)  # Compression
        #     ) for _ in range(6)
        # ])
        self.experts_regions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(160)
        ])

        self.experts_intra_regions_CON = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(32)
        ])
        self.experts_intra_regions_DMN = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(34)
        ])
        self.experts_intra_regions_FPN = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(21)
        ])
        self.experts_intra_regions_SMN = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(33)
        ])
        self.experts_intra_regions_CEB = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(18)
        ])
        self.experts_intra_regions_OCC = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(22)
        ])
        self.experts_total = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(2)
        ])
        self.experts_intra = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(2)
        ])
        self.experts_network = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(6)
        ])


        # Gating network to determine expert weights
        self.gate_all_brain = nn.Linear(d_model, 2)
        # self.gate_network = nn.Linear(d_model, num_network)
        self.gate_regions = nn.Linear(d_model, 160)
        self.gate_regions_intra_DMN = nn.Linear(d_model, 34)
        self.gate_regions_intra_FPN = nn.Linear(d_model, 21)
        self.gate_regions_intra_OCC = nn.Linear(d_model, 22)
        self.gate_regions_intra_SMN = nn.Linear(d_model, 33)
        self.gate_regions_intra_CON = nn.Linear(d_model, 32)
        self.gate_regions_intra_CEB = nn.Linear(d_model, 18)
        self.gate_intra = nn.Linear(d_model, 2)
        self.gate_total = nn.Linear(d_model, 2)
        self.gate_network = nn.Linear(d_model, 6)
    def forward(self, x):
        """For_intra_CEB = nn.Linear(d_model, 18)
ward pass with sparse expert activation

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Processed tensor of same shape as input
        """
        '''
        MoE setting
        directly added the network MOE do not result in NaN loss.
        '''
        batch_size, seq_len, _ = x.shape

        # Flatten batch and sequence dimensions
        flat_x = x.view(-1, x.size(-1))  # [batch*seq, d_model]

        # Compute gating scores (logits)
        gate_logits = self.gate_all_brain(flat_x)  # [batch*seq, num_experts]

        # Convert to probabilities via softmax
        gate_probs = F.softmax(gate_logits, dim=-1)  # [batch*seq, num_experts]
        topk_probs_all_brain, topk_indices_all_brain = torch.topk(gate_probs, 1, dim=-1)  # both [batch*seq, k]
        topk_probs_all_brain = topk_probs_all_brain / topk_probs_all_brain.sum(dim=-1, keepdim=True)
        out_all_brain = torch.zeros_like(flat_x)

        for expert_id, expert in enumerate(self.experts_all_brain):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices_all_brain == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs_all_brain[expert_mask,
                (topk_indices_all_brain[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(flat_x[expert_mask])
                out_all_brain[expert_mask] += expert_out * prob.unsqueeze(-1)
        out_all_brain = out_all_brain
        # gate_logits_roi = out_all_brain.view(batch_size*seq_len, -1).repeat_interleave(6, dim=-1)
        gate_logits_roi = self.gate_regions(out_all_brain)
        gate_probs_roi = F.softmax(gate_logits_roi , dim=-1)
        topk_probs_roi , topk_indices_roi = torch.topk(gate_probs_roi, 1, dim=-1)  # both [batch*seq, k]
        topk_probs_roi = topk_probs_roi / topk_probs_roi.sum(dim=-1, keepdim=True)
        out_roi = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts_regions):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices_roi == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs_roi[expert_mask,
                (topk_indices_roi[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(out_all_brain[expert_mask])
                out_roi[expert_mask] += expert_out * prob.unsqueeze(-1)
        out_roi = out_roi + out_all_brain
        # gate_logits_DMN = out_roi.view(batch_size * seq_len, -1).repeat_interleave(34, dim=-1)
        # gate_logits_FPN = out_roi.view(batch_size * seq_len, -1).repeat_interleave(21, dim=-1)
        # gate_logits_OCC = out_roi.view(batch_size * seq_len, -1).repeat_interleave(22, dim=-1)
        # gate_logits_SMN = out_roi.view(batch_size * seq_len, -1).repeat_interleave(33, dim=-1)
        # gate_logits_CEB = out_roi.view(batch_size * seq_len, -1).repeat_interleave(18, dim=-1)
        # gate_logits_CON = out_roi.view(batch_size * seq_len, -1).repeat_interleave(32, dim=-1)
        gate_logits_DMN = self.gate_regions_intra_DMN(out_roi)
        gate_logits_FPN = self.gate_regions_intra_FPN(out_roi)
        gate_logits_OCC = self.gate_regions_intra_OCC(out_roi)
        gate_logits_SMN = self.gate_regions_intra_SMN(out_roi)
        gate_logits_CEB = self.gate_regions_intra_CEB(out_roi)
        gate_logits_CON = self.gate_regions_intra_CON(out_roi)
        # # Select top-k experts for each token
        # topk_probs_DMN, topk_indices_DMN = torch.topk(gate_logits_DMN , 34, dim=-1)
        # topk_probs_FPN , topk_indices_FPN  = torch.topk(gate_logits_FPN, 21, dim=-1)
        # topk_probs_OCC, topk_indices_OCC = torch.topk(gate_logits_OCC, 22, dim=-1)
        # topk_probs_SMN, topk_indices_SMN = torch.topk(gate_logits_SMN, 33, dim=-1)
        # topk_probs_CEB, topk_indices_CEB = torch.topk(gate_logits_CEB, 18, dim=-1)
        # topk_probs_CON, topk_indices_CON = torch.topk(gate_logits_CON, 32, dim=-1)
        topk_probs_DMN, topk_indices_DMN = torch.topk(gate_logits_DMN , 1, dim=-1)
        topk_probs_FPN , topk_indices_FPN  = torch.topk(gate_logits_FPN, 1, dim=-1)
        topk_probs_OCC, topk_indices_OCC = torch.topk(gate_logits_OCC, 1, dim=-1)
        topk_probs_SMN, topk_indices_SMN = torch.topk(gate_logits_SMN, 1, dim=-1)
        topk_probs_CEB, topk_indices_CEB = torch.topk(gate_logits_CEB, 1, dim=-1)
        topk_probs_CON, topk_indices_CON = torch.topk(gate_logits_CON, 1, dim=-1)
        # # Normalize top-k probabilities
        topk_probs_DMN = topk_probs_DMN / topk_probs_DMN.sum(dim=-1, keepdim=True)  # [batch*seq, k]
        topk_probs_OCC = topk_probs_OCC / topk_probs_OCC.sum(dim=-1, keepdim=True)  # [batch*seq, k]
        topk_probs_CON = topk_probs_CON / topk_probs_CON.sum(dim=-1, keepdim=True)  # [batch*seq, k]
        topk_probs_CEB = topk_probs_CEB / topk_probs_CEB.sum(dim=-1, keepdim=True)  # [batch*seq, k]
        topk_probs_FPN = topk_probs_FPN / topk_probs_FPN.sum(dim=-1, keepdim=True)  # [batch*seq, k]
        topk_probs_SMN = topk_probs_SMN / topk_probs_SMN.sum(dim=-1, keepdim=True)  # [batch*seq, k]
        # # Initialize output tensor
        #
        out_DMN = torch.zeros_like(flat_x)
        out_OCC = torch.zeros_like(flat_x)
        out_CON = torch.zeros_like(flat_x)
        out_CEB = torch.zeros_like(flat_x)
        out_FPN  = torch.zeros_like(flat_x)
        out_SMN = torch.zeros_like(flat_x)
        # # Sparse computation: Only process activated experts
        for expert_id, expert in enumerate(self.experts_intra_regions_DMN):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices_DMN == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs_DMN[expert_mask,
                (topk_indices_DMN[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(out_roi[expert_mask])
                out_DMN[expert_mask] += expert_out * prob.unsqueeze(-1)
        for expert_id, expert in enumerate(self.experts_intra_regions_CON):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices_CON == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs_CON[expert_mask,
                (topk_indices_CON[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(out_roi[expert_mask])
                out_CON[expert_mask] += expert_out * prob.unsqueeze(-1)
        for expert_id, expert in enumerate(self.experts_intra_regions_SMN):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices_SMN == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs_SMN[expert_mask,
                (topk_indices_SMN[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(out_roi[expert_mask])
                out_SMN[expert_mask] += expert_out * prob.unsqueeze(-1)
        for expert_id, expert in enumerate(self.experts_intra_regions_OCC):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices_OCC == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs_OCC[expert_mask,
                (topk_indices_OCC[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(out_roi[expert_mask])
                out_OCC[expert_mask] += expert_out * prob.unsqueeze(-1)
        for expert_id, expert in enumerate(self.experts_intra_regions_FPN):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices_FPN == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs_FPN[expert_mask,
                (topk_indices_FPN[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(out_roi[expert_mask])
                out_FPN[expert_mask] += expert_out * prob.unsqueeze(-1)

        for expert_id, expert in enumerate(self.experts_intra_regions_CEB):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices_CEB == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs_CEB[expert_mask,
                (topk_indices_CEB[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(out_roi[expert_mask])
                out_CEB[expert_mask] += expert_out * prob.unsqueeze(-1)
        # # Restore original shape
        # # residual block is invalid for loss Nan
        # out_networks_region = torch.stack([out_OCC,out_SMN,out_CEB,out_FPN,out_DMN, out_CON],dim=2)
        out_networks_region = out_DMN + out_OCC + out_CEB + out_FPN + out_SMN + out_CON + out_roi
        # gate_logits_total = torch.cat([out_roi.view(batch_size * seq_len, -1),out_networks_region.view(batch_size * seq_len, -1)], dim=-1)
        gate_logits_total = self.gate_total(out_roi)
        gates_network = self.gate_network(out_networks_region)
        # gates_network = gates_network.squeeze()
        topk_probs_network, topk_indices_network = torch.topk(gates_network, 1, dim=-1)
        topk_probs_network = topk_probs_network / topk_probs_network.sum(dim=-1, keepdim=True)  # [batch*seq, k]
        out_network = torch.zeros_like(out_networks_region)
        for expert_id, expert in enumerate(self.experts_network):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices_network == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs_network[expert_mask,
                (topk_indices_network[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(out_networks_region[expert_mask])
                # print(expert_out.shape)
                out_network[expert_mask] += expert_out * prob.unsqueeze(-1)
        # out_network = torch.mean(out_network,dim=-1)
        gate_logits_intra = self.gate_intra(out_network)
        topk_probs_total, topk_indices_total = torch.topk(gate_logits_total, 1, dim=-1)
        topk_probs_total = topk_probs_total / topk_probs_total.sum(dim=-1, keepdim=True)  # [batch*seq, k]
        topk_probs_intra, topk_indices_intra = torch.topk(gate_logits_intra, 1, dim=-1)
        topk_probs_intra = topk_probs_intra/ topk_probs_intra.sum(dim=-1, keepdim=True)  # [batch*seq, k]
        out_total = torch.zeros_like(flat_x)
        out_intra = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts_total):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices_total == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs_total[expert_mask,
                (topk_indices_total[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(out_roi[expert_mask])
                out_total[expert_mask] += expert_out * prob.unsqueeze(-1)
        for expert_id, expert in enumerate(self.experts_intra):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices_intra == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs_intra[expert_mask,
                (topk_indices_intra[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(out_network[expert_mask])
                out_intra[expert_mask] += expert_out * prob.unsqueeze(-1)
        # out = torch.matmul(out_total, out_intra)/ torch.add(out_total, out_intra)
        # out_total = out_total + out_roi
        out = out_intra + out_total + out_network + out_roi
        # out = out_DMN + out_OCC + out_CON + out_CEB + out_FPN + out_SMN
        # dot_product = torch.sum(out_total * out_intra, dim=1, keepdim=True)
        # sum_result = torch.add(out_total, out_intra)
        # out = dot_product / sum_result
        return out.view(batch_size, seq_len, -1)

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

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        # self.moe = MoELayer(dim, mlp_dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout),
                MoELayer_hirachical(dim, mlp_dim)
            ]))
    '''
    the final must be ff(x), too much moe network lead to nan, directly employ moe result in Nan
    '''

    def forward(self, x):
        for attn, ff, moe in self.layers:
            # print('test')
            x = attn(x) + x
            x_ff = ff(x) + x
            x_moe = moe(x) + x
            x_ff_moe = moe(x_ff) + x_ff
            x_moe_ff = ff(x_moe) + x_moe
            group_mean_ff = torch.mean(x_moe_ff, dim = 0)
            group_mean_moe = torch.mean(x_ff_moe, dim = 0)
            # loss 0.2187 1e-5 0.0788 1e-4
            # x_sub = ff(x_ff_moe-x_moe_ff) - ff(x_moe_ff-x_ff_moe)
            '''loss 0.0609 1e-4'''
            # x_sub = ff(x_ff_moe*group_mean_moe) - ff(x_moe_ff*group_mean_ff)
            '''loss 0.0609 1e-4'''
            # x_sub = ff(x_ff_moe*group_mean_moe) + ff(x_moe_ff*group_mean_ff)
            '''loss 0.0963 1e-44'''
            # x_sub = ff(x_ff_moe*x_ff_moe*group_mean_moe) - ff(x_moe_ff*x_moe_ff*group_mean_ff)
            '''loss 0.0731 1e-4 loss 0.1048'''
            # group_weight = torch.max(group_mean_ff, group_mean_moe)
            # x_sub = ff(x_ff_moe*group_weight) - ff(x_moe_ff*group_weight)
            '''loss 0.0731 1e-4'''
            if opt.structure == 'Max':
                group_weight = torch.max(group_mean_ff, group_mean_moe)
                x_sub = ff(x_ff_moe*group_weight) * ff(x_moe_ff*group_weight)
            '''loss 0.0790 1e-4'''
            if opt.structure == 'Max_min_sub':
                group_weight_max = torch.max(group_mean_ff, group_mean_moe)
                group_weight_min = torch.min(group_mean_ff, group_mean_moe)
                x_sub_for = ff(x_ff_moe*group_weight_max) + ff(x_moe_ff*group_weight_max)
                x_sub_minus = ff(x_ff_moe*group_weight_min) + ff(x_moe_ff*group_weight_min)
                x_sub = x_sub_for - x_sub_minus
            '''loss 0.0766 1e-4'''
            if opt.structure == 'Max_min_mut':
                group_weight_max = torch.max(group_mean_ff, group_mean_moe)
                group_weight_min = torch.min(group_mean_ff, group_mean_moe)
                x_sub_for = ff(x_ff_moe*group_weight_max) + ff(x_moe_ff*group_weight_max)
                x_sub_minus = ff(x_ff_moe*group_weight_min) + ff(x_moe_ff*group_weight_min)
                x_sub = x_sub_for * x_sub_minus
            '''loss 0.0837 1e-4'''
            if opt.structure == 'Max_min_add':
                group_weight_max = torch.max(group_mean_ff, group_mean_moe)
                group_weight_min = torch.min(group_mean_ff, group_mean_moe)
                x_sub_for = ff(x_ff_moe*group_weight_max) + ff(x_moe_ff*group_weight_max)
                x_sub_minus = ff(x_ff_moe*group_weight_min) + ff(x_moe_ff*group_weight_min)
                x_sub = x_sub_for + x_sub_minus
            '''loss 0.0769'''
            if opt.structure == 'Min':
                group_weight_min = torch.min(group_mean_ff, group_mean_moe)
                x_sub_minus = ff(x_ff_moe*group_weight_min) + ff(x_moe_ff*group_weight_min)
                x_sub =  x_sub_minus
            '''loss (1e-5 ) nan'''
            # x_sub = moe(x_ff_moe-x_moe_ff) - moe(x_moe_ff-x_ff_moe)
            # final loss 0.08 (1e-5), loss (1e-4) nan, loss (1e-5) nan
            # x_sub = ff(x_ff_moe-x_moe_ff) - moe(x_moe_ff-x_ff_moe)
            '''loss (1e-5 ) nan'''
            # x_sub = ff(x_moe_ff-x_ff_moe) - moe(x_ff_moe-x_moe_ff)
            '''loss (1e-5 ) nan'''
            # x_sub = ff(x_moe_ff+x_ff_moe) - moe(x_ff_moe+x_moe_ff)

            ''' 0.0940 1e-4 0.0355 1e-3'''
            if opt.structure == 'None':
                x = x_ff
            else:
                x = x_ff + x_sub

        return self.norm(x)

class ViT_dualModal(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim,  channels=1,
                  dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)
        # patch_size_dMRI = 12
        patch_size_fMRI = 8
        # image_height_dMRI, image_width_dMRI = 96,96
        image_height_fMRI, image_width_fMRI = 64,64
        # image_depth_dMRI = 60
        image_depth_fMRI = 40
        # num_patches = ((image_height_dMRI // patch_size_dMRI) * (image_width_dMRI // patch_size_dMRI)
        #                 * (image_depth_dMRI // patch_size_dMRI))
        num_patches_fMRI = ((image_height_fMRI // patch_size_fMRI )
                            * (image_width_fMRI // patch_size_fMRI )) * (image_depth_fMRI // patch_size_fMRI )
        # patch_dim = channels * patch_size_dMRI* patch_size_dMRI * patch_size_dMRI
        patch_dim_fMRI = channels * patch_size_fMRI * patch_size_fMRI * patch_size_fMRI
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)',
        #               p1 = patch_size_dMRI, p2 = patch_size_dMRI, p3 = patch_size_dMRI),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # )
        self.to_patch_embedding_fMRI = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)',
                      p1 = patch_size_fMRI, p2 = patch_size_fMRI, p3 = patch_size_fMRI),
            nn.LayerNorm(patch_dim_fMRI),
            nn.Linear(patch_dim_fMRI, dim),
            nn.LayerNorm(dim),
        )
        # self.inverse_to_patch_embedding = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, patch_dim),
        #     nn.LayerNorm(patch_dim),
        #     Rearrange(' b (h w d) (p1 p2 p3 c) -> b c (h p1) (w p2) (d p3)',
        #               h=image_height_dMRI // patch_size_dMRI,
        #               w=image_width_dMRI // patch_size_dMRI,
        #               d= image_depth_dMRI // patch_size_dMRI,
        #               p1=patch_size_dMRI, p2=patch_size_dMRI, p3=patch_size_dMRI),
        # )
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

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.pos_embedding_fMRI = nn.Parameter(torch.randn(1, num_patches_fMRI, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer_fMRI = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
    #
    # def coff(self, arr, mask):
    #        mask_flatten = mask.view(arr.shape[0], arr.shape[1],-1)
    #        mean_region = torch.zeros(arr.shape[0],arr.shape[1],1)
    #        for cur_region in range(1,161,1):
    #            current_region_index = (mask_flatten == cur_region).float()
    #            cur_region_value = torch.mean(arr[current_region_index],dim=2,keepdim=True)
    #            mean_region= torch.cat((mean_region,cur_region_value),dim=2)
    #        mean_region= mean_region[:,:,1:]
    #        corr = torch.corrcoef(mean_region)
    #        return corr


    def forward(self, img_f):
        # x_d = self.to_patch_embedding(img_d)
        # b_d, n_d, _ = x_d.shape
        # x_d += self.pos_embedding[:, :(n_d)]
        # x_d = self.dropout(x_d)
        # x_d = self.transformer(x_d)
        # x_d = self.inverse_to_patch_embedding(x_d)
        # d_flatten = x_d.view(b_d, n_d,-1)
        # sc = self.coff(d_flatten, atlas_d)
        x_f = self.to_patch_embedding_fMRI(img_f)
        b_f, n_f, _ = x_f.shape
        x_f += self.pos_embedding_fMRI[:, :(n_f)]
        x_f = self.dropout(x_f)
        # print(x_f.shape)
        x_f = self.transformer_fMRI(x_f)
        # print(x_f.shape)
        x_f = self.inverse_to_patch_embedding_fMRI(x_f)
        # x_f_flatten = x_f.view(x_f.shape[0], x_f.shape[1], -1)
        # sc = self.coff(x_f_flatten, atlas_f)
        return x_f
