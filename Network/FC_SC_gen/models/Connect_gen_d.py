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
import torchvision.models as models
# from my_model_transformer import generator
from opts import parse_opts
# from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from collections import OrderedDict
from torch.nn.functional import interpolate
from einops.layers.torch import Rearrange
from torch.autograd import grad as torch_grad
def hinge_gen_loss(fake):
    return -fake.mean()
def log(t, eps=1e-10):
    return torch.log(t + eps)
def vgg():
        vgg = torchvision.models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        vgg.classifier = nn.Sequential(*vgg.classifier[:-2])
        _vgg = vgg.cuda()
        return _vgg
def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()
class MoELayer(nn.Module):
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
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),  # Expansion
                nn.GELU(),  # Activation
                nn.Linear(d_ff, d_model)  # Compression
            ) for _ in range(1)
        ])
        # Gating network to determine expert weights
        self.gate = nn.Linear(d_model, 1)
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
        gate_logits = self.gate(flat_x)  # [batch*seq, num_experts]
        # Convert to probabilities via softmax
        gate_probs = F.softmax(gate_logits, dim=-1)  # [batch*seq, num_experts]
        # Select top-k experts for each token
        topk_probs, topk_indices  = torch.topk(gate_probs, 1, dim=-1)  # both [batch*seq, k]
        # Normalize top-k probabilities
        topk_probs  = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # [batch*seq, k]
        # Initialize output tensor
        out  = torch.zeros_like(flat_x)  # [batch*seq, d_model]
        # Sparse computation: Only process activated experts
        for expert_id, expert in enumerate(self.experts ):
            # Create mask for tokens selecting current expert
            expert_mask = (topk_indices == expert_id).any(dim=-1)  # [batch*seq]

            if expert_mask.any():
                # Get probability weights for current expert
                prob = topk_probs [expert_mask,
                (topk_indices[expert_mask] == expert_id).nonzero()[:, 1]]

                # Compute and weight expert outputs
                expert_out = expert(flat_x[expert_mask])
                out [expert_mask] += expert_out * prob.unsqueeze(-1)
        out =  out
        # sim =

        # Restore original shape
        return out.view(batch_size, seq_len, -1)

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)
#
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#
#         self.norm = nn.LayerNorm(dim)
#
#         self.attend = nn.Softmax(dim = -1)
#         self.dropout = nn.Dropout(dropout)
#
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
#
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#
#     def forward(self, x):
#         x = self.norm(x)
#
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
#
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#
#         attn = self.attend(dots)
#         attn = self.dropout(attn)
#
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)
#
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.layers = nn.ModuleList([])
#         # self.moe = MoELayer(dim, dim_head, heads)
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
#                 FeedForward(dim, mlp_dim, dropout = dropout),
#             ]))
#
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#
#         return self.norm(x)
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
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                MoELayer(dim, mlp_dim)
                # FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, lay in self.layers:
            x = attn(x) + x
            x = lay(x) + x

        return self.norm(x)
class Transformer_discr(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                MoELayer(dim, mlp_dim)
                # FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, moe in self.layers:
            x = attn(x) + x
            x = moe(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_height, patch_size, dim, depth, heads, mlp_dim, pool='cls', channels=1,
                 image_width=160, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)
        patch_height = image_height
        patch_width = patch_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.inverse_to_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange(' b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=image_height // patch_height, w=image_width // patch_width,
                      p1=patch_height, p2=patch_width)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Linear(dim, num_classes)
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.inverse_to_patch_embedding(x)
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #
        # x = self.to_latent(x)
        return x
class ViT_discr(nn.Module):
    def __init__(self, *, image_height, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 1, image_width=160, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)
        patch_height = image_height
        patch_width = patch_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.inverse_to_patch_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange(' b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=image_height // patch_height, w=image_width // patch_width,
                      p1=patch_height, p2=patch_width)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_discr(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.inverse_to_patch_embedding(x)
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        #
        # x = self.to_latent(x)
        return x

# class Conn_gen(nn.Module):
#     def __init__(self, *, dim, depth, heads, mlp_dim, channels=1,
#                   dim_head=64, dropout=0., emb_dropout=0.):
#         super().__init__()
#         # image_height, image_width = pair(image_size)
#         # patch_height, patch_width = pair(patch_size)
#         patch_size = 16
#         image_height, image_width = 160,160
#         num_patches = ((image_height// patch_size) * (image_width // patch_size))
#
#         patch_dim = channels * patch_size * patch_size
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
#                       p1 = patch_size, p2 = patch_size),
#             nn.LayerNorm(patch_dim),
#             nn.Linear(patch_dim, dim),
#             nn.LayerNorm(dim),
#         )
#         self.inverse_to_patch_embedding = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, patch_dim),
#             nn.LayerNorm(patch_dim),
#             Rearrange(' b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
#                       h=image_height // patch_size,
#                       w=image_width // patch_size,
#                       p1=patch_size, p2=patch_size),
#         )
#
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
#         # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)
#
#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
#         # self.pool = pool
#         # self.to_latent = nn.Identity()
#
#     def forward(self, raw_con):
#         x = self.to_patch_embedding(raw_con)
#         b_d, n_d, _ = x.shape
#         x += self.pos_embedding[:, :(n_d)]
#         x = self.dropout(x)
#         x = self.transformer(x)
#         refine_connect = self.inverse_to_patch_embedding(x)
#         return refine_connect
# class generator(nn.Module):
#     def __init__(self, *, dim, depth, heads, mlp_dim, channels=1,
#                   dim_head=64, dropout=0., emb_dropout=0.):
#         super().__init__()
#         self.encoder = Conn_gen(dim=dim, depth=int(depth), heads=heads, mlp_dim=mlp_dim)
#         self.decoder = Conn_gen(dim=dim, depth=int(depth), heads=heads, mlp_dim=mlp_dim)
#     def forward(self, raw_con):
#         x = self.encoder(raw_con)
#         x = self.decoder(x)
#         return x
# class discriminator(nn.Module):
#     def __init__(self, *, dim, depth, heads, mlp_dim, channels=1,
#                   dim_head=64, dropout=0., emb_dropout=0.):
#         super().__init__()
#         self.encoder = Conn_gen(dim=dim, depth=int(depth*1.5), heads=heads, mlp_dim=mlp_dim)
#         self.decoder = Conn_gen(dim=dim, depth=int(depth*1.5), heads=heads, mlp_dim=mlp_dim)
#     def forward(self, raw_con):
#         x = self.encoder(raw_con)
#         x = self.decoder(x)
#         return x
class Transformer_encoder(nn.Module):
    def  __init__(self, opt, dim=256):
        super(Transformer_encoder, self).__init__()
        self.opt = opt
        self.dim = dim
        # self.codebook_size = codebook_size
        self.ViT_CEB = ViT(image_height=17, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d, depth=opt.depth_refine_d,
                       heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_CON = ViT(image_height=32, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d, depth=opt.depth_refine_d,
                       heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_DMN = ViT(image_height=34, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d, depth=opt.depth_refine_d,
                       heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_OCC = ViT(image_height=21, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d, depth=opt.depth_refine_d,
                       heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_FPN = ViT(image_height=22, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d, depth=opt.depth_refine_d,
                       heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_SMA = ViT(image_height=34, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d, depth=opt.depth_refine_d,
                       heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        # self.vae = VQGanVAE(dim=self.dim, opt=opt)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.apply_grad_penalty_every =4
        self.grad_accum_every = 1
        self.use_ema = True
    def forward(self, x):

            # torch.cuda.empty_cache()
            # shape_res_W = x.shape[3]
            # x_squeeze = x_res[0][1].squeeze()
        # loss_dirsc, x = self.vae(x_res[0][1], return_discr_loss= True, return_recons = True)
        # loss_auto, x = self.vae(x_res[0][1], return_loss=True, return_recons=True)
        x_encode_CEB = self.ViT_CEB(x[:,:,0:17,...])
        x_encode_CON = self.ViT_CON(x[:, :, 17:49, ...])
        x_encode_DMN = self.ViT_DMN(x[:, :, 49:83, ...])
        x_encode_OCC = self.ViT_OCC(x[:, :, 83:104, ...])
        x_encode_FPN = self.ViT_FPN(x[:, :, 104:126, ...])
        x_encode_SMA= self.ViT_SMA(x[:, :, 126:, ...])
        x_encode = torch.concatenate((x_encode_CEB,x_encode_CON,x_encode_DMN,x_encode_OCC,x_encode_FPN,x_encode_SMA),dim=2)
        # x_encode = self.ViT(x)
        # with torch.no_grad():
        #     checkpoint = torch.load(opt.resume_path)
        #     opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
        #     assert opt.arch == checkpoint['arch']
        #     new_state_dict = OrderedDict()
        #     for k,v in checkpoint['state_dict'].items():
        #         name=k[7:]
        #         new_state_dict[name]=v
        #     self.classifier.load_state_dict(new_state_dict)
        #     x = self.classifier(x.unsqueeze(4))
        # loss_ce = self.criterion(x, x_res[1])
        return x_encode
class Transformer_decoder(nn.Module):
    def  __init__(self, opt):
        super(Transformer_decoder, self).__init__()
        self.opt = opt
        # self.codebook_size = codebook_size
        # self.ViT = ViT(image_size=160, patch_size=opt.patch_size_refine, num_classes=opt.n_classes, dim=opt.dim_refine depth=opt.depth_refine_d,
        #                heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_CEB = ViT(image_height=17, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d,
                           depth=opt.depth_refine_d,
                           heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_CON = ViT(image_height=32, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d,
                           depth=opt.depth_refine_d,
                           heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_DMN = ViT(image_height=34, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d,
                           depth=opt.depth_refine_d,
                           heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_OCC = ViT(image_height=21, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d,
                           depth=opt.depth_refine_d,
                           heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_FPN = ViT(image_height=22, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d,
                           depth=opt.depth_refine_d,
                           heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_SMA = ViT(image_height=34, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d,
                           depth=opt.depth_refine_d,
                           heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.apply_grad_penalty_every =4
        self.grad_accum_every = 1
        self.use_ema = True
    def last_dec_layer(self):
        return self.ViT.transformer[-1].weight
    def forward(self, x):
        # x_res = x.copy()
            # torch.cuda.empty_cache()
            # shape_res_W = x.shape[3]
            # x_squeeze = x_res[0][1].squeeze()
        # loss_dirsc, x = self.vae(x_res[0][1], return_discr_loss= True, return_recons = True)
        # loss_auto, x = self.vae(x_res[0][1], return_loss=True, return_recons=True)
        # x_decode = self.ViT(x)
        x_decode_CEB = self.ViT_CEB(x[:,:,0:17,...])
        x_decode_CON = self.ViT_CON(x[:, :, 17:49, ...])
        x_decode_DMN = self.ViT_DMN(x[:, :, 49:83, ...])
        x_decode_OCC = self.ViT_OCC(x[:, :, 83:104, ...])
        x_decode_FPN = self.ViT_FPN(x[:, :, 104:126, ...])
        x_decode_SMA= self.ViT_SMA(x[:, :, 126:, ...])
        x_decode = torch.concatenate((x_decode_CEB,x_decode_CON,x_decode_DMN,x_decode_OCC,x_decode_FPN,x_decode_SMA),dim=2)
        # with torch.no_grad():
        #     checkpoint = torch.load(opt.resume_path)
        #     opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
        #     assert opt.arch == checkpoint['arch']
        #     new_state_dict = OrderedDict()
        #     for k,v in checkpoint['state_dict'].items():
        #         name=k[7:]
        #         new_state_dict[name]=v
        #     self.classifier.load_state_dict(new_state_dict)
        #     x = self.classifier(x.unsqueeze(4))
        # loss_ce = self.criterion(x, x_res[1])
        return x_decode
class Transformer_dirsc(nn.Module):
    def  __init__(self, opt):
        super(Transformer_dirsc, self).__init__()
        self.opt = opt
        # self.classifier = classifier(BasicBlock, [1, 1, 1, 1], opt)
        # self.codebook_size = codebook_size
        # self.ViT_discr = ViT_discr(image_size=160, patch_size=opt.patch_size_refine, num_classes=opt.n_classes, dim=opt.dim_refine depth=8,
        #                heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        # self.ViT_discr_label = ViT_discr(image_size=160, patch_size=opt.patch_size_refine, num_classes=opt.n_classes, dim=opt.dim_refine depth=8,
        #                heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_CEB = ViT_discr(image_height=17, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d,
                           depth=opt.depth_refine_d,
                           heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_CON = ViT_discr(image_height=32, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d,
                           depth=opt.depth_refine_d,
                           heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_DMN = ViT_discr(image_height=34, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d,
                           depth=opt.depth_refine_d,
                           heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_OCC = ViT_discr(image_height=21, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d,
                           depth=opt.depth_refine_d,
                           heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_FPN = ViT_discr(image_height=22, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d,
                           depth=opt.depth_refine_d,
                           heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.ViT_SMA = ViT_discr(image_height=34, patch_size=opt.patch_size_refine, dim=opt.dim_refine_d,
                           depth=opt.depth_refine_d,
                           heads=opt.heads, mlp_dim=opt.mlp_dim_refine_d)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.apply_grad_penalty_every =4
        self.grad_accum_every = 1
        self.use_ema = True
    def forward(self, x):
        # x_discr = self.ViT_discr(x)
        x_discr_CEB = self.ViT_CEB(x[:, :, 0:17, ...])
        x_discr_CON = self.ViT_CON(x[:, :, 17:49, ...])
        x_discr_DMN = self.ViT_DMN(x[:, :, 49:83, ...])
        x_discr_OCC = self.ViT_OCC(x[:, :, 83:104, ...])
        x_discr_FPN = self.ViT_FPN(x[:, :, 104:126, ...])
        x_discr_SMA = self.ViT_SMA(x[:, :, 126:, ...])
        x_discr = torch.concatenate(
            (x_discr_CEB, x_discr_CON, x_discr_DMN, x_discr_OCC, x_discr_FPN, x_discr_SMA), dim=2)
        return x_discr

# class Conn_discr(nn.Module):
#     def __init__(self, *, dim, depth, heads, mlp_dim, channels=1,
#                   dim_head=64, dropout=0., emb_dropout=0.):
#         super().__init__()
#         # image_height, image_width = pair(image_size)
#         # patch_height, patch_width = pair(patch_size)
#         patch_size = 8
#         image_height, image_width = 160,160
#         num_patches = ((image_height// patch_size) * (image_width // patch_size))
#
#         patch_dim = channels * patch_size * patch_size
#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
#                       p1 = patch_size, p2 = patch_size),
#             nn.LayerNorm(patch_dim),
#             nn.Linear(patch_dim, dim),
#             nn.LayerNorm(dim),
#         )
#         self.inverse_to_patch_embedding = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, patch_dim),
#             nn.LayerNorm(patch_dim),
#             Rearrange(' b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
#                       h=image_height // patch_size,
#                       w=image_width // patch_size,
#                       p1=patch_size, p2=patch_size),
#         )
#
#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
#         # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)
#
#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # self.pool = pool
        # self.to_latent = nn.Identity()

    # def forward(self, raw_con):
    #     x = self.to_patch_embedding(raw_con)
    #     b_d, n_d, _ = x.shape
    #     x += self.pos_embedding[:, :(n_d)]
    #     x = self.dropout(x)
    #     x = self.transformer(x)
    #     refine_connect = self.inverse_to_patch_embedding(x)
    #     return refine_connect
'''

'''
class generator(nn.Module):
    def __init__(self, opt):
        super(generator, self).__init__()
        self.Transformer_encode = Transformer_encoder(opt=opt)
        self.Transformer_decode = Transformer_decoder(opt=opt)
        self.Transformer_Individual_encode = Transformer_encoder(opt=opt)
        self.Transformer_Individual_decode = Transformer_decoder(opt=opt)
    def forward(self, individual, group):
        individual_out  = self.Transformer_Individual_encode(individual)
        # individual_out = self.Transformer_Individual_decode(individual_en)
        group_out  = self.Transformer_encode(group)
        # group_out = self.Transformer_decode(group_en)

        # label_prompt = self.label_generation(target_fc)
        return individual_out + group_out
class discriminator(nn.Module):
    def __init__(self, opt):
        super(discriminator, self).__init__()
        self.Transformer_encode = Transformer_dirsc(opt=opt)
        # self.Transformer_decode = Transformer_dirsc(opt=opt)

    def forward(self, arr):
        out = self.Transformer_encode(arr)
        # out = self.Transformer_decode(en_fc)

        return out

class Connect_gen_discr_d(nn.Module):
    def __init__(self, *, opt):
        super().__init__()
        self.generator = generator(opt=opt)
        self.discriminator = discriminator(opt=opt)
    def region_loss(self,fake, real):
        fake_discr_CEB, fake_discr_CEB_T, real_discr_CEB = fake[:, :, 0:17, ...], fake.permute(0,1,3,2)[:, :, 0:17, ...], real[:, :, 0:17, ...]
        fake_discr_CON, fake_discr_CON_T, real_discr_CON = fake[:, :, 17:49, ...],fake.permute(0,1,3,2)[:, :, 17:49, ...], real[:, :, 17:49, ...]
        fake_discr_DMN, fake_discr_DMN_T, real_discr_DMN = fake[:, :, 49:83, ...], fake.permute(0,1,3,2)[:, :, 49:83, ...], real[:, :, 49:83, ...]
        fake_discr_OCC, fake_discr_OCC_T, real_discr_OCC = fake[:, :, 83:104, ...], fake.permute(0,1,3,2)[:, :, 83:104, ...], real[:, :, 83:104, ...]
        fake_discr_FPN, fake_discr_FPN_T, real_discr_FPN = fake[:, :, 104:126, ...], fake.permute(0,1,3,2)[:, :, 104:126, ...], real[:, :, 104:126, ...]
        fake_discr_SMA, fake_discr_SMA_T, real_discr_SMA = fake[:, :, 126:, ...], fake.permute(0,1,3,2)[:, :, 126:, ...], real[:, :, 126:, ...]
        # region_discr_loss = bce_discr_loss(fake_discr_CEB, real_discr_CEB) + bce_discr_loss(fake_discr_CON, real_discr_CON) + bce_discr_loss(fake_discr_DMN, real_discr_DMN) \
        #                        + bce_discr_loss(fake_discr_OCC, real_discr_OCC) + bce_discr_loss(fake_discr_FPN, real_discr_FPN) + bce_discr_loss(fake_discr_SMA, real_discr_SMA) +\
        #                         + bce_discr_loss(fake_discr_CEB_T, real_discr_CEB) + bce_discr_loss(fake_discr_CON_T, real_discr_CON) + bce_discr_loss(fake_discr_DMN_T, real_discr_DMN) + \
        #                     bce_discr_loss(fake_discr_OCC_T, real_discr_OCC) + bce_discr_loss(fake_discr_FPN_T,real_discr_FPN) + bce_discr_loss(fake_discr_SMA_T, real_discr_SMA)
        region_discr_loss = F.mse_loss(fake_discr_CEB, real_discr_CEB) +F.mse_loss(fake_discr_CON, real_discr_CON) + F.mse_loss(fake_discr_DMN, real_discr_DMN) \
                               + F.mse_loss(fake_discr_OCC, real_discr_OCC) + F.mse_loss(fake_discr_FPN, real_discr_FPN) + F.mse_loss(fake_discr_SMA, real_discr_SMA) + \
                          F.mse_loss(fake_discr_CEB_T, real_discr_CEB) + F.mse_loss(fake_discr_CON_T,real_discr_CON) + F.mse_loss(fake_discr_DMN_T,real_discr_DMN)+\
                          F.mse_loss(fake_discr_OCC_T,real_discr_OCC) + F.mse_loss(fake_discr_FPN_T,real_discr_FPN) + F.mse_loss(fake_discr_SMA_T,real_discr_SMA)
        return region_discr_loss

    def region_loss_gen(self,fake, real):
        fake_discr_CEB, fake_discr_CEB_T, real_discr_CEB = fake[:, :, 0:17, ...], fake.permute(0,1,3,2)[:, :, 0:17, ...], real[:, :, 0:17, ...]
        fake_discr_CON, fake_discr_CON_T, real_discr_CON = fake[:, :, 17:49, ...],fake.permute(0,1,3,2)[:, :, 17:49, ...], real[:, :, 17:49, ...]
        fake_discr_DMN, fake_discr_DMN_T, real_discr_DMN = fake[:, :, 49:83, ...], fake.permute(0,1,3,2)[:, :, 49:83, ...], real[:, :, 49:83, ...]
        fake_discr_OCC, fake_discr_OCC_T, real_discr_OCC = fake[:, :, 83:104, ...], fake.permute(0,1,3,2)[:, :, 83:104, ...], real[:, :, 83:104, ...]
        fake_discr_FPN, fake_discr_FPN_T, real_discr_FPN = fake[:, :, 104:126, ...], fake.permute(0,1,3,2)[:, :, 104:126, ...], real[:, :, 104:126, ...]
        fake_discr_SMA, fake_discr_SMA_T, real_discr_SMA = fake[:, :, 126:, ...], fake.permute(0,1,3,2)[:, :, 126:, ...], real[:, :, 126:, ...]
        region_gen_loss = F.mse_loss(fake_discr_CEB, real_discr_CEB) +F.mse_loss(fake_discr_CON, real_discr_CON) + F.mse_loss(fake_discr_DMN, real_discr_DMN) \
                               + F.mse_loss(fake_discr_OCC, real_discr_OCC) + F.mse_loss(fake_discr_FPN, real_discr_FPN) + F.mse_loss(fake_discr_SMA, real_discr_SMA) + \
                          F.mse_loss(fake_discr_CEB_T, real_discr_CEB) + F.mse_loss(fake_discr_CON_T,real_discr_CON) + F.mse_loss(fake_discr_DMN_T,real_discr_DMN)+\
                          F.mse_loss(fake_discr_OCC_T,real_discr_OCC) + F.mse_loss(fake_discr_FPN_T,real_discr_FPN) + F.mse_loss(fake_discr_SMA_T,real_discr_SMA)
        return region_gen_loss
    def forward(self, raw_con, conn_tar,group_mat):
        gen_fc = self.generator(raw_con,group_mat)
        # gen_fc  = torch.clamp(gen_fc , -1, 1)

        # print(gen_fc.shape)
        '''
        perceptual_loss needed
         MoE loss 0.34
        '''
        # img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c=3),
        #                                         (gen_fc.float(), conn_tar.float().cuda()))
        # vgg_ = vgg()
        # img_vgg_feats = vgg_(img_vgg_input)
        # recon_vgg_feats = vgg_(fmap_vgg_input)
        # perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)

        # b_mean = gen_fc.mean(dim=0, keepdim=True)
        # b_mean_group = group_mat.mean(dim=0, keepdim=True)
        # fmap_discr_logits, img_discr_logits = map(self.discriminator, (gen_fc, conn_tar))
        # fmap_discr_logits_group, img_discr_logits_group = map(self.discriminator, (b_mean, b_mean_group))

        # fmap_discr_logits_, img_discr_logits_ = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c=3),
        #                                         (fmap_discr_logits.float(), img_discr_logits))
        # fmap_discr_logits_group_, img_discr_logits_group_ = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c=3),
        #                                         (fmap_discr_logits_group.float(), img_discr_logits_group))
        # fmap_discr_logits_vgg = vgg_(fmap_discr_logits_)
        # img_discr_logits_vgg = vgg_(img_discr_logits_)
        # fmap_discr_logits_vgg_group = vgg_(fmap_discr_logits_group_)
        # img_discr_logits_vgg_group = vgg_(img_discr_logits_group_)
        # perceptual_loss_discr = F.mse_loss(fmap_discr_logits_vgg, img_discr_logits_vgg) + F.mse_loss(
        #     fmap_discr_logits_vgg_group, img_discr_logits_vgg_group)

        # gen_loss = hinge_gen_loss(self.discriminator(gen_fc))
        # gen_loss_group = hinge_gen_loss(self.discriminator(b_mean_group))
        # loss_gen = ((F.mse_loss(gen_fc.float(), conn_tar.cuda().float())  + gen_loss)  +self.region_loss_gen(gen_fc, conn_tar) \
        #             + F.mse_loss(b_mean,b_mean_group) + self.region_loss_gen(b_mean,b_mean_group)) + gen_loss_group
        # discr_loss = (
        #          F.mse_loss(fmap_discr_logits, img_discr_logits) + F.mse_loss(fmap_discr_logits_group, img_discr_logits_group)
        #              +  self.region_loss(fmap_discr_logits_group, img_discr_logits_group)
        #               + self.region_loss(fmap_discr_logits, img_discr_logits)
        #               )
        '''
        discard group loss, only retain region loss, if not work
        '''
        return gen_fc