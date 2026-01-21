# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------


import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn


#################################################################################
#               Embedding Layers for Timesteps and Condition Inptus             #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    时间步嵌入器：将标量时间步编码为向量表示

    用于扩散模型，将扩散时间步t（0-1000）编码为向量，告诉模型当前处于扩散过程的哪个阶段。
    使用正弦余弦位置编码 + MLP的方式。
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        # MLP：将频率嵌入映射到隐藏空间
        # 结构：Linear(256 -> hidden_size) -> SiLU -> Linear(hidden_size -> hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),  # Swish激活函数：x * sigmoid(x)
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        创建正弦余弦时间步嵌入（类似Transformer的位置编码）

        参数：
            t: (N,) - 时间步标量，每个批次元素一个
            dim: 输出维度
            max_period: 控制嵌入的最小频率（周期）

        返回：
            (N, D) - 位置嵌入张量
        """
        # 参考：GLIDE的实现
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2

        # 计算频率：从高频到低频
        # freqs[i] = 1 / (10000^(i/(dim/2)))
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )

        # 计算相位：t * freqs
        # (B, 1) * (1, D/2) -> (B, D/2)
        args = t[:, None].float() * freqs[None]

        # 正弦余弦编码：[cos(t*freq), sin(t*freq)]
        # (B, D/2) -> (B, D)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # 如果维度是奇数，补零
        # (B, D) -> (B, D+1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        """
        前向传播：将时间步编码为向量

        参数：
            t: (B,) 或 (1,) - 扩散时间步

        返回：
            (B, hidden_size) - 时间步嵌入向量
        """
        # 步骤1：正弦余弦编码
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # 步骤2：通过MLP映射到隐藏空间
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                          Cross Attention Layers                               #
#################################################################################
class CrossAttention(nn.Module):
    """
    交叉注意力层：用于注入条件信息（语言、图像）

    核心思想：
    - Query来自主序列x（状态-动作序列）
    - Key和Value来自条件序列c（语言或图像条件）
    - 通过注意力机制，让主序列关注条件信息
    - 支持Flash Attention加速（如果可用）
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,
            proj_drop: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 缩放因子：1/sqrt(head_dim)
        self.fused_attn = use_fused_attn()  # 检查是否支持Flash Attention

        # Query投影：从主序列x生成Q
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # Key-Value投影：从条件序列c生成K和V
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # Query和Key的归一化（可选，用于稳定训练）
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 输出投影
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        交叉注意力前向传播

        参数：
            x: (B, N, C) - 主序列（状态-动作序列）
            c: (B, L, C) - 条件序列（语言或图像条件）
            mask: (B, L) - 条件mask（True表示有效token）

        返回：
            (B, N, C) - 注入条件信息后的主序列
        """
        B, N, C = x.shape  # N = 主序列长度
        _, L, _ = c.shape  # L = 条件序列长度

        # 生成Query：从主序列x生成
        q = self.q(x).reshape(B, N, self.num_heads,
                              # (B, H, N, D_h)
                              self.head_dim).permute(0, 2, 1, 3)

        # 生成Key和Value：从条件序列c生成
        kv = self.kv(c).reshape(B, L, 2, self.num_heads,
                                # (2, B, H, L, D_h)
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # k, v: (B, H, L, D_h)

        # 归一化Query和Key
        q, k = self.q_norm(q), self.k_norm(k)

        # 准备注意力mask：将(B, L)扩展到(B, 1, N, L)
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)

        # 计算注意力（使用Flash Attention或标准实现）
        if self.fused_attn:
            # Flash Attention：更快的融合实现
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            # 标准实现：手动计算注意力
            q = q * self.scale  # 缩放
            attn = q @ k.transpose(-2, -1)  # (B, H, N, L)

            # 应用mask：将无效位置设为-inf
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))

            attn = attn.softmax(dim=-1)  # Softmax归一化

            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)

            x = attn @ v  # 加权求和

        # 重塑并投影输出
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x


#################################################################################
#                                 RDT Block                                     #
#################################################################################
class RDTBlock(nn.Module):
    """
    RDT Transformer块：包含自注意力、交叉注意力和FFN

    架构（类似Transformer，但添加了交叉注意力）：
    1. 自注意力：主序列内部交互
    2. 交叉注意力：注入条件信息（语言或图像）
    3. FFN：特征变换

    使用Pre-Norm架构（归一化在注意力/FFN之前）和残差连接。
    """

    def __init__(self, hidden_size, num_heads, **block_kwargs):
        super().__init__()
        # Pre-Norm + 自注意力：主序列内部交互
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)  # RMS归一化（更稳定）
        self.attn = Attention(
            dim=hidden_size, num_heads=num_heads,
            qkv_bias=True, qk_norm=True,  # 启用QK归一化
            norm_layer=RmsNorm, **block_kwargs)

        # Pre-Norm + 交叉注意力：注入条件信息
        self.cross_attn = CrossAttention(
            hidden_size, num_heads=num_heads,
            qkv_bias=True, qk_norm=True,
            norm_layer=RmsNorm, **block_kwargs)

        # Pre-Norm + FFN：特征变换
        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        def approx_gelu(): return nn.GELU(approximate="tanh")  # 使用tanh近似的GELU（更快）
        self.ffn = Mlp(in_features=hidden_size,
                       hidden_features=hidden_size,  # 隐藏层维度等于输入维度
                       act_layer=approx_gelu, drop=0)
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)

    def forward(self, x, c, mask=None):
        """
        前向传播

        参数：
            x: (B, N, C) - 主序列（状态-动作序列）
            c: (B, L, C) - 条件序列（语言或图像）
            mask: (B, L) - 条件mask

        返回：
            (B, N, C) - 处理后的主序列
        """
        # 子块1：自注意力 + 残差
        origin_x = x
        x = self.norm1(x)  # Pre-Norm
        x = self.attn(x)  # 自注意力
        x = x + origin_x  # 残差连接

        # 子块2：交叉注意力 + 残差（注入条件）
        origin_x = x
        x = self.norm2(x)  # Pre-Norm
        x = self.cross_attn(x, c, mask)  # 交叉注意力
        x = x + origin_x  # 残差连接

        # 子块3：FFN + 残差
        origin_x = x
        x = self.norm3(x)  # Pre-Norm
        x = self.ffn(x)  # 前馈网络
        x = x + origin_x  # 残差连接

        return x


class FinalLayer(nn.Module):
    """
    RDT最终输出层：将隐藏状态映射到动作空间

    结构：RMS归一化 + MLP（映射到输出维度）
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        # MLP：hidden_size -> hidden_size -> out_channels
        self.ffn_final = Mlp(in_features=hidden_size,
                             hidden_features=hidden_size,
                             out_features=out_channels,  # 输出维度：128（动作维度）
                             act_layer=approx_gelu, drop=0)

    def forward(self, x):
        """
        前向传播：将隐藏状态映射到动作空间

        参数：
            x: (B, N, hidden_size) - 隐藏状态

        返回：
            (B, N, out_channels) - 动作预测
        """
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    生成1D正弦余弦位置编码（类似Transformer的位置编码）

    参数：
        embed_dim: 输出维度（必须是偶数）
        pos: 位置列表，形状为(M,)

    返回：
        (M, D) - 位置编码矩阵

    参考：Transformer的原始位置编码实现
    """
    assert embed_dim % 2 == 0

    # 计算频率：omega[i] = 1 / (10000^(i/(D/2)))
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    # 转换为numpy数组
    if not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    pos = pos.reshape(-1)  # (M,)

    # 外积：pos * omega，得到每个位置在每个频率下的相位
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    # 正弦余弦编码
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    # 拼接：[sin, cos]
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim, grid_sizes):
    """
    embed_dim: output dimension for each position
    grid_sizes: the grids sizes in each dimension (K,).
    out: (grid_sizes[0], ..., grid_sizes[K-1], D)
    """
    num_sizes = len(grid_sizes)
    # For grid size of 1, we do not need to add any positional embedding
    num_valid_sizes = len([x for x in grid_sizes if x > 1])
    emb = np.zeros(grid_sizes + (embed_dim,))
    # Uniformly divide the embedding dimension for each grid size
    dim_for_each_grid = embed_dim // num_valid_sizes
    # To make it even
    if dim_for_each_grid % 2 != 0:
        dim_for_each_grid -= 1
    valid_size_idx = 0
    for size_idx in range(num_sizes):
        grid_size = grid_sizes[size_idx]
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [dim_for_each_grid]
        posemb_shape[size_idx] = -1
        emb[..., valid_size_idx * dim_for_each_grid:(valid_size_idx + 1) * dim_for_each_grid] += \
            get_1d_sincos_pos_embed_from_grid(
                dim_for_each_grid, pos).reshape(posemb_shape)
        valid_size_idx += 1
    return emb


def get_multimodal_cond_pos_embed(embed_dim, mm_cond_lens: OrderedDict,
                                  embed_modality=True):
    """
    生成多模态条件的位置编码

    为不同模态（语言、图像等）生成位置编码，同时考虑：
    1. 模态信息：不同模态有不同的嵌入
    2. 位置信息：同一模态内的不同位置

    参数：
        embed_dim: 嵌入维度
        mm_cond_lens: OrderedDict，包含(模态名, 模态token长度)对
            - 对于"image"模态，值可以是多维元组（考虑时间-相机-空间结构）
            - 如果长度 < 0，表示该模态不使用位置编码
        embed_modality: 是否嵌入模态信息（默认True）
            - True: 前半部分嵌入模态信息，后半部分嵌入位置信息
            - False: 整个嵌入都用于位置信息

    返回：
        (总token数, embed_dim) - 多模态位置编码矩阵

    示例：
        mm_cond_lens = OrderedDict([
            ('timestep', 1),
            ('ctrl_freq', 1),
            ('state', 1),
            ('action', 64)
        ])
        将生成4个模态的位置编码，每个模态的token数分别为1, 1, 1, 64
    """
    num_modalities = len(mm_cond_lens)
    modality_pos_embed = np.zeros((num_modalities, embed_dim))

    if embed_modality:
        # 生成模态嵌入：为每个模态分配一个嵌入向量
        # 放在前半部分
        modality_sincos_embed = get_1d_sincos_pos_embed_from_grid(
            embed_dim // 2, torch.arange(num_modalities))
        modality_pos_embed[:, :embed_dim // 2] = modality_sincos_embed
        # 后半部分用于位置嵌入
        pos_embed_dim = embed_dim // 2
    else:
        # 整个嵌入都用于位置信息
        pos_embed_dim = embed_dim

    # 为每个模态内的位置生成嵌入
    c_pos_emb = np.zeros((0, embed_dim))
    for idx, (modality, cond_len) in enumerate(mm_cond_lens.items()):
        if modality == "image" and \
                (isinstance(cond_len, tuple) or isinstance(cond_len, list)):
            # 图像模态：使用多维位置编码（考虑时间-相机-空间结构）
            all_grid_sizes = tuple([abs(x) for x in cond_len])
            embed_grid_sizes = tuple([x if x > 0 else 1 for x in cond_len])
            cond_sincos_embed = get_nd_sincos_pos_embed_from_grid(
                pos_embed_dim, embed_grid_sizes)
            cond_pos_embed = np.zeros(all_grid_sizes + (embed_dim,))
            cond_pos_embed[..., -pos_embed_dim:] += cond_sincos_embed
            cond_pos_embed = cond_pos_embed.reshape((-1, embed_dim))
        else:
            # 其他模态：使用1D位置编码
            cond_sincos_embed = get_1d_sincos_pos_embed_from_grid(
                pos_embed_dim, torch.arange(cond_len if cond_len > 0 else 1))
            cond_pos_embed = np.zeros((abs(cond_len), embed_dim))
            cond_pos_embed[:, -pos_embed_dim:] += cond_sincos_embed

        # 添加模态嵌入
        cond_pos_embed += modality_pos_embed[idx]
        c_pos_emb = np.concatenate([c_pos_emb, cond_pos_embed], axis=0)

    return c_pos_emb
