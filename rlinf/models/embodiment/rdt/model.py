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
from collections import OrderedDict

import torch
import torch.nn as nn

from rlinf.models.embodiment.rdt.blocks import (FinalLayer, RDTBlock, TimestepEmbedder,
                               get_1d_sincos_pos_embed_from_grid,
                               get_multimodal_cond_pos_embed)


class RDT(nn.Module):
    """
    机器人扩散变换器 (Robotics Diffusion Transformer) 核心模型

    核心思想：
    1. 将机器人动作预测问题转化为扩散去噪问题
    2. 使用Transformer架构处理多模态条件（语言、图像、状态）
    3. 通过交叉注意力机制融合条件信息
    """

    def __init__(
        self,
        output_dim=128,          # 输出维度：统一动作向量的维度（128维）
        horizon=32,              # 预测时间步长：未来预测多少步动作（默认64步）
        hidden_size=1152,        # 隐藏层维度：Transformer的隐藏层大小
        depth=28,                # Transformer深度：RDT块的数量（1B模型为28层）
        num_heads=16,            # 注意力头数：多头注意力的头数（1B模型为32头）
        max_lang_cond_len=1024,  # 最大语言条件长度：语言指令的最大token数
        img_cond_len=4096,       # 图像条件长度：图像token的总数
        lang_pos_embed_config=None,  # 语言位置编码配置
        img_pos_embed_config=None,   # 图像位置编码配置
        dtype=torch.bfloat16     # 数据类型：使用bfloat16以节省显存
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.dtype = dtype
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config

        # 时间步嵌入器：将扩散时间步t编码为向量
        # 用于告诉模型当前处于扩散过程的哪个阶段
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)

        # 控制频率嵌入器：将机器人控制频率编码为向量
        # 不同数据集可能有不同的控制频率（如10Hz, 30Hz等）
        self.freq_embedder = TimestepEmbedder(hidden_size, dtype=dtype)

        # 输入序列位置编码：用于[时间步token; 频率token; 状态token; 动作tokens]
        # horizon+3 = 1(时间步) + 1(频率) + 1(状态) + horizon(动作)
        # 可训练的正弦余弦位置编码
        self.x_pos_embed = nn.Parameter(
            torch.zeros(1, horizon+3, hidden_size))

        # 语言条件位置编码：为语言指令token添加位置信息
        # 支持变长语言指令（最大1024个token）
        self.lang_cond_pos_embed = nn.Parameter(
            torch.zeros(1, max_lang_cond_len, hidden_size))

        # 图像条件位置编码：为图像token添加位置信息
        # 固定长度：img_history_size * num_cameras * num_patches
        self.img_cond_pos_embed = nn.Parameter(
            torch.zeros(1, img_cond_len, hidden_size))

        # Transformer块堆叠：深度为depth的RDT块
        # 每个RDT块包含：自注意力 + 交叉注意力 + FFN
        self.blocks = nn.ModuleList([
            RDTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # 最终输出层：将隐藏状态映射到动作空间
        self.final_layer = FinalLayer(hidden_size, output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        """
        权重初始化策略

        1. Transformer层：使用Xavier均匀初始化（标准做法）
        2. 位置编码：使用正弦余弦位置编码初始化（类似Transformer）
        3. 最终层：零初始化（扩散模型常见做法，确保训练初期输出接近零）
        """
        # 初始化Transformer层：所有Linear层使用Xavier均匀初始化
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # 初始化输入序列位置编码：使用多模态正弦余弦位置编码
        # 序列结构：[timestep(1) + ctrl_freq(1) + state(1) + action(horizon)]
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict([
                ('timestep', 1),      # 时间步token
                ('ctrl_freq', 1),     # 控制频率token
                ('state', 1),         # 状态token
                ('action', self.horizon),  # 动作tokens（horizon个）
            ])
        )
        # 为什么不能直接赋值？
        # 因为x_pos_embed是nn.Parameter，需要通过data.copy_()赋值
        # 而torch.from_numpy(x_pos_embed).float().unsqueeze(0)是Tensor
        # 所以需要通过data.copy_()赋值
        self.x_pos_embed.data.copy_(
            torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        # 初始化语言条件位置编码
        if self.lang_pos_embed_config is None:
            # 默认：使用1D正弦余弦位置编码（线性序列）
            lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.max_lang_cond_len))
        else:
            # 自定义：使用多模态位置编码（考虑语言token的模态信息）
            lang_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.lang_pos_embed_config),
                embed_modality=False
            )
        self.lang_cond_pos_embed.data.copy_(
            torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))

        # 初始化图像条件位置编码
        if self.img_pos_embed_config is None:
            # 默认：使用1D正弦余弦位置编码
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.img_cond_len))
        else:
            # 自定义：使用多模态位置编码（考虑图像的时间-相机-空间结构）
            img_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.img_pos_embed_config),
                embed_modality=False
            )
        self.img_cond_pos_embed.data.copy_(
            torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))

        # 初始化时间步和控制频率嵌入的MLP：使用小方差正态分布初始化
        # 这有助于训练稳定性
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)

        # 初始化最终输出层：零初始化（扩散模型标准做法）
        # 这确保训练初期模型输出接近零，避免不稳定的梯度
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

        # 将所有参数转换为指定数据类型（通常为bfloat16以节省显存）
        self.to(self.dtype)

    def forward(self, x, freq, t, lang_c, img_c, lang_mask=None, img_mask=None):
        """
        RDT前向传播

        核心流程：
        1. 将时间步t和控制频率freq嵌入为向量
        2. 拼接形成输入序列：[时间步; 频率; 状态; 动作tokens]
        3. 添加位置编码
        4. 通过Transformer块处理（交替使用语言和图像条件）
        5. 输出动作预测

        参数：
            x: (B, T, D) - 状态+动作token序列，T = horizon + 1
               其中包含1个状态token和horizon个动作tokens
            freq: (B,) - 控制频率标量（如10Hz, 30Hz等）
            t: (B,) 或 (1,) - 扩散时间步（0到num_train_timesteps-1）
            lang_c: (B, L_lang, D) - 语言条件tokens（变长）
                来自T5编码器的输出
            img_c: (B, L_img, D) - 图像条件tokens（固定长度）
                来自SigLIP编码器的输出
            lang_mask: (B, L_lang) - 语言mask（True表示有效token）
            img_mask: (B, L_img) - 图像mask（True表示有效token）

        返回：
            (B, horizon, output_dim) - 预测的动作序列
        """
        # 步骤1：将时间步和控制频率嵌入为向量
        t = self.t_embedder(t).unsqueeze(1)             # (B, 1, D) 或 (1, 1, D)
        freq = self.freq_embedder(freq).unsqueeze(1)    # (B, 1, D)

        # 如果时间步是广播的（只有1个样本），扩展到批次大小
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)

        # 步骤2：拼接输入序列 [时间步token; 频率token; 状态token; 动作tokens]
        # 最终形状：(B, 1+1+1+horizon, D) = (B, horizon+3, D)
        x = torch.cat([t, freq, x], dim=1)               # (B, T+2, D)

        # 步骤3：添加位置编码
        # 为输入序列添加位置信息
        x = x + self.x_pos_embed

        # 为语言条件添加位置编码（注意：语言是变长的）
        # 只取前lang_c.shape[1]个位置编码
        lang_c = lang_c + self.lang_cond_pos_embed[:, :lang_c.shape[1]]

        # 为图像条件添加位置编码（固定长度）
        img_c = img_c + self.img_cond_pos_embed

        # 步骤4：通过Transformer块处理
        # 交替注入语言和图像条件：奇数层使用语言，偶数层使用图像
        conds = [lang_c, img_c]
        masks = [lang_mask, img_mask]
        for i, block in enumerate(self.blocks):
            # 交替选择条件：i%2=0使用语言，i%2=1使用图像
            c, mask = conds[i % 2], masks[i % 2]
            # RDT块：自注意力 + 交叉注意力（条件注入）+ FFN
            x = block(x, c, mask)                       # (B, T+2, D)

        # 步骤5：通过最终输出层映射到动作空间
        x = self.final_layer(x)                         # (B, T+2, output_dim)

        # 步骤6：只保留动作tokens（去掉时间步、频率、状态token）
        # 输出形状：(B, horizon, output_dim)
        x = x[:, -self.horizon:]
        return x
