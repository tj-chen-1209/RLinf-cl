import os
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 跳过 diffusers 的 peft 版本检查
os.environ["_CHECK_PEFT"] = "0"

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

# from models.hub_mixin import CompatiblePyTorchModelHubMixin
from rlinf.models.embodiment.rdt.model import RDT


class RDTRunner(nn.Module):
    """
    扩散模型训练和推理封装类

    核心功能：
    1. 封装RDT模型和条件适配器
    2. 实现扩散训练损失计算（前向扩散过程）
    3. 实现条件采样（反向去噪过程）
    4. 管理噪声调度器（训练用DDPM，推理用DPM-Solver）
    """
    
    # HuggingFace Hub 相关配置
    _repo_url = "https://huggingface.co/robotics-diffusion-transformer/rdt-1b"

    def __init__(self, *, action_dim, pred_horizon, config,
                 lang_token_dim, img_token_dim, state_token_dim,
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None,
                 img_pos_embed_config=None, dtype=torch.bfloat16):
        super(RDTRunner, self).__init__()

        # 创建RDT核心模型：用于预测去噪后的动作
        hidden_size = config['rdt']['hidden_size']
        self.model = RDT(
            output_dim=action_dim,           # 输出维度：128维统一动作向量
            horizon=pred_horizon,            # 预测时间步：64步
            hidden_size=hidden_size,         # 隐藏层大小：2048（1B模型）
            depth=config['rdt']['depth'],    # 深度：28层
            num_heads=config['rdt']['num_heads'],  # 注意力头数：32
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
        )

        # 创建条件适配器：将不同模态的token映射到统一的隐藏空间
        # 语言适配器：T5编码器输出(4096维) -> RDT隐藏空间(2048维)
        self.lang_adaptor = self.build_condition_adapter(
            config['lang_adaptor'],      # 例如：'mlp2x_gelu'（2层MLP）
            in_features=lang_token_dim,  # 4096（T5-XXL输出维度）
            out_features=hidden_size     # 2048
        )

        # 图像适配器：SigLIP编码器输出(1152维) -> RDT隐藏空间(2048维)
        self.img_adaptor = self.build_condition_adapter(
            config['img_adaptor'],       # 例如：'mlp2x_gelu'
            in_features=img_token_dim,   # 1152（SigLIP输出维度）
            out_features=hidden_size     # 2048
        )

        # 状态适配器：状态向量(128维) + mask(128维) -> RDT隐藏空间(2048维)
        # 注意：state在这里指的是状态或动作向量，mask表示有效维度
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'],     # 例如：'mlp3x_gelu'（3层MLP）
            in_features=state_token_dim * 2,  # 128 * 2 = 256（状态+mask）
            out_features=hidden_size         # 2048
        )

        # 创建噪声调度器
        noise_scheduler_config = config['noise_scheduler']

        # 训练用调度器：DDPM（Denoising Diffusion Probabilistic Model）
        # 用于前向扩散过程（添加噪声）和训练损失计算
        self.noise_scheduler = DDPMScheduler(
            # 1000步
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            # 'squaredcos_cap_v2'
            beta_schedule=noise_scheduler_config['beta_schedule'],
            # 'sample'（预测样本）或'epsilon'（预测噪声）
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],  # False（不裁剪样本）
        )

        # 推理用调度器：DPM-Solver（更快的高阶求解器）
        # 用于反向去噪过程（采样生成），只需5步即可完成去噪
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            # 1000
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )

        # 1000
        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        # 5
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        # 'sample'
        self.prediction_type = noise_scheduler_config['prediction_type']

        self.pred_horizon = pred_horizon  # 64
        self.action_dim = action_dim       # 128

        # 打印模型参数量
        print("Diffusion params: %e" % sum(
            [p.numel() for p in self.model.parameters()] +
            [p.numel() for p in self.lang_adaptor.parameters()] +
            [p.numel() for p in self.img_adaptor.parameters()] +
            [p.numel() for p in self.state_adaptor.parameters()]))

    def build_condition_adapter(
            self, projector_type, in_features, out_features):
        """
        构建条件适配器：将不同模态的token映射到统一的隐藏空间

        支持的适配器类型：
        - 'linear': 单层线性映射
        - 'mlpNx_gelu': N层MLP，每层之间使用GELU激活

        例如：'mlp2x_gelu' 表示：
            Linear(in_features -> out_features)
            GELU
            Linear(out_features -> out_features)
        """
        projector = None
        if projector_type == 'linear':
            # 单层线性映射
            projector = nn.Linear(in_features, out_features)
        else:
            # 解析MLP配置：例如 'mlp2x_gelu' -> depth=2
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]  # 第一层
                # 后续层：每层都是 out_features -> out_features
                for _ in range(1, mlp_depth):
                    # GELU激活（使用tanh近似加速）
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector

    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        """
        将不同模态的条件token映射到统一的隐藏空间

        参数：
            lang_tokens: (B, L_lang, lang_token_dim) - T5编码的语言token
            img_tokens: (B, L_img, img_token_dim) - SigLIP编码的图像token
            state_tokens: (B, L_state, state_token_dim) - 状态/动作token（可能包含mask）

        返回：
            (adpated_lang, adpated_img, adpated_state) - 所有token都映射到hidden_size维度
        """
        adpated_lang = self.lang_adaptor(
            lang_tokens)    # (B, L_lang, hidden_size)
        adpated_img = self.img_adaptor(
            img_tokens)      # (B, L_img, hidden_size)
        adpated_state = self.state_adaptor(
            state_tokens)  # (B, L_state, hidden_size)

        return adpated_lang, adpated_img, adpated_state

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond,
                           state_traj, action_mask, ctrl_freqs):
        """
        条件采样：从纯噪声逐步去噪生成动作序列

        扩散采样过程（反向过程）：
        1. 从纯噪声 x_T 开始（T=1000）
        2. 逐步去噪：x_T -> x_{T-1} -> ... -> x_1 -> x_0
        3. 每步使用模型预测去噪方向，使用DPM-Solver快速求解

        参数：
            lang_cond: (B, L_lang, hidden_size) - 语言条件
            lang_attn_mask: (B, L_lang) - 语言mask
            img_cond: (B, L_img, hidden_size) - 图像条件
            state_traj: (B, 1, hidden_size) - 当前状态
            action_mask: (B, 1, action_dim) - 动作mask（0-1浮点张量）
            ctrl_freqs: (B,) - 控制频率

        返回：
            (B, horizon, action_dim) - 生成的动作序列
        """
        device = state_traj.device
        dtype = state_traj.dtype

        # 步骤1：初始化纯噪声动作序列
        # 从标准正态分布采样，形状为 (B, horizon, action_dim)
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_dim),
            dtype=dtype, device=device)

        # 扩展动作mask到所有时间步
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)

        # 步骤2：设置采样时间步（DPM-Solver只需要5步）
        # 从1000个训练时间步中选择5个时间步进行采样
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)

        # 步骤3：迭代去噪过程
        for t in self.noise_scheduler_sample.timesteps:
            # 准备状态-动作序列
            # 将动作和mask拼接，然后通过状态适配器
            # (B, horizon, action_dim*2)
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(
                action_traj)  # (B, horizon, hidden_size)
            # (B, 1+horizon, hidden_size)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)

            # 模型预测：预测去噪后的动作（或噪声）
            model_output = self.model(state_action_traj, ctrl_freqs,
                                      t.unsqueeze(-1).to(device),  # 当前扩散时间步
                                      lang_cond, img_cond, lang_mask=lang_attn_mask)

            # DPM-Solver一步去噪：x_t -> x_{t-1}
            # 使用高阶求解器，可以大幅减少采样步数
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)

        # 步骤4：应用动作mask，将无效动作维度置零
        noisy_action = noisy_action * action_mask

        return noisy_action

    # ========= Train  ============
    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens,
                     state_tokens, action_gt, action_mask, ctrl_freqs
                     ) -> torch.Tensor:
        """
        计算扩散训练损失

        扩散训练过程（前向过程）：
        1. 随机采样时间步 t ~ Uniform(0, T)
        2. 对真实动作添加噪声：x_0 -> x_t（前向扩散）
        3. 模型预测去噪结果（或预测噪声）
        4. 计算预测与目标的MSE损失

        参数：
            lang_tokens: (B, L_lang, lang_token_dim) - 语言token
            lang_attn_mask: (B, L_lang) - 语言mask
            img_tokens: (B, L_img, img_token_dim) - 图像token
            state_tokens: (B, 1, state_token_dim) - 当前状态
            action_gt: (B, horizon, state_token_dim) - 真实动作序列（ground truth）
            action_mask: (B, 1, state_token_dim) - 动作mask（0-1浮点张量）
            ctrl_freqs: (B,) - 控制频率

        返回：
            loss - 标量张量，MSE损失
        """
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device

        # 步骤1：采样噪声（标准正态分布）
        noise = torch.randn(
            action_gt.shape, dtype=action_gt.dtype, device=device
        )

        # 步骤2：随机采样扩散时间步 t ~ Uniform(0, num_train_timesteps)
        # 每个样本的时间步可以不同，增加训练多样性
        timesteps = torch.randint(
            0, self.num_train_timesteps,  # [0, 1000)
            (batch_size,), device=device
        ).long()

        # 步骤3：前向扩散过程：对真实动作添加噪声
        # 根据时间步t的噪声强度，将action_gt加噪得到noisy_action
        # 公式：x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
        noisy_action = self.noise_scheduler.add_noise(
            action_gt, noise, timesteps)

        # 步骤4：准备输入序列
        # 拼接状态和带噪动作
        # (B, 1+horizon, state_token_dim)
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)

        # 扩展动作mask到所有时间步，并拼接到序列中
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        # (B, 1+horizon, state_token_dim*2)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)

        # 步骤5：通过条件适配器映射到统一隐藏空间
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_action_traj)

        # 步骤6：模型预测去噪结果
        # 模型输入：带噪动作序列 + 条件 + 时间步
        pred = self.model(state_action_traj, ctrl_freqs,
                          timesteps, lang_cond, img_cond,
                          lang_mask=lang_attn_mask)

        # 步骤7：确定预测目标
        pred_type = self.prediction_type
        if pred_type == 'epsilon':
            # 预测噪声：模型直接预测添加的噪声
            target = noise
        elif pred_type == 'sample':
            # 预测样本：模型预测去噪后的动作（x_0）
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # 步骤8：计算MSE损失
        loss = F.mse_loss(pred, target)
        return loss

    # ========= Inference  ============
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                       action_mask, ctrl_freqs):
        """
        预测动作序列：推理接口

        流程：
        1. 准备状态和条件（通过适配器映射）
        2. 运行条件采样（从噪声生成动作）

        参数：
            lang_tokens: (B, L_lang, lang_token_dim) - 语言token
            lang_attn_mask: (B, L_lang) - 语言mask
            img_tokens: (B, L_img, img_token_dim) - 图像token
            state_tokens: (B, 1, state_token_dim) - 当前状态
            action_mask: (B, 1, action_dim) - 动作mask
            ctrl_freqs: (B,) - 控制频率

        返回：
            (B, horizon, action_dim) - 预测的动作序列
        """
        # 准备状态：将状态和mask拼接
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)

        # 通过条件适配器映射到统一隐藏空间
        lang_cond, img_cond, state_traj = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens)

        # 运行条件采样：从纯噪声生成动作序列
        action_pred = self.conditional_sample(
            lang_cond, lang_attn_mask, img_cond,
            state_traj, action_mask, ctrl_freqs,
        )

        return action_pred

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播：用于训练时计算损失
        """
        return self.compute_loss(*args, **kwargs)
