# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO:想要用DDPM去做log probs的计算，以支持RL后训练

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.modules.value_head import ValueHead

# from rlinf.models.embodiment.rdt.rdt_runner import RDTRunner
# from rlinf.models.embodiment.rdt.multimodal_encoder.t5_encoder import T5Embedder
# from rlinf.models.embodiment.rdt.multimodal_encoder.siglip_encoder import SiglipVisionTower


class RDTForRLActionPrediction(BasePolicy):
    """
    RDT扩散模型适配到RLinf的RL训练框架（与 diffusers DDPMScheduler 完全一致）
    
    设计思路：
    1. RDT本身是扩散模型（行为克隆），我们要让它支持RL训练
    2. RLinf期望的接口：default_forward() 和 predict_action_batch()
    3. 关键挑战：扩散模型不直接输出logprobs，需要转换
    
    架构：
    - 多模态编码器（T5 + SigLIP）-> 条件特征
    - RDT扩散模型 -> 动作生成
    - Value Head (可选) -> Critic估值
    
    关键修复（与 diffusers 完全一致）：
    ==========================================
    1. **ddpm_posterior_mean_variance()**：
       - 支持三种 prediction_type: epsilon, sample, v_prediction
       - 使用 scheduler._get_variance() 获取方差（与 scheduler.step 一致）
       - 使用稳定的后验均值公式（写法 B）：
         mu = sqrt(abar_prev)*x0_pred + sqrt(max(1-abar_prev-variance, 0))*eps_pred
       - 支持跳步 timesteps（t 和 prev_t 不连续）
       - 处理 prev_t=-1（最后一步，abar_prev=1）
    
    2. **compute_diffusion_step_with_logprob()**：
       - 正确获取 prev_t：scheduler.timesteps[t_idx+1] 或 -1
       - 使用 ddpm_posterior_mean_variance 计算 mu, sigma
       - train 模式：x_next = mu + sigma * randn；eval 模式：x_next = mu
    
    3. **predict_action_batch(mode="train")**：
       - 只用 DDPM scheduler 采样并记录 chain
       - 只对有效动作维度计算 logprob：idx=[39,40,41,42,43,44,10]
       - 保存 timesteps 用于 default_forward
    
    4. **default_forward()**：
       - 使用 rollout 保存的 timesteps（不重新生成）
       - 对每一步 (x_t, x_next) 重算 mu, sigma
       - 计算 log p(x_next|mu, sigma)，只对有效维度
       - Sanity check：验证 logprobs 与 prev_logprobs 的差异
    
    5. **extract_libero_action()**：
       - Gripper 保持连续值（clip 到 [-1, 1]），不二值化
       - 确保 logprob 对应同一连续动作
    """
    
    def __init__(
        self,
        config: dict,                      # RDT的配置（从base.yaml加载）
        model_path: str,                   # 预训练权重路径
        num_action_chunks: int = 64,      # 动作chunk数量
        denoising_steps: int = 5,         # 推理时去噪步数
        add_value_head: bool = False,     # 是否添加value head
        obs_converter_type: str = "libero", # 观测转换器类型
        torch_dtype: torch.dtype = torch.bfloat16,  # 数据类型
    ):
        super().__init__()
        
        self.config = config
        self.model_path = Path(model_path)
        self.num_action_chunks = num_action_chunks
        self.denoising_steps = denoising_steps
        self.torch_dtype = torch_dtype
        self.add_value_head = add_value_head
        
        # 提取常用的配置项（避免重复写config['xxx']）
        self.common_cfg = config['common']
        self.model_cfg = config['model']
        self.dataset_cfg = config['dataset']
        
        print("📌 Step 1/4: Initializing multimodal encoders...")
        self._init_encoders()
        
        print("📌 Step 2/4: Initializing RDT core model...")
        self._init_rdt_runner()
        
        if add_value_head:
            print("📌 Step 3/4: Initializing Value Head...")
            self._init_value_head()
        
        print("📌 Step 4/4: Setting up converters...")
        self.obs_converter_type = obs_converter_type
        self.obs_convert_fn = self._get_obs_converter(obs_converter_type)
        self.action_convert_fn = self._get_action_converter(obs_converter_type)
        self.action_mask_fn = self._get_action_mask_fn(obs_converter_type)
        self.action_dim = self._get_action_dim(obs_converter_type)
        
        if self.model_path.exists():
            print("📌 Loading pretrained weights...")
            self._load_pretrained_weights()
        
        # 历史帧缓存：用于存储 t-1 帧（RDT需要2帧图像）
        self.history_obs = None
        
        print("✅ RDT model initialization complete!")
    
    def _init_encoders(self):
        """
        初始化T5和SigLIP编码器
        """
        from rlinf.models.embodiment.rdt.multimodal_encoder.t5_encoder import T5Embedder
        from rlinf.models.embodiment.rdt.multimodal_encoder.siglip_encoder import SiglipVisionTower
        
        encoder_cfg = self.config.get('encoders', {})
        
        # 获取本地模型路径（使用相对于当前文件的路径）
        rdt_dir = Path(__file__).parent
        t5_local_path = str(rdt_dir / 'google' / 't5-v1_1-xxl')
        siglip_local_path = str(rdt_dir / 'google' / 'siglip-so400m-patch14-384')
        
        # T5语言编码器：文本 -> 4096维特征
        # 注意：需要修改 T5Embedder 以支持本地路径
        # 临时方案：直接传递本地路径，但需要修改 T5Embedder 的 assertion
        self.lang_encoder = T5Embedder(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            from_pretrained=encoder_cfg.get('t5_path', t5_local_path),
            model_max_length=self.dataset_cfg['tokenizer_max_length'],
            torch_dtype=self.torch_dtype,
            local_files_only=True,  # 使用本地文件
        )
        
        # SigLIP视觉编码器：图像 -> 1152维特征
        # 创建 args 对象（SiglipVisionTower 需要）
        class Args:
            mm_vision_select_feature = 'patch'
            unfreeze_mm_vision_tower = False
        
        self.vision_encoder = SiglipVisionTower(
            vision_tower=encoder_cfg.get('siglip_path', siglip_local_path),
            args=Args(),
            delay_load=False,
        )
        
        # 冻结编码器参数（重要！）
        for param in self.lang_encoder.model.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        print(f"  ✓ T5 encoder loaded (frozen)")
        print(f"  ✓ SigLIP encoder loaded (frozen)")
    
    def _init_rdt_runner(self):
        """
        初始化RDT扩散模型
        
        思考：
        - RDT需要什么输入？语言、图像、状态条件
        - RDT输出什么？动作序列（通过扩散采样）
        """
        # 延迟导入
        from rlinf.models.embodiment.rdt.rdt_runner import RDTRunner
        
        # 计算图像条件的总长度
        # 公式：历史帧数 × 相机数 × 每张图的patch数
        img_cond_len = (
            self.common_cfg['img_history_size'] *    # 例如：2帧历史
            self.common_cfg['num_cameras'] *          # 例如：3个相机
            self.vision_encoder.num_patches          # 例如：729个patches
        )
        
        # 创建RDT Runner
        self.rdt_runner = RDTRunner(
            action_dim=self.common_cfg['state_dim'],           # 128维动作
            pred_horizon=self.common_cfg['action_chunk_size'], # 64步预测
            config=self.model_cfg,
            lang_token_dim=self.model_cfg['lang_token_dim'],   # 4096
            img_token_dim=self.model_cfg['img_token_dim'],     # 1152
            state_token_dim=self.model_cfg['state_token_dim'], # 128
            max_lang_cond_len=self.dataset_cfg['tokenizer_max_length'],
            img_cond_len=img_cond_len,
            dtype=self.torch_dtype,
        )
        
        # 设置推理时的去噪步数
        # 思考：为什么要设置？
        # - 训练时用1000步DDPM
        # - 推理时用5步DPM-Solver（更快）
        self.rdt_runner.noise_scheduler_sample.set_timesteps(self.denoising_steps)
        
        # 为 RL 训练添加一个 DDPM scheduler（用于 rollout 时计算 log probs）
        # 使用 DDPM 而不是 DPM-Solver 的原因：
        # 1. DDPM 每步都有明确的均值和方差，可以计算 log p(x_{t-1}|x_t)
        # 2. DPM-Solver 是确定性 ODE 求解器，没有方差信息
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        
        noise_scheduler_config = self.model_cfg['noise_scheduler']
        self.ddpm_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],  # 1000
            beta_schedule=noise_scheduler_config['beta_schedule'],  # 'squaredcos_cap_v2'
            prediction_type=noise_scheduler_config['prediction_type'],  # 'sample'
            clip_sample=noise_scheduler_config['clip_sample'],  # False
        )
        # 设置 RL 训练时的采样步数（10~50 步）
        self.ddpm_scheduler.set_timesteps(self.denoising_steps)
        
        print(f"  ✓ RDT Runner initialized")
        print(f"    - Action chunks: {self.num_action_chunks}")
        print(f"    - Denoising steps (inference): {self.denoising_steps}")
        print(f"    - DDPM scheduler added for RL training (log prob calculation)")
    
    # TODO:value_input_dim 这个东西不懂
    def _init_value_head(self):
        """
        初始化Value Head用于Critic
        
        思考：
        Q: 输入是什么？
        A: 拼接所有特征：语言特征 + 图像特征 + 状态
        
        Q: 输出是什么？
        A: 单个标量，表示状态价值V(s)
        """
        # 计算输入维度：使用 mean pooling 后的特征维度
        # lang_feat: mean of (B, L_lang, 4096) -> (B, 4096)
        # img_feat: mean of (B, L_img, 1152) -> (B, 1152) 
        # state_feat: flatten of (B, 1, 128) -> (B, 128)
        value_input_dim = (
            self.model_cfg['lang_token_dim'] +        # 4096
            self.model_cfg['img_token_dim'] +         # 1152 (mean pooling)
            self.common_cfg['state_dim']              # 128
        )  # Total: 4096 + 1152 + 128 = 5376
        
        # 创建Value Head（3层MLP）
        self.value_head = ValueHead(
            input_dim=value_input_dim,
            hidden_sizes=(2048, 1024, 512),  # 逐步降维
            output_dim=1,
            activation="relu",
            bias_last=True,
        )
        
        print(f"  ✓ Value Head initialized (input_dim={value_input_dim})")

    def _get_obs_converter(self, obs_converter_type: str):
        """获取观测转换函数"""
        OBS_CONVERSION = {
            "libero": convert_libero_obs_to_rdt_format,
            # 可以添加其他环境的转换
            # "droid": convert_droid_obs_to_rdt_format,
            # "aloha": convert_aloha_obs_to_rdt_format,
        }
        
        if obs_converter_type not in OBS_CONVERSION:
            raise ValueError(f"Unknown obs_converter_type: {obs_converter_type}")
        
        return OBS_CONVERSION[obs_converter_type]
    
    def _get_action_converter(self, obs_converter_type: str):
        """获取动作转换函数"""
        ACTION_CONVERSION = {
            "libero": extract_libero_action,
            # 可以添加其他环境的转换
            # "droid": extract_droid_action,
            # "aloha": extract_aloha_action,
        }
        
        if obs_converter_type not in ACTION_CONVERSION:
            raise ValueError(f"Unknown obs_converter_type: {obs_converter_type}")
        
        return ACTION_CONVERSION[obs_converter_type]
    
    def _get_action_mask_fn(self, obs_converter_type: str):
        """获取动作 mask 生成函数"""
        ACTION_MASK_FN = {
            "libero": get_libero_action_mask,
            # 可以添加其他环境的 mask 函数
            # "droid": get_droid_action_mask,
            # "aloha": get_aloha_action_mask,
        }
        
        if obs_converter_type not in ACTION_MASK_FN:
            raise ValueError(f"Unknown obs_converter_type: {obs_converter_type}")
        
        return ACTION_MASK_FN[obs_converter_type]
    
    def _get_action_dim(self, obs_converter_type: str) -> int:
        """获取环境特定的动作维度"""
        ACTION_DIM = {
            "libero": 7,  # 6 EEF velocities + 1 gripper
            # "droid": 7,
            # "aloha": 14,  # 双臂
        }
        
        if obs_converter_type not in ACTION_DIM:
            raise ValueError(f"Unknown obs_converter_type: {obs_converter_type}")
        
        return ACTION_DIM[obs_converter_type]
    
    def _load_pretrained_weights(self):
        """加载预训练权重"""
        print(f"Loading weights from {self.model_path}")
        
        # 检查是否是 PyTorch 格式的 checkpoint
        if (self.model_path / "pytorch_model.bin").exists():
            checkpoint = torch.load(
                self.model_path / "pytorch_model.bin", 
                map_location='cpu'
            )
            
            # 加载到 rdt_runner
            if 'module' in checkpoint:
                # DeepSpeed 格式
                self.rdt_runner.load_state_dict(checkpoint['module'], strict=False)
            else:
                # 标准格式
                self.rdt_runner.load_state_dict(checkpoint, strict=False)
            
            # 显式转换 rdt_runner 的 dtype 以匹配输入数据
            self.rdt_runner = self.rdt_runner.to(dtype=self.torch_dtype)
            
            print("✓ Pretrained weights loaded")
        else:
            print("⚠ No pretrained weights found, using random initialization")

    # ========================================================================
    # Log Probability 计算相关方法（参考 GR00T 的 joint_logprob 实现）
    # ========================================================================
    
    def get_logprob_norm(self, sample: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        计算高斯分布的 log probability
        
        公式: log p(x|μ,σ) = -log(σ) - 0.5*log(2π) - 0.5*((x-μ)/σ)²
        
        Args:
            sample: 采样值 x, shape (B, H, D)
            mu: 均值 μ, shape (B, H, D)
            sigma: 标准差 σ, shape (B, H, D)
        
        Returns:
            log_prob: log p(x|μ,σ), shape (B, H, D)
        """
        # 处理 sigma=0 的情况（确定性步骤）
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        
        # 常数项: -log(σ) - 0.5*log(2π)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        
        # 指数项: -0.5*((x-μ)/σ)²
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        
        # 总 log probability
        log_prob = constant_term + exponent_term
        
        # 对于 sigma=0 的情况，设置 log_prob=0（确定性）
        log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        
        return log_prob
    
    def _get_libero_valid_action_indices(self) -> torch.Tensor:
        """
        获取 LIBERO 的有效动作维度索引
        
        Returns:
            indices: torch.Tensor, shape (7,) - [39,40,41,42,43,44,10]
        """
        eef_indices = [39, 40, 41, 42, 43, 44]  # EEF velocities
        gripper_idx = [10]  # Gripper
        return torch.tensor(eef_indices + gripper_idx, dtype=torch.long)
    
    def _extract_valid_action_logprobs(
        self, 
        logprobs: torch.Tensor, 
        device: torch.device
    ) -> torch.Tensor:
        """
        从 128 维 logprobs 中提取 LIBERO 的有效动作维度
        
        关键修复：只对有效动作维度计算 logprob，避免无效维度污染 PPO ratio
        
        LIBERO 的有效动作维度：
        - EEF velocities: indices [39, 40, 41, 42, 43, 44] (6维)
        - Gripper: index [10] (1维)
        - 总共 7 维
        
        为什么重要？
        - 无效维度的 logprob 是常数（因为 action_mask=0），会污染 ratio 计算
        - 只对有效维度计算，确保 PPO ratio 正确
        
        Args:
            logprobs: (B, num_steps, H, 128) 或 (B, H, 128)
            device: torch device
        
        Returns:
            valid_logprobs: (B, num_steps, H, 7) 或 (B, H, 7)
        """
        valid_indices = self._get_libero_valid_action_indices().to(device)
        
        # 处理不同的输入形状
        if logprobs.dim() == 4:  # (B, num_steps, H, 128)
            B, num_steps, H, D = logprobs.shape
            # 使用 index_select 提取有效维度
            valid_logprobs = logprobs.index_select(dim=-1, index=valid_indices)
            assert valid_logprobs.shape == (B, num_steps, H, 7), \
                f"Expected shape (B, num_steps, H, 7), got {valid_logprobs.shape}"
        elif logprobs.dim() == 3:  # (B, H, 128)
            B, H, D = logprobs.shape
            valid_logprobs = logprobs.index_select(dim=-1, index=valid_indices)
            assert valid_logprobs.shape == (B, H, 7), \
                f"Expected shape (B, H, 7), got {valid_logprobs.shape}"
        else:
            raise ValueError(f"Unexpected logprobs shape: {logprobs.shape}")
        
        return valid_logprobs
    
    def ddpm_mean_std_like_diffusers(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        timestep: int,
        scheduler: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完全对齐 diffusers DDPMScheduler.step() 的 mean 和 std 计算
        
        关键对齐点：
        1. 使用 scheduler.previous_timestep() 获取 prev_t（支持跳步）
        2. 使用 diffusers 的系数命名（pred_original_sample_coeff, current_sample_coeff）
        3. 正确处理 variance_type（fixed_small, fixed_small_log, learned_range 等）
        4. t=0 时 std=0（不加噪）
        5. 支持三种 prediction_type: epsilon, sample, v_prediction
        
        Args:
            model_output: 模型预测, (B, H, D)
            x_t: 当前噪声动作, (B, H, D)
            timestep: 当前时间步（int）
            scheduler: DDPMScheduler 实例
        
        Returns:
            mean: 后验均值, (B, H, D)
            std: 后验标准差, (B, H, D)
        """
        device = x_t.device
        dtype = x_t.dtype
        
        # 1. 获取 prev_timestep（使用 scheduler 的方法，支持跳步）
        # 参考 diffusers/schedulers/scheduling_ddpm.py
        if hasattr(scheduler, 'previous_timestep'):
            prev_timestep = scheduler.previous_timestep(timestep)
        else:
            # 备用方案：从 timesteps 中查找
            timesteps_list = scheduler.timesteps.tolist()
            if timestep in timesteps_list:
                idx = timesteps_list.index(timestep)
                if idx + 1 < len(timesteps_list):
                    prev_timestep = timesteps_list[idx + 1]
                else:
                    prev_timestep = -1  # 最后一步
            else:
                # 如果 timestep 不在 timesteps 中，手动计算
                prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_timesteps
        
        # 2. 获取 alpha_cumprod
        alpha_prod_t = scheduler.alphas_cumprod[timestep].to(device=device, dtype=dtype)
        if prev_timestep >= 0:
            alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep].to(device=device, dtype=dtype)
        else:
            # 最后一步（去噪到 x_0）
            alpha_prod_t_prev = torch.tensor(1.0, device=device, dtype=dtype)
        
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        # 3. 根据 prediction_type 计算 pred_original_sample
        # 参考 diffusers/schedulers/scheduling_ddpm.py line 379-397
        prediction_type = scheduler.config.prediction_type
        
        if prediction_type == "epsilon":
            # model_output 是噪声 ε
            pred_original_sample = (x_t - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif prediction_type == "sample":
            # model_output 就是 x_0
            pred_original_sample = model_output
        elif prediction_type == "v_prediction":
            # v = sqrt(abar_t)*eps - sqrt(1-abar_t)*x0
            # => x0 = sqrt(abar_t)*x_t - sqrt(1-abar_t)*v
            pred_original_sample = alpha_prod_t.sqrt() * x_t - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")
        
        # 4. 计算 pred_prev_sample 的均值（与 diffusers 完全一致）
        # 参考 diffusers/schedulers/scheduling_ddpm.py line 404-407
        # 使用 diffusers 的系数命名
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * (1 - alpha_prod_t / alpha_prod_t_prev)) / beta_prod_t
        current_sample_coeff = (alpha_prod_t ** 0.5 * beta_prod_t_prev) / beta_prod_t
        
        pred_prev_sample_mean = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * x_t
        
        # 5. 计算标准差（根据 variance_type）
        # 参考 diffusers/schedulers/scheduling_ddpm.py line 411-430
        variance_type = scheduler.config.variance_type
        
        # 特殊处理：t=0 时不加噪声
        if timestep == 0 or prev_timestep < 0:
            variance = torch.tensor(0.0, device=device, dtype=dtype)
            std = torch.tensor(0.0, device=device, dtype=dtype)
        else:
            if variance_type == "fixed_small":
                # variance = _get_variance(t)
                if hasattr(scheduler, '_get_variance'):
                    variance = scheduler._get_variance(timestep, predicted_variance=None)
                else:
                    # 手动计算（与 diffusers 一致）
                    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
                    variance = torch.clamp(variance, min=1e-20)
                std = variance ** 0.5
                
            elif variance_type == "fixed_small_log":
                # 返回的是 log variance
                if hasattr(scheduler, '_get_variance'):
                    variance = scheduler._get_variance(timestep, predicted_variance=None)
                else:
                    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
                    variance = torch.clamp(variance, min=1e-20)
                # log variance clipping（与 diffusers 一致）
                variance = torch.log(torch.clamp(variance, min=1e-20))
                std = torch.exp(0.5 * variance)
                
            elif variance_type == "learned_range":
                # model_output 包含 variance 信息（需要特殊处理）
                # 这里暂不支持，使用 fixed_small 作为备用
                if hasattr(scheduler, '_get_variance'):
                    variance = scheduler._get_variance(timestep, predicted_variance=None)
                else:
                    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
                    variance = torch.clamp(variance, min=1e-20)
                std = variance ** 0.5
                
            else:
                # 默认：fixed_small
                if hasattr(scheduler, '_get_variance'):
                    variance = scheduler._get_variance(timestep, predicted_variance=None)
                else:
                    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
                    variance = torch.clamp(variance, min=1e-20)
                std = variance ** 0.5
        
        # 6. 转换为 tensor 并广播到 x_t 的形状
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, device=device, dtype=dtype)
        else:
            std = std.to(device=device, dtype=dtype)
        
        if std.dim() == 0:
            std = std * torch.ones_like(pred_prev_sample_mean)
        else:
            while std.dim() < pred_prev_sample_mean.dim():
                std = std.unsqueeze(-1)
            std = std.expand_as(pred_prev_sample_mean)
        
        return pred_prev_sample_mean, std
    
    def compute_diffusion_step_with_logprob(
        self,
        x_t: torch.Tensor,
        t_idx: int,
        timestep: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_attn_mask: torch.Tensor,
        img_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        action_mask: torch.Tensor,
        ctrl_freqs: torch.Tensor,
        scheduler: Any,
        mode: Literal["train", "eval"] = "eval",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行一步扩散去噪并计算 log probability（使用 DDPM）
        
        关键修复：
        1. 使用 ddpm_mean_std() 正确计算后验分布的 mean 和 std（不是 prev_sample）
        2. 支持跳步采样（5-step sampling）
        3. 确保与 scheduler 的方差计算一致
        
        参考 GR00T 的 sample_mean_var_val 方法
        
        Args:
            x_t: 当前噪声动作, (B, H, 128)
            t_idx: 时间步索引（在 timesteps 数组中的位置）
            timestep: 当前时间步的实际值, scalar tensor
            lang_tokens, img_tokens, state_tokens: 条件特征
            action_mask: 动作mask, (B, 1, 128)
            ctrl_freqs: 控制频率, (B,)
            scheduler: DDPMScheduler 实例
            mode: "train" (带随机性) 或 "eval" (确定性)
        
        Returns:
            x_next: 下一步的动作, (B, H, 128)
            mu: 后验均值, (B, H, 128)
            sigma: 后验标准差, (B, H, 128)
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # 准备状态序列：state + action_mask 拼接后通过 state_adaptor
        # state: (B, 1, 128) + mask: (B, 1, 128) -> (B, 1, 256)
        state_with_mask = torch.cat([state_tokens, action_mask], dim=2)
        # 通过 state_adaptor: (B, 1, 256) -> (B, 1, hidden_size=2048)
        state_traj = self.rdt_runner.state_adaptor(state_with_mask)
        
        # 准备动作序列：扩展 action_mask 到所有时间步
        action_mask_expanded = action_mask.expand(-1, self.rdt_runner.pred_horizon, -1)
        # action: (B, H, 128) + mask: (B, H, 128) -> (B, H, 256)
        action_traj = torch.cat([x_t, action_mask_expanded], dim=2)
        # 通过 state_adaptor: (B, H, 256) -> (B, H, hidden_size=2048)
        action_traj = self.rdt_runner.state_adaptor(action_traj)
        
        # 拼接状态和动作轨迹: (B, 1, 2048) + (B, H, 2048) -> (B, 1+H, 2048)
        state_action_traj = torch.cat([state_traj, action_traj], dim=1)
        
        # 适配语言和图像条件特征（state_action_traj 已经在隐藏空间了）
        lang_cond = self.rdt_runner.lang_adaptor(lang_tokens)  # (B, L_lang, hidden_size)
        img_cond = self.rdt_runner.img_adaptor(img_tokens)     # (B, L_img, hidden_size)
        
        # 扩展 timestep 到 batch 维度: (B,)
        # 注意：RDT model 期望 timestep 是 (B,) 或 (1,)，会在内部自动嵌入
        timestep_batch = torch.full((batch_size,), timestep.item(), device=device, dtype=torch.long)
        
        # 模型预测（预测去噪后的 x_0 或噪声）
        model_output = self.rdt_runner.model(
            state_action_traj, ctrl_freqs,
            timestep_batch,  # (B,) 而不是 (B, 1)
            lang_cond, img_cond,
            lang_mask=lang_attn_mask
        )
        
        # 使用与 diffusers 完全对齐的方法计算后验分布的 mean 和 std
        # 关键：DDPMScheduler.step() 的 prev_sample 不是 mean，需要手动计算
        t_value = int(timestep.item())
        mu, sigma = self.ddpm_mean_std_like_diffusers(
            model_output=model_output,
            x_t=x_t,
            timestep=t_value,
            scheduler=scheduler,
        )
        
        # 采样或确定性预测
        if mode == "eval":
            # 确定性采样，无随机性
            x_next = mu
            sigma = torch.zeros_like(mu)  # eval 模式下 sigma=0
        else:
            # 训练模式：采样 x_next ~ N(mu, sigma^2)
            noise = torch.randn_like(mu)
            x_next = mu + noise * sigma
        
        # 应用 action_mask（确保无效维度为 0）
        x_next = x_next * action_mask_expanded
        x_next = x_next.to(state_tokens.dtype)
        
        return x_next, mu, sigma

    # ========================================================================
    # 接口1: Rollout (推理/采样) - predict_action_batch
    # ========================================================================
    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: Dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """
        环境交互时预测动作（完全对齐 RDT 原始输入格式）
        
        流程：
        1. 观测转换 + 历史帧处理（env_obs -> rdt_obs with 2 frames）
        2. 编码器提取特征（2帧×2相机 = 4张图像）
        3. RDT 扩散采样生成动作
        4. 动作转换（RDT格式 -> 环境格式）
        5. 准备训练数据（保存 chains, logprobs, values）
        
        注意：RDT 需要 2 帧图像（历史帧 t-1 + 当前帧 t）
        """
        # ====================================================================
        # 步骤1: 观测格式转换 + 历史帧处理
        # ====================================================================
        device = next(self.parameters()).device
        
        # 转换观测（包含历史帧）
        rdt_obs = self.obs_convert_fn(env_obs, self.history_obs)
        
        # 更新历史帧缓存（保存当前帧作为下一次的历史帧）
        self.history_obs = {
            "main_images": env_obs["main_images"].clone(),
            "wrist_images": env_obs["wrist_images"].clone(),
        }
        
        batch_size = rdt_obs["video.image"].shape[0]
        
        # ====================================================================
        # 步骤2: 多模态编码（参考 train.py line 504-521）
        # ====================================================================
        with torch.no_grad():
            # 2.1 语言编码：List[str] -> (B, L, 4096)
            instructions = rdt_obs["annotation.human.action.task_description"]
            text_embeds, lang_attn_mask = self.lang_encoder.get_text_embeddings(instructions)
            text_embeds = text_embeds.to(device, dtype=self.torch_dtype)
            lang_attn_mask = lang_attn_mask.to(device).bool()
            
            # 2.2 图像编码：组织成 [t-1_main, t-1_wrist, None, t_main, t_wrist, None]
            # 输入：(B, T=2, H, W, C) -> 目标：编码 2帧×2相机 = 4张图像
            
            main_images = torch.from_numpy(rdt_obs["video.image"])    # (B, 2, H, W, C)
            wrist_images = torch.from_numpy(rdt_obs["video.wrist_image"])  # (B, 2, H, W, C)
            
            # 转换为 (B, T, C, H, W) 格式（SigLIP 需要 BCHW）
            main_images = main_images.permute(0, 1, 4, 2, 3).to(device, dtype=self.torch_dtype)
            wrist_images = wrist_images.permute(0, 1, 4, 2, 3).to(device, dtype=self.torch_dtype)
            
            # 组织成 [t-1_main, t-1_wrist, t-1_dummy, t_main, t_wrist, t_dummy] 格式
            # LIBERO 只有 2 个实际相机，需要为第 3 个相机位置添加 dummy
            images_to_encode = []
            num_actual_cameras = 2  # LIBERO: main + wrist
            
            for t in range(2):  # t=0 (历史), t=1 (当前)
                images_to_encode.append(main_images[:, t])   # Camera 1: main
                images_to_encode.append(wrist_images[:, t])  # Camera 2: wrist
                
                # Camera 3: dummy（用零填充，与真实图像尺寸相同）
                if self.common_cfg['num_cameras'] > num_actual_cameras:
                    dummy_image = torch.zeros_like(main_images[:, t])
                    images_to_encode.append(dummy_image)
            
            # 堆叠并展平：(B, 6, C, H, W) -> (B*6, C, H, W)
            all_images = torch.stack(images_to_encode, dim=1)  # (B, 6, C, H, W)
            B, T_CAM, C, H, W = all_images.shape
            all_images = all_images.reshape(B * T_CAM, C, H, W)  # (B*6, C, H, W)
            
            # SigLIP 编码
            image_embeds = self.vision_encoder(all_images)  # (B*6, num_patches, 1152)
            
            # 重塑：(B*6, num_patches, 1152) -> (B, 6*num_patches, 1152)
            num_patches = image_embeds.shape[1]
            image_embeds = image_embeds.reshape(
                batch_size, T_CAM * num_patches, self.vision_encoder.hidden_size
            )  # (B, 6*729=4374, 1152)
            
            # 2.3 状态编码：已经是 128 维，直接使用
            state_tokens = torch.from_numpy(rdt_obs["state.joint_states"]).to(device, dtype=self.torch_dtype)  # (B, 1, 128)
            
            # 准备 action_mask：使用环境特定的 mask 函数
            action_mask = self.action_mask_fn(
                batch_size=batch_size,
                state_dim=self.common_cfg['state_dim'],
                device=device,
                dtype=self.torch_dtype
            )
            
            # 准备控制频率（LIBERO 默认 20Hz）
            ctrl_freqs = torch.full((batch_size,), 20.0, device=device, dtype=self.torch_dtype)
        
        # ====================================================================
        # 步骤3: RDT 扩散采样（手动执行去噪循环并记录 chains 和 logprobs）
        # ====================================================================
        # 3.1 选择 scheduler：训练用 DDPM，推理用 DPM-Solver
        if mode == "train":
            scheduler = self.ddpm_scheduler
            scheduler.set_timesteps(self.denoising_steps)
        else:
            scheduler = self.rdt_runner.noise_scheduler_sample
            scheduler.set_timesteps(self.denoising_steps)
        
        # 3.2 初始化噪声 x_T
        x_t = torch.randn(
            size=(batch_size, self.rdt_runner.pred_horizon, self.common_cfg['state_dim']),
            dtype=self.torch_dtype,
            device=device,
        )
        
        # 3.3 记录 chains, logprobs, denoise_inds
        chains = [x_t]  # [x_T, x_{T-1}, ..., x_0]
        log_probs = []  # 每步的 log probability（只对有效动作维度）
        denoise_inds = []  # 时间步索引
        
        # 初始噪声的 log prob: log N(0, I)
        # 注意：只对有效动作维度计算 logprob
        if mode == "train":
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
            )
            # 提取有效动作维度的 logprob
            initial_log_prob = self._extract_valid_action_logprobs(
                initial_log_prob, device
            )  # (B, H, 7)
            log_probs.append(initial_log_prob)
        
        # 3.4 去噪循环
        timesteps = scheduler.timesteps
        for idx, t in enumerate(timesteps):
            # 记录当前时间步索引
            denoise_inds.append(idx)
            
            # 执行一步去噪并计算 log prob
            x_t, mu_t, sigma_t = self.compute_diffusion_step_with_logprob(
                x_t=x_t,
                t_idx=idx,
                timestep=t,
                lang_tokens=text_embeds,
                lang_attn_mask=lang_attn_mask,
                img_tokens=image_embeds,
                state_tokens=state_tokens,
                action_mask=action_mask,
                ctrl_freqs=ctrl_freqs,
                scheduler=scheduler,
                mode=mode,
            )
            
            # 记录轨迹
            chains.append(x_t)
            
            # 计算 log prob（只对有效动作维度）
            if mode == "train":
                # 特殊处理 t=0：std=0，logprob=0（确定性，不计入 PPO ratio）
                t_value = int(t.item())
                if t_value == 0 or (sigma_t == 0).all():
                    # t=0 时是确定性去噪，logprob 应该为 0
                    log_prob = torch.zeros(batch_size, self.rdt_runner.pred_horizon, 7, device=device, dtype=self.torch_dtype)
                else:
                    # 计算所有维度的 logprob
                    log_prob_full = self.get_logprob_norm(x_t, mu_t, sigma_t)  # (B, H, 128)
                    # 提取有效动作维度 [39-44, 10]
                    log_prob = self._extract_valid_action_logprobs(log_prob_full, device)  # (B, H, 7)
                log_probs.append(log_prob)
        
        # 最终去噪后的动作
        actions = x_t  # (B, pred_horizon=64, 128)
        
        # 堆叠 chains 和 log_probs
        chains = torch.stack(chains, dim=1)  # (B, num_steps+1, H, 128)
        denoise_inds = torch.tensor(denoise_inds, dtype=torch.long, device=device)  # (num_steps,)
        
        if mode == "train":
            log_probs = torch.stack(log_probs, dim=1)  # (B, num_steps+1, H, 7)
            # 只取前 num_action_chunks
            log_probs = log_probs[:, :, :self.num_action_chunks, :]  # (B, num_steps+1, num_action_chunks, 7)
        
        # ====================================================================
        # 步骤4: 动作转换（RDT 128维 -> 环境特定维度）
        # ====================================================================
        # 使用环境特定的动作转换函数
        # BFloat16 需要先转 float32 再转 numpy（numpy 1.26 不支持 bfloat16）
        actions_cpu = actions[:, :self.num_action_chunks].cpu()
        if actions_cpu.dtype == torch.bfloat16:
            actions_cpu = actions_cpu.float()
        actions_numpy = actions_cpu.numpy()  # (B, H, 128)
        raw_actions = self.action_convert_fn(actions_numpy)  # (B, H, action_dim)
        
        # ====================================================================
        # 步骤5: 准备训练数据（用于RLinf的训练流程）
        # ====================================================================
        result = {}
        if mode == "train":
            # 计算 values（如果有 value head）
            if self.add_value_head:
                lang_feat = text_embeds.mean(dim=1)
                img_feat = image_embeds.mean(dim=1)
                state_feat = state_tokens.flatten(1)
                value_input = torch.cat([lang_feat, img_feat, state_feat], dim=-1)
                # 转换为 float32（value_head 的参数是 float32）
                values = self.value_head(value_input.float()).squeeze(-1)
            else:
                values = torch.zeros(batch_size, device=device, dtype=self.torch_dtype)
            
            # 保存用于训练的所有数据
            # 关键：prev_logprobs 只包含有效动作维度 (B, num_steps+1, num_action_chunks, 7)
            result = {
                'prev_logprobs': log_probs.cpu(),  # (B, num_steps+1, num_action_chunks, 7) - 只包含有效维度
                'prev_values': values.cpu(),        # (B,)
                'forward_inputs': {
                    'lang_tokens': text_embeds.cpu(),
                    'lang_attn_mask': lang_attn_mask.cpu(),
                    'img_tokens': image_embeds.cpu(),
                    'state_tokens': state_tokens.cpu(),
                    'action_mask': action_mask.cpu(),
                    'ctrl_freqs': ctrl_freqs.cpu(),
                    'chains': chains.cpu(),          # [x_T, x_{T-1}, ..., x_0], (B, num_steps+1, H, 128)
                    'denoise_inds': denoise_inds.cpu(),  # 时间步索引, (num_steps,)
                    'timesteps': timesteps.cpu(),    # 时间步值, (num_steps,)
                }
            }
        
        return raw_actions, result

    # ========================================================================
    # 接口2: Training (PPO训练) - default_forward
    # ========================================================================
    def default_forward(
        self,
        data: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = True,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        """
        PPO 训练时的前向传播，重新计算 log probabilities
        
        参考 GR00T 的 default_forward 实现
        
        流程：
        1. 从 data 中提取 forward_inputs (来自 predict_action_batch)
        2. 使用保存的 chains 重新前向传播
        3. 计算新的 log probabilities
        4. 返回 logprobs, prev_logprobs, values 用于 PPO 更新
        
        为什么这样能让 PPO ratio 成立？
        ====================================
        
        PPO 需要计算 ratio = exp(log π_new(a|s) - log π_old(a|s))
        
        对于扩散模型，动作 a 是通过多步去噪生成的：
        a = x_0 = f(x_T, x_{T-1}, ..., x_1)
        
        关键点：
        1. **同一转移分布**：rollout 和 update 都计算 p(x_{t-1}|x_t)，使用相同的 chains
        2. **同一动作维度**：只对有效动作维度 [39-44, 10] 计算 logprob，避免无效维度污染
        3. **连续动作空间**：gripper 保持连续值（不二值化），确保 logprob 对应同一动作
        
        具体实现：
        - Rollout (predict_action_batch): 
          * 采样 x_{t-1} ~ N(μ_old, σ_old²)，记录 chains 和 log p_old(x_{t-1}|x_t)
          * 只对有效维度计算 logprob: log p_old(a_valid|s)
        
        - Update (default_forward):
          * 沿着保存的 chains，对同一对 (x_t, x_{t-1}) 重新计算 μ_new, σ_new
          * 计算 log p_new(x_{t-1}|x_t)，只对有效维度: log p_new(a_valid|s)
          * ratio = exp(log p_new - log p_old) 成立，因为：
            - 同一转移分布 p(x_{t-1}|x_t)
            - 同一动作维度（有效维度对齐）
            - 同一动作值（chains 保存了 rollout 时的采样值）
        
        Args:
            data: 包含 forward_inputs 和 prev_logprobs 的字典
            compute_logprobs: 是否计算 log probabilities
            compute_entropy: 是否计算熵（暂不支持）
            compute_values: 是否计算 values
        
        Returns:
            dict with keys:
                - logprobs: 新策略的 log prob, (B, num_action_chunks, 7)
                - prev_logprobs: 旧策略的 log prob, (B, num_action_chunks, 7)
                - values: critic 估值, (B,)
                - entropy: 熵（暂不支持，返回 None）
        """
        # ====================================================================
        # 步骤1: 提取 rollout 时保存的数据
        # ====================================================================
        forward_inputs = data['forward_inputs']
        
        lang_tokens = forward_inputs['lang_tokens']
        lang_attn_mask = forward_inputs['lang_attn_mask']
        img_tokens = forward_inputs['img_tokens']
        state_tokens = forward_inputs['state_tokens']
        action_mask = forward_inputs['action_mask']
        ctrl_freqs = forward_inputs['ctrl_freqs']
        chains = forward_inputs['chains']          # (B, num_steps+1, H, 128)
        denoise_inds = forward_inputs['denoise_inds']  # (num_steps,)
        timesteps = forward_inputs['timesteps']    # (num_steps,)
        
        batch_size = chains.shape[0]
        num_steps = chains.shape[1] - 1  # 去掉初始噪声 x_T
        
        # 获取模型所在的设备
        device = next(self.parameters()).device
        
        # 将所有数据移到模型所在的设备
        chains = chains.to(device)
        lang_tokens = lang_tokens.to(device)
        lang_attn_mask = lang_attn_mask.to(device)
        img_tokens = img_tokens.to(device)
        state_tokens = state_tokens.to(device)
        action_mask = action_mask.to(device)
        ctrl_freqs = ctrl_freqs.to(device)
        timesteps = timesteps.to(device)
        
        # ====================================================================
        # 步骤2: 沿着保存的 chains 重新计算 log probabilities
        # ====================================================================
        # 使用 DDPM scheduler（与 rollout 时保持一致）
        # 关键：必须使用 rollout 时保存的 timesteps，不要重新生成
        scheduler = self.ddpm_scheduler
        # 注意：这里不能调用 set_timesteps，因为会改变 timesteps 序列
        # 直接使用保存的 timesteps
        
        log_probs_list = []
        
        # 初始噪声的 log prob: log N(0, I)
        x_T = chains[:, 0]  # 初始噪声 x_T
        initial_log_prob_full = self.get_logprob_norm(
            x_T, torch.zeros_like(x_T), torch.ones_like(x_T)
        )  # (B, H, 128)
        # 提取有效动作维度
        initial_log_prob = self._extract_valid_action_logprobs(
            initial_log_prob_full, device
        )  # (B, H, 7)
        log_probs_list.append(initial_log_prob)
        
        # 对每一步去噪重新计算 log probability
        for idx in range(num_steps):
            x_t = chains[:, idx]      # x_t (rollout 时的状态)
            x_next = chains[:, idx+1]  # x_{t-1} (rollout 时采样的下一状态)
            t = timesteps[idx]        # 当前时间步（使用 rollout 保存的）
            t_value = int(t.item())
            
            # 重新前向传播，获取当前策略的 mu 和 sigma
            # 关键：使用与 rollout 相同的条件，但策略参数可能已更新
            batch_size = x_t.shape[0]
            device = x_t.device
            
            # 准备状态序列
            state_with_mask = torch.cat([state_tokens, action_mask], dim=2)
            state_traj = self.rdt_runner.state_adaptor(state_with_mask)
            
            action_mask_expanded = action_mask.expand(-1, self.rdt_runner.pred_horizon, -1)
            action_traj = torch.cat([x_t, action_mask_expanded], dim=2)
            action_traj = self.rdt_runner.state_adaptor(action_traj)
            
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
            lang_cond = self.rdt_runner.lang_adaptor(lang_tokens)
            img_cond = self.rdt_runner.img_adaptor(img_tokens)
            
            timestep_batch = torch.full((batch_size,), t_value, device=device, dtype=torch.long)
            
            # 模型预测
            model_output = self.rdt_runner.model(
                state_action_traj, ctrl_freqs,
                timestep_batch,
                lang_cond, img_cond,
                lang_mask=lang_attn_mask
            )
            
            # 计算后验分布（使用与 diffusers 对齐的方法）
            mu_t, sigma_t = self.ddpm_mean_std_like_diffusers(
                model_output=model_output,
                x_t=x_t,
                timestep=t_value,
                scheduler=scheduler,
            )
            
            # 计算 rollout 时采样的 x_next 在当前策略下的 log probability
            # 这就是 PPO 中的 log π_new(a|s)
            # 关键：必须对同一转移分布 p(x_{t-1}|x_t) 计算 logprob
            
            # 特殊处理 t=0：std=0，logprob=0（确定性，不计入 PPO ratio）
            if t_value == 0 or (sigma_t == 0).all():
                # t=0 时是确定性去噪，logprob 应该为 0 或跳过
                log_prob = torch.zeros(batch_size, self.rdt_runner.pred_horizon, 7, device=device, dtype=x_t.dtype)
            else:
                log_prob_full = self.get_logprob_norm(x_next, mu_t, sigma_t)  # (B, H, 128)
                # 提取有效动作维度 [39-44, 10]
                log_prob = self._extract_valid_action_logprobs(log_prob_full, device)  # (B, H, 7)
            
            log_probs_list.append(log_prob)
        
        # 堆叠所有步骤的 log probs
        log_probs = torch.stack(log_probs_list, dim=1)  # (B, num_steps+1, H, 7)
        
        # 只取前 num_action_chunks
        log_probs = log_probs[:, :, :self.num_action_chunks, :]  # (B, num_steps+1, num_action_chunks, 7)
        
        # ====================================================================
        # 步骤3: 计算 values
        # ====================================================================
        if compute_values and self.add_value_head:
            lang_feat = lang_tokens.mean(dim=1)
            img_feat = img_tokens.mean(dim=1)
            state_feat = state_tokens.flatten(1)
            value_input = torch.cat([lang_feat, img_feat, state_feat], dim=-1)
            # 转换为 float32（value_head 的参数是 float32）
            values = self.value_head(value_input.float()).squeeze(-1)
        else:
            values = torch.zeros(batch_size, device=device, dtype=self.torch_dtype)
        
        # ====================================================================
        # 步骤4: 提取旧策略的 log probs（与 GR00T 对齐）
        # ====================================================================
        prev_logprobs = data['prev_logprobs']  # (B, num_steps+1, num_action_chunks, 7)
        
        # 移到正确的设备
        prev_logprobs = prev_logprobs.to(device)
        
        # 确保形状对齐
        assert log_probs.shape == prev_logprobs.shape, \
            f"Shape mismatch: log_probs {log_probs.shape} vs prev_logprobs {prev_logprobs.shape}"
        
        # ====================================================================
        # Sanity Check: 验证 logprob 计算的一致性
        # ====================================================================
        # 如果模型参数没有更新，重新计算的 logprobs 应该与 prev_logprobs 非常接近
        # 这是验证我们的实现是否正确的关键检查
        with torch.no_grad():
            logprob_diff = (log_probs - prev_logprobs).abs().mean().item()
            if logprob_diff > 0.1:  # 允许小的数值误差
                print(f"⚠ Warning: Large logprob difference detected: {logprob_diff:.6f}")
                print(f"  This may indicate:")
                print(f"  1. Model parameters were updated between rollout and update")
                print(f"  2. Different random seeds were used")
                print(f"  3. Implementation inconsistency (需要检查)")
            # 调试信息（可选）
            # print(f"✓ Logprob sanity check: mean_diff={logprob_diff:.6f}")
        
        # 按照 GR00T 的 joint_logprob 方式聚合：
        # 1. 先对 action_dim 求和（dim=-1）
        # 2. 再对 denoise_steps 求平均（dim=1）
        # 参考 GR00T: log_probs.mean(dim=1) 
        log_probs_sum_action = log_probs.sum(dim=-1)  # (B, num_steps+1, num_action_chunks)
        prev_logprobs_sum_action = prev_logprobs.sum(dim=-1)  # (B, num_steps+1, num_action_chunks)
        
        # 对 denoise_steps 求平均
        log_probs_mean = log_probs_sum_action.mean(dim=1)  # (B, num_action_chunks)
        prev_logprobs_mean = prev_logprobs_sum_action.mean(dim=1)  # (B, num_action_chunks)
        
        # ====================================================================
        # 步骤5: 返回结果（与 GR00T 接口对齐）
        # ====================================================================
        # 注意：logprobs 已聚合为 (B, num_action_chunks)
        # - action_dim 求和（7维）
        # - denoise_steps 求平均
        return {
            "logprobs": log_probs_mean.float(),       # 新策略 π_new(a|s), (B, num_action_chunks)
            "prev_logprobs": prev_logprobs_mean.float(),  # 旧策略 π_old(a|s), (B, num_action_chunks)
            "values": values.float(),  # (B,)
            "entropy": None,  # 扩散模型的熵难以计算，暂不支持
        }


# ======================
# LIBERO <-> RDT 转换函数
# ======================

# 定义 LIBERO 在 128 维向量中的索引映射
LIBERO_STATE_VEC_IDX_MAPPING = {
    # [0:7): 7-DoF 关节位置
    **{f'arm_joint_{i}_pos': i for i in range(7)},
    # [10:12): 2 个夹爪关节位置
    'gripper_joint_0_pos': 10,
    'gripper_joint_1_pos': 11,
    'gripper_open': 10,  # alias
    # [39:45): EEF velocities (用于 action)
    'eef_vel_x': 39,
    'eef_vel_y': 40,
    'eef_vel_z': 41,
    'eef_angular_vel_roll': 42,
    'eef_angular_vel_pitch': 43,
    'eef_angular_vel_yaw': 44,
}

def fill_in_libero_state(values: np.ndarray, state_dim: int = 128) -> np.ndarray:
    """
    将 LIBERO 的 9 维状态填充到 RDT 的 128 维向量中
    
    输入：values (B, T, 9) - [joint1-7, gripper1-2]
    输出：uni_vec (B, T, 128) - 128 维统一状态向量
    
    映射规则（参考 RDT hdf5_libero_sft_dataset.py line 362-372）：
    - 关节位置: values[:, :, 0:7] -> uni_vec[:, :, 0:7]
    - 夹爪位置: values[:, :, 7:9] -> uni_vec[:, :, 10:12]
    """
    # 计算目标索引
    joint_indices = [LIBERO_STATE_VEC_IDX_MAPPING[f'arm_joint_{i}_pos'] for i in range(7)]
    gripper_indices = [LIBERO_STATE_VEC_IDX_MAPPING['gripper_joint_0_pos'],
                       LIBERO_STATE_VEC_IDX_MAPPING['gripper_joint_1_pos']]
    
    # 创建零向量
    uni_vec = np.zeros(values.shape[:-1] + (state_dim,), dtype=values.dtype)
    
    # 填充关节位置
    uni_vec[..., joint_indices] = values[..., 0:7]
    # 填充夹爪位置
    uni_vec[..., gripper_indices] = values[..., 7:9]
    
    return uni_vec

def get_libero_action_mask(batch_size: int, state_dim: int = 128, device='cpu', dtype=torch.float32) -> torch.Tensor:
    """
    生成 LIBERO 的 action mask
    
    输入：
        batch_size: batch 大小
        state_dim: 状态/动作空间维度 (默认 128)
        device: torch device
        dtype: torch dtype
    
    输出：
        action_mask (B, 1, 128) - 标记有效的动作维度
    
    LIBERO 有效的动作维度：
    - EEF velocities [39:45]: 6 维
    - Gripper [10]: 1 维
    """
    action_mask = torch.zeros(batch_size, 1, state_dim, device=device, dtype=dtype)
    # EEF velocities 有效
    action_mask[:, :, 39:45] = 1.0  # 6维: x,y,z, roll,pitch,yaw
    # Gripper 有效
    action_mask[:, :, 10] = 1.0  # 1维: gripper open/close
    return action_mask

def extract_libero_action(uni_action: np.ndarray) -> np.ndarray:
    """
    从 RDT 的 128 维动作向量中提取 LIBERO 的 7 维动作
    
    输入：uni_action (B, H, 128) - RDT 预测的 128 维动作
    输出：libero_action (B, H, 7) - LIBERO 需要的 7 维动作
    
    映射规则（参考 RDT hdf5_libero_sft_dataset.py line 380-390）：
    - EEF velocities: uni_action[:, :, 39:45] (6维: x,y,z, roll,pitch,yaw)
    - Gripper: uni_action[:, :, 10] (1维: open/close)
    
    关键修复：gripper 保持连续值（不二值化），确保 logprob 一致性
    - 训练时：gripper 值直接用于环境（或 clip 到 [-1, 1]）
    - 这样可以保证 logprob 对应同一连续动作，PPO ratio 成立
    
    注意：LIBERO 的动作空间是 7 维 EEF velocities + gripper
    """
    # 提取 EEF velocities (6维)
    eef_indices = [
        LIBERO_STATE_VEC_IDX_MAPPING['eef_vel_x'],
        LIBERO_STATE_VEC_IDX_MAPPING['eef_vel_y'],
        LIBERO_STATE_VEC_IDX_MAPPING['eef_vel_z'],
        LIBERO_STATE_VEC_IDX_MAPPING['eef_angular_vel_roll'],
        LIBERO_STATE_VEC_IDX_MAPPING['eef_angular_vel_pitch'],
        LIBERO_STATE_VEC_IDX_MAPPING['eef_angular_vel_yaw'],
    ]
    eef_actions = uni_action[..., eef_indices]  # (B, H, 6)
    
    # 提取 gripper (1维)
    gripper_action = uni_action[..., LIBERO_STATE_VEC_IDX_MAPPING['gripper_open']:LIBERO_STATE_VEC_IDX_MAPPING['gripper_open']+1]  # (B, H, 1)
    
    # 关键修复：gripper 保持连续值，不二值化
    # 使用 tanh 或 clip 确保值在合理范围内，但保持连续
    # 这样 logprob 可以正确计算，PPO ratio 成立
    gripper_action = np.clip(gripper_action, -1.0, 1.0)  # 限制到 [-1, 1]，但保持连续
    
    # 拼接为 7 维
    libero_action = np.concatenate([eef_actions, gripper_action], axis=-1)  # (B, H, 7)
    
    return libero_action

def expand2square_np(image: np.ndarray, background_color: np.ndarray) -> np.ndarray:
    """
    将图像 padding 到正方形（防止 resize 时变形）
    
    参考 RDT dataset.py line 389-404
    
    Args:
        image: (H, W, C) 图像
        background_color: (3,) 背景色
    
    Returns:
        正方形图像
    """
    H, W = image.shape[:2]
    if H == W:
        return image
    elif W > H:
        # 宽度更大，上下 padding
        result = np.full((W, W, image.shape[2]), background_color, dtype=image.dtype)
        pad_top = (W - H) // 2
        result[pad_top:pad_top+H, :] = image
        return result
    else:
        # 高度更大，左右 padding
        result = np.full((H, H, image.shape[2]), background_color, dtype=image.dtype)
        pad_left = (H - W) // 2
        result[:, pad_left:pad_left+W] = image
        return result


def convert_libero_obs_to_rdt_format(env_obs, history_obs=None, target_size=(384, 384), background_color=(0.5, 0.5, 0.5)):
    """
    Convert the observation to the format expected by the RDT model.
    
    与 RDT 训练时的图像处理完全一致：
    1. expand2square - padding 到正方形（防止变形）
    2. resize 到 target_size
    
    Input env_obs:
        - main_images: (B, H, W, C) RGB images (当前帧, uint8 [0-255])
        - wrist_images: (B, H, W, C) wrist camera images (当前帧, uint8 [0-255])
        - states_joint: (B, 9) [joint1-7, gripper1, gripper2]
        - task_descriptions: List[str]
    
    Input history_obs (optional):
        - 如果为 None，则复制当前帧作为历史帧
        - 否则使用 history_obs 作为 t-1 帧
    
    Input target_size:
        - 目标图像尺寸 (H, W)，默认 (384, 384) for SigLIP-so400m-patch14-384
    
    Input background_color:
        - Padding 的背景色，默认 (0.5, 0.5, 0.5) 对应 SigLIP 的 image_mean
    
    Output rdt_obs:
        - video.image: (B, T=2, H, W, C) - [t-1 帧的 main_images, t 帧的 main_images]
        - video.wrist_image: (B, T=2, H, W, C) - [t-1 帧的 wrist_images, t 帧的 wrist_images]
        - state.joint_states: (B, T=1, 128) - 128维统一状态向量（正确映射索引）
        - annotation.human.action.task_description: List[str]
    """
    rdt_obs = {}
    
    # 处理历史帧：如果没有历史，则复制当前帧
    if history_obs is None:
        # 第一次推理，复制当前帧
        prev_main = env_obs["main_images"]
        prev_wrist = env_obs["wrist_images"]
    else:
        prev_main = history_obs["main_images"]
        prev_wrist = history_obs["wrist_images"]
    
    # 图像: 拼接历史帧和当前帧 -> [B, T=2, H, W, C]
    main_stacked = torch.stack([prev_main, env_obs["main_images"]], dim=1).cpu().numpy()  # (B, 2, H, W, C)
    wrist_stacked = torch.stack([prev_wrist, env_obs["wrist_images"]], dim=1).cpu().numpy()  # (B, 2, H, W, C)
    
    # 将背景色转换为 uint8
    bg_color_uint8 = np.array([int(c * 255) for c in background_color], dtype=np.uint8)
    
    # 处理图像：expand2square + resize
    B, T, H, W, C = main_stacked.shape
    target_h, target_w = target_size
    
    main_processed = np.zeros((B, T, target_h, target_w, C), dtype=main_stacked.dtype)
    wrist_processed = np.zeros((B, T, target_h, target_w, C), dtype=wrist_stacked.dtype)
    
    for b in range(B):
        for t in range(T):
            # Main camera: expand2square -> resize
            main_square = expand2square_np(main_stacked[b, t], bg_color_uint8)
            main_processed[b, t] = cv2.resize(
                main_square, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
            
            # Wrist camera: expand2square -> resize
            wrist_square = expand2square_np(wrist_stacked[b, t], bg_color_uint8)
            wrist_processed[b, t] = cv2.resize(
                wrist_square, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
    
    rdt_obs["video.image"] = main_processed
    rdt_obs["video.wrist_image"] = wrist_processed
    
    # 状态: [B, 9] -> [B, T=1, 128] 使用正确的索引映射
    states_9d = env_obs["states_joint"].unsqueeze(1).cpu().numpy()  # (B, 1, 9)
    states_128d = fill_in_libero_state(states_9d, state_dim=128)  # (B, 1, 128)
    rdt_obs["state.joint_states"] = states_128d
    
    # 任务描述
    rdt_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]
    
    return rdt_obs