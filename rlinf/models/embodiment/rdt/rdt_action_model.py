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

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.base_policy import BasePolicy


class RDTForRLActionPrediction(BasePolicy):
    """

    
    """
    
    def __init__(
        self,
        config: dict,                      # RDT的配置（从base.yaml加载）
        model_path: str,                   # 预训练权重路径
        output_action_chunks: int = 8,    # 输出动作chunk数量（与GR00T对齐）
        denoising_steps: int = 5,         # 推理时去噪步数
        obs_converter_type: str = "libero", # 观测转换器类型
        torch_dtype: torch.dtype = torch.bfloat16,  # 数据类型
    ):
        super().__init__()
        
        self.config = config
        self.model_path = Path(model_path)
        self.output_action_chunks = output_action_chunks
        self.denoising_steps = denoising_steps
        self.torch_dtype = torch_dtype
        
        # 提取常用的配置项
        self.common_cfg = config['common']
        self.model_cfg = config['model']
        self.dataset_cfg = config['dataset']
        
        print("📌 Step 1/3: Initializing multimodal encoders...")
        self._init_encoders()
        
        print("📌 Step 2/3: Initializing RDT core model...")
        self._init_rdt_runner()
        
        print("📌 Step 3/3: Setting up converters...")
        self.obs_converter_type = obs_converter_type
        self.obs_convert_fn = self._get_obs_converter(obs_converter_type)
        self.action_convert_fn = self._get_action_converter(obs_converter_type)
        self.action_mask_fn = self._get_action_mask_fn(obs_converter_type)
        self.action_dim = self._get_action_dim(obs_converter_type)
        
        if self.model_path.exists():
            print("📌 Loading pretrained weights...")
            self._load_pretrained_weights()
        
        # 历史帧缓存
        self.history_obs = None
        
        # 统一转换所有参数为目标 dtype (包括冻结的参数)
        # 这是为了确保 FSDP 可以正常工作，它要求所有参数类型一致
        self.to(dtype=self.torch_dtype)
        
        print("✅ RDT model initialization complete!")
    
    def _init_encoders(self):
        """初始化T5和SigLIP编码器"""
        from rlinf.models.embodiment.rdt.multimodal_encoder.t5_encoder import T5Embedder
        from rlinf.models.embodiment.rdt.multimodal_encoder.siglip_encoder import SiglipVisionTower
        
        encoder_cfg = self.config.get('encoders', {})
        
        # 获取本地模型路径
        rdt_dir = Path(__file__).parent
        t5_local_path = str(rdt_dir / 'google' / 't5-v1_1-xxl')
        siglip_local_path = str(rdt_dir / 'google' / 'siglip-so400m-patch14-384')
        
        # T5语言编码器
        self.lang_encoder = T5Embedder(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            from_pretrained=encoder_cfg.get('t5_path', t5_local_path),
            model_max_length=self.dataset_cfg['tokenizer_max_length'],
            torch_dtype=self.torch_dtype,
            local_files_only=True,
        )
        
        # SigLIP视觉编码器
        class Args:
            mm_vision_select_feature = 'patch'
            unfreeze_mm_vision_tower = False
        
        self.vision_encoder = SiglipVisionTower(
            vision_tower=encoder_cfg.get('siglip_path', siglip_local_path),
            args=Args(),
            delay_load=False,
        )
        
        # 冻结编码器参数以节省显存和计算
        # 需要 FSDP 配置中设置 use_orig_params=True 才能支持不同的 requires_grad
        for param in self.lang_encoder.model.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        print(f"  ✓ T5 encoder loaded (frozen)")
        print(f"  ✓ SigLIP encoder loaded (frozen)")
    
    def _init_rdt_runner(self):
        """初始化RDT扩散模型"""
        from rlinf.models.embodiment.rdt.rdt_runner import RDTRunner
        
        # 计算图像条件长度
        img_cond_len = (
            self.common_cfg['img_history_size'] *
            self.common_cfg['num_cameras'] *
            self.vision_encoder.num_patches
        )
        
        # 创建RDT Runner
        self.rdt_runner = RDTRunner(
            action_dim=self.common_cfg['state_dim'],
            pred_horizon=self.common_cfg['action_chunk_size'],
            config=self.model_cfg,
            lang_token_dim=self.model_cfg['lang_token_dim'],
            img_token_dim=self.model_cfg['img_token_dim'],
            state_token_dim=self.model_cfg['state_token_dim'],
            max_lang_cond_len=self.dataset_cfg['tokenizer_max_length'],
            img_cond_len=img_cond_len,
            # 图像位置编码配置：(历史帧数, 相机数, -patch数)
            # 负号表示不使用初始位置编码，因为 ViT 已经包含了位置信息
            img_pos_embed_config=[
                ("image", (
                    self.common_cfg['img_history_size'],  # 2
                    self.common_cfg['num_cameras'],       # 3
                    -self.vision_encoder.num_patches      # -729
                )),
            ],
            # 语言位置编码配置：负号表示不使用初始位置编码
            lang_pos_embed_config=[
                ("lang", -self.dataset_cfg['tokenizer_max_length']),  # -1024
            ],
            dtype=self.torch_dtype,
        )
        
        # 设置推理去噪步数（只使用DPM-Solver）
        self.rdt_runner.noise_scheduler_sample.set_timesteps(self.denoising_steps)
        
        print(f"  ✓ RDT Runner initialized")
        print(f"    - Output action chunks: {self.output_action_chunks}")
        print(f"    - Denoising steps: {self.denoising_steps}")
        print(f"    - img_cond_len: {img_cond_len} (history={self.common_cfg['img_history_size']} × cameras={self.common_cfg['num_cameras']} × patches={self.vision_encoder.num_patches})")
        print(f"    - max_lang_cond_len: {self.dataset_cfg['tokenizer_max_length']}")

    def _get_obs_converter(self, obs_converter_type: str):
        """获取观测转换函数"""
        OBS_CONVERSION = {
            "libero": convert_libero_obs_to_rdt_format,
        }
        
        if obs_converter_type not in OBS_CONVERSION:
            raise ValueError(f"Unknown obs_converter_type: {obs_converter_type}")
        
        return OBS_CONVERSION[obs_converter_type]
    
    def _get_action_converter(self, obs_converter_type: str):
        """获取动作转换函数"""
        ACTION_CONVERSION = {
            "libero": extract_libero_action,
        }
        
        if obs_converter_type not in ACTION_CONVERSION:
            raise ValueError(f"Unknown obs_converter_type: {obs_converter_type}")
        
        return ACTION_CONVERSION[obs_converter_type]
    
    def _get_action_mask_fn(self, obs_converter_type: str):
        """获取动作mask生成函数"""
        ACTION_MASK_FN = {
            "libero": get_libero_action_mask,
        }
        
        if obs_converter_type not in ACTION_MASK_FN:
            raise ValueError(f"Unknown obs_converter_type: {obs_converter_type}")
        
        return ACTION_MASK_FN[obs_converter_type]
    
    def _get_action_dim(self, obs_converter_type: str) -> int:
        """获取环境特定的动作维度"""
        ACTION_DIM = {
            "libero": 7,  # 6 EEF velocities + 1 gripper
        }
        
        if obs_converter_type not in ACTION_DIM:
            raise ValueError(f"Unknown obs_converter_type: {obs_converter_type}")
        
        return ACTION_DIM[obs_converter_type]
    
    def _load_pretrained_weights(self):
        """加载预训练权重"""
        print(f"Loading weights from {self.model_path}")
        
        if (self.model_path / "pytorch_model.bin").exists():
            checkpoint = torch.load(
                self.model_path / "pytorch_model.bin", 
                map_location='cpu',
                weights_only=False  # PyTorch 2.6+ compatibility
            )
            
            if 'module' in checkpoint:
                self.rdt_runner.load_state_dict(checkpoint['module'], strict=False)
            else:
                self.rdt_runner.load_state_dict(checkpoint, strict=False)
            
            self.rdt_runner = self.rdt_runner.to(dtype=self.torch_dtype)
            
            print("✓ Pretrained weights loaded")
        else:
            print("⚠ No pretrained weights found, using random initialization")

    def preprocess_env_obs(self, env_obs):
        """
        预处理环境观测（基类方法override）
        
        RDT 不需要特殊预处理，直接返回环境观测。
        RDT 需要的键会在 predict_action_batch 中通过 obs_convert_fn 处理。
        """
        return env_obs
    # TODO
    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: Dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """
        预测动作
        
        流程：
        1. 观测转换 + 历史帧处理
        2. 编码器提取特征
        3. RDT扩散采样生成动作
        4. 动作转换输出
        """
        device = next(self.parameters()).device
        
        # 步骤1: 观测格式转换（使用 image_processor）
        # 从 config 读取 auto_adjust_brightness 设置
        auto_adjust_brightness = self.dataset_cfg.get('auto_adjust_image_brightness', False)
        
        # DEBUG: 打印输入观测信息（首次）
        if not hasattr(self, '_input_debug_done'):
            self._input_debug_done = True
            print(f"\n[RDT INPUT DEBUG]")
            print(f"  main_images shape: {env_obs['main_images'].shape}")  # Should be (B, H, W, C)
            print(f"  wrist_images shape: {env_obs['wrist_images'].shape}")
            print(f"  states_joint shape: {env_obs['states_joint'].shape}")  # Should be (B, 9)
            print(f"  states_joint[0]: {env_obs['states_joint'][0].cpu().numpy()}")
            print(f"  states_joint gripper[0,-2:]: {env_obs['states_joint'][0,-2:].cpu().numpy()}")
            print(f"  auto_adjust_brightness: {auto_adjust_brightness}")
        
        rdt_obs = self.obs_convert_fn(
            env_obs, 
            self.history_obs,
            image_processor=self.vision_encoder.image_processor,
            auto_adjust_brightness=auto_adjust_brightness
        )
        
        # 更新历史帧缓存
        self.history_obs = {
            "main_images": env_obs["main_images"].clone(),
            "wrist_images": env_obs["wrist_images"].clone(),
        }
        
        batch_size = rdt_obs["video.image"].shape[0]
        
        # 步骤2: 多模态编码
        # 语言编码
        instructions = rdt_obs["annotation.human.action.task_description"]
        text_embeds, lang_attn_mask = self.lang_encoder.get_text_embeddings(instructions)
        text_embeds = text_embeds.to(device, dtype=self.torch_dtype)
        lang_attn_mask = lang_attn_mask.to(device).bool()
        
        # 图像编码（已经由 image_processor 归一化，格式为 (B, T, C, H, W)）
        main_images = rdt_obs["video.image"].to(device, dtype=self.torch_dtype)
        wrist_images = rdt_obs["video.wrist_image"].to(device, dtype=self.torch_dtype)
        
        # 组织图像: [t-1_main, t-1_wrist, t-1_dummy, t_main, t_wrist, t_dummy]
        images_to_encode = []
        num_actual_cameras = 2  # LIBERO: main + wrist
        
        for t in range(2):
            images_to_encode.append(main_images[:, t])
            images_to_encode.append(wrist_images[:, t])
            
            if self.common_cfg['num_cameras'] > num_actual_cameras:
                dummy_image = torch.zeros_like(main_images[:, t])
                images_to_encode.append(dummy_image)
        
        all_images = torch.stack(images_to_encode, dim=1)
        B, T_CAM, C, H, W = all_images.shape
        all_images = all_images.reshape(B * T_CAM, C, H, W)
        
        image_embeds = self.vision_encoder(all_images)
        
        num_patches = image_embeds.shape[1]
        image_embeds = image_embeds.reshape(
            batch_size, T_CAM * num_patches, self.vision_encoder.hidden_size
        )
        
        # 状态编码
        state_tokens = torch.from_numpy(rdt_obs["state.joint_states"]).to(device, dtype=self.torch_dtype)
        
        # 准备action_mask
        action_mask = self.action_mask_fn(
            batch_size=batch_size,
            state_dim=self.common_cfg['state_dim'],
            device=device,
            dtype=self.torch_dtype
        )
        
        # 控制频率
        ctrl_freqs = torch.full((batch_size,), 20.0, device=device, dtype=self.torch_dtype)
        
        # 步骤3: RDT扩散采样 (使用DPM-Solver)
        actions = self.rdt_runner.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=lang_attn_mask,
            img_tokens=image_embeds,
            state_tokens=state_tokens,
            action_mask=action_mask,
            ctrl_freqs=ctrl_freqs,
        )  # (B, pred_horizon, 128)
        
        # 步骤4: 动作转换
        # 取前 output_action_chunks 个动作
        actions_cpu = actions[:, :self.output_action_chunks].cpu()
        if actions_cpu.dtype == torch.bfloat16:
            actions_cpu = actions_cpu.float()
        actions_numpy = actions_cpu.numpy()
        
        # DEBUG: 打印维度和统计信息（首次或每100步打印一次）
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        if self._debug_counter % 100 == 0:
            print(f"\n[RDT DEBUG - Step {self._debug_counter}]")
            print(f"  actions shape: {actions_numpy.shape}")  # Should be (B, 8, 128)
            print(f"  actions[0,0,10] (gripper): {actions_numpy[0, 0, 10]:.4f}")
            print(f"  actions[0,0,39:45] (eef vel): {actions_numpy[0, 0, 39:45]}")
            print(f"  actions range: [{actions_numpy.min():.4f}, {actions_numpy.max():.4f}]")
        self._debug_counter += 1
        
        # 传递 chunk_size 给 action_convert_fn
        raw_actions = self.action_convert_fn(actions_numpy, chunk_size=self.output_action_chunks)
        
        # DEBUG: 打印转换后的动作
        if self._debug_counter % 100 == 1:
            print(f"  raw_actions shape: {raw_actions.shape}")  # Should be (B, 8, 7)
            print(f"  raw_actions[0,0] (7D): {raw_actions[0, 0]}")
            print(f"  gripper after binarize: {raw_actions[0, 0, -1]:.4f}")
        

        # TODO
        result = {
            "prev_logprobs": torch.zeros(batch_size, self.output_action_chunks, device=device),
            "prev_values": torch.zeros(batch_size, 1, device=device),  # [B, 1]
            "forward_inputs": {},  # 占位符
        }
        return raw_actions, result


# ======================
# LIBERO <-> RDT 转换函数
# ======================

# LIBERO在128维向量中的索引映射
LIBERO_STATE_VEC_IDX_MAPPING = {
    **{f'arm_joint_{i}_pos': i for i in range(7)},
    'gripper_joint_0_pos': 10,
    'gripper_joint_1_pos': 11,
    'gripper_open': 10,
    'eef_vel_x': 39,
    'eef_vel_y': 40,
    'eef_vel_z': 41,
    'eef_angular_vel_roll': 42,
    'eef_angular_vel_pitch': 43,
    'eef_angular_vel_yaw': 44,
}

def fill_in_libero_state(values: np.ndarray, state_dim: int = 128) -> np.ndarray:
    """将LIBERO的9维状态填充到RDT的128维向量中"""
    joint_indices = [LIBERO_STATE_VEC_IDX_MAPPING[f'arm_joint_{i}_pos'] for i in range(7)]
    gripper_indices = [LIBERO_STATE_VEC_IDX_MAPPING['gripper_joint_0_pos'],
                       LIBERO_STATE_VEC_IDX_MAPPING['gripper_joint_1_pos']]
    
    uni_vec = np.zeros(values.shape[:-1] + (state_dim,), dtype=values.dtype)
    uni_vec[..., joint_indices] = values[..., 0:7]
    uni_vec[..., gripper_indices] = values[..., 7:9]
    
    return uni_vec

def get_libero_action_mask(batch_size: int, state_dim: int = 128, device='cpu', dtype=torch.float32) -> torch.Tensor:
    """生成LIBERO的action mask"""
    action_mask = torch.zeros(batch_size, 1, state_dim, device=device, dtype=dtype)
    action_mask[:, :, 39:45] = 1.0  # EEF velocities
    action_mask[:, :, 10] = 1.0  # Gripper
    return action_mask

def extract_libero_action(uni_action: np.ndarray, chunk_size: int = 1) -> np.ndarray:
    """
    从RDT的128维动作向量中提取LIBERO的7维动作
    
    Args:
        uni_action: shape (B, chunk_size, 128) 或 (B, 128)
        chunk_size: 输出的动作chunk数量
    
    Returns:
        libero_action: shape (B, chunk_size, 7) 或 (B, 7)
    
    Note:
        与 Libero_RDT 项目的 eval 代码保持一致：
        - gripper < 0 → -1 (闭合)
        - gripper >= 0 → 1 (打开)
    """
    eef_indices = [39, 40, 41, 42, 43, 44]
    eef_actions = uni_action[..., eef_indices]  # (B, chunk_size, 6)
    
    gripper_action = uni_action[..., 10:11]  # (B, chunk_size, 1)
    
    # 二值化 gripper 动作（与 Libero_RDT eval 一致）
    # gripper_open: 负值表示闭合 (-1)，正值表示打开 (1)
    gripper_action = np.where(
        gripper_action < 0,
        -1.0,
        1.0
    )
    
    libero_action = np.concatenate([eef_actions, gripper_action], axis=-1)  # (B, chunk_size, 7)
    
    return libero_action

def expand2square_np(image: np.ndarray, background_color: np.ndarray) -> np.ndarray:
    """将图像padding到正方形（防止resize时变形）"""
    H, W = image.shape[:2]
    if H == W:
        return image
    elif W > H:
        result = np.full((W, W, image.shape[2]), background_color, dtype=image.dtype)
        pad_top = (W - H) // 2
        result[pad_top:pad_top+H, :] = image
        return result
    else:
        result = np.full((H, H, image.shape[2]), background_color, dtype=image.dtype)
        pad_left = (H - W) // 2
        result[:, pad_left:pad_left+W] = image
        return result


def convert_libero_obs_to_rdt_format(env_obs, history_obs=None, image_processor=None, 
                                     auto_adjust_brightness=False):
    """
    将LIBERO观测转换为RDT格式（与 Libero_RDT eval 对齐）
    
    处理流程：
    0. **反转图像** - RLinf 环境会翻转 180°，但 Libero_RDT 训练时未翻转，需要反转回来
    1. 可选：亮度调整（暗图增强）
    2. expand2square - padding到正方形（防止变形）
    3. image_processor.preprocess() - 归一化到 [0,1]
    
    Args:
        env_obs: 环境观测 (main_images, wrist_images, states_joint, task_descriptions)
        history_obs: 历史观测（可选）
        image_processor: SigLIP 图像预处理器
        auto_adjust_brightness: 是否自动调整亮度（暗图增强）
    
    Note:
        RLinf 的 LiberoEnv 使用 get_libero_image() 会翻转图像 180°，
        但 Libero_RDT 训练时未翻转，所以这里需要反转回来以保持一致性。
    """
    from PIL import Image
    from torchvision import transforms
    
    rdt_obs = {}
    
    # 处理历史帧
    if history_obs is None:
        prev_main = env_obs["main_images"]
        prev_wrist = env_obs["wrist_images"]
    else:
        prev_main = history_obs["main_images"]
        prev_wrist = history_obs["wrist_images"]
    
    # 拼接历史帧和当前帧
    main_stacked = torch.stack([prev_main, env_obs["main_images"]], dim=1).cpu().numpy()
    wrist_stacked = torch.stack([prev_wrist, env_obs["wrist_images"]], dim=1).cpu().numpy()
    
    # ⚠️ 关键：反转图像以匹配 Libero_RDT 训练数据
    # RLinf 环境在 get_libero_image() 中已经翻转了 180°，
    # 但 Libero_RDT 训练时使用的是未翻转的原始 LIBERO 图像，
    # 所以这里需要反转回来
    # 测试结果证明：双重翻转（回到原图）是正确的
    main_stacked = main_stacked[:, :, ::-1, ::-1, :].copy()  # (B, T, H, W, C)
    wrist_stacked = wrist_stacked[:, :, ::-1, ::-1, :].copy()  # (B, T, H, W, C)
    
    # 获取背景色（用于 padding）
    if image_processor is not None:
        background_color = tuple(int(x * 255) for x in image_processor.image_mean)
    else:
        background_color = (int(0.5 * 255), int(0.5 * 255), int(0.5 * 255))
    
    # 创建背景图像（用于占位）
    if image_processor is not None:
        background_image = np.ones((
            image_processor.size["height"], 
            image_processor.size["width"], 3
        ), dtype=np.uint8) * np.array(background_color, dtype=np.uint8).reshape(1, 1, 3)
    else:
        background_image = None
    
    # 定义 expand2square 函数（PIL 版本）
    def expand2square_pil(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    # 处理所有图像
    B, T, H, W, C = main_stacked.shape
    main_processed_list = []
    wrist_processed_list = []
    
    for b in range(B):
        batch_main_images = []
        batch_wrist_images = []
        
        for t in range(T):
            # 处理主相机图像
            main_img = Image.fromarray(main_stacked[b, t])
            
            # 可选：亮度调整
            if auto_adjust_brightness:
                pixel_values = list(main_img.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    main_img = transforms.ColorJitter(brightness=(1.75, 1.75))(main_img)
            
            # expand2square + preprocess
            main_img = expand2square_pil(main_img, background_color)
            if image_processor is not None:
                main_tensor = image_processor.preprocess(main_img, return_tensors='pt')['pixel_values'][0]
            else:
                # 回退方案：手动处理
                main_img = main_img.resize((384, 384))
                main_tensor = torch.from_numpy(np.array(main_img)).float() / 255.0
                main_tensor = main_tensor.permute(2, 0, 1)
            batch_main_images.append(main_tensor)
            
            # 处理腕部相机图像
            wrist_img = Image.fromarray(wrist_stacked[b, t])
            
            if auto_adjust_brightness:
                pixel_values = list(wrist_img.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    wrist_img = transforms.ColorJitter(brightness=(1.75, 1.75))(wrist_img)
            
            wrist_img = expand2square_pil(wrist_img, background_color)
            if image_processor is not None:
                wrist_tensor = image_processor.preprocess(wrist_img, return_tensors='pt')['pixel_values'][0]
            else:
                wrist_img = wrist_img.resize((384, 384))
                wrist_tensor = torch.from_numpy(np.array(wrist_img)).float() / 255.0
                wrist_tensor = wrist_tensor.permute(2, 0, 1)
            batch_wrist_images.append(wrist_tensor)
        
        # (T, C, H, W)
        main_processed_list.append(torch.stack(batch_main_images, dim=0))
        wrist_processed_list.append(torch.stack(batch_wrist_images, dim=0))
    
    # (B, T, C, H, W)
    rdt_obs["video.image"] = torch.stack(main_processed_list, dim=0)
    rdt_obs["video.wrist_image"] = torch.stack(wrist_processed_list, dim=0)
    
    # 状态转换
    states_9d = env_obs["states_joint"].unsqueeze(1).cpu().numpy()

    # gripper归一化：RDT要求
    gripper_min = -0.04245
    gripper_max = 0.05185
    states_9d[..., -2:] = (states_9d[..., -2:] - gripper_min) / \
        (gripper_max - gripper_min)

    states_128d = fill_in_libero_state(states_9d, state_dim=128)
    rdt_obs["state.joint_states"] = states_128d
    
    # 任务描述
    rdt_obs["annotation.human.action.task_description"] = env_obs["task_descriptions"]
    
    return rdt_obs
