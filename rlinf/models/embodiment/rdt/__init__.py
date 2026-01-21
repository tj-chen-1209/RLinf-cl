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

"""
RDT (Robotics Diffusion Transformer) Model for RLinf

Integration of RDT diffusion policy into RLinf RL training framework.
"""

from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from rlinf.models.embodiment.rdt.rdt_action_model import RDTForRLActionPrediction


def get_model(cfg: Dict[str, Any], torch_dtype: torch.dtype = torch.bfloat16):
    """
    加载 RDT 模型用于 RLinf 训练框架
    
    Args:
        cfg: RLinf 配置对象，应包含以下字段：
            - model_path: 预训练权重路径
            - obs_converter_type: 环境观测转换器类型（如 "libero"）
            - num_action_chunks: 动作 chunk 数量（默认 64）
            - denoising_steps: 推理去噪步数（默认 5）
            - add_value_head: 是否添加 value head（默认 False）
        torch_dtype: 模型数据类型（默认 bfloat16）
    
    Returns:
        RDTForRLActionPrediction 模型实例
    
    Example:
        >>> from omegaconf import DictConfig
        >>> cfg = DictConfig({
        ...     "model_type": "rdt",
        ...     "model_path": "/path/to/rdt/checkpoint",
        ...     "obs_converter_type": "libero",
        ...     "num_action_chunks": 64,
        ...     "denoising_steps": 5,
        ...     "add_value_head": False,
        ... })
        >>> model = get_model(cfg)
    """
    # 加载 RDT 配置文件（base.yaml）
    rdt_dir = Path(__file__).parent
    config_path = rdt_dir / "config" / "base.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"RDT config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        rdt_config = yaml.safe_load(f)
    
    # 从 RLinf cfg 中提取参数
    model_path = cfg.get("model_path", "")
    obs_converter_type = cfg.get("obs_converter_type", "libero")
    output_action_chunks = cfg.get("num_action_chunks", 8)  # 配置文件中使用 num_action_chunks，但传给模型时使用 output_action_chunks
    denoising_steps = cfg.get("denoising_steps", 5)
    
    # 创建模型
    model = RDTForRLActionPrediction(
        config=rdt_config,
        model_path=model_path,
        output_action_chunks=output_action_chunks,
        denoising_steps=denoising_steps,
        obs_converter_type=obs_converter_type,
        torch_dtype=torch_dtype,
    )
    
    return model


__all__ = [
    "RDTForRLActionPrediction",
    "get_model",
]
    