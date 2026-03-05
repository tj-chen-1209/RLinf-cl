# RLinf-CL: Continuous Learning for Embodied AI

[![English](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-red.svg)](README.zh-CN.md)

---

本项目实现了基于强化学习的机器人操作持续学习框架，支持多种策略架构和训练算法。

## 🆕 最新进展

### ✅ RDT (Robot Diffusion Transformer) 集成完成
- **完成日期**: 2026-02
- **功能**: 成功将 RDT 扩散策略模型接入 RLinf 框架
- **支持环境**: LIBERO (Spatial, Goal, Object, Long)
- **当前状态**: ✅ 评估（行为克隆） | ⏳ RL 训练（开发中）
- **详细文档**: 📖 [RDT Integration Guide](docs/RDT_INTEGRATION.md) - 完整的技术文档（中英双语）

---

## 目录
- [模型支持](#模型支持)
- [快速开始](#快速开始)
  - [1. RDT 策略部署与训练](#1-rdt-策略部署与训练)
  - [2. Residual SAC 训练](#2-residual-sac-训练)
- [实现细节](#实现细节)

---

## 模型支持

### 1. RDT (Robot Diffusion Transformer)
- 基于扩散模型的机器人策略
- 支持视觉-语言多模态输入
- Action chunking 推理
- LIBERO 多任务微调
- 配置文件: `examples/embodiment/config/model/rdt.yaml`

### 2. Residual Policy (OpenVLA-OFT)
- 基于 OpenVLA 的残差策略
- LoRA 高效微调
- 支持 SAC/PPO 算法
- 配置文件: `examples/embodiment/config/model/residual_policy.yaml`

---

## 快速开始

## 1. RDT 策略部署与训练

### 1.1 环境部署

使用优化的 Docker 镜像（国内镜像加速 + 完整依赖）：

```bash
cd /home/zhukefei/chensiqi/rlinf_workspace

# 构建 Docker 镜像（已优化，使用清华源）
sudo docker build \
  -f RLinf-cl/docker/Dockerfile.rdt.simple.optimized \
  -t rlinf-rdt:optimized \
  --progress=plain \
  .

# 启动容器（后台运行）
sudo docker run -d --gpus all \
  --shm-size 100g \
  --net=host \
  --name rlinf-rdt \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e MUJOCO_GL=egl \
  -e PYOPENGL_PLATFORM=egl \
  -v /home/zhukefei/chensiqi/rlinf_workspace:/workspace \
  -v /home/zhukefei/.cache/huggingface:/root/.cache/huggingface \
  rlinf-rdt:optimized tail -f /dev/null

# 进入容器
sudo docker exec -it rlinf-rdt bash

# 容器内安装缺失依赖（首次使用）
pip install ray datasets==3.6.0 einops scipy sentencepiece wandb
```

### 1.2 RDT 模型准备

**从 HuggingFace 下载预训练模型：**

```bash
# 在容器内执行
cd /workspace

# 下载 RDT-1B LIBERO 模型系列
huggingface-cli download TJ-chen/RDT-1B-LIBERO-Base --local-dir ./checkpoints/libero_base
huggingface-cli download TJ-chen/RDT-1B-LIBERO-Spatial --local-dir ./checkpoints/libero_spatial
huggingface-cli download TJ-chen/RDT-1B-LIBERO-Goal --local-dir ./checkpoints/libero_goal
huggingface-cli download TJ-chen/RDT-1B-LIBERO-Object --local-dir ./checkpoints/libero_object
huggingface-cli download TJ-chen/RDT-1B-LIBERO-Long --local-dir ./checkpoints/libero_long

# 设置环境变量
export RDT_CHECKPOINT_PATH="/workspace/checkpoints"
```

**或使用本地已有模型：**

```bash
# 本地 checkpoint 路径
export RDT_CHECKPOINT_PATH="/workspace/Libero_RDT/RDT_libero_finetune/checkpoints/best_checkpoints"
```

### 1.3 训练（PPO + RDT）

```bash
cd /workspace/RLinf-cl

# LIBERO Spatial 任务
bash examples/embodiment/run_libero_rdt.sh

# 自定义配置
python examples/embodiment/train_embodied_agent.py \
  --config-path examples/embodiment/config/ \
  --config-name libero_spatial_ppo_rdt \
  actor.model.model_path=${RDT_CHECKPOINT_PATH}/libero_spatial_best_ckpt \
  rollout.model.model_path=${RDT_CHECKPOINT_PATH}/libero_spatial_best_ckpt
```

### 1.4 评估

```bash
# 单任务评估
bash examples/embodiment/eval_libero_rdt.sh

# 指定 checkpoint 评估
python examples/embodiment/eval_embodied_agent.py \
  --config-path examples/embodiment/config/ \
  --config-name libero_spatial_rdt_eval \
  rollout.model.model_path=${RDT_CHECKPOINT_PATH}/libero_spatial_best_ckpt
```

---

## 2. Residual SAC 训练

### 2.0 Deployment

```bash
docker pull rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0

docker run -it --gpus all \
   --shm-size 100g \
   --net=host \
   --name rlinf \
   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
   -v $RLINF_DIR:/workspace/RLinf \
   -v $HF_HOME:/workspace/hf \
   -v $LOG_DIR:/workspace/RLinf/logs \
   rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 /bin/bash

source switch_env openvla-oft
```

### 2.1 Training

```bash
bash examples/embodiment/run_embodiment.sh libero_spatial_task0_lora_residual_sac_openvlaoft
```

### 2.2 Eval single-task succ

```bash
bash examples/embodiment/eval_embodiment.sh libero_spatial_task0_lora_residual_sac_openvlaoft
```

### 2.3 Eval multi-task correction Field

```bash
bash examples/embodiment/eval/correction_field_analysis/run_correction_field_analysis.sh
```

在run_correction_field_analysis.sh脚本中，指定：
- `TASK_ID`
- checkpoint路径
- eval模式（base_rollout, demo, both_demo)

---

## 实现细节

### RDT 集成实现

#### 1. 模型架构
- **位置**: `rlinf/models/embodiment/rdt/`
- **核心文件**:
  - `rdt_policy.py` - RDT 策略包装器
  - `obs_converter.py` - LIBERO 观测转换器
- **特性**:
  - 支持扩散推理（DDIM scheduler）
  - Action chunking 推理模式
  - 视觉-语言多模态输入
  - BF16/FP16 混合精度

#### 2. 配置文件
- **模型配置**: `examples/embodiment/config/model/rdt.yaml`
  ```yaml
  model_type: rdt
  model_path: ${RDT_CHECKPOINT_PATH}
  obs_converter_type: libero
  num_action_chunks: 64
  denoising_steps: 5
  precision: bf16
  ```

- **训练配置**: `examples/embodiment/config/libero_spatial_ppo_rdt.yaml`
- **评估配置**: `examples/embodiment/config/libero_spatial_rdt_eval.yaml`

#### 3. 数据流
```
LIBERO Env
  ↓ raw_obs (agentview, eye_in_hand, ee_pos, etc.)
ObsConverter
  ↓ formatted_obs (main_images, wrist_images, proprio, task_descriptions)
RDT Policy
  ↓ diffusion inference + action chunking
Action (7-dim: 3 pos + 4 quat)
  ↓
LIBERO Env
```

#### 4. 关键文件
- **训练脚本**: `examples/embodiment/run_libero_rdt.sh`
- **评估脚本**: `examples/embodiment/eval_libero_rdt.sh`
- **集成测试**: `test_rdt_with_libero_env.py`

---

### Residual SAC 实现

#### 1. 模型架构
- **位置**: `rlinf/models/embodiment/residual_policy/`

#### 2. 模型配置
- **配置文件**:
  - `embodiment/config/model/residual_policy.yaml`
  - `embodiment/config/model/lora_residual_policy.yaml`

#### 3. 运行配置
- **配置**: `embodiment/config/libero_spatial_task{$ID}_lora_residual_sac_openvlaoft.yaml`

#### 4. 训练pipeline

- 计算rollout，填充replay buffer，传入rollout_batch
  - `rlinf/workers/rollout/hf/residual_rollout_worker.py`
  - 在`ChunkStepResult`中存储(obs, next_obs, base_action, base_next_action)
  - 特别地：访问base model，将当前obs的base action存入`last_forward_inputs`，存入rollout_batch

- 接收rollout_batch，从replay buffer中采样，训练actor和critic
  - `rlinf/workers/actor/residual_fsdp_sac_policy_worker.py`
  > 继承自`rlinf/workers/actor/fsdp_sac_policy_worker.py`
  > 重新实现了`forward_sac`, `forward_critic`等SAC相关方法
  > 不访问base model，直接从rollout_batch中读取base action

#### 5. 数据通信

- rollout worker -> rollout_batch(`EmbodiedRolloutResult`) -> actor worker

---

## 🐳 Docker 优化说明

Docker 构建已优化，使用国内镜像源加速：
- ✅ **PyPI**: 清华大学镜像
- ✅ **APT**: 阿里云镜像  
- ✅ **PyTorch**: 官方源（镜像源不支持最新版本）

预期构建时间：5-10 分钟（原 20-30 分钟）

---

## 📁 项目结构

```
RLinf-cl/
├── rlinf/
│   ├── models/embodiment/
│   │   ├── rdt/              # RDT 模型实现
│   │   └── residual_policy/  # Residual 策略实现
│   ├── workers/
│   │   ├── rollout/          # Rollout worker
│   │   └── actor/            # Actor worker
│   └── envs/
│       └── libero/           # LIBERO 环境封装
├── examples/embodiment/
│   ├── config/               # 配置文件
│   ├── run_libero_rdt.sh    # RDT 训练脚本
│   └── eval_libero_rdt.sh   # RDT 评估脚本
└── docker/
    ├── Dockerfile.rdt.simple  # 优化的 Dockerfile
    └── requirements_core.txt  # 核心依赖
```

---

## 📝 开发日志

### 2026-02 RDT 集成
- ✅ RDT 模型接入 RLinf 框架
- ✅ LIBERO 环境观测转换器
- ✅ PPO 训练流程
- ✅ 评估脚本和配置
- ✅ Docker 环境优化（国内镜像加速）
- ✅ 完整测试脚本

---

## 🔗 相关资源

- **RDT 论文**: [Robot Diffusion Transformer](https://arxiv.org/abs/2410.07494)
- **LIBERO Benchmark**: [LIBERO: Lifelong Robot Learning](https://lifelong-robot-learning.cs.utexas.edu/LIBERO.html)
- **RLinf Framework**: 分布式强化学习框架

---

## 📧 联系方式

如有问题，请联系项目维护者或提交 Issue。

