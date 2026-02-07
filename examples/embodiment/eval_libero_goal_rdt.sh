#!/bin/bash

# ================================================
# RDT Evaluation Script for LIBERO Goal Tasks
# ================================================

# 设置环境路径
export EMBODIED_PATH=$(pwd)
export PYTHONPATH="${EMBODIED_PATH}:${PYTHONPATH}"
export LIBERO_BASE=/home/zhukefei/chensiqi/rlinf_workspace/LIBERO

# 显示配置信息
echo "=========================================="
echo "   RDT LIBERO Goal Evaluation"
echo "=========================================="
echo "EMBODIED_PATH: ${EMBODIED_PATH}"
echo "LIBERO_BASE: ${LIBERO_BASE}"
echo "=========================================="

# 模型路径
MODEL_PATH="/share_data/zhukefei/siqi_rdt/best_checkpoints/libero_goal_best_ckpt/libero_goal_best_ckpt"

# 运行评估（使用专用 eval 脚本）
python examples/embodiment/eval_embodied_agent.py \
    --config-name=libero_goal_rdt_eval \
    rollout.model.model_path="${MODEL_PATH}" \
    actor.model.model_path="${MODEL_PATH}" \
    runner.logger.experiment_name="libero_goal_rdt_eval_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="



