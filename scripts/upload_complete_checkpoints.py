#!/usr/bin/env python3
"""
Upload complete RDT checkpoints to HuggingFace
Includes all files needed for both inference and continued training
"""

import os
import argparse
import logging
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def upload_checkpoint(checkpoint_path: str, repo_id: str, description: str):
    """
    Upload complete checkpoint directory to HuggingFace
    
    Args:
        checkpoint_path: Path to checkpoint directory
        repo_id: HuggingFace repository ID (e.g., "TJ-chen/RDT-1B-LIBERO-Base")
        description: Model description
    """
    api = HfApi()
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    # Create repository if it doesn't exist
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        logger.info(f"Repository '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        logger.info(f"Creating repository: {repo_id}")
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    
    # Create model card
    model_card = f"""---
license: apache-2.0
tags:
- robotics
- rdt
- libero
- diffusion
- transformers
---

# RDT-1B LIBERO Checkpoint

{description}

## Model Information
- Base Model: RDT-1B (Residual Diffusion Transformer)
- Training Framework: DeepSpeed ZeRO Stage 2
- Precision: BF16

## Checkpoint Contents

This checkpoint includes:

### For Inference
- `ema/model.safetensors` - EMA model weights (recommended for inference)
- `config.json` - Model configuration

### For Training
- `pytorch_model/` - DeepSpeed distributed training checkpoint
  - `bf16_zero_pp_rank_*_optim_states.pt` - Optimizer states (ZeRO Stage 2)
  - `mp_rank_00_model_states.pt` - Model states
- `scheduler.bin` - Learning rate scheduler state
- `random_states_*.pkl` - Random number generator states
- `zero_to_fp32.py` - Utility to convert DeepSpeed checkpoint to FP32

## Usage

### For Inference

```python
from transformers import AutoModel
import torch

# Load the EMA model for inference
model = AutoModel.from_pretrained(
    "{repo_id}",
    subfolder="ema",
    trust_remote_code=True
)
model.eval()
```

### For Continued Training

Download the complete checkpoint and use DeepSpeed to resume training:

```bash
# The checkpoint can be loaded with DeepSpeed ZeRO Stage 2
# Make sure your training script is configured with the same DeepSpeed settings
```

## Citation

If you use this model, please cite:

```bibtex
@article{{rdt2024,
  title={{Residual Diffusion Transformer for Robotic Manipulation}},
  author={{Your Name}},
  journal={{arXiv preprint}},
  year={{2024}}
}}
```

## License

Apache 2.0
"""
    
    # Save model card
    readme_path = checkpoint_path / "README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)
    
    logger.info(f"Uploading checkpoint from: {checkpoint_path}")
    logger.info(f"To repository: {repo_id}")
    logger.info(f"Checkpoint size: {get_dir_size(checkpoint_path):.2f} GB")
    
    # Upload entire directory
    try:
        api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload complete RDT training checkpoint",
            ignore_patterns=["*.log", "*.txt", "eval_results/*"],  # Exclude logs and eval results
        )
        logger.info(f"✅ Upload complete! View at: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        return False


def get_dir_size(path: Path) -> float:
    """Get directory size in GB"""
    total = 0
    for entry in path.rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total / (1024**3)


def main():
    # Define checkpoints to upload
    base_path = Path("/home/zhukefei/chensiqi/rlinf_workspace/Libero_RDT/RDT_libero_finetune/checkpoints/best_checkpoints")
    
    checkpoints = [
        {
            "path": base_path / "libero_base" / "checkpoint-65000",
            "repo": "TJ-chen/RDT-1B-LIBERO-Base",
            "desc": "RDT-1B full fine-tuned on LIBERO-90 dataset (checkpoint-65000). Base model for all LIBERO tasks.",
        },
        {
            "path": base_path / "libero_spatial_best_ckpt" / "libero_spatial_best_ckpt",
            "repo": "TJ-chen/RDT-1B-LIBERO-Spatial",
            "desc": "RDT-1B fine-tuned on LIBERO Spatial benchmark. Best performing checkpoint.",
        },
        {
            "path": base_path / "libero_goal_best_ckpt" / "libero_goal_best_ckpt",
            "repo": "TJ-chen/RDT-1B-LIBERO-Goal",
            "desc": "RDT-1B fine-tuned on LIBERO Goal benchmark. Best performing checkpoint.",
        },
        {
            "path": base_path / "libero_object_best_ckpt" / "libero_object_best_ckpt",
            "repo": "TJ-chen/RDT-1B-LIBERO-Object",
            "desc": "RDT-1B fine-tuned on LIBERO Object benchmark. Best performing checkpoint.",
        },
        {
            "path": base_path / "libero_long_best_ckpt" / "checkpoint-38000",
            "repo": "TJ-chen/RDT-1B-LIBERO-Long",
            "desc": "RDT-1B fine-tuned on LIBERO-10 (Long) from LIBERO-90 base (checkpoint-38000).",
        },
    ]
    
    logger.info("="*60)
    logger.info("Uploading Complete RDT Checkpoints to HuggingFace")
    logger.info("="*60)
    
    success_count = 0
    for ckpt in checkpoints:
        logger.info("")
        logger.info("="*60)
        if upload_checkpoint(str(ckpt["path"]), ckpt["repo"], ckpt["desc"]):
            success_count += 1
    
    logger.info("")
    logger.info("="*60)
    logger.info(f"Upload Summary: {success_count}/{len(checkpoints)} successful")
    logger.info("="*60)


if __name__ == "__main__":
    main()


