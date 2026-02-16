"""
DeepSpeed configuration presets for multi-GPU training.

Provides ZeRO Stage 2 and Stage 3 configurations optimized for
different model sizes and GPU memory constraints.

ZeRO-2: Partitions optimizer states + gradients (good for 7B–13B)
ZeRO-3: Partitions everything including params (needed for 30B+)
"""

import logging
import torch

logger = logging.getLogger("meowllm.deepspeed")


def get_gpu_count() -> int:
    """Detect number of available CUDA GPUs."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def get_deepspeed_config(stage: int = 2, gpu_count: int = 1) -> dict:
    """
    Get a DeepSpeed configuration dict.

    Args:
        stage: ZeRO stage (2 or 3)
        gpu_count: Number of GPUs available

    Returns:
        DeepSpeed config dict ready to pass to TrainingArguments
    """
    if stage == 3:
        return _zero3_config(gpu_count)
    return _zero2_config(gpu_count)


def auto_select_config(model_size_b: float = 7.0, gpu_count: int = 1) -> dict | None:
    """
    Auto-select the best DeepSpeed config based on model size and GPU count.

    Args:
        model_size_b: Model size in billions of parameters
        gpu_count: Number of available GPUs

    Returns:
        DeepSpeed config dict, or None if single GPU (DeepSpeed not needed)
    """
    if gpu_count <= 1:
        return None

    # ZeRO-3 for very large models, ZeRO-2 for everything else
    if model_size_b >= 30:
        logger.info("Auto-selected DeepSpeed ZeRO-3 for %.1fB param model on %d GPUs", model_size_b, gpu_count)
        return _zero3_config(gpu_count)
    else:
        logger.info("Auto-selected DeepSpeed ZeRO-2 for %.1fB param model on %d GPUs", model_size_b, gpu_count)
        return _zero2_config(gpu_count)


def _zero2_config(gpu_count: int) -> dict:
    """
    ZeRO Stage 2: Partition optimizer states and gradients.

    Best for 7B–13B models on 2–8 GPUs.
    Lower communication overhead than ZeRO-3.
    """
    return {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
        "bf16": {
            "enabled": "auto",
        },
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
    }


def _zero3_config(gpu_count: int) -> dict:
    """
    ZeRO Stage 3: Partition everything (params + gradients + optimizer states).

    Required for 30B+ models or when GPUs are memory-constrained.
    Higher communication overhead but enables much larger models.
    """
    return {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
        "bf16": {
            "enabled": "auto",
        },
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
    }
