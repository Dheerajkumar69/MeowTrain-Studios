"""
DPO (Direct Preference Optimization) and ORPO trainer for MeowLLM.

Uses the TRL library to train language models with preference data
(chosen/rejected pairs) for alignment. This is the industry-standard
approach for making models actually useful and safe post-SFT.

Supports:
  - DPO: Direct Preference Optimization (Rafailov et al., 2023)
  - ORPO: Odds Ratio Preference Optimization (Hong et al., 2024)
  - Both integrate with LoRA/QLoRA for efficient training
"""

import logging
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

logger = logging.getLogger("meowllm.dpo_trainer")


def create_dpo_trainer(
    model_name: str,
    training_method: str,
    hyperparameters: dict,
    train_dataset,
    eval_dataset,
    tokenizer,
    output_dir: str,
    metrics,
    checkpoint_manager=None,
):
    """
    Create a DPO trainer using the TRL library.

    The dataset must have columns: prompt, chosen, rejected
    (as produced by load_preference_dataset in data_loader).

    Args:
        model_name: HuggingFace model ID
        training_method: 'dpo' (uses LoRA by default)
        hyperparameters: Training config dict
        train_dataset: HF Dataset with prompt/chosen/rejected columns
        eval_dataset: HF Dataset or None
        tokenizer: Loaded tokenizer
        output_dir: Checkpoint directory
        metrics: Shared TrainingMetrics
        checkpoint_manager: Optional CheckpointManager

    Returns:
        Configured DPOTrainer instance
    """
    try:
        from trl import DPOTrainer, DPOConfig
    except ImportError:
        raise RuntimeError(
            "TRL library is required for DPO training. "
            "Install it with: pip install trl>=0.7.0"
        )

    from app.ml.trainer import MeowTrainerCallback, _load_model, _apply_lora

    logger.info("Creating DPO trainer: model=%s", model_name)

    # Load model
    model = _load_model(model_name, "lora")  # DPO always uses LoRA
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA
    model = _apply_lora(model, hyperparameters)

    # Enable gradient checkpointing
    if hyperparameters.get("gradient_checkpointing", True):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

    # Reference model (frozen copy for DPO KL divergence)
    ref_model = _load_model(model_name, "full")
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Training arguments
    epochs = hyperparameters.get("epochs", 1)
    batch_size = hyperparameters.get("batch_size", 2)
    lr = hyperparameters.get("learning_rate", 5e-5)
    beta = hyperparameters.get("dpo_beta", 0.1)
    max_tokens = hyperparameters.get("max_tokens", 512)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=hyperparameters.get("gradient_accumulation_steps", 4),
        learning_rate=lr,
        beta=beta,
        max_length=max_tokens,
        max_prompt_length=max_tokens // 2,
        warmup_steps=hyperparameters.get("warmup_steps", 10),
        weight_decay=hyperparameters.get("weight_decay", 0.01),
        logging_steps=1,
        save_steps=hyperparameters.get("save_steps", 100),
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=hyperparameters.get("eval_steps", 50) if eval_dataset is not None else None,
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
        report_to="none",
        lr_scheduler_type=hyperparameters.get("lr_scheduler_type", "cosine"),
        optim="paged_adamw_8bit",
        seed=42,
        disable_tqdm=True,
        gradient_checkpointing=hyperparameters.get("gradient_checkpointing", True),
    )

    # Update metrics
    if train_dataset is not None:
        steps_per_epoch = max(1, len(train_dataset) // (batch_size * dpo_config.gradient_accumulation_steps))
        metrics.total_steps = steps_per_epoch * epochs
    metrics.total_epochs = epochs

    # Callbacks
    callbacks = [MeowTrainerCallback(metrics, checkpoint_manager)]

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info(
        "DPO Trainer created: %d samples, beta=%.2f, lr=%.2e",
        len(train_dataset) if train_dataset else 0, beta, lr,
    )

    return trainer


def create_orpo_trainer(
    model_name: str,
    training_method: str,
    hyperparameters: dict,
    train_dataset,
    eval_dataset,
    tokenizer,
    output_dir: str,
    metrics,
    checkpoint_manager=None,
):
    """
    Create an ORPO trainer using the TRL library.

    ORPO does NOT need a reference model (unlike DPO), making it
    more memory-efficient. The dataset format is the same as DPO.
    """
    try:
        from trl import ORPOTrainer, ORPOConfig
    except ImportError:
        raise RuntimeError(
            "TRL library >= 0.8.0 is required for ORPO training. "
            "Install it with: pip install trl>=0.8.0"
        )

    from app.ml.trainer import MeowTrainerCallback, _load_model, _apply_lora

    logger.info("Creating ORPO trainer: model=%s", model_name)

    # Load model (no reference model needed for ORPO)
    model = _load_model(model_name, "lora")
    model.resize_token_embeddings(len(tokenizer))
    model = _apply_lora(model, hyperparameters)

    if hyperparameters.get("gradient_checkpointing", True):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

    epochs = hyperparameters.get("epochs", 1)
    batch_size = hyperparameters.get("batch_size", 2)
    lr = hyperparameters.get("learning_rate", 5e-5)
    orpo_alpha = hyperparameters.get("orpo_alpha", 1.0)
    max_tokens = hyperparameters.get("max_tokens", 512)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    orpo_config = ORPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=hyperparameters.get("gradient_accumulation_steps", 4),
        learning_rate=lr,
        alpha=orpo_alpha,
        max_length=max_tokens,
        max_prompt_length=max_tokens // 2,
        warmup_steps=hyperparameters.get("warmup_steps", 10),
        weight_decay=hyperparameters.get("weight_decay", 0.01),
        logging_steps=1,
        save_steps=hyperparameters.get("save_steps", 100),
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=hyperparameters.get("eval_steps", 50) if eval_dataset is not None else None,
        fp16=use_fp16,
        bf16=use_bf16,
        remove_unused_columns=False,
        report_to="none",
        lr_scheduler_type=hyperparameters.get("lr_scheduler_type", "cosine"),
        optim="paged_adamw_8bit",
        seed=42,
        disable_tqdm=True,
        gradient_checkpointing=hyperparameters.get("gradient_checkpointing", True),
    )

    if train_dataset is not None:
        steps_per_epoch = max(1, len(train_dataset) // (batch_size * orpo_config.gradient_accumulation_steps))
        metrics.total_steps = steps_per_epoch * epochs
    metrics.total_epochs = epochs

    callbacks = [MeowTrainerCallback(metrics, checkpoint_manager)]

    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info(
        "ORPO Trainer created: %d samples, alpha=%.2f, lr=%.2e",
        len(train_dataset) if train_dataset else 0, orpo_alpha, lr,
    )

    return trainer
