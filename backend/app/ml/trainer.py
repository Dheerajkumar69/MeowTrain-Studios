"""
HuggingFace Trainer wrapper for MeowLLM.

Supports:
  - LoRA fine-tuning via PEFT
  - QLoRA (4-bit quantized LoRA) via bitsandbytes
  - Full fine-tuning
  - Real-time metrics streaming via a shared state dict
  - Graceful pause/stop via callback flag checks
"""

import logging
import time
import gc
import threading
from typing import Optional
from dataclasses import dataclass, field

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    DataCollatorForLanguageModeling,
)

logger = logging.getLogger("meowllm.trainer")


@dataclass
class TrainingMetrics:
    """Thread-safe shared state for real-time metrics."""

    status: str = "initializing"
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    learning_rate_current: Optional[float] = None
    tokens_per_sec: float = 0.0
    error_message: Optional[str] = None
    log_history: list = field(default_factory=list)

    # Control flags (set by routes, read by callback)
    pause_requested: bool = False
    stop_requested: bool = False

    # Lock for thread-safe log_history access
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append_log(self, entry: dict):
        """Thread-safe append to log_history with automatic pruning."""
        with self._lock:
            self.log_history.append(entry)
            if len(self.log_history) > 10000:
                self.log_history = self.log_history[-5000:]

    def get_log_history(self) -> list:
        """Thread-safe copy of log_history."""
        with self._lock:
            return list(self.log_history)


class MeowTrainerCallback(TrainerCallback):
    """
    Custom callback that streams metrics to the shared state and
    handles pause/stop commands from the API.
    """

    def __init__(self, metrics: TrainingMetrics, checkpoint_manager=None):
        self.metrics = metrics
        self.checkpoint_manager = checkpoint_manager
        self._step_start_time = None
        self._tokens_processed = 0

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.metrics.status = "running"
        self.metrics.total_steps = state.max_steps
        self.metrics.total_epochs = int(args.num_train_epochs)
        logger.info("Training started: %d steps, %d epochs", state.max_steps, args.num_train_epochs)

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self._step_start_time = time.time()

        # Handle pause
        if self.metrics.pause_requested:
            self.metrics.status = "paused"
            logger.info("Training paused at step %d", state.global_step)
            while self.metrics.pause_requested and not self.metrics.stop_requested:
                time.sleep(0.5)
            if not self.metrics.stop_requested:
                self.metrics.status = "running"
                logger.info("Training resumed at step %d", state.global_step)

        # Handle stop
        if self.metrics.stop_requested:
            self.metrics.status = "stopping"
            logger.info("Stop requested at step %d — saving checkpoint...", state.global_step)
            control.should_training_stop = True

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return

        current_loss = logs.get("loss")
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch", 0)

        if current_loss is not None:
            self.metrics.current_loss = round(current_loss, 6)
            if self.metrics.best_loss is None or current_loss < self.metrics.best_loss:
                self.metrics.best_loss = round(current_loss, 6)

        if lr is not None:
            self.metrics.learning_rate_current = lr

        self.metrics.current_epoch = int(epoch)
        self.metrics.current_step = state.global_step

        # Calculate tokens/sec: batch_size * seq_length / elapsed_time
        if self._step_start_time is not None:
            elapsed = time.time() - self._step_start_time
            if elapsed > 0:
                seq_len = getattr(args, '_max_seq_length', 512)
                step_tokens = args.per_device_train_batch_size * seq_len
                self.metrics.tokens_per_sec = round(step_tokens / max(elapsed, 0.001), 1)

        # Save to log history (thread-safe)
        log_entry = {
            "step": state.global_step,
            "loss": current_loss,
            "learning_rate": lr,
            "epoch": epoch,
            "timestamp": time.time(),
        }
        self.metrics.append_log(log_entry)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get("eval_loss")
            if eval_loss is not None:
                self.metrics.validation_loss = round(eval_loss, 6)
                logger.info("Validation loss: %.6f", eval_loss)

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info("Checkpoint saved at step %d", state.global_step)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.metrics.stop_requested:
            self.metrics.status = "stopped"
        elif self.metrics.error_message:
            self.metrics.status = "error"
        else:
            self.metrics.status = "completed"
        logger.info("Training ended with status: %s", self.metrics.status)


def create_model_and_trainer(
    model_name: str,
    training_method: str,
    hyperparameters: dict,
    train_dataset,
    eval_dataset,
    tokenizer,
    output_dir: str,
    metrics: TrainingMetrics,
    checkpoint_manager=None,
) -> Trainer:
    """
    Create a fully configured HuggingFace Trainer.

    Args:
        model_name: HuggingFace model ID (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        training_method: 'lora', 'qlora', or 'full'
        hyperparameters: Dict from TrainingConfigRequest
        train_dataset: Tokenized HF Dataset
        eval_dataset: Tokenized HF Dataset or None
        tokenizer: Loaded tokenizer
        output_dir: Where to save checkpoints
        metrics: Shared TrainingMetrics for real-time updates
        checkpoint_manager: Optional CheckpointManager

    Returns:
        Configured Trainer instance
    """
    logger.info(
        "Creating trainer: model=%s method=%s epochs=%d",
        model_name, training_method, hyperparameters.get("epochs", 3),
    )

    # ── Load Model ────────────────────────────────────────────────
    model = _load_model(model_name, training_method)

    # Resize embeddings if tokenizer modified
    model.resize_token_embeddings(len(tokenizer))

    # ── Apply LoRA/QLoRA ──────────────────────────────────────────
    if training_method in ("lora", "qlora"):
        model = _apply_lora(model, hyperparameters)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # ── Training Arguments ────────────────────────────────────────
    epochs = hyperparameters.get("epochs", 3)
    batch_size = hyperparameters.get("batch_size", 4)
    lr = hyperparameters.get("learning_rate", 2e-4)
    warmup_steps = hyperparameters.get("warmup_steps", 10)
    grad_accum = hyperparameters.get("gradient_accumulation_steps", 4)
    max_tokens = hyperparameters.get("max_tokens", 512)
    save_steps = hyperparameters.get("save_steps", 100)

    # Calculate total steps for progress tracking
    if train_dataset is not None:
        steps_per_epoch = max(1, len(train_dataset) // (batch_size * grad_accum))
        total_steps = steps_per_epoch * epochs
    else:
        total_steps = 0

    metrics.total_steps = total_steps
    metrics.total_epochs = epochs

    # Determine device
    use_fp16 = torch.cuda.is_available() and training_method != "qlora"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and training_method != "qlora"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_steps=1,  # Log every step for real-time updates
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=save_steps if eval_dataset is not None else None,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        fp16=use_fp16 and not use_bf16,
        bf16=use_bf16,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",  # We handle our own reporting via callbacks
        lr_scheduler_type="cosine",
        optim="adamw_torch" if training_method != "qlora" else "paged_adamw_8bit",
        max_grad_norm=1.0,
        seed=42,
        dataloader_num_workers=0,  # Avoid multiprocessing issues in thread
        disable_tqdm=True,  # We have our own progress tracking
    )

    # Store max_seq_length for tokens/sec calculation
    training_args._max_seq_length = max_tokens

    # ── Data Collator ─────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # ── Callback ──────────────────────────────────────────────────
    callback = MeowTrainerCallback(metrics, checkpoint_manager)

    # ── Create Trainer ────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[callback],
    )

    logger.info(
        "Trainer created: %d train samples, %s eval samples, %d total steps",
        len(train_dataset) if train_dataset else 0,
        len(eval_dataset) if eval_dataset else "no",
        total_steps,
    )

    return trainer


def _load_model(model_name: str, training_method: str):
    """Load the base model with appropriate quantization."""
    logger.info("Loading model: %s (method=%s)", model_name, training_method)

    from app.config import TRUST_REMOTE_CODE
    load_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "low_cpu_mem_usage": True,
    }

    if training_method == "qlora":
        # 4-bit quantization for QLoRA
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
            logger.info("QLoRA: Using 4-bit quantization")
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to standard LoRA")
            load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    elif training_method == "lora":
        load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

    else:  # full fine-tune
        load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        logger.info("Model loaded successfully: %s", model_name)
        return model
    except torch.cuda.OutOfMemoryError:
        _cleanup_gpu()
        raise RuntimeError(
            f"GPU out of memory loading {model_name}. "
            "Try a smaller model, QLoRA method, or reduce batch size."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")


def _apply_lora(model, hyperparameters: dict):
    """Apply LoRA adapter to the model."""
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    lora_rank = hyperparameters.get("lora_rank", 16)
    lora_alpha = hyperparameters.get("lora_alpha", 32)
    lora_dropout = hyperparameters.get("lora_dropout", 0.05)

    # Prepare model for k-bit training if quantized
    if getattr(model, "is_quantized", False) or hasattr(model, "quantization_method"):
        try:
            model = prepare_model_for_kbit_training(model)
            logger.info("Model prepared for k-bit training")
        except Exception as e:
            logger.warning("prepare_model_for_kbit_training failed (non-fatal): %s", e)

    # Find target modules (common across architectures)
    target_modules = _find_target_modules(model)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total if total > 0 else 0
    logger.info(
        "LoRA applied: %d trainable / %d total params (%.2f%%)",
        trainable, total, pct,
    )

    return model


def _find_target_modules(model) -> list[str]:
    """
    Auto-detect linear layer names for LoRA targeting.
    Works across LLaMA, Mistral, Phi, Gemma, etc.
    """
    # Common attention projection names across architectures
    common_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # LLaMA, Mistral
        "gate_proj", "up_proj", "down_proj",       # MLP layers
        "query_key_value",                          # Some architectures
        "dense", "dense_h_to_4h", "dense_4h_to_h", # GPT-NeoX style
    ]

    # Check which modules actually exist in the model
    model_modules = set()
    for name, _ in model.named_modules():
        parts = name.split(".")
        model_modules.update(parts)

    found = [t for t in common_targets if t in model_modules]

    if not found:
        # Fallback: target all Linear layers
        logger.warning("No common target modules found — using all linear layers")
        found = ["all-linear"]

    logger.info("LoRA target modules: %s", found)
    return found


def _cleanup_gpu():
    """Force GPU memory cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleaned up")
