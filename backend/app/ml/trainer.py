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
import math
import time
import gc
from typing import Optional

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


def get_best_device() -> str:
    """
    Return the best available training device: 'cuda', 'mps', or 'cpu'.
    Works across all platforms: NVIDIA (CUDA), Apple Silicon (MPS),
    AMD (ROCm via CUDA-compatible API), and CPU-only.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TrainingMetrics:
    """
    Shared training state.

    Two modes of operation:
      1. **Local** (no shared_dict): plain Python attrs, used inside tests or
         when the caller doesn't need cross-process sharing.
      2. **Shared** (shared_dict from multiprocessing.Manager): every read/write
         transparently proxies through the Manager dict, so the API process
         can read metrics that the child training process writes.

    pause_event / stop_event are multiprocessing.Event objects when running
    in a subprocess, or simple flag booleans when running locally.
    """

    _FIELDS = (
        "status", "current_step", "total_steps", "current_epoch",
        "total_epochs", "current_loss", "best_loss", "validation_loss",
        "perplexity", "learning_rate_current", "tokens_per_sec",
        "error_message", "log_history",
    )

    def __init__(self, shared_dict=None, pause_event=None, stop_event=None):
        self._shared = shared_dict  # None → local mode
        self._pause_event = pause_event
        self._stop_event = stop_event

        # Local fallback storage (used when shared_dict is None)
        if self._shared is None:
            self._local = {
                "status": "initializing",
                "current_step": 0,
                "total_steps": 0,
                "current_epoch": 0,
                "total_epochs": 0,
                "current_loss": None,
                "best_loss": None,
                "validation_loss": None,
                "perplexity": None,
                "learning_rate_current": None,
                "tokens_per_sec": 0.0,
                "error_message": None,
                "log_history": [],
            }
            self._pause_flag = False
            self._stop_flag = False
        else:
            self._local = None
            self._pause_flag = None
            self._stop_flag = None

    # ── Attribute access proxied to the correct backing store ──

    def __getattr__(self, name: str):
        if name.startswith("_") or name not in self._FIELDS:
            raise AttributeError(name)
        store = self._shared if self._shared is not None else self._local
        return store.get(name)

    def __setattr__(self, name: str, value):
        if name.startswith("_") or name not in self._FIELDS:
            super().__setattr__(name, value)
            return
        store = self._shared if self._shared is not None else self._local
        store[name] = value

    # ── Control flags via Events or plain bools ──

    @property
    def pause_requested(self) -> bool:
        if self._pause_event is not None:
            return self._pause_event.is_set()
        return bool(self._pause_flag)

    @pause_requested.setter
    def pause_requested(self, value: bool):
        if self._pause_event is not None:
            self._pause_event.set() if value else self._pause_event.clear()
        else:
            self._pause_flag = value

    @property
    def stop_requested(self) -> bool:
        if self._stop_event is not None:
            return self._stop_event.is_set()
        return bool(self._stop_flag)

    @stop_requested.setter
    def stop_requested(self, value: bool):
        if self._stop_event is not None:
            self._stop_event.set() if value else self._stop_event.clear()
        else:
            self._stop_flag = value

    # ── Log history helpers ──

    def append_log(self, entry: dict):
        """Append to log_history with automatic pruning."""
        try:
            history = self.log_history or []
            history = list(history)  # defensive copy for proxy
            history.append(entry)
            if len(history) > 10000:
                history = history[-5000:]
            self.log_history = history
        except Exception:
            pass  # Don't crash training for a logging failure

    def get_log_history(self) -> list:
        """Return a copy of log_history."""
        try:
            return list(self.log_history or [])
        except Exception:
            return []


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
                # Compute perplexity: exp(loss), capped to avoid overflow
                try:
                    ppl = math.exp(min(eval_loss, 100))
                    self.metrics.perplexity = round(ppl, 2)
                except OverflowError:
                    self.metrics.perplexity = float('inf')
                logger.info("Validation loss: %.6f  Perplexity: %.2f", eval_loss, self.metrics.perplexity)

                # Add eval metrics to log history
                eval_entry = {
                    "step": state.global_step,
                    "eval_loss": round(eval_loss, 6),
                    "perplexity": self.metrics.perplexity,
                    "epoch": metrics.get("epoch", 0),
                    "timestamp": time.time(),
                }
                self.metrics.append_log(eval_entry)

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

    # Enable gradient checkpointing if configured (saves ~40% VRAM)
    use_gradient_checkpointing = hyperparameters.get("gradient_checkpointing", True)
    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
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
    weight_decay = hyperparameters.get("weight_decay", 0.01)
    lr_scheduler_type = hyperparameters.get("lr_scheduler_type", "cosine")
    eval_steps = hyperparameters.get("eval_steps", 50)
    early_stopping_patience = hyperparameters.get("early_stopping_patience", 3)
    early_stopping_threshold = hyperparameters.get("early_stopping_threshold", 0.01)
    use_gradient_checkpointing = hyperparameters.get("gradient_checkpointing", True)

    # Calculate total steps for progress tracking
    if train_dataset is not None:
        steps_per_epoch = max(1, len(train_dataset) // (batch_size * grad_accum))
        total_steps = steps_per_epoch * epochs
    else:
        total_steps = 0

    metrics.total_steps = total_steps
    metrics.total_epochs = epochs

    # ── Universal mixed-precision detection ─────────────────────────
    device = get_best_device()
    if device == "cuda":
        use_bf16 = torch.cuda.is_bf16_supported() and training_method != "qlora"
        use_fp16 = not use_bf16 and training_method != "qlora"
    elif device == "mps":
        # MPS handles precision internally — don't set fp16/bf16 flags
        use_bf16 = False
        use_fp16 = False
    else:
        # CPU — no mixed precision
        use_bf16 = False
        use_fp16 = False

    # ── Optimizer selection ────────────────────────────────────────
    # paged_adamw_8bit requires bitsandbytes (NVIDIA CUDA only)
    if training_method == "qlora" and device == "cuda":
        optim = "paged_adamw_8bit"
    else:
        optim = "adamw_torch"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_steps=1,  # Log every step for real-time updates
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=eval_steps if eval_dataset is not None else None,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_pin_memory=(device == "cuda"),  # Only beneficial for CUDA
        remove_unused_columns=False,
        report_to="none",  # We handle our own reporting via callbacks
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        max_grad_norm=1.0,
        seed=42,
        dataloader_num_workers=0,  # Avoid multiprocessing issues in thread
        disable_tqdm=True,  # We have our own progress tracking
        gradient_checkpointing=use_gradient_checkpointing,
    )

    # ── DeepSpeed Multi-GPU ───────────────────────────────────────
    if hyperparameters.get("multi_gpu", False):
        try:
            from app.ml.deepspeed_configs import get_deepspeed_config, get_gpu_count
            gpu_count = get_gpu_count()
            if gpu_count > 1:
                ds_stage = hyperparameters.get("deepspeed_stage", 2)
                ds_config = get_deepspeed_config(stage=ds_stage, gpu_count=gpu_count)
                training_args.deepspeed = ds_config
                logger.info(
                    "DeepSpeed ZeRO-%d enabled for %d GPUs",
                    ds_stage, gpu_count,
                )
            else:
                logger.warning("multi_gpu requested but only %d GPU(s) detected", gpu_count)
        except ImportError:
            logger.warning("DeepSpeed not installed, multi-GPU disabled")

    # Store max_seq_length for tokens/sec calculation
    training_args._max_seq_length = max_tokens

    # ── Data Collator ─────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # ── Callbacks ─────────────────────────────────────────────────
    callbacks = [MeowTrainerCallback(metrics, checkpoint_manager)]

    # Early stopping (only if we have eval data and patience > 0)
    if eval_dataset is not None and early_stopping_patience > 0:
        from transformers import EarlyStoppingCallback as _ESCallback
        callbacks.append(_ESCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        ))
        logger.info("Early stopping enabled: patience=%d threshold=%.4f", early_stopping_patience, early_stopping_threshold)

    # ── Create Trainer ────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    logger.info(
        "Trainer created: %d train samples, %s eval samples, %d total steps, "
        "scheduler=%s, weight_decay=%.4f, gradient_checkpointing=%s",
        len(train_dataset) if train_dataset else 0,
        len(eval_dataset) if eval_dataset else "no",
        total_steps,
        lr_scheduler_type,
        weight_decay,
        use_gradient_checkpointing,
    )

    return trainer


def _get_model_dtype(device: str) -> torch.dtype:
    """Choose the best dtype for the given device."""
    if device == "cuda":
        return torch.float16
    elif device == "mps":
        return torch.float32  # MPS works best with float32 for stability
    return torch.float32


def _load_model(model_name: str, training_method: str):
    """Load the base model with appropriate quantization for any device."""
    device = get_best_device()
    logger.info("Loading model: %s (method=%s, device=%s)", model_name, training_method, device)

    from app.config import TRUST_REMOTE_CODE
    load_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "low_cpu_mem_usage": True,
    }

    if training_method == "qlora":
        # 4-bit quantization for QLoRA — requires bitsandbytes (CUDA only)
        if device == "cuda":
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
                logger.info("QLoRA: Using 4-bit quantization (CUDA)")
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to standard LoRA on CUDA")
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
        else:
            # MPS / CPU: bitsandbytes not supported, fall back to LoRA-compatible loading
            logger.warning(
                "QLoRA requires bitsandbytes (CUDA only). "
                "Falling back to standard LoRA on %s.", device
            )
            load_kwargs["torch_dtype"] = _get_model_dtype(device)
            if device == "mps":
                load_kwargs["device_map"] = {"" : "mps"}

    elif training_method in ("lora", "full"):
        load_kwargs["torch_dtype"] = _get_model_dtype(device)
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
        elif device == "mps":
            load_kwargs["device_map"] = {"" : "mps"}
        # CPU: no device_map needed, model stays on CPU

    else:
        # DPO / ORPO or unknown — same device-aware loading
        load_kwargs["torch_dtype"] = _get_model_dtype(device)
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
        elif device == "mps":
            load_kwargs["device_map"] = {"" : "mps"}

    try:
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        logger.info("Model loaded successfully on %s: %s", device, model_name)
        return model
    except torch.cuda.OutOfMemoryError:
        _cleanup_gpu()
        raise RuntimeError(
            f"GPU out of memory loading {model_name}. "
            "Try a smaller model, QLoRA method, or reduce batch size."
        )
    except RuntimeError as e:
        if "MPS" in str(e) or "mps" in str(e):
            _cleanup_gpu()
            raise RuntimeError(
                f"MPS out of memory loading {model_name}. "
                "Try a smaller model or reduce batch size. "
                "Apple Silicon has shared memory — close other apps to free RAM."
            )
        raise RuntimeError(f"Failed to load model {model_name}: {e}")
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
    Works across LLaMA, Mistral, Phi, Gemma, GPT-2, etc.
    """
    import torch.nn as nn
    try:
        from transformers.pytorch_utils import Conv1D  # GPT-2 style layers
    except ImportError:
        Conv1D = None

    # Common attention projection names across architectures
    common_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # LLaMA, Mistral
        "gate_proj", "up_proj", "down_proj",       # MLP layers
        "query_key_value",                          # Some architectures
        "dense", "dense_h_to_4h", "dense_4h_to_h", # GPT-NeoX style
        "c_attn", "c_proj", "c_fc",               # GPT-2 style
    ]

    # Check which modules actually exist in the model
    model_modules = set()
    for name, _ in model.named_modules():
        parts = name.split(".")
        model_modules.update(parts)

    found = [t for t in common_targets if t in model_modules]

    if not found:
        # Fallback: collect the leaf names of all actual Linear / Conv1D layers
        logger.warning("No common target modules found — scanning model for linear layers")
        linear_types = (nn.Linear,) + ((Conv1D,) if Conv1D is not None else ())
        leaf_names: set[str] = set()
        for name, module in model.named_modules():
            if isinstance(module, linear_types):
                leaf_names.add(name.split(".")[-1])
        found = sorted(leaf_names)
        if not found:
            # Last resort – let PEFT handle it (supported in newer peft versions)
            logger.warning("No linear layers found by scan — falling back to 'all-linear'")
            found = ["all-linear"]

    logger.info("LoRA target modules: %s", found)
    return found


def _cleanup_gpu():
    """Force GPU memory cleanup for any device (CUDA, MPS, or CPU)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA GPU memory cleaned up")
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
            logger.info("MPS memory cleaned up")
        except Exception:
            pass
