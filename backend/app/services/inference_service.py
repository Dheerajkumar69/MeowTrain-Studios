"""
Native inference service for MeowLLM.

Loads fine-tuned models from the project output directory and generates
text using HuggingFace's model.generate(). The loaded model is cached
in memory (singleton per model path) to avoid reloading on each request.

Falls back gracefully if no model is available or GPU runs out of memory.
"""

import gc
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import PROJECTS_DIR

logger = logging.getLogger("meowllm.inference_service")

# ── Singleton model cache ────────────────────────────────────────────
_loaded_model = None
_loaded_tokenizer = None
_loaded_path: Optional[str] = None
_model_lock = threading.Lock()


def _get_model_path_for_project(project_id: int) -> Optional[str]:
    """
    Find the latest completed training output for a project.
    Returns the path to the output directory, or None if not found.
    """
    output_base = PROJECTS_DIR / str(project_id) / "output"
    if not output_base.exists():
        return None

    # Find the latest run output (run_N directories, sorted descending)
    run_dirs = sorted(output_base.iterdir(), reverse=True)
    for run_dir in run_dirs:
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            # Check if it has model files (config.json is a reliable indicator)
            if (run_dir / "config.json").exists():
                return str(run_dir)
            # Also check for adapter_config.json (LoRA adapters)
            if (run_dir / "adapter_config.json").exists():
                return str(run_dir)

    return None


def _load_model(model_path: str):
    """Load model and tokenizer from a path, with GPU/CPU fallback."""
    global _loaded_model, _loaded_tokenizer, _loaded_path

    with _model_lock:
        # Already loaded?
        if _loaded_path == model_path and _loaded_model is not None:
            return _loaded_model, _loaded_tokenizer

        # Unload previous model
        _unload_model_internal()

        logger.info("Loading model from: %s", model_path)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True,
            )

            # Ensure pad token
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            # Determine device and dtype
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.float16
            else:
                device_map = "cpu"
                torch_dtype = torch.float32

            # Check if this is a PEFT adapter
            adapter_config_path = Path(model_path) / "adapter_config.json"
            if adapter_config_path.exists():
                # Load as a PEFT model
                import json
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)

                base_model_name = adapter_config.get("base_model_name_or_path", "")
                if not base_model_name:
                    raise ValueError("adapter_config.json missing base_model_name_or_path")

                logger.info("Loading PEFT adapter, base model: %s", base_model_name)
                from peft import PeftModel

                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                model = PeftModel.from_pretrained(base_model, model_path)
                model = model.merge_and_unload()
            else:
                # Load as a full model
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )

            model.eval()
            model.resize_token_embeddings(len(tokenizer))

            _loaded_model = model
            _loaded_tokenizer = tokenizer
            _loaded_path = model_path

            logger.info("Model loaded successfully from %s", model_path)
            return model, tokenizer

        except torch.cuda.OutOfMemoryError:
            _cleanup_gpu()
            raise RuntimeError(
                "GPU out of memory loading model for inference. "
                "Try freeing GPU memory or restarting the server."
            )
        except Exception as e:
            _cleanup_gpu()
            raise RuntimeError(f"Failed to load model: {e}")


def _unload_model_internal():
    """Internal: unload model without lock (caller must hold lock)."""
    global _loaded_model, _loaded_tokenizer, _loaded_path
    if _loaded_model is not None:
        del _loaded_model
        _loaded_model = None
    if _loaded_tokenizer is not None:
        del _loaded_tokenizer
        _loaded_tokenizer = None
    _loaded_path = None
    _cleanup_gpu()


def unload_model():
    """Public: unload the cached model to free memory."""
    with _model_lock:
        _unload_model_internal()
    logger.info("Model unloaded from memory")


def generate_response(
    project_id: int,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> dict:
    """
    Generate a response using the project's fine-tuned model.

    Returns dict with: response, tokens_used, model_path
    Raises RuntimeError if no model is available.
    """
    model_path = _get_model_path_for_project(project_id)
    if not model_path:
        raise RuntimeError(
            "No fine-tuned model found for this project. "
            "Train a model first, or connect LM Studio."
        )

    model, tokenizer = _load_model(model_path)

    # Build the prompt
    full_prompt = f"### System:\n{system_prompt}\n\n### User:\n{prompt}\n\n### Assistant:\n"

    start_time = time.time()

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 0.01),  # Avoid division by zero
            do_sample=temperature > 0,
            top_p=0.9 if temperature > 0 else 1.0,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    elapsed_ms = (time.time() - start_time) * 1000
    total_tokens = len(generated_tokens)

    return {
        "response": response_text,
        "tokens_used": total_tokens,
        "generation_time_ms": round(elapsed_ms, 2),
        "model_path": model_path,
    }


def get_model_info(project_id: int) -> Optional[dict]:
    """Get info about the loaded/available model for a project."""
    model_path = _get_model_path_for_project(project_id)
    if not model_path:
        return None

    is_loaded = _loaded_path == model_path and _loaded_model is not None

    return {
        "model_path": model_path,
        "is_loaded": is_loaded,
        "device": str(next(_loaded_model.parameters()).device) if is_loaded else None,
    }


def _cleanup_gpu():
    """Force GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
