"""
Native inference service for MeowLLM.

Loads fine-tuned models from the project output directory and generates
text using HuggingFace's model.generate(). Loaded models are cached
in an LRU pool (default max_size=2) to reduce evict-reload storms when
multiple projects are queried back-to-back.

Falls back gracefully if no model is available or GPU runs out of memory.
"""

import gc
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import PROJECTS_DIR, TRUST_REMOTE_CODE

logger = logging.getLogger("meowllm.inference_service")

# ── LRU model pool ──────────────────────────────────────────────
_MAX_CACHED_MODELS = int(os.getenv("MEOWLLM_MAX_CACHED_MODELS", "2"))

# _cache: OrderedDict[model_path → {"model", "tokenizer", "loaded_at", "last_used"}]
_cache: OrderedDict[str, dict] = OrderedDict()
_cache_lock = threading.Lock()


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
    """Load model and tokenizer from a path, with GPU/CPU fallback and LRU caching."""
    with _cache_lock:
        # Already loaded? → move to end (MRU)
        if model_path in _cache:
            entry = _cache[model_path]
            entry["last_used"] = time.time()
            _cache.move_to_end(model_path)
            return entry["model"], entry["tokenizer"]

        # Need to evict?
        while len(_cache) >= _MAX_CACHED_MODELS:
            evicted_path, evicted = _cache.popitem(last=False)  # pop LRU
            age = time.time() - evicted["loaded_at"]
            logger.info(
                "LRU evicting model '%s' (cached %.0fs, last used %.0fs ago)",
                evicted_path, age, time.time() - evicted["last_used"],
            )
            del evicted["model"]
            del evicted["tokenizer"]
            _cleanup_gpu()

        logger.info("Loading model from: %s", model_path)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=TRUST_REMOTE_CODE,
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
                    trust_remote_code=TRUST_REMOTE_CODE,
                    low_cpu_mem_usage=True,
                )
                model = PeftModel.from_pretrained(base_model, model_path)
                model = model.merge_and_unload()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    trust_remote_code=TRUST_REMOTE_CODE,
                    low_cpu_mem_usage=True,
                )

            model.eval()
            model.resize_token_embeddings(len(tokenizer))

            now = time.time()
            _cache[model_path] = {
                "model": model,
                "tokenizer": tokenizer,
                "loaded_at": now,
                "last_used": now,
            }

            logger.info("Model loaded successfully from %s (pool size: %d/%d)",
                        model_path, len(_cache), _MAX_CACHED_MODELS)
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


def unload_model(model_path: str = None):
    """
    Public: unload a specific cached model, or ALL if model_path is None.
    """
    with _cache_lock:
        if model_path:
            entry = _cache.pop(model_path, None)
            if entry:
                del entry["model"]
                del entry["tokenizer"]
        else:
            _cache.clear()
    _cleanup_gpu()
    logger.info("Model(s) unloaded from memory (path=%s)", model_path or "ALL")


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

    # Build the prompt using architecture-aware template
    from app.utils.prompt_templates import format_prompt
    full_prompt = format_prompt(model_path, system_prompt, prompt)

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


def generate_response_stream(
    project_id: int,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    max_tokens: int = 512,
):
    """
    Streaming version of generate_response.
    Yields partial text chunks as they are decoded, for SSE streaming.
    """
    from transformers import TextIteratorStreamer

    model_path = _get_model_path_for_project(project_id)
    if not model_path:
        raise RuntimeError(
            "No fine-tuned model found for this project. "
            "Train a model first, or connect LM Studio."
        )

    model, tokenizer = _load_model(model_path)

    from app.utils.prompt_templates import format_prompt
    full_prompt = format_prompt(model_path, system_prompt, prompt)

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "temperature": max(temperature, 0.01),
        "do_sample": temperature > 0,
        "top_p": 0.9 if temperature > 0 else 1.0,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for text_chunk in streamer:
        if text_chunk:
            yield text_chunk

    thread.join()


def get_model_info(project_id: int) -> Optional[dict]:
    """Get info about the loaded/available model for a project."""
    model_path = _get_model_path_for_project(project_id)
    if not model_path:
        return None

    with _cache_lock:
        entry = _cache.get(model_path)
    is_loaded = entry is not None

    return {
        "model_path": model_path,
        "is_loaded": is_loaded,
        "device": str(next(entry["model"].parameters()).device) if is_loaded else None,
    }


def _cleanup_gpu():
    """Force GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
