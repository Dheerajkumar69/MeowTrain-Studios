"""
Per-model-architecture prompt formatting.

Instead of hardcoding a single prompt template, we detect the model architecture
from its config and apply the correct chat format.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("meowllm.prompt_templates")


def _detect_architecture(model_path: str) -> str:
    """Detect model architecture from config.json or adapter_config.json."""
    path = Path(model_path)

    config_file = path / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
            arch = config.get("model_type", "").lower()
            name = config.get("_name_or_path", "").lower()
            if "llama" in arch or "llama" in name:
                return "llama"
            if "mistral" in arch or "mistral" in name:
                return "mistral"
            if "phi" in arch or "phi" in name:
                return "phi"
            if "gpt2" in arch or "qwen" in arch:
                return "chatml"
        except Exception:
            pass

    adapter_config = path / "adapter_config.json"
    if adapter_config.exists():
        try:
            with open(adapter_config) as f:
                config = json.load(f)
            base = config.get("base_model_name_or_path", "").lower()
            if "llama" in base:
                return "llama"
            if "mistral" in base:
                return "mistral"
            if "phi" in base:
                return "phi"
        except Exception:
            pass

    return "generic"


def format_prompt(model_path: str, system_prompt: str, user_prompt: str) -> str:
    """Format a prompt using the architecture-appropriate template."""
    arch = _detect_architecture(model_path)
    logger.debug("Detected architecture: %s for %s", arch, model_path)
    return _apply_template(arch, system_prompt, user_prompt)


def _apply_template(arch: str, system: str, user: str) -> str:
    """Apply the architecture-specific prompt template."""
    if arch == "llama":
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            + system
            + "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            + user
            + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif arch == "mistral":
        return "<s>[INST] " + system + "\n\n" + user + " [/INST]"
    elif arch == "phi":
        return (
            "<|system|>\n" + system + "<|end|>\n"
            "<|user|>\n" + user + "<|end|>\n"
            "<|assistant|>\n"
        )
    elif arch == "chatml":
        return (
            "<|im_start|>system\n" + system + "<|im_end|>\n"
            "<|im_start|>user\n" + user + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        # Generic fallback
        return (
            "### System:\n" + system + "\n\n"
            "### User:\n" + user + "\n\n"
            "### Assistant:\n"
        )
