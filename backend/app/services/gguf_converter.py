"""
GGUF Converter Service for MeowTrain.

Converts trained HuggingFace models to GGUF format compatible with LM Studio.

Process:
  1. If LoRA adapter → merge with base model first
  2. Convert merged model to GGUF using llama.cpp conversion
  3. Optionally quantize (Q4_K_M for speed, Q8_0 for quality)
  4. Save .gguf file ready for LM Studio
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from app.config import PROJECTS_DIR

logger = logging.getLogger("meowllm.gguf_converter")

# Valid quantization options
VALID_QUANTIZATIONS = {"f16", "Q8_0", "Q4_K_M"}


def _check_disk_space(path: Path, required_bytes: int) -> None:
    """Raise if less than required_bytes of disk space available."""
    try:
        stat = shutil.disk_usage(str(path))
        if stat.free < required_bytes:
            free_gb = round(stat.free / (1024**3), 1)
            need_gb = round(required_bytes / (1024**3), 1)
            raise RuntimeError(
                f"Insufficient disk space: {free_gb} GB free, ~{need_gb} GB needed. "
                "Free disk space and try again."
            )
    except OSError:
        logger.warning("Could not check disk space for %s — proceeding anyway", path)


def _estimate_model_size(model_dir: str) -> int:
    """Estimate total model size in bytes from directory contents."""
    total = 0
    model_path = Path(model_dir)
    if model_path.is_dir():
        for f in model_path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def _find_convert_script() -> Optional[Path]:
    """Find the llama.cpp convert script bundled with llama-cpp-python."""
    try:
        import llama_cpp
        pkg_dir = Path(llama_cpp.__file__).parent
        # The conversion script may be bundled with llama-cpp-python
        for name in ("convert_hf_to_gguf.py", "convert.py"):
            script = pkg_dir / name
            if script.exists():
                return script
    except ImportError:
        pass

    # Fallback: check if convert script is on PATH
    for name in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
        result = shutil.which(name)
        if result:
            return Path(result)

    return None


def merge_lora_adapter(
    base_model_path: str,
    adapter_path: str,
    output_dir: str,
) -> str:
    """Merge a LoRA adapter with its base model into a full model.

    Loads on CPU to avoid GPU OOM. Raises MemoryError with guidance
    if system RAM is insufficient.
    """
    logger.info("Merging LoRA adapter: %s + %s -> %s", base_model_path, adapter_path, output_dir)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency for LoRA merge: {e}. "
            "Install with: pip install transformers peft"
        )

    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            device_map="cpu",  # Merge on CPU to avoid OOM
            low_cpu_mem_usage=True,
        )

        # Load and merge adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = model.merge_and_unload()

        # Save merged model
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(out), safe_serialization=True)

        # Also save the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(str(out))

        logger.info("Merge complete: %s", output_dir)
        return str(out)

    except MemoryError:
        raise RuntimeError(
            "Not enough RAM to merge LoRA adapter with base model. "
            "The base model is too large for this machine's available memory. "
            "Try: 1) Close other applications, 2) Use a smaller base model, "
            "3) Increase system swap space."
        )
    except Exception as e:
        raise RuntimeError(f"LoRA merge failed: {e}")


def convert_to_gguf(
    model_dir: str,
    output_path: str,
    quantization: str = "Q8_0",
) -> str:
    """
    Convert a HuggingFace model directory to GGUF format.

    quantization options:
      - "f16" = full float16, no quantization (largest, best quality)
      - "Q8_0" = 8-bit quantization (good balance)
      - "Q4_K_M" = 4-bit quantization (smallest, fastest)
    """
    if quantization not in VALID_QUANTIZATIONS:
        raise ValueError(f"Invalid quantization: {quantization}. Choose from: {', '.join(VALID_QUANTIZATIONS)}")

    logger.info("Converting to GGUF: %s -> %s (quant=%s)", model_dir, output_path, quantization)

    # Check disk space (GGUF output is roughly same size as model for Q8, half for Q4)
    model_size = _estimate_model_size(model_dir)
    estimated_output = model_size if quantization == "Q8_0" else model_size // 2
    _check_disk_space(Path(output_path).parent, estimated_output)

    convert_script = _find_convert_script()

    if convert_script:
        # Use llama.cpp's convert script
        cmd = [
            sys.executable,
            str(convert_script),
            model_dir,
            "--outfile", output_path,
            "--outtype", quantization.lower(),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=str(Path(model_dir).parent),  # Set working directory
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "GGUF conversion timed out after 1 hour. "
                "The model may be too large for this machine."
            )
        if result.returncode != 0:
            stderr = result.stderr[:2000] if result.stderr else "No error output"
            raise RuntimeError(f"GGUF conversion failed (exit code {result.returncode}): {stderr}")
    else:
        # Fallback: save as safetensors with clear warning
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_dir)

            model.save_pretrained(output_path, safe_serialization=True)
            tokenizer.save_pretrained(output_path)
            logger.warning(
                "GGUF conversion NOT performed: llama.cpp convert script not found. "
                "Saved as safetensors instead. Install llama-cpp-python for proper GGUF export."
            )
            raise RuntimeError(
                "llama.cpp conversion script not found. Cannot create GGUF file. "
                "Install with: pip install llama-cpp-python  "
                "(requires C++ build tools: sudo apt install build-essential)"
            )
        except RuntimeError:
            raise  # Re-raise our own error
        except Exception as e:
            raise RuntimeError(
                f"GGUF conversion failed. llama.cpp convert script not found and "
                f"fallback also failed: {e}"
            )

    logger.info("GGUF conversion complete: %s", output_path)
    return output_path


def export_project_gguf(
    project_id: int,
    quantization: str = "Q8_0",
    status_dict: Optional[dict] = None,
) -> str:
    """
    Full pipeline: find trained model → merge LoRA if needed → convert to GGUF.

    Args:
        project_id: the project to export
        quantization: "Q8_0" (default), "Q4_K_M", or "f16"
        status_dict: mutable dict for tracking progress from outside

    Returns:
        Path to the generated GGUF file.
    """
    if status_dict is None:
        status_dict = {}

    project_dir = PROJECTS_DIR / str(project_id)
    output_dir = project_dir / "output"
    adapters_dir = project_dir / "adapters"
    gguf_dir = project_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)

    gguf_filename = f"model-{quantization.lower()}.gguf"
    gguf_path = gguf_dir / gguf_filename
    merged_dir = None  # Track for cleanup

    try:
        # Step 1: Determine model source
        status_dict.update({"step": "preparing", "progress": 10, "message": "Finding trained model..."})

        model_source = None

        # Check for merged output first
        if output_dir.exists() and any(output_dir.glob("*.safetensors")):
            model_source = str(output_dir)
            logger.info("Using merged output model: %s", model_source)
        elif output_dir.exists() and any(output_dir.glob("*.bin")):
            model_source = str(output_dir)
        elif adapters_dir.exists() and any(adapters_dir.iterdir()):
            # Need to merge LoRA adapter first
            status_dict.update({"step": "merging", "progress": 20, "message": "Merging LoRA adapter with base model..."})

            # Find adapter config to get base model name
            import json
            adapter_config_path = adapters_dir / "adapter_config.json"
            if not adapter_config_path.exists():
                # Check subdirectories
                for sub in adapters_dir.iterdir():
                    if sub.is_dir() and (sub / "adapter_config.json").exists():
                        adapter_config_path = sub / "adapter_config.json"
                        break

            if not adapter_config_path.exists():
                raise FileNotFoundError("No adapter_config.json found — cannot determine base model")

            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "")

            if not base_model_name:
                raise ValueError("adapter_config.json missing 'base_model_name_or_path'")

            # Merge
            merged_dir = project_dir / "merged_for_gguf"
            model_source = merge_lora_adapter(
                base_model_path=base_model_name,
                adapter_path=str(adapter_config_path.parent),
                output_dir=str(merged_dir),
            )
        else:
            raise FileNotFoundError(
                "No trained model found. Complete training first. "
                "Expected model files in 'output/' or 'adapters/' directory."
            )

        # Step 2: Convert to GGUF
        status_dict.update({"step": "converting", "progress": 60, "message": f"Converting to GGUF ({quantization})..."})

        convert_to_gguf(
            model_dir=model_source,
            output_path=str(gguf_path),
            quantization=quantization,
        )

        # Verify output file exists and is non-empty
        if not gguf_path.exists() or gguf_path.stat().st_size == 0:
            raise RuntimeError("GGUF conversion produced no output file")

        status_dict.update({
            "step": "completed",
            "progress": 100,
            "message": f"GGUF export complete! File: {gguf_filename}",
            "gguf_path": str(gguf_path),
            "gguf_filename": gguf_filename,
            "gguf_size_mb": round(gguf_path.stat().st_size / (1024 * 1024), 1),
        })

        return str(gguf_path)

    except Exception as e:
        logger.error("GGUF export failed for project %d: %s", project_id, e)
        status_dict.update({
            "step": "error",
            "progress": 0,
            "message": str(e),
            "error": str(e),
        })
        raise

    finally:
        # Clean up temporary merged model directory
        if merged_dir and merged_dir.exists():
            try:
                shutil.rmtree(str(merged_dir))
                logger.info("Cleaned up temporary merged model: %s", merged_dir)
            except OSError as e:
                logger.warning("Failed to clean up %s: %s", merged_dir, e)
