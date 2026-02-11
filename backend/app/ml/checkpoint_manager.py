"""
Checkpoint manager for MeowLLM training pipeline.

Handles:
  - Saving intermediate checkpoints during training
  - Keeping only the N most recent checkpoints (disk-safe)
  - Tracking the best checkpoint by validation loss
  - Merging LoRA adapters back into the base model on completion
  - Exporting the final model to the project output directory
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

from app.config import PROJECTS_DIR

logger = logging.getLogger("meowllm.checkpoint_manager")

DEFAULT_MAX_CHECKPOINTS = 3


class CheckpointManager:
    """
    Manages training checkpoints for a single training run.
    
    Directory layout:
        data/projects/{project_id}/
            checkpoints/
                run_{run_id}/
                    checkpoint-100/
                    checkpoint-200/
                    best/
            output/
                run_{run_id}/
    """

    def __init__(
        self,
        project_id: int,
        run_id: int,
        max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
    ):
        self.project_id = project_id
        self.run_id = run_id
        self.max_checkpoints = max(1, max_checkpoints)

        self.base_dir = PROJECTS_DIR / str(project_id)
        self.checkpoint_dir = self.base_dir / "checkpoints" / f"run_{run_id}"
        self.output_dir = self.base_dir / "output" / f"run_{run_id}"
        self.best_checkpoint_dir = self.checkpoint_dir / "best"

        # Create dirs
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._best_loss: Optional[float] = None
        self._checkpoint_steps: list[int] = []

        logger.info(
            "CheckpointManager initialized: project=%d run=%d dir=%s",
            project_id, run_id, self.checkpoint_dir,
        )

    @property
    def best_loss(self) -> Optional[float]:
        return self._best_loss

    def get_checkpoint_path(self, step: int) -> Path:
        """Get the path for a checkpoint at a given step."""
        return self.checkpoint_dir / f"checkpoint-{step}"

    def get_output_path(self) -> Path:
        """Get the final model output path."""
        return self.output_dir

    def save_checkpoint(
        self,
        trainer,
        step: int,
        loss: Optional[float] = None,
    ) -> Path:
        """
        Save a checkpoint using the HuggingFace Trainer.

        Also handles:
          - Pruning old checkpoints beyond max_checkpoints
          - Tracking best checkpoint by loss
        """
        checkpoint_path = self.get_checkpoint_path(step)

        try:
            trainer.save_model(str(checkpoint_path))
            logger.info("Checkpoint saved at step %d: %s", step, checkpoint_path)
        except Exception as e:
            logger.error("Failed to save checkpoint at step %d: %s", step, e)
            raise

        # Track this checkpoint
        self._checkpoint_steps.append(step)

        # Update best checkpoint
        if loss is not None:
            if self._best_loss is None or loss < self._best_loss:
                self._best_loss = loss
                self._save_best_checkpoint(checkpoint_path)
                logger.info("New best checkpoint at step %d (loss=%.6f)", step, loss)

        # Prune old checkpoints
        self._prune_old_checkpoints()

        return checkpoint_path

    def _save_best_checkpoint(self, source_path: Path):
        """Copy the current checkpoint as the best."""
        try:
            if self.best_checkpoint_dir.exists():
                shutil.rmtree(self.best_checkpoint_dir)
            shutil.copytree(source_path, self.best_checkpoint_dir)
        except Exception as e:
            logger.error("Failed to save best checkpoint: %s", e)

    def _prune_old_checkpoints(self):
        """Remove checkpoints beyond the max, keeping the most recent ones."""
        if len(self._checkpoint_steps) <= self.max_checkpoints:
            return

        steps_to_remove = self._checkpoint_steps[:-self.max_checkpoints]
        self._checkpoint_steps = self._checkpoint_steps[-self.max_checkpoints:]

        for step in steps_to_remove:
            ckpt_path = self.get_checkpoint_path(step)
            if ckpt_path.exists():
                try:
                    shutil.rmtree(ckpt_path)
                    logger.info("Pruned old checkpoint: step %d", step)
                except Exception as e:
                    logger.warning("Failed to prune checkpoint step %d: %s", step, e)

    def merge_and_export(
        self,
        base_model_name: str,
        training_method: str,
        tokenizer=None,
    ) -> Optional[str]:
        """
        After training completes, merge LoRA adapters into the base model
        and save the final model to the output directory.

        For full fine-tune, just copies from the last checkpoint.

        Returns the output path as a string, or None on failure.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine source: best checkpoint if available, else latest
        source_dir = None
        if self.best_checkpoint_dir.exists():
            source_dir = self.best_checkpoint_dir
        elif self._checkpoint_steps:
            source_dir = self.get_checkpoint_path(self._checkpoint_steps[-1])

        if source_dir is None or not source_dir.exists():
            logger.error("No checkpoint found to export.")
            return None

        if training_method in ("lora", "qlora"):
            return self._merge_lora(base_model_name, source_dir, tokenizer)
        else:
            return self._export_full(source_dir, tokenizer)

    def _merge_lora(
        self,
        base_model_name: str,
        adapter_dir: Path,
        tokenizer=None,
    ) -> Optional[str]:
        """Merge LoRA/QLoRA adapters into the base model."""
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM

            logger.info("Loading base model %s for LoRA merge...", base_model_name)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            logger.info("Loading adapter from %s...", adapter_dir)
            model = PeftModel.from_pretrained(base_model, str(adapter_dir))

            logger.info("Merging LoRA weights...")
            merged = model.merge_and_unload()

            logger.info("Saving merged model to %s...", self.output_dir)
            merged.save_pretrained(str(self.output_dir))

            if tokenizer is not None:
                tokenizer.save_pretrained(str(self.output_dir))

            logger.info("LoRA merge complete: %s", self.output_dir)
            return str(self.output_dir)

        except Exception as e:
            logger.error("LoRA merge failed: %s", e)
            # Fallback: just copy the adapter
            try:
                if self.output_dir.exists():
                    shutil.rmtree(self.output_dir)
                shutil.copytree(adapter_dir, self.output_dir)
                logger.info("Copied adapter (unmerged) to %s", self.output_dir)
                return str(self.output_dir)
            except Exception as e2:
                logger.error("Adapter copy also failed: %s", e2)
                return None

    def _export_full(self, source_dir: Path, tokenizer=None) -> Optional[str]:
        """Export a full fine-tuned model."""
        try:
            if self.output_dir.exists():
                shutil.rmtree(self.output_dir)
            shutil.copytree(source_dir, self.output_dir)

            if tokenizer is not None:
                tokenizer.save_pretrained(str(self.output_dir))

            logger.info("Full model exported to %s", self.output_dir)
            return str(self.output_dir)

        except Exception as e:
            logger.error("Model export failed: %s", e)
            return None

    def cleanup_checkpoints(self):
        """Remove ALL checkpoints for this run (called after successful export)."""
        try:
            if self.checkpoint_dir.exists():
                shutil.rmtree(self.checkpoint_dir)
                logger.info("Cleaned up checkpoints for run %d", self.run_id)
        except Exception as e:
            logger.warning("Checkpoint cleanup failed: %s", e)

    def get_disk_usage_mb(self) -> float:
        """Get total disk usage of checkpoints + output in MB."""
        total = 0
        for d in [self.checkpoint_dir, self.output_dir]:
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        try:
                            total += f.stat().st_size
                        except OSError:
                            pass
        return round(total / (1024 * 1024), 1)
