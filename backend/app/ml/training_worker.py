"""
Training worker — background thread that runs the full training lifecycle.

Bridges the API routes and the ML trainer:
  1. Loads datasets from disk
  2. Creates model + trainer
  3. Runs training in a background thread
  4. Pushes real-time metrics into a shared state (read by WebSocket)
  5. Updates the database with progress and final results
  6. Handles errors, pause/resume, and stop gracefully
"""

import gc
import logging
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from app.database import SessionLocal
from app.models.training_run import TrainingRun
from app.models.dataset import Dataset
from app.models.project import Project

# Heavy ML imports are deferred to _run_training()
# from app.ml.data_loader import load_datasets_for_project
# from app.ml.trainer import TrainingMetrics, create_model_and_trainer
# from app.ml.checkpoint_manager import CheckpointManager

logger = logging.getLogger("meowllm.training_worker")

# How often (in seconds) to sync metrics to the DB
DB_SYNC_INTERVAL = 5.0


class TrainingWorker:
    """
    Manages a single training run in a background thread.
    
    Each TrainingWorker corresponds to one training run for one project.
    The worker is created when /start is called and runs until completion,
    stop, or error.
    """

    def __init__(self, project_id: int, run_id: int, config: dict):
        self.project_id = project_id
        self.run_id = run_id
        self.config = config
        # Lazy import TrainingMetrics
        from app.ml.trainer import TrainingMetrics
        self.metrics = TrainingMetrics()
        self._thread: Optional[threading.Thread] = None
        self._trainer = None
        self._checkpoint_manager = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        """Launch training in a background thread."""
        if self.is_alive:
            raise RuntimeError("Training is already running for this project.")

        self._thread = threading.Thread(
            target=self._run_training,
            name=f"training-project-{self.project_id}-run-{self.run_id}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Training worker started: project=%d run=%d",
            self.project_id, self.run_id,
        )

    def pause(self):
        """Request training to pause at the next step boundary."""
        self.metrics.pause_requested = True
        logger.info("Pause requested for project %d", self.project_id)

    def resume(self):
        """Resume paused training."""
        self.metrics.pause_requested = False
        logger.info("Resume requested for project %d", self.project_id)

    def stop(self):
        """Request graceful stop with checkpoint save."""
        self.metrics.stop_requested = True
        self.metrics.pause_requested = False  # Unpause if paused
        logger.info("Stop requested for project %d", self.project_id)

    def get_status(self) -> dict:
        """Get current training status for WebSocket/API."""
        m = self.metrics
        return {
            "status": m.status,
            "current_step": m.current_step,
            "total_steps": m.total_steps,
            "current_epoch": m.current_epoch,
            "total_epochs": m.total_epochs,
            "current_loss": m.current_loss,
            "best_loss": m.best_loss,
            "validation_loss": m.validation_loss,
            "learning_rate_current": m.learning_rate_current,
            "tokens_per_sec": m.tokens_per_sec,
            "error_message": m.error_message,
        }

    def _run_training(self):
        """
        Main training loop — runs in a background thread.
        
        This is the heart of the training pipeline. It:
          1. Loads datasets
          2. Creates the model and trainer
          3. Runs training (with periodic DB sync)
          4. Saves the final model
          5. Updates the database
        """
        db = SessionLocal()
        start_time = time.time()

        # Import heavy ML modules here (not at module level)
        from app.ml.data_loader import load_datasets_for_project
        from app.ml.trainer import create_model_and_trainer
        from app.ml.checkpoint_manager import CheckpointManager

        try:
            self.metrics.status = "initializing"
            self._update_db_status(db, "running")

            # ── Step 1: Load Datasets ──
            logger.info("Step 1: Loading datasets for project %d", self.project_id)
            self.metrics.status = "loading_data"

            datasets = db.query(Dataset).filter(
                Dataset.project_id == self.project_id,
                Dataset.status == "ready",
            ).all()

            model_name = self.config.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            max_tokens = self.config.get("max_tokens", 512)
            train_split = self.config.get("train_split", 0.9)

            data_result = load_datasets_for_project(
                project_id=self.project_id,
                dataset_records=datasets,
                tokenizer_name=model_name,
                max_tokens=max_tokens,
                train_split=train_split,
            )

            train_dataset = data_result["train_dataset"]
            eval_dataset = data_result["eval_dataset"]
            tokenizer = data_result["tokenizer"]
            total_samples = data_result["total_samples"]

            logger.info(
                "Data loaded: %d total samples, %d train, %s eval",
                total_samples,
                len(train_dataset),
                len(eval_dataset) if eval_dataset else "none",
            )

            # ── Step 2: Setup Checkpoint Manager ──
            self._checkpoint_manager = CheckpointManager(
                project_id=self.project_id,
                run_id=self.run_id,
            )

            # ── Step 3: Create Model & Trainer ──
            logger.info("Step 2: Creating model & trainer")
            self.metrics.status = "loading_model"

            training_method = self.config.get("method", "lora")
            output_dir = str(self._checkpoint_manager.checkpoint_dir)

            self._trainer = create_model_and_trainer(
                model_name=model_name,
                training_method=training_method,
                hyperparameters=self.config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                output_dir=output_dir,
                metrics=self.metrics,
                checkpoint_manager=self._checkpoint_manager,
            )

            # Update DB with total steps
            self._update_db_progress(db, total_steps=self.metrics.total_steps)

            # ── Step 4: Start periodic DB sync + Train ──
            logger.info("Step 3: Starting training")
            self.metrics.status = "running"

            # Launch a periodic DB sync thread
            self._db_sync_stop = threading.Event()
            db_sync_thread = threading.Thread(
                target=self._periodic_db_sync,
                name=f"db-sync-project-{self.project_id}",
                daemon=True,
            )
            db_sync_thread.start()

            train_result = self._trainer.train()

            # Stop the DB sync thread
            self._db_sync_stop.set()
            db_sync_thread.join(timeout=3)

            # ── Step 5: Post-training ──
            if self.metrics.stop_requested:
                logger.info("Training stopped by user after %d steps", self.metrics.current_step)
                self.metrics.status = "stopped"
                # Save final checkpoint
                try:
                    self._checkpoint_manager.save_checkpoint(
                        self._trainer, self.metrics.current_step, self.metrics.current_loss
                    )
                except Exception as e:
                    logger.warning("Failed to save stop checkpoint: %s", e)
            else:
                logger.info("Training completed successfully")
                self.metrics.status = "completing"

                # Save final checkpoint
                try:
                    self._checkpoint_manager.save_checkpoint(
                        self._trainer, self.metrics.current_step, self.metrics.current_loss
                    )
                except Exception as e:
                    logger.warning("Failed to save final checkpoint: %s", e)

                # Merge and export
                try:
                    output_path = self._checkpoint_manager.merge_and_export(
                        base_model_name=model_name,
                        training_method=training_method,
                        tokenizer=tokenizer,
                    )
                    if output_path:
                        self._update_db_output_path(db, output_path)
                        logger.info("Model exported to: %s", output_path)
                except Exception as e:
                    logger.error("Model export failed (non-fatal): %s", e)

                self.metrics.status = "completed"

            # ── Step 6: Final DB Update ──
            elapsed = time.time() - start_time
            self._finalize_db(db, self.metrics.status, elapsed)

        except torch.cuda.OutOfMemoryError:
            error_msg = (
                "GPU ran out of memory during training. Try:\n"
                "• Use QLoRA instead of LoRA\n"
                "• Reduce batch size\n"
                "• Reduce max tokens\n"
                "• Use a smaller model"
            )
            logger.error("OOM Error: %s", error_msg)
            self.metrics.error_message = error_msg
            self.metrics.status = "error"
            self._finalize_db(db, "error", time.time() - start_time, error_msg)
            _cleanup_gpu()

        except ValueError as e:
            # Data validation errors (no datasets, empty data, etc.)
            error_msg = str(e)
            logger.error("Validation error: %s", error_msg)
            self.metrics.error_message = error_msg
            self.metrics.status = "error"
            self._finalize_db(db, "error", time.time() - start_time, error_msg)

        except RuntimeError as e:
            error_msg = str(e)
            logger.error("Runtime error: %s", error_msg)
            self.metrics.error_message = error_msg
            self.metrics.status = "error"
            self._finalize_db(db, "error", time.time() - start_time, error_msg)
            if "CUDA" in error_msg or "out of memory" in error_msg.lower():
                _cleanup_gpu()

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error("Unexpected error:\n%s", traceback.format_exc())
            self.metrics.error_message = error_msg
            self.metrics.status = "error"
            self._finalize_db(db, "error", time.time() - start_time, error_msg)

        finally:
            db.close()
            # Clean up model from memory
            if self._trainer is not None:
                try:
                    del self._trainer.model
                    del self._trainer
                    self._trainer = None
                except Exception:
                    pass
            _cleanup_gpu()
            logger.info(
                "Training worker finished: project=%d run=%d status=%s",
                self.project_id, self.run_id, self.metrics.status,
            )

    def _update_db_status(self, db, status: str):
        """Update training run status in DB."""
        try:
            run = db.query(TrainingRun).filter(TrainingRun.id == self.run_id).first()
            if run:
                run.status = status
                db.commit()
        except Exception as e:
            logger.error("DB status update failed: %s", e)
            db.rollback()

    def _update_db_progress(self, db, **kwargs):
        """Update training run progress fields in DB."""
        try:
            run = db.query(TrainingRun).filter(TrainingRun.id == self.run_id).first()
            if run:
                for key, value in kwargs.items():
                    if hasattr(run, key):
                        setattr(run, key, value)
                db.commit()
        except Exception as e:
            logger.error("DB progress update failed: %s", e)
            db.rollback()

    def _update_db_output_path(self, db, output_path: str):
        """Save the output model path to DB."""
        try:
            run = db.query(TrainingRun).filter(TrainingRun.id == self.run_id).first()
            if run:
                run.output_path = output_path
                db.commit()
        except Exception as e:
            logger.error("DB output path update failed: %s", e)
            db.rollback()

    def _finalize_db(self, db, status: str, elapsed: float, error_msg: str = None):
        """Final DB update when training ends."""
        try:
            run = db.query(TrainingRun).filter(TrainingRun.id == self.run_id).first()
            if run:
                run.status = status
                run.completed_at = datetime.now(timezone.utc)
                run.current_step = self.metrics.current_step
                run.current_epoch = self.metrics.current_epoch
                run.current_loss = self.metrics.current_loss
                run.best_loss = self.metrics.best_loss
                run.tokens_per_sec = self.metrics.tokens_per_sec
                run.validation_loss = self.metrics.validation_loss
                run.learning_rate_current = self.metrics.learning_rate_current
                if error_msg:
                    run.error_message = error_msg
                # Save log history as JSON
                if self.metrics.log_history:
                    run.log_history = self.metrics.log_history[-5000:]  # Cap storage
                db.commit()

            # Update project status
            project = db.query(Project).filter(Project.id == self.project_id).first()
            if project:
                if status == "completed":
                    project.status = "trained"
                elif status == "error":
                    project.status = "created"
                elif status == "stopped":
                    project.status = "created"
                db.commit()

            logger.info(
                "DB finalized: run=%d status=%s elapsed=%.1fs",
                self.run_id, status, elapsed,
            )
        except Exception as e:
            logger.error("DB finalization failed: %s", e)
            db.rollback()

    def _periodic_db_sync(self):
        """
        Periodically sync training metrics to the DB so that
        the WebSocket and /status always have fresh data,
        even if the worker crashes before finalization.
        """
        while not self._db_sync_stop.is_set():
            self._db_sync_stop.wait(DB_SYNC_INTERVAL)
            if self._db_sync_stop.is_set():
                break
            db = SessionLocal()
            try:
                run = db.query(TrainingRun).filter(TrainingRun.id == self.run_id).first()
                if run:
                    run.status = self.metrics.status
                    run.current_step = self.metrics.current_step
                    run.current_epoch = self.metrics.current_epoch
                    run.current_loss = self.metrics.current_loss
                    run.best_loss = self.metrics.best_loss
                    run.tokens_per_sec = self.metrics.tokens_per_sec
                    run.validation_loss = self.metrics.validation_loss
                    run.learning_rate_current = self.metrics.learning_rate_current
                    if self.metrics.error_message:
                        run.error_message = self.metrics.error_message
                    db.commit()
            except Exception as e:
                logger.warning("Periodic DB sync failed: %s", e)
                try:
                    db.rollback()
                except Exception:
                    pass
            finally:
                db.close()


def _cleanup_gpu():
    """Force GPU memory cleanup."""
    try:
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
