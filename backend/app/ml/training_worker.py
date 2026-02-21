"""
Training worker -- isolated subprocess that runs the full training lifecycle.

Bridges the API routes and the ML trainer:
  1. Loads datasets from disk
  2. Creates model + trainer
  3. Runs training in an **isolated child process**
  4. Pushes real-time metrics into shared memory (read by WebSocket)
  5. Updates the database with progress and final results
  6. Handles errors, pause/resume, and stop gracefully

**Why a process, not a thread?**
  - A CUDA segfault in a thread kills the entire API server.
  - Python's GIL prevents true CPU parallelism with threads.
  - multiprocessing.Process gives full crash isolation: if the
    training process dies the API keeps serving requests and the
    worker is marked as "error" on the next health check.
"""

import gc
import logging
import multiprocessing
import multiprocessing.managers
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore

logger = logging.getLogger("meowllm.training_worker")

# How often (in seconds) to sync metrics to the DB
DB_SYNC_INTERVAL = 5.0

# CUDA requires spawn (not fork) to avoid corrupted GPU context in child
_mp_ctx = multiprocessing.get_context("spawn")


# =========================================================================
# Default metrics template
# =========================================================================

_DEFAULT_METRICS: dict = {
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


def _new_shared_metrics(manager: multiprocessing.managers.SyncManager) -> dict:
    """Create a Manager-backed dict with default metric keys."""
    return manager.dict(_DEFAULT_METRICS)


# =========================================================================
# TrainingWorker -- wrapper that sits in the API process
# =========================================================================

class TrainingWorker:
    """
    Manages a single training run in an **isolated child process**.

    Each TrainingWorker corresponds to one training run for one project.
    The worker is created when /start is called and runs until completion,
    stop, or error.  A crash in the child process does NOT affect the
    FastAPI server.
    """

    def __init__(self, project_id: int, run_id: int, config: dict):
        self.project_id = project_id
        self.run_id = run_id
        self.config = config

        # Shared state (survives child process crash)
        self._manager = _mp_ctx.Manager()
        self._shared_metrics = _new_shared_metrics(self._manager)
        self._pause_event = _mp_ctx.Event()
        self._stop_event = _mp_ctx.Event()

        self._process: Optional[multiprocessing.process.BaseProcess] = None
        self._db_sync_thread: Optional[threading.Thread] = None
        self._db_sync_stop = threading.Event()

    # -- Lifecycle -------------------------------------------------

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def pid(self) -> int | None:
        """OS process ID of the training child process (None if not started)."""
        return self._process.pid if self._process else None

    def start(self):
        """Launch training in a child process."""
        if self.is_alive:
            raise RuntimeError("Training is already running for this project.")

        self._process = _mp_ctx.Process(
            target=_training_process_entry,
            name=f"training-project-{self.project_id}-run-{self.run_id}",
            kwargs=dict(
                project_id=self.project_id,
                run_id=self.run_id,
                config=self.config,
                shared_metrics=self._shared_metrics,
                pause_event=self._pause_event,
                stop_event=self._stop_event,
            ),
            daemon=True,
        )
        self._process.start()

        # Start DB-sync thread (runs in main process, safe)
        self._db_sync_stop.clear()
        self._db_sync_thread = threading.Thread(
            target=self._periodic_db_sync,
            name=f"db-sync-project-{self.project_id}",
            daemon=True,
        )
        self._db_sync_thread.start()

        logger.info(
            "Training worker started: project=%d run=%d pid=%d",
            self.project_id, self.run_id, self._process.pid,
        )

    def pause(self):
        """Request training to pause at the next step boundary."""
        self._pause_event.set()
        logger.info("Pause requested for project %d", self.project_id)

    def resume(self):
        """Resume paused training."""
        self._pause_event.clear()
        logger.info("Resume requested for project %d", self.project_id)

    def stop(self):
        """Request graceful stop with checkpoint save."""
        self._stop_event.set()
        self._pause_event.clear()  # Unpause so the trainer can stop
        logger.info("Stop requested for project %d", self.project_id)

    def get_status(self) -> dict:
        """Get current training status for WebSocket/API."""
        try:
            return dict(self._shared_metrics)
        except Exception:
            exitcode = self._process.exitcode if self._process else None
            return {
                **_DEFAULT_METRICS,
                "status": "error",
                "error_message": (
                    f"Training process died unexpectedly (exit code {exitcode}). "
                    "This may be caused by a GPU driver crash or out-of-memory. "
                    "Check the server logs for details."
                ),
            }

    def join(self, timeout: float = 30):
        """Wait for the child process to finish."""
        if self._process is not None:
            self._process.join(timeout=timeout)
        self._db_sync_stop.set()

    def kill(self):
        """Force-kill the child process (last resort)."""
        if self._process is not None and self._process.is_alive():
            self._process.kill()
            logger.warning(
                "Force-killed training process for project %d",
                self.project_id,
            )
        self._db_sync_stop.set()

    def cleanup(self):
        """Release Manager resources after the worker is done."""
        try:
            self._db_sync_stop.set()
            self._manager.shutdown()
        except Exception:
            pass

    # -- DB sync (runs in main process) ---------------------------

    def _periodic_db_sync(self):
        """
        Periodically sync training metrics to the DB.
        Runs in the **main** (API) process, not in the child.
        """
        from app.database import SessionLocal
        from app.models.training_run import TrainingRun
        from app.models.project import Project

        while not self._db_sync_stop.is_set():
            self._db_sync_stop.wait(DB_SYNC_INTERVAL)
            if self._db_sync_stop.is_set():
                break

            # Check if child process died unexpectedly
            if self._process and not self._process.is_alive():
                exitcode = self._process.exitcode
                if exitcode is not None and exitcode != 0:
                    logger.error(
                        "Training process for project %d exited with code %d",
                        self.project_id, exitcode,
                    )
                    db = SessionLocal()
                    try:
                        run = db.query(TrainingRun).filter(
                            TrainingRun.id == self.run_id
                        ).first()
                        if run and run.status not in ("completed", "error", "stopped"):
                            run.status = "error"
                            run.error_message = (
                                f"Training process crashed (exit code {exitcode}). "
                                "This may indicate a GPU driver issue or OOM."
                            )
                            run.completed_at = datetime.now(timezone.utc)
                            db.commit()
                        project = db.query(Project).filter(
                            Project.id == self.project_id
                        ).first()
                        if project:
                            project.status = "created"
                            db.commit()
                    except Exception as e:
                        logger.error("DB crash-finalize failed: %s", e)
                        try:
                            db.rollback()
                        except Exception:
                            pass
                    finally:
                        db.close()
                break

            # Normal sync
            db = SessionLocal()
            try:
                status_snapshot = self.get_status()
                run = db.query(TrainingRun).filter(
                    TrainingRun.id == self.run_id
                ).first()
                if run:
                    run.status = status_snapshot.get("status", run.status)
                    run.current_step = status_snapshot.get("current_step", 0)
                    run.current_epoch = status_snapshot.get("current_epoch", 0)
                    run.current_loss = status_snapshot.get("current_loss")
                    run.best_loss = status_snapshot.get("best_loss")
                    run.tokens_per_sec = status_snapshot.get("tokens_per_sec", 0.0)
                    run.validation_loss = status_snapshot.get("validation_loss")
                    run.learning_rate_current = status_snapshot.get("learning_rate_current")
                    err = status_snapshot.get("error_message")
                    if err:
                        run.error_message = err
                    db.commit()
            except Exception as e:
                logger.warning("Periodic DB sync failed: %s", e)
                try:
                    db.rollback()
                except Exception:
                    pass
            finally:
                db.close()


# =========================================================================
# Child-process entry point (top-level function, must be picklable)
# =========================================================================

def _training_process_entry(
    project_id: int,
    run_id: int,
    config: dict,
    shared_metrics: dict,
    pause_event,
    stop_event,
):
    """
    Top-level function executed in the child process.

    Everything in here is fully isolated -- a segfault or OOM
    in this process does NOT affect the API server.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    child_logger = logging.getLogger("meowllm.training_child")
    child_logger.info(
        "Child process started: project=%d run=%d pid=%d",
        project_id, run_id, os.getpid(),
    )

    # Import everything fresh inside the child process
    from app.database import SessionLocal
    from app.models.training_run import TrainingRun
    from app.models.dataset import Dataset
    from app.models.project import Project
    from app.ml.trainer import TrainingMetrics

    metrics = TrainingMetrics(
        shared_dict=shared_metrics,
        pause_event=pause_event,
        stop_event=stop_event,
    )

    db = SessionLocal()
    start_time = time.time()
    _trainer = None

    try:
        metrics.status = "initializing"
        _update_db_status(db, run_id, "running")

        # -- Step 1: Load Datasets --
        child_logger.info("Step 1: Loading datasets for project %d", project_id)
        metrics.status = "loading_data"

        datasets = db.query(Dataset).filter(
            Dataset.project_id == project_id,
            Dataset.status == "ready",
        ).all()

        model_name = config.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        max_tokens = config.get("max_tokens", 512)
        train_split = config.get("train_split", 0.9)
        training_method = config.get("method", "lora")
        is_alignment = training_method in ("dpo", "orpo")

        if is_alignment:
            from app.ml.data_loader import load_preference_dataset
            data_result = load_preference_dataset(
                project_id=project_id,
                dataset_records=datasets,
                tokenizer_name=model_name,
                max_tokens=max_tokens,
                train_split=train_split,
            )
        else:
            from app.ml.data_loader import load_datasets_for_project
            data_result = load_datasets_for_project(
                project_id=project_id,
                dataset_records=datasets,
                tokenizer_name=model_name,
                max_tokens=max_tokens,
                train_split=train_split,
            )

        train_dataset = data_result["train_dataset"]
        eval_dataset = data_result["eval_dataset"]
        tokenizer = data_result["tokenizer"]
        total_samples = data_result["total_samples"]

        child_logger.info(
            "Data loaded: %d total samples, %d train, %s eval",
            total_samples,
            len(train_dataset),
            len(eval_dataset) if eval_dataset else "none",
        )

        # -- Step 2: Setup Checkpoint Manager --
        from app.ml.checkpoint_manager import CheckpointManager
        checkpoint_manager = CheckpointManager(
            project_id=project_id,
            run_id=run_id,
        )

        # -- Step 3: Create Model & Trainer --
        child_logger.info("Step 2: Creating model & trainer")
        metrics.status = "loading_model"

        output_dir = str(checkpoint_manager.checkpoint_dir)

        if training_method == "dpo":
            from app.ml.dpo_trainer import create_dpo_trainer
            _trainer = create_dpo_trainer(
                model_name=model_name,
                training_method=training_method,
                hyperparameters=config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                output_dir=output_dir,
                metrics=metrics,
                checkpoint_manager=checkpoint_manager,
            )
        elif training_method == "orpo":
            from app.ml.dpo_trainer import create_orpo_trainer
            _trainer = create_orpo_trainer(
                model_name=model_name,
                training_method=training_method,
                hyperparameters=config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                output_dir=output_dir,
                metrics=metrics,
                checkpoint_manager=checkpoint_manager,
            )
        else:
            from app.ml.trainer import create_model_and_trainer
            _trainer = create_model_and_trainer(
                model_name=model_name,
                training_method=training_method,
                hyperparameters=config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                output_dir=output_dir,
                metrics=metrics,
                checkpoint_manager=checkpoint_manager,
            )

        # Update DB with total steps
        _update_db_progress(db, run_id, total_steps=metrics.total_steps)

        # -- Step 4: Train --
        child_logger.info("Step 3: Starting training")
        metrics.status = "running"

        resume_ckpt = None
        if config.get("resume_from_checkpoint", False):
            resume_ckpt = checkpoint_manager.get_latest_checkpoint()
            if resume_ckpt:
                child_logger.info("Resuming from checkpoint: %s", resume_ckpt)
            else:
                child_logger.info("No checkpoint found, starting fresh")

        _trainer.train(resume_from_checkpoint=resume_ckpt)

        # -- Step 5: Post-training --
        if metrics.stop_requested:
            child_logger.info("Training stopped by user after %d steps", metrics.current_step)
            metrics.status = "stopped"
            try:
                checkpoint_manager.save_checkpoint(
                    _trainer, metrics.current_step, metrics.current_loss
                )
            except Exception as e:
                child_logger.warning("Failed to save stop checkpoint: %s", e)
        else:
            child_logger.info("Training completed successfully")
            metrics.status = "completing"

            try:
                checkpoint_manager.save_checkpoint(
                    _trainer, metrics.current_step, metrics.current_loss
                )
            except Exception as e:
                child_logger.warning("Failed to save final checkpoint: %s", e)

            try:
                output_path = checkpoint_manager.merge_and_export(
                    base_model_name=model_name,
                    training_method=training_method,
                    tokenizer=tokenizer,
                )
                if output_path:
                    _update_db_output_path(db, run_id, output_path)
                    child_logger.info("Model exported to: %s", output_path)
            except Exception as e:
                child_logger.error("Model export failed (non-fatal): %s", e)

            metrics.status = "completed"

        # -- Step 6: Final DB Update --
        elapsed = time.time() - start_time
        _finalize_db(db, run_id, project_id, metrics, elapsed)

    except Exception as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "OutOfMemoryError" in type(e).__name__:
            error_msg = (
                "GPU ran out of memory during training. Try:\n"
                "  Use QLoRA instead of LoRA\n"
                "  Reduce batch size\n"
                "  Reduce max tokens\n"
                "  Use a smaller model"
            )
        elif not error_msg:
            error_msg = f"Unexpected error: {type(e).__name__}"
        child_logger.error("Training error:\n%s", traceback.format_exc())
        metrics.error_message = error_msg
        metrics.status = "error"
        _finalize_db(db, run_id, project_id, metrics, time.time() - start_time, error_msg)
        _cleanup_gpu()

    finally:
        db.close()
        if _trainer is not None:
            try:
                del _trainer.model
                del _trainer
            except Exception:
                pass
        _cleanup_gpu()
        child_logger.info(
            "Training child process finished: project=%d run=%d status=%s pid=%d",
            project_id, run_id, metrics.status, os.getpid(),
        )


# =========================================================================
# DB helpers (called from child process)
# =========================================================================

def _update_db_status(db, run_id: int, status: str):
    from app.models.training_run import TrainingRun
    try:
        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run:
            run.status = status
            db.commit()
    except Exception as e:
        logger.error("DB status update failed: %s", e)
        db.rollback()


def _update_db_progress(db, run_id: int, **kwargs):
    from app.models.training_run import TrainingRun
    try:
        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run:
            for key, value in kwargs.items():
                if hasattr(run, key):
                    setattr(run, key, value)
            db.commit()
    except Exception as e:
        logger.error("DB progress update failed: %s", e)
        db.rollback()


def _update_db_output_path(db, run_id: int, output_path: str):
    from app.models.training_run import TrainingRun
    try:
        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run:
            run.output_path = output_path
            db.commit()
    except Exception as e:
        logger.error("DB output path update failed: %s", e)
        db.rollback()


def _finalize_db(db, run_id: int, project_id: int, metrics, elapsed: float, error_msg: str = None):
    from app.models.training_run import TrainingRun
    from app.models.project import Project
    status = metrics.status
    try:
        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run:
            run.status = status
            run.completed_at = datetime.now(timezone.utc)
            run.current_step = metrics.current_step
            run.current_epoch = metrics.current_epoch
            run.current_loss = metrics.current_loss
            run.best_loss = metrics.best_loss
            run.tokens_per_sec = metrics.tokens_per_sec
            run.validation_loss = metrics.validation_loss
            run.perplexity = metrics.perplexity
            run.learning_rate_current = metrics.learning_rate_current
            if error_msg:
                run.error_message = error_msg
            log_hist = metrics.log_history
            if log_hist:
                run.log_history = log_hist[-5000:]
            db.commit()

        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            if status == "completed":
                project.status = "trained"
            elif status in ("error", "stopped"):
                project.status = "created"
            db.commit()

        logger.info("DB finalized: run=%d status=%s elapsed=%.1fs", run_id, status, elapsed)
    except Exception as e:
        logger.error("DB finalization failed: %s", e)
        db.rollback()


def _cleanup_gpu():
    """Force GPU/device memory cleanup (CUDA, MPS, or CPU)."""
    try:
        gc.collect()
        if torch is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
    except Exception:
        pass
