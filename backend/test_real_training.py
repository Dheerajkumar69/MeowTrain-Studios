#!/usr/bin/env python3
"""
MeowTrain — End-to-End Real Training Verification Script

This script directly invokes the ML engine to verify that training
actually works with the provided dataset and TinyLlama model.

It bypasses the web API and tests the raw ML pipeline:
  1. Loads the training dataset (Alpaca format)
  2. Tokenizes it using TinyLlama's tokenizer
  3. Creates a QLoRA-configured model + HuggingFace Trainer
  4. Runs 1 epoch of real training
  5. Reports results (loss curve, speed, success/failure)
"""

import json
import os
import shutil
import sys
import time
import tempfile

# Make sure we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set minimal env vars
os.environ.setdefault("MEOWLLM_JWT_SECRET", "test-secret")
os.environ.setdefault("MEOWLLM_DATABASE_URL", "sqlite:///./test_training.db")


def main():
    print("=" * 70)
    print("  MeowTrain — Real Training Verification")
    print("=" * 70)
    print()

    # ─── Step 0: Check Hardware ───────────────────────────────────────
    print("🔍 Step 0: Checking hardware...")
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   ✅ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("   ⚠️  No GPU detected — training will use CPU (very slow)")
    print()

    # ─── Step 1: Load Dataset ─────────────────────────────────────────
    print("📂 Step 1: Loading training dataset...")
    dataset_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "training dataset.txt"
    )
    dataset_path = os.path.normpath(dataset_path)

    if not os.path.exists(dataset_path):
        print(f"   ❌ Dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r") as f:
        raw_data = json.load(f)

    print(f"   ✅ Loaded {len(raw_data)} training examples (Alpaca format)")
    print(f"   📝 Sample: \"{raw_data[0]['instruction'][:60]}...\"")
    print()

    # ─── Step 2: Tokenize ─────────────────────────────────────────────
    print("🔤 Step 2: Loading tokenizer & tokenizing data...")
    from transformers import AutoTokenizer
    from datasets import Dataset as HFDataset

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_tokens = 256  # Keep short for memory efficiency

    # Format as Alpaca-style text
    formatted_texts = []
    for item in raw_data:
        instruction = item.get("instruction", "")
        inp = item.get("input", "")
        output = item.get("output", "")
        if inp:
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        formatted_texts.append(text)

    # Tokenize
    def tokenize_fn(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_tokens,
            padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = HFDataset.from_dict({"text": formatted_texts})
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    # Split 90/10
    split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"   ✅ Tokenized: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
    print(f"   📏 Max tokens: {max_tokens}")
    print()

    # ─── Step 3: Create Model + Trainer ───────────────────────────────
    print("🧠 Step 3: Loading TinyLlama with QLoRA (4-bit)...")
    start_load = time.time()

    from app.ml.trainer import TrainingMetrics, create_model_and_trainer

    # Use local metrics (no multiprocessing needed for this test)
    metrics = TrainingMetrics()

    # Create a temp output dir for checkpoints
    output_dir = tempfile.mkdtemp(prefix="meow_test_training_")

    hyperparams = {
        "method": "qlora",
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 2e-4,
        "max_tokens": max_tokens,
        "warmup_steps": 2,
        "gradient_accumulation_steps": 2,
        "gradient_checkpointing": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "eval_steps": 10000,  # Won't trigger with so few steps
        "save_steps": 10000,  # Don't save checkpoints during test
        "early_stopping_patience": 0,  # Disable early stopping
    }

    try:
        trainer = create_model_and_trainer(
            model_name=model_name,
            training_method="qlora",
            hyperparameters=hyperparams,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            output_dir=output_dir,
            metrics=metrics,
        )
        load_time = time.time() - start_load
        print(f"   ✅ Model loaded & LoRA applied in {load_time:.1f}s")
    except Exception as e:
        print(f"   ❌ Failed to create model/trainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print model info
    if hasattr(trainer.model, "get_nb_trainable_parameters"):
        trainable, total = trainer.model.get_nb_trainable_parameters()
        pct = 100 * trainable / total if total > 0 else 0
        print(f"   📊 Parameters: {trainable:,} trainable / {total:,} total ({pct:.2f}%)")

    print(f"   📊 Total training steps: {metrics.total_steps}")
    print()

    # ─── Step 4: Train! ───────────────────────────────────────────────
    print("🚀 Step 4: Starting REAL training (1 epoch)...")
    print("   This will take a few minutes depending on your hardware.")
    print()

    start_train = time.time()
    try:
        trainer.train()
        train_time = time.time() - start_train
        print()
        print(f"   ✅ Training completed in {train_time:.1f}s!")
    except torch.cuda.OutOfMemoryError:
        print()
        print("   ❌ GPU OUT OF MEMORY!")
        print("   💡 Try: reduce batch_size to 1, max_tokens to 128, or use CPU")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"   ❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ─── Step 5: Verify Results ───────────────────────────────────────
    print()
    print("=" * 70)
    print("  📊 RESULTS")
    print("=" * 70)
    print()

    log_history = metrics.get_log_history()
    losses = [entry["loss"] for entry in log_history if entry.get("loss") is not None]

    print(f"   Status:        {metrics.status}")
    print(f"   Steps:         {metrics.current_step} / {metrics.total_steps}")
    print(f"   Epochs:        {metrics.current_epoch} / {metrics.total_epochs}")
    print(f"   Final Loss:    {metrics.current_loss}")
    print(f"   Best Loss:     {metrics.best_loss}")
    print(f"   Tokens/sec:    {metrics.tokens_per_sec}")
    print(f"   Training Time: {train_time:.1f}s")
    print()

    if losses:
        print("   📈 Loss Curve:")
        for i, entry in enumerate(log_history):
            if entry.get("loss") is not None:
                bar_len = min(50, max(1, int(entry["loss"] * 10)))
                bar = "█" * bar_len
                print(f"      Step {entry.get('step', i):3d}: {entry['loss']:.4f}  {bar}")
        print()

        # Check if loss decreased
        if len(losses) >= 2:
            first_half = sum(losses[:len(losses)//2]) / max(1, len(losses)//2)
            second_half = sum(losses[len(losses)//2:]) / max(1, len(losses) - len(losses)//2)
            decreased = second_half < first_half
        else:
            decreased = False

        print("   ─── Verification ───")
        print(f"   {'✅' if metrics.status == 'completed' else '❌'} Training completed: {metrics.status}")
        print(f"   {'✅' if metrics.current_step > 0 else '❌'} Steps executed: {metrics.current_step}")
        print(f"   {'✅' if metrics.current_loss is not None else '❌'} Loss recorded: {metrics.current_loss}")
        if len(losses) >= 2:
            print(f"   {'✅' if decreased else '⚠️ '} Loss trend: {'decreasing ✓' if decreased else 'not decreasing (normal with tiny dataset/few steps)'}")
            print(f"      First half avg:  {first_half:.4f}")
            print(f"      Second half avg: {second_half:.4f}")
        print()

        all_passed = (
            metrics.status == "completed"
            and metrics.current_step > 0
            and metrics.current_loss is not None
        )

        if all_passed:
            print("   🎉 ═══════════════════════════════════════════")
            print("   🎉  TRAINING IS REAL AND WORKING!")
            print("   🎉  The ML pipeline trains models for real.")
            print("   🎉 ═══════════════════════════════════════════")
        else:
            print("   ⚠️  Training completed but some checks failed.")
    else:
        print("   ❌ No loss values recorded — training may not have run properly.")

    # Cleanup
    print()
    print("🧹 Cleaning up temp files...")
    shutil.rmtree(output_dir, ignore_errors=True)
    if os.path.exists("test_training.db"):
        os.remove("test_training.db")
    print("   ✅ Done!")
    print()


if __name__ == "__main__":
    main()
