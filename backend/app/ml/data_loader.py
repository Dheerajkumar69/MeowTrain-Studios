"""
Dataset loader for MeowLLM training pipeline.

Loads project datasets from disk, extracts text, tokenizes, and returns
HuggingFace Dataset objects ready for the Trainer.

Supports three data formats:
  1. JSON instruction-response pairs  →  chat/instruct fine-tuning
  2. CSV with text columns            →  structured text training
  3. Raw text (.txt, .md, .pdf, .docx)→  causal language modelling
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from app.config import PROJECTS_DIR
from app.utils.text_extractor import extract_text

logger = logging.getLogger("meowllm.data_loader")

# Maximum number of raw text characters to load per project (safety cap)
MAX_TEXT_CHARS = 50_000_000  # ~50 MB of text


def load_datasets_for_project(
    project_id: int,
    dataset_records: list,
    tokenizer_name: str,
    max_tokens: int = 512,
    train_split: float = 0.9,
) -> dict:
    """
    Load all ready datasets for a project, tokenize, and split.

    Args:
        project_id: DB project ID
        dataset_records: list of Dataset ORM objects (with .filename, .file_type, .status)
        tokenizer_name: HuggingFace model ID for the tokenizer
        max_tokens: Max sequence length per training example
        train_split: Fraction for training (rest is validation)

    Returns:
        {
            "train_dataset": HFDataset,
            "eval_dataset": HFDataset or None,
            "total_samples": int,
            "tokenizer": PreTrainedTokenizerBase,
        }

    Raises:
        ValueError: if no usable data is found
    """
    ready_datasets = [d for d in dataset_records if d.status == "ready"]
    if not ready_datasets:
        raise ValueError("No datasets with status 'ready' found for this project.")

    # Load tokenizer
    tokenizer = _load_tokenizer(tokenizer_name, max_tokens)

    # Gather raw text examples from all datasets
    all_examples = []
    dataset_dir = PROJECTS_DIR / str(project_id) / "datasets"

    for ds in ready_datasets:
        file_path = dataset_dir / ds.filename
        if not file_path.exists():
            logger.warning("Dataset file missing on disk: %s", file_path)
            continue

        try:
            examples = _extract_examples(file_path, ds.file_type)
            all_examples.extend(examples)
            logger.info(
                "Loaded %d examples from %s (%s)",
                len(examples), ds.original_name, ds.file_type,
            )
        except Exception as e:
            logger.error("Failed to load dataset %s: %s", ds.original_name, e)
            continue

    if not all_examples:
        raise ValueError(
            "Could not extract any training examples from the uploaded datasets. "
            "Ensure files contain text and are not corrupted."
        )

    # Tokenize
    tokenized = _tokenize_examples(all_examples, tokenizer, max_tokens)

    if len(tokenized["input_ids"]) == 0:
        raise ValueError(
            "Tokenization produced 0 samples. Your data may be too short or empty."
        )

    hf_dataset = HFDataset.from_dict(tokenized)
    total_samples = len(hf_dataset)

    logger.info("Total tokenized samples: %d", total_samples)

    # Split
    if total_samples < 2 or train_split >= 0.99:
        logger.info("Dataset too small for split or split=1.0 — using all for training.")
        return {
            "train_dataset": hf_dataset,
            "eval_dataset": None,
            "total_samples": total_samples,
            "tokenizer": tokenizer,
        }

    split = hf_dataset.train_test_split(
        test_size=max(1 - train_split, 1 / total_samples),
        seed=42,
    )

    logger.info(
        "Split: %d train, %d eval",
        len(split["train"]), len(split["test"]),
    )

    return {
        "train_dataset": split["train"],
        "eval_dataset": split["test"],
        "total_samples": total_samples,
        "tokenizer": tokenizer,
    }


# ── Tokenizer Loading ────────────────────────────────────────────────


def _load_tokenizer(
    model_name: str, max_tokens: int
) -> PreTrainedTokenizerBase:
    """Load and configure tokenizer with proper padding."""
    logger.info("Loading tokenizer for %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    # Ensure pad token exists (many models lack one)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tokenizer.model_max_length = max_tokens
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    return tokenizer


# ── Example Extraction ───────────────────────────────────────────────


def _extract_examples(file_path: Path, file_type: str) -> list[dict]:
    """
    Extract training examples from a file.

    Returns a list of dicts, each with at least a "text" key.
    For instruction data, also has "instruction" and "response" keys.
    """
    file_type = file_type.lower()

    # JSON: try to detect instruction/response format
    if file_type == ".json":
        return _extract_json_examples(file_path)

    # CSV: try to detect instruction/response columns
    if file_type == ".csv":
        return _extract_csv_examples(file_path)

    # Everything else: raw text
    extracted = extract_text(file_path, file_type)
    text = extracted.get("text", "")
    if not text.strip():
        return []

    # Enforce text size cap
    text = text[:MAX_TEXT_CHARS]

    # Split into paragraphs for better training chunks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    # Merge very short paragraphs, split very long ones
    examples = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) < 2000:
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            if buffer:
                examples.append({"text": buffer.strip()})
            buffer = para

    if buffer:
        examples.append({"text": buffer.strip()})

    return examples


def _extract_json_examples(file_path: Path) -> list[dict]:
    """Extract from JSON: supports instruction/response, conversations, or plain text."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in %s: %s", file_path.name, e)
            return []

    if not isinstance(data, list):
        data = [data]

    examples = []
    for item in data:
        if not isinstance(item, dict):
            # Plain text item
            examples.append({"text": str(item)})
            continue

        # Alpaca-style: instruction + input + output
        if "instruction" in item:
            instr = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            out = item.get("output", item.get("response", "")).strip()
            if instr and out:
                full_input = f"{instr}\n{inp}" if inp else instr
                examples.append({
                    "text": f"### Instruction:\n{full_input}\n\n### Response:\n{out}",
                    "instruction": full_input,
                    "response": out,
                })
            continue

        # ShareGPT / conversations format
        if "conversations" in item:
            conv = item["conversations"]
            if isinstance(conv, list):
                parts = []
                for msg in conv:
                    role = msg.get("from", msg.get("role", "user"))
                    content = msg.get("value", msg.get("content", ""))
                    parts.append(f"### {role.capitalize()}:\n{content}")
                if parts:
                    examples.append({"text": "\n\n".join(parts)})
            continue

        # Question/Answer format
        if "question" in item and "answer" in item:
            q = item["question"].strip()
            a = item["answer"].strip()
            if q and a:
                examples.append({
                    "text": f"### Question:\n{q}\n\n### Answer:\n{a}",
                    "instruction": q,
                    "response": a,
                })
            continue

        # Prompt/completion format
        if "prompt" in item and "completion" in item:
            p = item["prompt"].strip()
            c = item["completion"].strip()
            if p and c:
                examples.append({
                    "text": f"{p}\n{c}",
                    "instruction": p,
                    "response": c,
                })
            continue

        # Plain text field
        if "text" in item:
            t = item["text"].strip()
            if t:
                examples.append({"text": t})
            continue

        # Fallback: serialize entire object
        examples.append({"text": json.dumps(item, ensure_ascii=False)})

    return examples


def _extract_csv_examples(file_path: Path) -> list[dict]:
    """Extract from CSV: auto-detect instruction/response columns."""
    try:
        import pandas as pd
        df = pd.read_csv(file_path, nrows=50000)  # Safety cap
    except Exception as e:
        logger.error("Failed to read CSV %s: %s", file_path.name, e)
        return []

    if df.empty:
        return []

    columns_lower = {c.lower().strip(): c for c in df.columns}

    # Try to find instruction/response columns
    instr_col = None
    resp_col = None
    text_col = None

    for alias in ["instruction", "input", "prompt", "question"]:
        if alias in columns_lower:
            instr_col = columns_lower[alias]
            break

    for alias in ["output", "response", "answer", "completion"]:
        if alias in columns_lower:
            resp_col = columns_lower[alias]
            break

    for alias in ["text", "content", "body", "message"]:
        if alias in columns_lower:
            text_col = columns_lower[alias]
            break

    examples = []

    if instr_col and resp_col:
        for _, row in df.iterrows():
            instr = str(row.get(instr_col, "")).strip()
            resp = str(row.get(resp_col, "")).strip()
            if instr and resp and instr != "nan" and resp != "nan":
                examples.append({
                    "text": f"### Instruction:\n{instr}\n\n### Response:\n{resp}",
                    "instruction": instr,
                    "response": resp,
                })
    elif text_col:
        for _, row in df.iterrows():
            t = str(row.get(text_col, "")).strip()
            if t and t != "nan":
                examples.append({"text": t})
    else:
        # Fallback: concatenate all columns
        for _, row in df.iterrows():
            parts = []
            for col in df.columns:
                val = row.get(col)
                if pd.notna(val):
                    parts.append(f"{col}: {val}")
            if parts:
                examples.append({"text": " | ".join(parts)})

    return examples


# ── Tokenization ─────────────────────────────────────────────────────


def _tokenize_examples(
    examples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
) -> dict:
    """
    Tokenize examples for causal language modelling.

    Returns dict with input_ids, attention_mask, labels.
    Labels are a copy of input_ids (for causal LM the model predicts next token).
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for ex in examples:
        text = ex.get("text", "")
        if not text.strip():
            continue

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_tokens,
            padding="max_length",
            return_tensors=None,  # Return plain lists
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Labels = input_ids, but with padding tokens masked as -100
        labels = [
            token_id if mask == 1 else -100
            for token_id, mask in zip(input_ids, attention_mask)
        ]

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }
