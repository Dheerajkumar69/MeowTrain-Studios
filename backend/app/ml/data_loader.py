"""
Dataset loader for MeowLLM training pipeline.

Loads project datasets from disk, extracts text, tokenizes, and returns
HuggingFace Dataset objects ready for the Trainer.

Supports structured data formats:
  1. Alpaca (instruction/input/output)
  2. ShareGPT (conversations with from/value)
  3. OpenAI messages (messages array with role/content)
  4. Q&A (question/answer)
  5. Prompt/Completion (prompt/completion)
  6. Plain text (.txt, .md, .pdf, .docx, etc.)

Supports file formats: JSON, JSONL, CSV, and all text_extractor types.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset as HFDataset
    from transformers import PreTrainedTokenizerBase

from app.config import PROJECTS_DIR
from app.utils.text_extractor import extract_text

logger = logging.getLogger("meowllm.data_loader")


def _lazy_hf_dataset():
    """Lazily import HFDataset to avoid crashing when 'datasets' isn't installed."""
    from datasets import Dataset as HFDataset
    return HFDataset


def _lazy_auto_tokenizer():
    """Lazily import AutoTokenizer to avoid crashing when 'transformers' isn't installed."""
    from transformers import AutoTokenizer
    return AutoTokenizer


# Maximum number of raw text characters to load per project (safety cap)
MAX_TEXT_CHARS = 50_000_000  # ~50 MB of text

# Dataset format names (for detection/display)
FORMAT_ALPACA = "alpaca"
FORMAT_SHAREGPT = "sharegpt"
FORMAT_OPENAI = "openai_messages"
FORMAT_QA = "question_answer"
FORMAT_PROMPT_COMPLETION = "prompt_completion"
FORMAT_PLAIN_TEXT = "plain_text"
FORMAT_CSV_INSTRUCTION = "csv_instruction"
FORMAT_CSV_TEXT = "csv_text"
FORMAT_PREFERENCE = "preference_pairs"
FORMAT_UNKNOWN = "unknown"


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
            "detected_formats": list[str],
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
    detected_formats = set()
    dataset_dir = PROJECTS_DIR / str(project_id) / "datasets"

    for ds in ready_datasets:
        file_path = dataset_dir / ds.filename
        if not file_path.exists():
            logger.warning("Dataset file missing on disk: %s", file_path)
            continue

        try:
            examples, fmt = _extract_examples(file_path, ds.file_type)
            all_examples.extend(examples)
            detected_formats.add(fmt)
            logger.info(
                "Loaded %d examples from %s (%s, format=%s)",
                len(examples), ds.original_name, ds.file_type, fmt,
            )
        except Exception as e:
            logger.error("Failed to load dataset %s: %s", ds.original_name, e)
            continue

    if not all_examples:
        raise ValueError(
            "Could not extract any training examples from the uploaded datasets. "
            "Ensure files contain text and are not corrupted."
        )

    # Tokenize — use chat template if available and data has instruction format
    has_instruction_data = any(ex.get("messages") or ex.get("instruction") for ex in all_examples)
    use_chat_template = has_instruction_data and _has_chat_template(tokenizer)

    if use_chat_template:
        logger.info("Using model's chat template for tokenization")
        tokenized = _tokenize_with_chat_template(all_examples, tokenizer, max_tokens)
    else:
        tokenized = _tokenize_examples(all_examples, tokenizer, max_tokens)

    if len(tokenized["input_ids"]) == 0:
        raise ValueError(
            "Tokenization produced 0 samples. Your data may be too short or empty."
        )

    hf_dataset = _lazy_hf_dataset().from_dict(tokenized)
    total_samples = len(hf_dataset)

    logger.info("Total tokenized samples: %d (chat_template=%s)", total_samples, use_chat_template)

    # Split
    if total_samples < 2 or train_split >= 0.99:
        logger.info("Dataset too small for split or split=1.0 — using all for training.")
        return {
            "train_dataset": hf_dataset,
            "eval_dataset": None,
            "total_samples": total_samples,
            "tokenizer": tokenizer,
            "detected_formats": list(detected_formats),
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
        "detected_formats": list(detected_formats),
    }


# ── Tokenizer Loading ────────────────────────────────────────────────


def _load_tokenizer(
    model_name: str, max_tokens: int
) -> PreTrainedTokenizerBase:
    """Load and configure tokenizer with proper padding."""
    logger.info("Loading tokenizer for %s", model_name)

    from app.config import TRUST_REMOTE_CODE
    tokenizer = _lazy_auto_tokenizer().from_pretrained(
        model_name,
        trust_remote_code=TRUST_REMOTE_CODE,
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


def _has_chat_template(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Check if the tokenizer has a chat template defined."""
    try:
        return hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    except Exception:
        return False


# ── Format Detection ─────────────────────────────────────────────────


def detect_dataset_format(file_path: Path, file_type: str) -> dict:
    """
    Detect the format of a dataset file without full extraction.

    Returns:
        {
            "format": str,       # alpaca, sharegpt, openai_messages, etc.
            "sample_count": int,
            "samples": list,     # up to 3 sample items (raw dicts for JSON)
            "description": str,  # human-readable format description
        }
    """
    file_type = file_type.lower()

    if file_type in (".json", ".jsonl"):
        return _detect_json_format(file_path, file_type)
    elif file_type == ".csv":
        return _detect_csv_format(file_path)
    else:
        return {
            "format": FORMAT_PLAIN_TEXT,
            "sample_count": 1,
            "samples": [],
            "description": "Raw text — will be split into training chunks",
        }


def _detect_json_format(file_path: Path, file_type: str) -> dict:
    """Detect format of JSON/JSONL file."""
    items = _read_json_items(file_path, file_type, max_items=5)

    if not items:
        return {"format": FORMAT_UNKNOWN, "sample_count": 0, "samples": [], "description": "Empty or invalid JSON"}

    first = items[0] if isinstance(items[0], dict) else {}

    # OpenAI messages format
    if "messages" in first and isinstance(first["messages"], list):
        return {
            "format": FORMAT_OPENAI,
            "sample_count": len(items),
            "samples": items[:3],
            "description": "OpenAI messages format (role/content pairs)",
        }

    # Alpaca format
    if "instruction" in first:
        return {
            "format": FORMAT_ALPACA,
            "sample_count": len(items),
            "samples": items[:3],
            "description": "Alpaca format (instruction/input/output)",
        }

    # ShareGPT format
    if "conversations" in first:
        return {
            "format": FORMAT_SHAREGPT,
            "sample_count": len(items),
            "samples": items[:3],
            "description": "ShareGPT format (multi-turn conversations)",
        }

    # Q&A format
    if "question" in first and "answer" in first:
        return {
            "format": FORMAT_QA,
            "sample_count": len(items),
            "samples": items[:3],
            "description": "Question & Answer format",
        }

    # Prompt/Completion format
    if "prompt" in first and "completion" in first:
        return {
            "format": FORMAT_PROMPT_COMPLETION,
            "sample_count": len(items),
            "samples": items[:3],
            "description": "Prompt/Completion format",
        }

    # Plain text
    if "text" in first:
        return {
            "format": FORMAT_PLAIN_TEXT,
            "sample_count": len(items),
            "samples": items[:3],
            "description": "Plain text entries",
        }

    return {
        "format": FORMAT_UNKNOWN,
        "sample_count": len(items),
        "samples": items[:3],
        "description": "Unknown JSON structure — will attempt best-effort extraction",
    }


def _detect_csv_format(file_path: Path) -> dict:
    """Detect CSV column structure."""
    try:
        import pandas as pd
        df = pd.read_csv(file_path, nrows=5)
    except Exception:
        return {"format": FORMAT_UNKNOWN, "sample_count": 0, "samples": [], "description": "Failed to read CSV"}

    cols_lower = {c.lower().strip() for c in df.columns}

    if cols_lower & {"instruction", "input", "prompt", "question"} and cols_lower & {"output", "response", "answer", "completion"}:
        return {
            "format": FORMAT_CSV_INSTRUCTION,
            "sample_count": len(df),
            "samples": df.head(3).to_dict("records"),
            "description": f"CSV with instruction/response columns: {list(df.columns)}",
        }
    elif cols_lower & {"text", "content", "body", "message"}:
        return {
            "format": FORMAT_CSV_TEXT,
            "sample_count": len(df),
            "samples": df.head(3).to_dict("records"),
            "description": f"CSV with text column: {list(df.columns)}",
        }
    else:
        return {
            "format": FORMAT_CSV_TEXT,
            "sample_count": len(df),
            "samples": df.head(3).to_dict("records"),
            "description": f"CSV — all columns will be concatenated: {list(df.columns)}",
        }


# ── Example Extraction ───────────────────────────────────────────────


def _read_json_items(file_path: Path, file_type: str, max_items: int = 0) -> list:
    """Read JSON or JSONL file into a list of items."""
    if file_type == ".jsonl":
        items = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSONL line %d in %s", i + 1, file_path.name)
                    continue
                if max_items and len(items) >= max_items:
                    break
        return items
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON in %s: %s", file_path.name, e)
                return []
        if not isinstance(data, list):
            data = [data]
        if max_items:
            return data[:max_items]
        return data


def _extract_examples(file_path: Path, file_type: str) -> tuple[list[dict], str]:
    """
    Extract training examples from a file.

    Returns a tuple of (examples, detected_format).
    Each example dict has at least a "text" key.
    For instruction data, also has "instruction", "response", and/or "messages" keys.
    """
    file_type = file_type.lower()

    # JSON / JSONL: structured format detection
    if file_type in (".json", ".jsonl"):
        return _extract_json_examples(file_path, file_type)

    # CSV: try to detect instruction/response columns
    if file_type == ".csv":
        examples = _extract_csv_examples(file_path)
        fmt = FORMAT_CSV_INSTRUCTION if any(ex.get("instruction") for ex in examples) else FORMAT_CSV_TEXT
        return examples, fmt

    # Everything else: raw text
    extracted = extract_text(file_path, file_type)
    text = extracted.get("text", "")
    if not text.strip():
        return [], FORMAT_PLAIN_TEXT

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

    return examples, FORMAT_PLAIN_TEXT


def _extract_json_examples(file_path: Path, file_type: str) -> tuple[list[dict], str]:
    """Extract from JSON/JSONL: supports all structured formats."""
    items = _read_json_items(file_path, file_type)

    if not items:
        return [], FORMAT_UNKNOWN

    examples = []
    detected_format = FORMAT_UNKNOWN

    for item in items:
        if not isinstance(item, dict):
            # Plain text item
            examples.append({"text": str(item)})
            detected_format = FORMAT_PLAIN_TEXT
            continue

        # ── OpenAI messages format ──
        if "messages" in item and isinstance(item["messages"], list):
            messages = item["messages"]
            # Validate message structure
            valid_messages = []
            for msg in messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    valid_messages.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })

            if valid_messages:
                # Build text representation
                parts = []
                for msg in valid_messages:
                    role = msg["role"].capitalize()
                    parts.append(f"### {role}:\n{msg['content']}")

                text = "\n\n".join(parts)
                examples.append({
                    "text": text,
                    "messages": valid_messages,
                })
                detected_format = FORMAT_OPENAI
            continue

        # ── Alpaca-style: instruction + input + output ──
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
                    "messages": [
                        {"role": "user", "content": full_input},
                        {"role": "assistant", "content": out},
                    ],
                })
                detected_format = FORMAT_ALPACA
            continue

        # ── ShareGPT / conversations format ──
        if "conversations" in item:
            conv = item["conversations"]
            if isinstance(conv, list):
                parts = []
                messages = []
                for msg in conv:
                    role = msg.get("from", msg.get("role", "user"))
                    content = msg.get("value", msg.get("content", ""))
                    # Normalize role names
                    if role in ("human", "user"):
                        role = "user"
                    elif role in ("gpt", "assistant", "bot"):
                        role = "assistant"
                    parts.append(f"### {role.capitalize()}:\n{content}")
                    messages.append({"role": role, "content": content})
                if parts:
                    examples.append({
                        "text": "\n\n".join(parts),
                        "messages": messages,
                    })
                    detected_format = FORMAT_SHAREGPT
            continue

        # ── Question/Answer format ──
        if "question" in item and "answer" in item:
            q = item["question"].strip()
            a = item["answer"].strip()
            if q and a:
                examples.append({
                    "text": f"### Question:\n{q}\n\n### Answer:\n{a}",
                    "instruction": q,
                    "response": a,
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                })
                detected_format = FORMAT_QA
            continue

        # ── Prompt/completion format ──
        if "prompt" in item and "completion" in item:
            p = item["prompt"].strip()
            c = item["completion"].strip()
            if p and c:
                examples.append({
                    "text": f"{p}\n{c}",
                    "instruction": p,
                    "response": c,
                    "messages": [
                        {"role": "user", "content": p},
                        {"role": "assistant", "content": c},
                    ],
                })
                detected_format = FORMAT_PROMPT_COMPLETION
            continue

        # ── Plain text field ──
        if "text" in item:
            t = item["text"].strip()
            if t:
                examples.append({"text": t})
                detected_format = FORMAT_PLAIN_TEXT
            continue

        # ── Fallback: serialize entire object ──
        examples.append({"text": json.dumps(item, ensure_ascii=False)})

    return examples, detected_format


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
        import pandas as pd
        for _, row in df.iterrows():
            instr = str(row.get(instr_col, "")).strip()
            resp = str(row.get(resp_col, "")).strip()
            if instr and resp and instr != "nan" and resp != "nan":
                examples.append({
                    "text": f"### Instruction:\n{instr}\n\n### Response:\n{resp}",
                    "instruction": instr,
                    "response": resp,
                    "messages": [
                        {"role": "user", "content": instr},
                        {"role": "assistant", "content": resp},
                    ],
                })
    elif text_col:
        for _, row in df.iterrows():
            t = str(row.get(text_col, "")).strip()
            if t and t != "nan":
                examples.append({"text": t})
    else:
        # Fallback: concatenate all columns
        import pandas as pd
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


def _tokenize_with_chat_template(
    examples: list[dict],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
) -> dict:
    """
    Tokenize examples using the model's native chat template.

    This produces properly formatted input for chat models (e.g., ChatML,
    Llama chat, Mistral Instruct). Falls back to plain tokenization for
    examples without messages.
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for ex in examples:
        messages = ex.get("messages")

        if messages and len(messages) > 0:
            # Use the model's chat template
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as e:
                logger.debug("Chat template failed for example, falling back: %s", e)
                text = ex.get("text", "")
        else:
            text = ex.get("text", "")

        if not text.strip():
            continue

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_tokens,
            padding="max_length",
            return_tensors=None,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

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


# ═══════════════════════════════════════════════════════════════════════
# Preference Pair Loading (for DPO / ORPO)
# ═══════════════════════════════════════════════════════════════════════

def load_preference_dataset(
    project_id: int,
    dataset_records: list,
    tokenizer_name: str,
    max_tokens: int = 512,
    train_split: float = 0.9,
) -> dict:
    """
    Load preference pair datasets for DPO/ORPO training.

    Expects JSON/JSONL with one of these structures:
      - { "chosen": "...", "rejected": "..." }
      - { "prompt": "...", "chosen": "...", "rejected": "..." }
      - { "chosen": [messages], "rejected": [messages] }

    Returns:
        {
            "train_dataset": HFDataset with columns: prompt, chosen, rejected
            "eval_dataset": HFDataset or None
            "total_samples": int
            "tokenizer": PreTrainedTokenizerBase
        }
    """
    ready_datasets = [d for d in dataset_records if d.status == "ready"]
    if not ready_datasets:
        raise ValueError("No datasets with status 'ready' found for this project.")

    tokenizer = _load_tokenizer(tokenizer_name, max_tokens)

    all_pairs = []
    dataset_dir = PROJECTS_DIR / str(project_id) / "datasets"

    for ds in ready_datasets:
        file_path = dataset_dir / ds.filename
        if not file_path.exists():
            logger.warning("Dataset file missing on disk: %s", file_path)
            continue

        try:
            items = _read_json_items(file_path, ds.file_type)
            pairs = _extract_preference_pairs(items)
            all_pairs.extend(pairs)
            logger.info(
                "Loaded %d preference pairs from %s",
                len(pairs), ds.original_name,
            )
        except Exception as e:
            logger.error("Failed to load preference dataset %s: %s", ds.original_name, e)
            continue

    if not all_pairs:
        raise ValueError(
            "Could not extract any preference pairs from the uploaded datasets. "
            "Datasets for DPO/ORPO must contain 'chosen' and 'rejected' fields."
        )

    logger.info("Total preference pairs: %d", len(all_pairs))

    # Build HF Dataset
    hf_dataset = _lazy_hf_dataset().from_dict({
        "prompt": [p["prompt"] for p in all_pairs],
        "chosen": [p["chosen"] for p in all_pairs],
        "rejected": [p["rejected"] for p in all_pairs],
    })

    # Split
    if train_split < 1.0 and len(hf_dataset) > 2:
        split = hf_dataset.train_test_split(test_size=1.0 - train_split, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = hf_dataset
        eval_dataset = None

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "total_samples": len(all_pairs),
        "tokenizer": tokenizer,
    }


def _extract_preference_pairs(items: list[dict]) -> list[dict]:
    """Extract preference pairs from JSON objects."""
    pairs = []
    for item in items:
        chosen = item.get("chosen")
        rejected = item.get("rejected")

        if chosen is None or rejected is None:
            continue

        prompt = item.get("prompt", "")

        # Handle message-format chosen/rejected (list of dicts)
        if isinstance(chosen, list):
            chosen = _messages_to_text(chosen)
        if isinstance(rejected, list):
            rejected = _messages_to_text(rejected)
        if isinstance(prompt, list):
            prompt = _messages_to_text(prompt)

        # Ensure strings
        chosen = str(chosen).strip()
        rejected = str(rejected).strip()
        prompt = str(prompt).strip()

        if chosen and rejected:
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

    return pairs


def _messages_to_text(messages: list) -> str:
    """Convert a list of message dicts to plain text."""
    parts = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", msg.get("from", "unknown"))
            content = msg.get("content", msg.get("value", ""))
            parts.append(f"{role}: {content}")
        elif isinstance(msg, str):
            parts.append(msg)
    return "\n".join(parts)

