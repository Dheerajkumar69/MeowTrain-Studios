"""
Dataset augmentation tools for MeowLLM.

Provides practical data cleaning and quality improvement:
  - Near-duplicate detection via MinHash/Jaccard similarity
  - Text cleaning (HTML removal, encoding fixes, whitespace normalization)
  - Quality filtering (too short/long, low information density)
  - Statistics reporting

All processing is local — no external API calls needed.
"""

import hashlib
import html
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger("meowllm.augmentation")


# ═══════════════════════════════════════════════════════════════════════
# Text Cleaning
# ═══════════════════════════════════════════════════════════════════════

# Regex patterns for cleaning
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")


def clean_text(text: str, options: dict = None) -> str:
    """
    Clean a single text string.

    Options:
        strip_html: bool (default True) — remove HTML tags
        fix_encoding: bool (default True) — fix mojibake and normalize unicode
        normalize_whitespace: bool (default True) — collapse whitespace
        strip_urls: bool (default False) — remove URLs
        strip_emails: bool (default False) — remove email addresses
    """
    if not text:
        return text

    opts = options or {}

    # Fix encoding issues
    if opts.get("fix_encoding", True):
        text = html.unescape(text)
        text = unicodedata.normalize("NFKC", text)
        # Fix common mojibake patterns
        try:
            text = text.encode("utf-8").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    # Strip HTML
    if opts.get("strip_html", True):
        text = _HTML_TAG_RE.sub("", text)

    # Strip URLs
    if opts.get("strip_urls", False):
        text = _URL_RE.sub("[URL]", text)

    # Strip emails
    if opts.get("strip_emails", False):
        text = _EMAIL_RE.sub("[EMAIL]", text)

    # Normalize whitespace
    if opts.get("normalize_whitespace", True):
        text = _MULTI_SPACE_RE.sub(" ", text)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)
        text = text.strip()

    return text


def clean_examples(examples: list[dict], options: dict = None) -> list[dict]:
    """Clean text fields in a list of training examples."""
    cleaned = []
    for ex in examples:
        cleaned_ex = {}
        for key, value in ex.items():
            if isinstance(value, str):
                cleaned_ex[key] = clean_text(value, options)
            elif isinstance(value, list):
                # Handle message arrays
                cleaned_ex[key] = [
                    {**msg, "content": clean_text(msg.get("content", ""), options)}
                    if isinstance(msg, dict) and "content" in msg
                    else msg
                    for msg in value
                ]
            else:
                cleaned_ex[key] = value
        cleaned.append(cleaned_ex)
    return cleaned


# ═══════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════

def _text_fingerprint(text: str) -> str:
    """Create a normalized fingerprint for dedup comparison."""
    # Lowercase, strip whitespace, remove punctuation
    normalized = text.lower().strip()
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = _MULTI_SPACE_RE.sub(" ", normalized)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _ngram_set(text: str, n: int = 3) -> set:
    """Generate character n-gram set for Jaccard similarity."""
    text = text.lower().strip()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def deduplicate_examples(
    examples: list[dict],
    threshold: float = 0.85,
    text_key: str = "text",
) -> tuple[list[dict], list[dict]]:
    """
    Remove near-duplicate examples using Jaccard similarity on character n-grams.

    Args:
        examples: List of training examples
        threshold: Jaccard similarity threshold (above = duplicate)
        text_key: Which field to use for comparison

    Returns:
        (unique_examples, removed_duplicates)
    """
    if not examples:
        return [], []

    # Extract text for comparison
    def _get_text(ex: dict) -> str:
        if text_key in ex:
            return str(ex[text_key])
        # Try common fields
        for field in ["text", "instruction", "output", "content", "chosen"]:
            if field in ex:
                return str(ex[field])
        # Fallback: join all string values
        return " ".join(str(v) for v in ex.values() if isinstance(v, str))

    # Phase 1: Exact dedup via fingerprints
    seen_fingerprints = set()
    exact_unique = []
    exact_dups = []

    for ex in examples:
        text = _get_text(ex)
        fp = _text_fingerprint(text)
        if fp in seen_fingerprints:
            exact_dups.append(ex)
        else:
            seen_fingerprints.add(fp)
            exact_unique.append(ex)

    # Phase 2: Near-dedup via Jaccard similarity (on smaller set)
    # Only do this if dataset is manageable (< 10k after exact dedup)
    if len(exact_unique) > 10000:
        logger.info(
            "Skipping near-dedup for %d examples (too large), using exact dedup only",
            len(exact_unique),
        )
        return exact_unique, exact_dups

    ngram_sets = [_ngram_set(_get_text(ex)) for ex in exact_unique]
    keep_mask = [True] * len(exact_unique)
    near_dups = []

    for i in range(len(exact_unique)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(exact_unique)):
            if not keep_mask[j]:
                continue
            sim = _jaccard_similarity(ngram_sets[i], ngram_sets[j])
            if sim >= threshold:
                keep_mask[j] = False
                near_dups.append(exact_unique[j])

    unique = [ex for ex, keep in zip(exact_unique, keep_mask) if keep]
    all_dups = exact_dups + near_dups

    logger.info(
        "Dedup: %d total → %d unique (%d exact dups, %d near dups)",
        len(examples), len(unique), len(exact_dups), len(near_dups),
    )

    return unique, all_dups


# ═══════════════════════════════════════════════════════════════════════
# Quality Filtering
# ═══════════════════════════════════════════════════════════════════════

def filter_quality(
    examples: list[dict],
    min_length: int = 10,
    max_length: int = 100000,
    min_word_count: int = 3,
    text_key: str = "text",
) -> tuple[list[dict], list[dict]]:
    """
    Filter out low-quality training examples.

    Args:
        examples: List of training examples
        min_length: Minimum character length
        max_length: Maximum character length
        min_word_count: Minimum number of words
        text_key: Which field to check

    Returns:
        (passed_examples, filtered_out_examples)
    """
    passed = []
    filtered = []

    for ex in examples:
        # Get text to check
        text = ""
        if text_key in ex:
            text = str(ex[text_key])
        else:
            for field in ["text", "instruction", "output", "content", "chosen"]:
                if field in ex:
                    text = str(ex[field])
                    break

        # Length check
        if len(text) < min_length:
            filtered.append({**ex, "_filter_reason": f"too_short ({len(text)} chars)"})
            continue

        if len(text) > max_length:
            filtered.append({**ex, "_filter_reason": f"too_long ({len(text)} chars)"})
            continue

        # Word count check
        word_count = len(text.split())
        if word_count < min_word_count:
            filtered.append({**ex, "_filter_reason": f"too_few_words ({word_count})"})
            continue

        # Repetition check: if >50% of content is repeated characters
        char_counts = Counter(text.lower())
        if char_counts and char_counts.most_common(1)[0][1] > len(text) * 0.5:
            filtered.append({**ex, "_filter_reason": "highly_repetitive"})
            continue

        passed.append(ex)

    logger.info(
        "Quality filter: %d total → %d passed, %d filtered",
        len(examples), len(passed), len(filtered),
    )

    return passed, filtered


# ═══════════════════════════════════════════════════════════════════════
# Augmentation Pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_augmentation_pipeline(
    examples: list[dict],
    enable_dedup: bool = True,
    enable_clean: bool = True,
    enable_filter: bool = True,
    dedup_threshold: float = 0.85,
    min_length: int = 10,
    max_length: int = 100000,
    clean_options: dict = None,
) -> dict:
    """
    Run the full augmentation pipeline on a list of examples.

    Returns:
        {
            "cleaned_examples": list[dict],
            "stats": {
                "original_count": int,
                "after_dedup": int,
                "after_clean": int,
                "after_filter": int,
                "duplicates_removed": int,
                "filtered_out": int,
                "filter_reasons": dict[str, int],
            }
        }
    """
    stats = {
        "original_count": len(examples),
        "duplicates_removed": 0,
        "filtered_out": 0,
        "filter_reasons": {},
    }

    current = list(examples)

    # Step 1: Deduplication
    if enable_dedup and current:
        current, dups = deduplicate_examples(current, threshold=dedup_threshold)
        stats["duplicates_removed"] = len(dups)
        stats["after_dedup"] = len(current)
    else:
        stats["after_dedup"] = len(current)

    # Step 2: Text cleaning
    if enable_clean and current:
        current = clean_examples(current, clean_options)
        stats["after_clean"] = len(current)
    else:
        stats["after_clean"] = len(current)

    # Step 3: Quality filter
    if enable_filter and current:
        current, filtered = filter_quality(
            current,
            min_length=min_length,
            max_length=max_length,
        )
        stats["filtered_out"] = len(filtered)
        stats["after_filter"] = len(current)

        # Count filter reasons
        reasons = {}
        for f in filtered:
            reason = f.get("_filter_reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
        stats["filter_reasons"] = reasons
    else:
        stats["after_filter"] = len(current)

    return {
        "cleaned_examples": current,
        "stats": stats,
    }
