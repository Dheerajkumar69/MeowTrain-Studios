"""
Centralised lazy-import helpers for heavy ML / third-party packages.

Importing ``transformers``, ``torch``, ``peft``, ``trl``, ``huggingface_hub``
or ``sse_starlette`` at module-load time adds seconds of startup latency
and pulls in GPU memory.  These helpers defer the import until first use
and cache the result, so subsequent accesses are free.

Usage
-----
::

    from app.utils.lazy_imports import transformers, torch, huggingface_hub

    tok = transformers().AutoTokenizer.from_pretrained(...)
    snap = huggingface_hub().snapshot_download(...)

Each function raises a clear ``ImportError`` if the package is absent.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any


def _lazy(module_name: str) -> Any:
    """Import *module_name* once and cache the result."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"Required package '{module_name}' is not installed. "
            f"Install it with:  pip install {module_name}"
        )


# ── Public accessors ────────────────────────────────────────────

@lru_cache(maxsize=1)
def transformers():  # noqa: ANN201
    """``import transformers`` — deferred."""
    return _lazy("transformers")


@lru_cache(maxsize=1)
def torch():  # noqa: ANN201
    """``import torch`` — deferred."""
    return _lazy("torch")


@lru_cache(maxsize=1)
def peft():  # noqa: ANN201
    """``import peft`` — deferred."""
    return _lazy("peft")


@lru_cache(maxsize=1)
def trl():  # noqa: ANN201
    """``import trl`` — deferred."""
    return _lazy("trl")


@lru_cache(maxsize=1)
def huggingface_hub():  # noqa: ANN201
    """``import huggingface_hub`` — deferred."""
    return _lazy("huggingface_hub")


@lru_cache(maxsize=1)
def sse_starlette():  # noqa: ANN201
    """``import sse_starlette`` — deferred."""
    return _lazy("sse_starlette")


@lru_cache(maxsize=1)
def datasets_lib():  # noqa: ANN201
    """``import datasets`` — deferred (named to avoid clash with route)."""
    return _lazy("datasets")
