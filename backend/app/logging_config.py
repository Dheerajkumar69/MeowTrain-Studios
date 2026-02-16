"""
Structured logging configuration for MeowTrain.

Supports two formats controlled by ``LOG_FORMAT`` env var:

* ``text`` (default) — human-readable, great for local development
* ``json`` — structured JSON, one object per line, for production log
  aggregators (ELK, Datadog, CloudWatch, Loki …)

Both formats automatically include the ``request_id`` correlation ID
injected by :class:`app.middleware.RequestIDLogFilter`.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Fields emitted:
        timestamp, level, logger, message, request_id,
        module, funcName, lineno, exc_info (if present)
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }

        # Include exception info when present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exc_type"] = record.exc_info[0].__name__
            log_entry["exc_message"] = str(record.exc_info[1])
            log_entry["exc_traceback"] = traceback.format_exception(*record.exc_info)

        # Include any extra fields attached to the record
        for key in ("status_code", "method", "path", "duration_ms", "client_ip"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


# ── Human-readable text format ──
TEXT_FORMAT = (
    "%(asctime)s | %(name)-30s | %(levelname)-7s | rid=%(request_id)s | %(message)s"
)
TEXT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: str = "INFO",
    log_format: str = "text",
) -> None:
    """Set up root logger with the chosen format and level.

    Parameters
    ----------
    level:
        Logging level name (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``).
    log_format:
        Either ``"text"`` or ``"json"``.
    """
    from app.middleware import RequestIDLogFilter

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Clear existing handlers on the root logger
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    # Attach request-ID filter
    handler.addFilter(RequestIDLogFilter())

    if log_format.lower() == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(TEXT_FORMAT, datefmt=TEXT_DATEFMT))

    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in (
        "uvicorn.access",
        "uvicorn.error",
        "httpcore",
        "httpx",
        "watchfiles",
        "multipart",
        "urllib3",
        "transformers",
        "accelerate",
    ):
        logging.getLogger(noisy).setLevel(max(numeric_level, logging.WARNING))
