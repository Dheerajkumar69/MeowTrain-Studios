"""
Request-correlation middleware.

Assigns a unique ``X-Request-ID`` to every incoming HTTP request so that
every log line emitted during that request can be correlated.  The ID is
also returned in the response headers so the frontend / curl user can
quote it in bug reports.

Usage
-----
The middleware stores the current request ID in a ``contextvars.ContextVar``
that the custom log filter reads.  Any logger using the project-wide format
will automatically include ``request_id`` in its output.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ── context var holds the ID for the duration of one request ──
REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)

_REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every request/response cycle."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Accept a caller-supplied ID (e.g. from a gateway) or generate one
        request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
        REQUEST_ID_CTX.set(request_id)

        response: Response = await call_next(request)
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response


class RequestIDLogFilter(logging.Filter):
    """Inject ``request_id`` into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID_CTX.get("-")  # type: ignore[attr-defined]
        return True
