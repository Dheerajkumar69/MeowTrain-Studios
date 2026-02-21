"""
Security audit logger for MeowTrain.

Writes structured security events to a dedicated logger so that
login attempts, account changes, admin actions, and data deletions
produce a forensic trail.

Events are written to the standard Python logging system under the
``meowllm.security`` logger.  In production, configure a dedicated
FileHandler (e.g. ``security.log``) or ship to a SIEM.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("meowllm.security")

# Ensure we have at least a handler so events don't vanish silently.
# In production this should be replaced by a proper file/SIEM handler.
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] SECURITY %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def audit(
    event: str,
    *,
    user_id: Optional[int] = None,
    email: Optional[str] = None,
    ip: Optional[str] = None,
    detail: str = "",
) -> None:
    """Log a structured security event.

    Args:
        event:   Short event name, e.g. ``login_success``, ``account_deleted``.
        user_id: Numeric user id if available.
        email:   User email if available.
        ip:      Request IP address.
        detail:  Free-text context.
    """
    ts = datetime.now(timezone.utc).isoformat()
    parts = [
        f"event={event}",
        f"ts={ts}",
    ]
    if user_id is not None:
        parts.append(f"user_id={user_id}")
    if email:
        parts.append(f"email={email}")
    if ip:
        parts.append(f"ip={ip}")
    if detail:
        parts.append(f"detail={detail}")

    logger.info(" | ".join(parts))
