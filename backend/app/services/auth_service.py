import logging
import secrets

import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from app.config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRY_HOURS
from app.models.user import User

logger = logging.getLogger("meowllm.auth_service")

# Maximum age (in days) of an expired token that can still be refreshed
_REFRESH_GRACE_DAYS = 7

# Maximum token length to prevent DoS via huge payloads
_MAX_TOKEN_LENGTH = 4096


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Timing-safe password verification."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed hash — always return False, but constant-time delay
        secrets.compare_digest("dummy", "dummy")
        return False


def create_token(user_id: int, token_version: int = 0) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "user_id": user_id,
        "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": now,
        "tv": token_version,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    if not token or len(token) > _MAX_TOKEN_LENGTH:
        raise ValueError("Invalid token")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        # Validate required claims
        if "user_id" not in payload:
            raise ValueError("Invalid token: missing user_id claim")
        if not isinstance(payload["user_id"], int):
            raise ValueError("Invalid token: malformed user_id")
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")


def decode_token_allow_expired(token: str) -> dict:
    """Decode a token even if expired (for refresh). Rejects if expired > GRACE days."""
    if not token or len(token) > _MAX_TOKEN_LENGTH:
        raise ValueError("Invalid token")
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        # Decode without verification to check how old it is
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"verify_exp": False})
        if "user_id" not in payload or "exp" not in payload:
            raise ValueError("Invalid token")
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        if datetime.now(timezone.utc) - exp > timedelta(days=_REFRESH_GRACE_DAYS):
            raise ValueError("Token expired too long ago — please log in again")
        return payload
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")


def get_current_user(db: Session, token: str) -> User:
    payload = decode_token(token)
    user = db.query(User).filter(User.id == payload["user_id"]).first()
    if not user:
        raise ValueError("User not found")
    # Reject tokens issued before a token_version bump (password change, logout-all)
    token_tv = payload.get("tv", 0)
    if token_tv < getattr(user, "token_version", 0):
        raise ValueError("Token has been revoked — please log in again")
    return user


def get_user_from_header(db: Session, authorization: str) -> User:
    if not authorization or not isinstance(authorization, str):
        raise ValueError("Missing or invalid authorization header")
    if not authorization.startswith("Bearer "):
        raise ValueError("Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise ValueError("Missing or invalid authorization header")
    return get_current_user(db, token)


def require_role(user: User, *roles: str) -> None:
    """Raise ValueError unless the user's role is in *roles*."""
    if user.role not in roles:
        raise ValueError(f"This action requires one of roles: {', '.join(roles)}")


def require_non_guest(user: User) -> None:
    """Raise ValueError if the user is a guest."""
    if user.is_guest:
        raise ValueError("Guest users cannot perform this action. Please register.")

