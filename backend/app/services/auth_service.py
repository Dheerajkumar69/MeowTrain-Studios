import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from app.config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRY_HOURS
from app.models.user import User

# Maximum age (in days) of an expired token that can still be refreshed
_REFRESH_GRACE_DAYS = 7


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")


def decode_token_allow_expired(token: str) -> dict:
    """Decode a token even if expired (for refresh). Rejects if expired > GRACE days."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        # Decode without verification to check how old it is
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"verify_exp": False})
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
    return user


def get_user_from_header(db: Session, authorization: str) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise ValueError("Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    return get_current_user(db, token)

