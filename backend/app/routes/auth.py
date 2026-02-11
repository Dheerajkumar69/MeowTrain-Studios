import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Header, Request
from sqlalchemy.orm import Session
from typing import Optional
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.database import get_db
from app.models.user import User
from app.schemas import RegisterRequest, LoginRequest, AuthResponse, UserResponse, ProfileUpdateRequest, PasswordChangeRequest
from app.services.auth_service import hash_password, verify_password, create_token, get_current_user, get_user_from_header, decode_token_allow_expired
from app.config import RATE_LIMIT_AUTH

router = APIRouter(prefix="/auth", tags=["Auth"])

# Route-level limiter that shares state with the app limiter (same key_func)
_limiter = Limiter(key_func=get_remote_address)


def _validate_password(password: str):
    """Enforce minimum password strength."""
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long.")
    if not re.search(r"\d", password):
        raise HTTPException(status_code=400, detail="Password must contain at least one digit.")
    if not re.search(r"[a-zA-Z]", password):
        raise HTTPException(status_code=400, detail="Password must contain at least one letter.")


@router.post("/register", response_model=AuthResponse)
@_limiter.limit(RATE_LIMIT_AUTH)
def register(request: Request, req: RegisterRequest, db: Session = Depends(get_db)):
    # Validate password strength
    _validate_password(req.password)

    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=req.email,
        password_hash=hash_password(req.password),
        display_name=req.display_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_token(user.id)
    return AuthResponse(token=token, user=UserResponse.model_validate(user))


@router.post("/login", response_model=AuthResponse)
@_limiter.limit(RATE_LIMIT_AUTH)
def login(request: Request, req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user.id)
    return AuthResponse(token=token, user=UserResponse.model_validate(user))


@router.post("/guest", response_model=AuthResponse)
@_limiter.limit("3/minute")
def guest_login(request: Request, db: Session = Depends(get_db)):
    guest = User(
        display_name="Guest",
        is_guest=True,
    )
    db.add(guest)
    db.commit()
    db.refresh(guest)
    token = create_token(guest.id)
    return AuthResponse(token=token, user=UserResponse.model_validate(guest))


@router.get("/me", response_model=UserResponse)
def me(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return UserResponse.model_validate(user)


@router.patch("/profile", response_model=UserResponse)
def update_profile(
    req: ProfileUpdateRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    if user.is_guest:
        raise HTTPException(status_code=403, detail="Guest users cannot update profile")

    update_data = req.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(user, key, value)

    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@router.post("/password")
def change_password(
    req: PasswordChangeRequest,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    if user.is_guest:
        raise HTTPException(status_code=403, detail="Guest users cannot change password")

    if not verify_password(req.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    _validate_password(req.new_password)

    user.password_hash = hash_password(req.new_password)
    db.commit()
    return {"detail": "Password changed successfully"}


@router.post("/refresh", response_model=AuthResponse)
@_limiter.limit(RATE_LIMIT_AUTH)
def refresh_token(request: Request, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    """Issue a fresh JWT from a valid or recently-expired token (up to 7 days old)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    old_token = authorization.split(" ", 1)[1]
    try:
        payload = decode_token_allow_expired(old_token)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    user = db.query(User).filter(User.id == payload["user_id"]).first()
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")

    new_token = create_token(user.id)
    return AuthResponse(token=new_token, user=UserResponse.model_validate(user))
