import re
import secrets
import hashlib
import hmac
import html as _html
import logging
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, Header, Request, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional
from slowapi import Limiter
from slowapi.util import get_remote_address
import httpx

from app.database import get_db
from app.models.user import User
from app.models.project import Project
from app.models.dataset import Dataset
from app.models.training_run import TrainingRun
from app.models.model_config import ModelConfig
from app.models.prompt_template import PromptTemplate
from app.models.background_task import BackgroundTask
from app.schemas import (
    RegisterRequest, LoginRequest, AuthResponse, UserResponse,
    ProfileUpdateRequest, PasswordChangeRequest, DetailResponse,
    ForgotPasswordRequest, ResetPasswordRequest,
    VerifyEmailRequest, ResendVerificationRequest,
)
from app.services.auth_service import hash_password, verify_password, create_token, get_current_user, get_user_from_header, decode_token_allow_expired
from app.services.email_service import send_verification_email, send_password_reset_email
from app.services.audit import audit as security_audit
from app.config import (
    RATE_LIMIT_AUTH, PROJECTS_DIR,
    OAUTH_GOOGLE_CLIENT_ID, OAUTH_GOOGLE_CLIENT_SECRET,
    OAUTH_GITHUB_CLIENT_ID, OAUTH_GITHUB_CLIENT_SECRET,
    OAUTH_REDIRECT_BASE, SMTP_ENABLED,
    RATE_LIMIT_PASSWORD_RESET,
    ACCOUNT_LOCKOUT_ATTEMPTS, ACCOUNT_LOCKOUT_MINUTES,
)

logger = logging.getLogger("meowllm.routes.auth")

router = APIRouter(prefix="/auth", tags=["Auth"])

# Route-level limiter that shares state with the app limiter (same key_func)
_limiter = Limiter(key_func=get_remote_address)

# Limits to prevent abuse
_MAX_GUEST_ACCOUNTS_PER_HOUR = 10  # per IP via rate limiter
_MAX_DISPLAY_NAME_LENGTH = 100

# Allowed characters in display names (strip HTML tags)
_DISPLAY_NAME_RE = re.compile(r"<[^>]+>")


def _generate_oauth_state() -> str:
    """Generate a cryptographically random state token for OAuth CSRF protection."""
    return secrets.token_urlsafe(32)


def _sign_state(state: str) -> str:
    """Create an HMAC signature of the state token using JWT_SECRET."""
    from app.config import JWT_SECRET
    return hmac.new(JWT_SECRET.encode(), state.encode(), hashlib.sha256).hexdigest()


def _verify_oauth_state(state: str, signature: str) -> bool:
    """Verify the HMAC signature of an OAuth state token."""
    expected = _sign_state(state)
    return hmac.compare_digest(expected, signature)


def _validate_password(password: str):
    """Enforce minimum password strength."""
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long.")
    if len(password) > 128:
        raise HTTPException(status_code=400, detail="Password must be at most 128 characters long.")
    if not re.search(r"\d", password):
        raise HTTPException(status_code=400, detail="Password must contain at least one digit.")
    if not re.search(r"[a-zA-Z]", password):
        raise HTTPException(status_code=400, detail="Password must contain at least one letter.")


def _normalize_email(email: str) -> str:
    """Normalize email for consistent storage and lookup."""
    return email.strip().lower()


@router.post("/register", response_model=AuthResponse)
@_limiter.limit(RATE_LIMIT_AUTH)
def register(request: Request, req: RegisterRequest, db: Session = Depends(get_db)):
    # Validate password strength
    _validate_password(req.password)

    normalized_email = _normalize_email(req.email)

    existing = db.query(User).filter(User.email == normalized_email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Generate email verification token
    verification_token = secrets.token_urlsafe(48)

    # Sanitize display name — strip HTML tags
    safe_display = _DISPLAY_NAME_RE.sub("", req.display_name).strip()[:_MAX_DISPLAY_NAME_LENGTH]
    if not safe_display:
        safe_display = "User"

    user = User(
        email=normalized_email,
        password_hash=hash_password(req.password),
        display_name=safe_display,
        email_verified=False,
        email_verification_token=verification_token,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    security_audit("register", user_id=user.id, email=normalized_email, ip=request.client.host)

    # Send verification email (non-blocking — failure doesn't prevent registration)
    email_sent = send_verification_email(normalized_email, verification_token)
    if not email_sent:
        logger.info(
            "[DEV] Email verification token for %s: %s",
            normalized_email, verification_token,
        )

    token = create_token(user.id, user.token_version)
    return AuthResponse(token=token, user=UserResponse.model_validate(user))


@router.post("/login", response_model=AuthResponse)
@_limiter.limit(RATE_LIMIT_AUTH)
def login(request: Request, req: LoginRequest, db: Session = Depends(get_db)):
    normalized_email = _normalize_email(req.email)
    user = db.query(User).filter(User.email == normalized_email).first()

    # Account lockout check
    if user and user.locked_until:
        # SQLite stores naive datetimes — compare consistently
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if user.locked_until > now:
            remaining = int((user.locked_until - now).total_seconds() / 60) + 1
            security_audit("login_locked", email=normalized_email, ip=request.client.host,
                           detail=f"Account still locked for ~{remaining} min")
            raise HTTPException(
                status_code=429,
                detail=f"Account temporarily locked due to too many failed attempts. "
                       f"Try again in {remaining} minute(s).",
            )

    # Use generic error message to prevent user enumeration
    if not user or not verify_password(req.password, user.password_hash or ""):
        # Track failed attempts for lockout
        if user:
            user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
            if user.failed_login_attempts >= ACCOUNT_LOCKOUT_ATTEMPTS:
                user.locked_until = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=ACCOUNT_LOCKOUT_MINUTES)
                security_audit("account_locked", user_id=user.id, email=normalized_email,
                               ip=request.client.host,
                               detail=f"Locked for {ACCOUNT_LOCKOUT_MINUTES}m after {user.failed_login_attempts} failures")
            db.commit()
        security_audit("login_failed", email=normalized_email, ip=request.client.host)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Successful login — reset lockout counters
    user.failed_login_attempts = 0
    user.locked_until = None
    db.commit()

    security_audit("login_success", user_id=user.id, email=normalized_email, ip=request.client.host)
    token = create_token(user.id, user.token_version)
    return AuthResponse(token=token, user=UserResponse.model_validate(user))


@router.post("/guest", response_model=AuthResponse)
@_limiter.limit("3/minute")
def guest_login(request: Request, db: Session = Depends(get_db)):
    guest = User(
        display_name="Guest",
        is_guest=True,
        role="guest",
    )
    db.add(guest)
    db.commit()
    db.refresh(guest)
    token = create_token(guest.id, guest.token_version)
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

    # Only allow explicit safe fields — prevent mass assignment
    _ALLOWED_PROFILE_FIELDS = {"display_name"}
    update_data = req.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if key not in _ALLOWED_PROFILE_FIELDS:
            continue
        if isinstance(value, str):
            # Strip HTML tags and trim
            value = _DISPLAY_NAME_RE.sub("", value).strip()[:_MAX_DISPLAY_NAME_LENGTH]
            if not value:
                continue  # skip empty after sanitization
        setattr(user, key, value)

    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)


@router.post("/password")
def change_password(
    request: Request,
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

    # OAuth-only users don't have a password set
    if not user.password_hash:
        raise HTTPException(
            status_code=400,
            detail="Your account uses OAuth login and has no password set. "
                   "Use the 'Forgot Password' flow to set one."
        )

    if not verify_password(req.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    _validate_password(req.new_password)

    user.password_hash = hash_password(req.new_password)
    # Revoke all existing tokens by bumping version
    user.token_version = (user.token_version or 0) + 1
    db.commit()

    security_audit("password_changed", user_id=user.id, email=user.email, ip=request.client.host if hasattr(request, 'client') else None)
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

    new_token = create_token(user.id, user.token_version)
    return AuthResponse(token=new_token, user=UserResponse.model_validate(user))


@router.post("/logout-all", response_model=DetailResponse)
def logout_all(
    request: Request,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Revoke ALL existing tokens for this user by bumping token_version."""
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    user.token_version = (user.token_version or 0) + 1
    db.commit()

    security_audit("logout_all", user_id=user.id, email=user.email, ip=request.client.host)
    return {"detail": "All sessions have been revoked. Please log in again."}


@router.delete("/account", response_model=DetailResponse)
def delete_account(
    request: Request,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Permanently delete the authenticated user's account and all related data.

    Non-guest, non-OAuth users must provide their current password in the
    request body as {"password": "..."} to confirm deletion.
    """
    try:
        user = get_user_from_header(db, authorization)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    # Require password confirmation for registered (non-guest, non-OAuth) accounts
    if not user.is_guest and user.password_hash:
        try:
            import json as _json
            body = _json.loads(request._body.decode()) if hasattr(request, '_body') else {}
        except Exception:
            body = {}
        # Also check query param as fallback for DELETE body support
        password = body.get("password", "")
        if not password:
            raise HTTPException(
                status_code=400,
                detail="Password confirmation required to delete your account.",
            )
        if not verify_password(password, user.password_hash):
            raise HTTPException(
                status_code=403,
                detail="Incorrect password. Account deletion cancelled.",
            )

    # Get all projects owned by this user for filesystem cleanup
    projects = db.query(Project).filter(Project.user_id == user.id).all()

    # Delete all project files from disk
    import shutil
    for project in projects:
        project_dir = PROJECTS_DIR / str(project.id)
        if project_dir.exists():
            shutil.rmtree(project_dir, ignore_errors=True)
            logger.info("Deleted project directory for project %d (user %d)", project.id, user.id)

    # Clean up background tasks associated with user's projects
    project_ids = [p.id for p in projects]
    if project_ids:
        db.query(BackgroundTask).filter(
            BackgroundTask.task_type.in_(("gguf", "augment")),
            BackgroundTask.task_key.in_([str(pid) for pid in project_ids]),
        ).delete(synchronize_session=False)

    # CASCADE handles: Projects -> Datasets, TrainingRuns, ModelConfigs, PromptTemplates
    # But we explicitly delete to be safe
    for project in projects:
        db.query(PromptTemplate).filter(PromptTemplate.project_id == project.id).delete(synchronize_session=False)
        db.query(Dataset).filter(Dataset.project_id == project.id).delete(synchronize_session=False)
        db.query(TrainingRun).filter(TrainingRun.project_id == project.id).delete(synchronize_session=False)
        db.query(ModelConfig).filter(ModelConfig.project_id == project.id).delete(synchronize_session=False)

    db.query(Project).filter(Project.user_id == user.id).delete(synchronize_session=False)
    db.delete(user)
    db.commit()

    security_audit("account_deleted", user_id=user.id, email=user.email, ip=request.client.host)
    logger.info("Deleted account for user %d (%s)", user.id, user.email or "guest")
    return {"detail": "Account and all associated data permanently deleted"}


@router.post("/forgot-password", response_model=DetailResponse)
@_limiter.limit(RATE_LIMIT_PASSWORD_RESET)
def forgot_password(
    request: Request,
    req: ForgotPasswordRequest,
    db: Session = Depends(get_db),
):
    """
    Request a password reset token.

    In production this would send an email; in local-dev mode the token is
    returned in the response for convenience. The response is always the same
    regardless of whether the email exists (prevents user enumeration).
    """
    normalized_email = _normalize_email(req.email)
    user = db.query(User).filter(User.email == normalized_email).first()

    # Always return success message to prevent user enumeration
    _success_msg = "If an account with that email exists, a reset link has been generated."

    if not user or user.is_guest:
        return {"detail": _success_msg}

    # Generate a secure reset token
    reset_token = secrets.token_urlsafe(48)
    user.password_reset_token = reset_token
    user.password_reset_expires = datetime.now(timezone.utc) + timedelta(hours=1)
    db.commit()

    # Send email if SMTP configured, otherwise log
    email_sent = send_password_reset_email(normalized_email, reset_token)
    if not email_sent:
        logger.info(
            "Password reset token generated for %s: %s (expires in 1 hour)",
            normalized_email,
            reset_token,
        )

    # Always return the same generic message (never expose token in response)
    return {"detail": _success_msg}


@router.post("/reset-password", response_model=DetailResponse)
@_limiter.limit(RATE_LIMIT_AUTH)
def reset_password(
    request: Request,
    req: ResetPasswordRequest,
    db: Session = Depends(get_db),
):
    """Reset a user's password using a valid reset token."""
    user = db.query(User).filter(
        User.password_reset_token == req.token,
    ).first()

    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    if not user.password_reset_expires or user.password_reset_expires < datetime.now(timezone.utc):
        # Expired — clear the token
        user.password_reset_token = None
        user.password_reset_expires = None
        db.commit()
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    # Validate new password
    _validate_password(req.new_password)

    # Apply the new password and clear the reset token
    user.password_hash = hash_password(req.new_password)
    user.password_reset_token = None
    user.password_reset_expires = None
    db.commit()

    logger.info("Password reset successful for user %d", user.id)
    return {"detail": "Password has been reset successfully. You can now log in with your new password."}


# ─────────────────────────────────────────────────────
# Email Verification
# ─────────────────────────────────────────────────────

@router.post("/verify-email", response_model=DetailResponse)
def verify_email(
    req: VerifyEmailRequest,
    db: Session = Depends(get_db),
):
    """Verify user's email with the token sent during registration."""
    user = db.query(User).filter(
        User.email_verification_token == req.token,
    ).first()

    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")

    if user.email_verified:
        return {"detail": "Email already verified"}

    user.email_verified = True
    user.email_verification_token = None
    db.commit()

    logger.info("Email verified for user %d (%s)", user.id, user.email)
    return {"detail": "Email verified successfully"}


@router.post("/resend-verification", response_model=DetailResponse)
@_limiter.limit(RATE_LIMIT_AUTH)
def resend_verification(
    request: Request,
    req: ResendVerificationRequest,
    db: Session = Depends(get_db),
):
    """Resend the email verification token."""
    normalized_email = _normalize_email(req.email)
    user = db.query(User).filter(User.email == normalized_email).first()

    # Always return success to prevent enumeration
    _msg = "If the account exists and is unverified, a new verification link has been sent."

    if not user or user.email_verified or user.is_guest:
        return {"detail": _msg}

    # Generate new token
    new_token = secrets.token_urlsafe(48)
    user.email_verification_token = new_token
    db.commit()

    email_sent = send_verification_email(normalized_email, new_token)
    if not email_sent:
        logger.info("[DEV] Resent verification token for %s: %s", normalized_email, new_token)

    return {"detail": _msg}


# ─────────────────────────────────────────────────────
# OAuth — Google
# ─────────────────────────────────────────────────────

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


@router.get("/oauth/google")
def oauth_google_redirect():
    """Redirect user to Google's OAuth consent screen."""
    if not OAUTH_GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")

    # Generate CSRF state token
    state = _generate_oauth_state()
    state_sig = _sign_state(state)

    redirect_uri = f"{OAUTH_REDIRECT_BASE}/api/auth/oauth/google/callback"
    params = {
        "client_id": OAUTH_GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    url = f"{_GOOGLE_AUTH_URL}?" + "&".join(f"{k}={v}" for k, v in params.items())
    response = RedirectResponse(url=url)
    # Store state signature in a secure cookie for verification on callback
    response.set_cookie(
        key="oauth_state_sig",
        value=state_sig,
        max_age=600,  # 10 minutes
        httponly=True,
        samesite="lax",
    )
    return response


@router.get("/oauth/google/callback")
def oauth_google_callback(
    request: Request,
    code: str = Query(...),
    state: str = Query(""),
    db: Session = Depends(get_db),
):
    """Handle Google OAuth callback, create or log in user."""
    if not OAUTH_GOOGLE_CLIENT_ID or not OAUTH_GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")

    # Verify CSRF state token
    state_sig = request.cookies.get("oauth_state_sig", "")
    if not state or not state_sig or not _verify_oauth_state(state, state_sig):
        raise HTTPException(status_code=400, detail="Invalid OAuth state — possible CSRF attack. Please try again.")

    redirect_uri = f"{OAUTH_REDIRECT_BASE}/api/auth/oauth/google/callback"

    # Exchange code for tokens
    try:
        token_resp = httpx.post(_GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": OAUTH_GOOGLE_CLIENT_ID,
            "client_secret": OAUTH_GOOGLE_CLIENT_SECRET,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }, timeout=10)
        token_resp.raise_for_status()
        tokens = token_resp.json()
    except Exception as e:
        logger.error("Google OAuth token exchange failed: %s", e)
        raise HTTPException(status_code=400, detail="OAuth token exchange failed")

    # Fetch user info
    try:
        userinfo_resp = httpx.get(_GOOGLE_USERINFO_URL, headers={
            "Authorization": f"Bearer {tokens['access_token']}",
        }, timeout=10)
        userinfo_resp.raise_for_status()
        userinfo = userinfo_resp.json()
    except Exception as e:
        logger.error("Google userinfo fetch failed: %s", e)
        raise HTTPException(status_code=400, detail="Failed to fetch user info from Google")

    google_id = userinfo.get("id")
    email = userinfo.get("email", "").strip().lower()
    name = userinfo.get("name", "Google User")

    if not google_id or not email:
        raise HTTPException(status_code=400, detail="Google did not return required user info")

    # Find or create user
    user = db.query(User).filter(
        (User.oauth_provider == "google") & (User.oauth_id == google_id)
    ).first()

    if not user:
        # Check if email already exists (link accounts)
        user = db.query(User).filter(User.email == email).first()
        if user:
            user.oauth_provider = "google"
            user.oauth_id = google_id
            user.email_verified = True
        else:
            user = User(
                email=email,
                display_name=name[:100],
                oauth_provider="google",
                oauth_id=google_id,
                email_verified=True,
            )
            db.add(user)

    db.commit()
    db.refresh(user)

    jwt_token = create_token(user.id)
    logger.info("Google OAuth login for user %d (%s)", user.id, email)

    # Redirect to frontend with token
    response = RedirectResponse(
        url=f"{OAUTH_REDIRECT_BASE}/oauth/callback?token={jwt_token}",
        status_code=302,
    )
    # Clear the CSRF state cookie
    response.delete_cookie("oauth_state_sig")
    return response


# ─────────────────────────────────────────────────────
# OAuth — GitHub
# ─────────────────────────────────────────────────────

_GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
_GITHUB_USER_URL = "https://api.github.com/user"
_GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


@router.get("/oauth/github")
def oauth_github_redirect():
    """Redirect user to GitHub's OAuth consent screen."""
    if not OAUTH_GITHUB_CLIENT_ID:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured")

    # Generate CSRF state token
    state = _generate_oauth_state()
    state_sig = _sign_state(state)

    redirect_uri = f"{OAUTH_REDIRECT_BASE}/api/auth/oauth/github/callback"
    params = {
        "client_id": OAUTH_GITHUB_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": "user:email",
        "state": state,
    }
    url = f"{_GITHUB_AUTH_URL}?" + "&".join(f"{k}={v}" for k, v in params.items())
    response = RedirectResponse(url=url)
    response.set_cookie(
        key="oauth_state_sig",
        value=state_sig,
        max_age=600,
        httponly=True,
        samesite="lax",
    )
    return response


@router.get("/oauth/github/callback")
def oauth_github_callback(
    request: Request,
    code: str = Query(...),
    state: str = Query(""),
    db: Session = Depends(get_db),
):
    """Handle GitHub OAuth callback, create or log in user."""
    if not OAUTH_GITHUB_CLIENT_ID or not OAUTH_GITHUB_CLIENT_SECRET:
        raise HTTPException(status_code=501, detail="GitHub OAuth not configured")

    # Verify CSRF state token
    state_sig = request.cookies.get("oauth_state_sig", "")
    if not state or not state_sig or not _verify_oauth_state(state, state_sig):
        raise HTTPException(status_code=400, detail="Invalid OAuth state — possible CSRF attack. Please try again.")

    # Exchange code for access token
    try:
        token_resp = httpx.post(_GITHUB_TOKEN_URL, data={
            "client_id": OAUTH_GITHUB_CLIENT_ID,
            "client_secret": OAUTH_GITHUB_CLIENT_SECRET,
            "code": code,
        }, headers={"Accept": "application/json"}, timeout=10)
        token_resp.raise_for_status()
        tokens = token_resp.json()
    except Exception as e:
        logger.error("GitHub OAuth token exchange failed: %s", e)
        raise HTTPException(status_code=400, detail="OAuth token exchange failed")

    access_token = tokens.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="GitHub did not return an access token")

    # Fetch user info
    auth_headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    try:
        user_resp = httpx.get(_GITHUB_USER_URL, headers=auth_headers, timeout=10)
        user_resp.raise_for_status()
        gh_user = user_resp.json()
    except Exception as e:
        logger.error("GitHub user fetch failed: %s", e)
        raise HTTPException(status_code=400, detail="Failed to fetch user info from GitHub")

    github_id = str(gh_user.get("id", ""))
    name = gh_user.get("name") or gh_user.get("login", "GitHub User")
    email = gh_user.get("email", "")

    # If email is not public, fetch from emails endpoint
    if not email:
        try:
            emails_resp = httpx.get(_GITHUB_EMAILS_URL, headers=auth_headers, timeout=10)
            emails_resp.raise_for_status()
            emails = emails_resp.json()
            primary = next((e for e in emails if e.get("primary")), None)
            if primary:
                email = primary.get("email", "")
        except Exception:
            pass

    email = email.strip().lower() if email else ""

    if not github_id:
        raise HTTPException(status_code=400, detail="GitHub did not return required user info")

    # Find or create user
    user = db.query(User).filter(
        (User.oauth_provider == "github") & (User.oauth_id == github_id)
    ).first()

    if not user:
        if email:
            user = db.query(User).filter(User.email == email).first()
        if user:
            user.oauth_provider = "github"
            user.oauth_id = github_id
            user.email_verified = True
        else:
            user = User(
                email=email or None,
                display_name=name[:100],
                oauth_provider="github",
                oauth_id=github_id,
                email_verified=bool(email),
            )
            db.add(user)

    db.commit()
    db.refresh(user)

    jwt_token = create_token(user.id)
    logger.info("GitHub OAuth login for user %d (%s)", user.id, email or github_id)

    response = RedirectResponse(
        url=f"{OAUTH_REDIRECT_BASE}/oauth/callback?token={jwt_token}",
        status_code=302,
    )
    response.delete_cookie("oauth_state_sig")
    return response
