import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from app.database import create_tables
from app.config import CORS_ORIGINS, LOG_LEVEL, LOG_FORMAT, ENFORCE_HTTPS
from app.middleware import RequestIDMiddleware, RequestIDLogFilter
from app.logging_config import configure_logging

# ── Logging (with request-ID correlation) ──
configure_logging(level=LOG_LEVEL, log_format=LOG_FORMAT)

logger = logging.getLogger("meowllm")

# ── Rate limiter (shared across routes) ──
limiter = Limiter(key_func=get_remote_address)


def _migrate_database():
    """Run Alembic migrations if available, otherwise fall back to create_all().

    In production, Alembic is the source of truth for schema changes,
    preventing the conflict where create_all() creates tables that skip
    migration version tracking.

    For fresh installs or dev setups without Alembic configured,
    falls back gracefully to create_all().
    """
    try:
        from alembic.config import Config as AlembicConfig
        from alembic import command as alembic_command
        import os

        alembic_ini = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "alembic.ini"
        )
        if os.path.exists(alembic_ini):
            alembic_cfg = AlembicConfig(alembic_ini)
            # Suppress alembic's own logging to avoid noise
            alembic_cfg.set_main_option("script_location",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "alembic"))
            alembic_command.upgrade(alembic_cfg, "head")
            logger.info("Database migrations applied successfully (Alembic)")
            return
        else:
            logger.info("No alembic.ini found — falling back to create_all()")
    except ImportError:
        logger.info("Alembic not installed — falling back to create_all()")
    except Exception as e:
        logger.warning("Alembic migration failed (%s) — falling back to create_all()", e)

    # Fallback: create_all() for fresh installs / dev without Alembic
    create_tables()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — run Alembic migrations (falls back to create_all for fresh installs)
    _migrate_database()
    _reconcile_interrupted_tasks()
    _cleanup_stale_guests()

    # Recover download/GGUF tasks that were interrupted by server restart
    from app.routes.models import recover_interrupted_downloads
    recover_interrupted_downloads()

    # Auto-detect hardware & validate device configuration
    from app.device_config import startup_device_check
    startup_device_check()

    logger.info("MeowLLM Studio started — CORS origins: %s", CORS_ORIGINS)
    yield
    # Shutdown (cleanup goes here if needed)
    logger.info("MeowLLM Studio shutting down")


def _reconcile_interrupted_tasks():
    """Mark any background tasks / training runs stuck in 'running' as interrupted.
    
    Also kills orphaned training processes via their stored worker_pid.
    """
    import os
    import signal

    from app.database import SessionLocal
    from app.models.training_run import TrainingRun
    from app.models.background_task import BackgroundTask  # noqa: ensure table created

    # ── Security reminder ──
    from app.config import JWT_SECRET
    _default_secret = "meowllm-local-secret-key-change-in-prod"
    if JWT_SECRET == _default_secret:
        logger.warning(
            "⚠️  SECURITY: Using default JWT secret. Fine for local dev, "
            "but MUST be changed before exposing to the network."
        )
    logger.info(
        "🔒 PRODUCTION REMINDER: MeowTrain serves plain HTTP. "
        "For production deployments, place behind a TLS-terminating reverse proxy "
        "(Caddy, nginx, Traefik) and restrict CORS origins."
    )

    db = SessionLocal()
    try:
        # Training runs
        stuck_runs = db.query(TrainingRun).filter(
            TrainingRun.status.in_(("running", "paused", "initializing"))
        ).all()
        for run in stuck_runs:
            # Try to kill the orphaned training process if PID is stored
            if run.worker_pid:
                try:
                    os.kill(run.worker_pid, signal.SIGTERM)
                    logger.warning(
                        "Sent SIGTERM to orphaned training process PID %d (run %d)",
                        run.worker_pid, run.id,
                    )
                except ProcessLookupError:
                    logger.info("Orphan PID %d already dead (run %d)", run.worker_pid, run.id)
                except PermissionError:
                    logger.warning(
                        "Cannot kill PID %d — permission denied (run %d)",
                        run.worker_pid, run.id,
                    )

            run.status = "error"
            run.worker_pid = None  # Clear stale PID
            run.error_message = (
                "Server restarted while this training run was in progress. "
                "Please start a new run."
            )
            logger.warning("Marked interrupted training run %d as error", run.id)

        # Background tasks (downloads, GGUF, etc.)
        stuck_tasks = db.query(BackgroundTask).filter(
            BackgroundTask.status == "running"
        ).all()
        for task in stuck_tasks:
            task.status = "interrupted"
            task.error = "Server restarted while task was running"
            logger.warning("Marked interrupted %s task '%s' as interrupted", task.task_type, task.task_key)

        if stuck_runs or stuck_tasks:
            db.commit()
            logger.info("Reconciled %d training runs and %d background tasks",
                        len(stuck_runs), len(stuck_tasks))
    except Exception as e:
        logger.error("Failed to reconcile interrupted tasks: %s", e)
        db.rollback()
    finally:
        db.close()


def _cleanup_stale_guests():
    """Delete guest accounts older than GUEST_CLEANUP_DAYS and their associated data."""
    from datetime import datetime, timedelta, timezone
    from app.database import SessionLocal
    from app.models.user import User
    from app.models.project import Project
    from app.config import GUEST_CLEANUP_DAYS, PROJECTS_DIR
    import shutil

    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=GUEST_CLEANUP_DAYS)
        stale_guests = db.query(User).filter(
            User.is_guest == True,  # noqa: E712
            User.created_at < cutoff,
        ).all()

        if not stale_guests:
            return

        deleted_count = 0
        for guest in stale_guests:
            # Delete filesystem data for all guest's projects
            projects = db.query(Project).filter(Project.user_id == guest.id).all()
            for project in projects:
                project_dir = PROJECTS_DIR / str(project.id)
                if project_dir.exists():
                    shutil.rmtree(project_dir, ignore_errors=True)

            # CASCADE handles DB children, but explicit delete to be safe
            db.query(Project).filter(Project.user_id == guest.id).delete(synchronize_session=False)
            db.delete(guest)
            deleted_count += 1

        db.commit()
        logger.info(
            "Guest cleanup: purged %d guest accounts older than %d days",
            deleted_count,
            GUEST_CLEANUP_DAYS,
        )
    except Exception as e:
        logger.error("Failed to clean up stale guests: %s", e)
        db.rollback()
    finally:
        db.close()


# ── App ──
app = FastAPI(title="MeowLLM Studio", version="0.4.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Prometheus metrics ──
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    _instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics", "/api/health"],
    )
    _instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    logger.info("Prometheus metrics enabled at /metrics")
except ImportError:
    logger.info("prometheus-fastapi-instrumentator not installed — metrics disabled")


# ── Global exception handler — catch-all to prevent stack traces leaking ──
@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions, log them with full trace, return safe JSON."""
    logger.error(
        "Unhandled %s on %s %s: %s\n%s",
        type(exc).__name__,
        request.method,
        request.url.path,
        exc,
        traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."},
    )


# ── Request body size limit middleware ──
_MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MB for JSON endpoints
# Upload routes have their own limits, so we exempt them
_UPLOAD_PATHS = {"/api/projects/import", "/api/datasets"}


class RequestBodyLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized request bodies to prevent DoS via huge JSON payloads."""
    async def dispatch(self, request: Request, call_next):
        # Skip size check for file upload endpoints (they enforce their own limits)
        if any(request.url.path.startswith(p) for p in _UPLOAD_PATHS):
            return await call_next(request)

        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body too large. Maximum size is {_MAX_BODY_SIZE // (1024*1024)} MB."},
            )
        return await call_next(request)


# ── Security headers middleware ──
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security-hardening response headers to every response."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

        # Cache-Control: no-store for API responses, allow caching for static assets
        path = request.url.path
        if path.startswith("/assets/") or path.endswith((".js", ".css", ".png", ".jpg", ".svg", ".woff2")):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-store"

        # CSP — allow self-hosted scripts/styles + Google Fonts for the frontend
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'none'"
        )
        # HSTS — only when explicitly enabled for production
        if ENFORCE_HTTPS:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


# ── CSRF protection middleware ──
class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """Validate Origin/Referer on state-changing requests to prevent CSRF.

    Since MeowTrain uses Bearer token auth (not cookies), CSRF risk is
    low — but this adds defense-in-depth for any future cookie-based flows
    (e.g. OAuth redirect cookies).
    """
    _SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}

    async def dispatch(self, request: Request, call_next):
        if request.method in self._SAFE_METHODS:
            return await call_next(request)

        # Skip CSRF check for non-browser clients (no Origin header)
        origin = request.headers.get("origin")
        referer = request.headers.get("referer")
        if not origin and not referer:
            # No browser indicators — likely API client, allow through
            return await call_next(request)

        # Build set of allowed origins: CORS origins + the server's own origin
        # (same-origin is always safe — this covers bare-metal/HPC deployments
        #  where the backend serves the frontend directly)
        allowed = set(CORS_ORIGINS) if CORS_ORIGINS != ["*"] else None
        if allowed is not None:
            # Also allow the request's own host (same-origin)
            scheme = "https" if request.url.scheme == "https" else "http"
            host_header = request.headers.get("host", "")
            if host_header:
                allowed.add(f"{scheme}://{host_header}")

            check_origin = origin or (referer.split("/", 3)[:3] if referer else [""])
            if isinstance(check_origin, list):
                check_origin = "/".join(check_origin)
            if check_origin not in allowed:
                logger.warning(
                    "CSRF: rejected %s %s from origin %s (allowed: %s)",
                    request.method, request.url.path, check_origin, allowed,
                )
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin request blocked (CSRF protection)"},
                )

        return await call_next(request)


app.add_middleware(CSRFProtectionMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestBodyLimitMiddleware)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# ── Request-ID correlation ──
app.add_middleware(RequestIDMiddleware)

# ── Routes (all under /api prefix to match frontend baseURL) ──
from app.routes import auth, projects, datasets, models, training, inference, hardware, lmstudio, augmentation
from app.routes import admin, backup, lineage

app.include_router(auth.router, prefix="/api")
app.include_router(projects.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(training.router, prefix="/api")
app.include_router(inference.router, prefix="/api")
app.include_router(hardware.router, prefix="/api")
app.include_router(lmstudio.router, prefix="/api")
app.include_router(augmentation.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(backup.router, prefix="/api")
app.include_router(lineage.router, prefix="/api")


from app.schemas import HealthResponse


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check with DB connectivity probe."""
    import time
    from app.database import SessionLocal
    from sqlalchemy import text

    db_connected = False
    db_latency_ms = None
    try:
        t0 = time.monotonic()
        session = SessionLocal()
        try:
            session.execute(text("SELECT 1"))
            db_connected = True
            db_latency_ms = round((time.monotonic() - t0) * 1000, 2)
        finally:
            session.close()
    except Exception:
        db_connected = False

    status = "healthy" if db_connected else "degraded"
    return {
        "status": status,
        "version": "0.4.0",
        "db_connected": db_connected,
        "db_latency_ms": db_latency_ms,
    }


# ── Static file serving for bare-metal / HPC deployment ──
# When running WITHOUT Docker/nginx (e.g. on a college HPC),
# the backend serves the pre-built frontend from ../frontend/dist.
# This is a no-op if the dist directory doesn't exist (dev mode uses Vite proxy).
import os as _os
from pathlib import Path as _Path

_FRONTEND_DIST = _Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"

if _FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse as _FileResponse

    _index_html = _FRONTEND_DIST / "index.html"

    # Serve static assets (JS, CSS, images) at /assets
    _assets_dir = _FRONTEND_DIST / "assets"
    if _assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="frontend-assets")

    # SPA catch-all: any non-API GET request returns index.html
    # so React Router can handle client-side routing
    @app.get("/{full_path:path}", include_in_schema=False)
    async def _serve_spa(full_path: str):
        # Don't intercept /api routes — let them 404 normally
        if full_path.startswith("api/") or full_path == "api":
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        # Serve real static files from frontend/dist
        file_path = _FRONTEND_DIST / full_path
        if full_path and file_path.is_file() and _FRONTEND_DIST in file_path.resolve().parents:
            return _FileResponse(str(file_path))
        return _FileResponse(str(_index_html))

    logger.info("Serving frontend from %s (bare-metal mode)", _FRONTEND_DIST)

