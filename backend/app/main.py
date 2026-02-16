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
from app.config import CORS_ORIGINS, LOG_LEVEL, LOG_FORMAT
from app.middleware import RequestIDMiddleware, RequestIDLogFilter
from app.logging_config import configure_logging

# ── Logging (with request-ID correlation) ──
configure_logging(level=LOG_LEVEL, log_format=LOG_FORMAT)

logger = logging.getLogger("meowllm")

# ── Rate limiter (shared across routes) ──
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan (replaces deprecated on_event) ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_tables()
    _reconcile_interrupted_tasks()
    _cleanup_stale_guests()
    logger.info("MeowLLM Studio started — CORS origins: %s", CORS_ORIGINS)
    yield
    # Shutdown (cleanup goes here if needed)
    logger.info("MeowLLM Studio shutting down")


def _reconcile_interrupted_tasks():
    """Mark any background tasks / training runs stuck in 'running' as interrupted."""
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
        "🔒 PRODUCTION REMINDER: MeowTrain serves plain HTTP with no CSRF protection. "
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
            run.status = "error"
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


# ── Security headers middleware ──
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security-hardening response headers to every response."""
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Cache-Control"] = "no-store"
        # CSP is light — API-only backend, no inline scripts
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        return response


app.add_middleware(SecurityHeadersMiddleware)

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

