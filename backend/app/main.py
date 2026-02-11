import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.database import create_tables
from app.config import CORS_ORIGINS

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("meowllm")

# ── Rate limiter (shared across routes) ──
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan (replaces deprecated on_event) ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_tables()
    logger.info("MeowLLM Studio started — CORS origins: %s", CORS_ORIGINS)
    yield
    # Shutdown (cleanup goes here if needed)
    logger.info("MeowLLM Studio shutting down")


# ── App ──
app = FastAPI(title="MeowLLM Studio", version="0.3.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──
from app.routes import auth, projects, datasets, models, training, inference, hardware, lmstudio

app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(hardware.router)
app.include_router(lmstudio.router)


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "version": "0.3.0"}

