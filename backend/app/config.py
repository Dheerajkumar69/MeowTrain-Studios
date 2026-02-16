import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# ===== Database =====
DATABASE_URL = os.getenv("MEOWLLM_DATABASE_URL", "sqlite:///./meowllm.db")

# ===== Paths =====
DATA_DIR = Path(os.getenv("MEOWLLM_DATA_DIR", "./data"))
PROJECTS_DIR = DATA_DIR / "projects"
MODEL_CACHE_DIR = DATA_DIR / "model_cache"

for d in (DATA_DIR, PROJECTS_DIR, MODEL_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ===== Security =====
_default_secret = "meowllm-local-secret-key-change-in-prod"
JWT_SECRET = os.getenv("MEOWLLM_JWT_SECRET", _default_secret)
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

if JWT_SECRET == _default_secret:
    warnings.warn(
        "\n⚠️  MEOWLLM_JWT_SECRET is using the DEFAULT value!\n"
        "   This is fine for local development but MUST be changed in production.\n"
        "   Set it in your .env file or environment:\n"
        '   MEOWLLM_JWT_SECRET="$(python -c \'import secrets; print(secrets.token_urlsafe(32))\')"',
        stacklevel=2,
    )

# ===== CORS =====
_cors_raw = os.getenv("MEOWLLM_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
CORS_ORIGINS = [origin.strip() for origin in _cors_raw.split(",") if origin.strip()]

# ===== Upload Limits =====
ALLOWED_EXTENSIONS = {
    # Text
    ".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".pdf", ".docx",
    # Structured
    ".xlsx", ".xls", ".html", ".xml", ".yaml", ".yml", ".parquet",
    # Images (OCR)
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
}
try:
    MAX_UPLOAD_SIZE_MB = int(os.getenv("MEOWLLM_MAX_UPLOAD_MB", "100"))
except (ValueError, TypeError):
    MAX_UPLOAD_SIZE_MB = 100
MAX_UPLOAD_SIZE = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# ===== HuggingFace =====
HF_TOKEN = os.getenv("HF_TOKEN", "")
# Trust remote code in HuggingFace models — SECURITY RISK, keep False unless you
# audit every model repo you load.  Set to "true" only if needed.
TRUST_REMOTE_CODE = os.getenv("MEOWLLM_TRUST_REMOTE_CODE", "false").lower() in ("true", "1", "yes")

# ===== Rate Limiting =====
RATE_LIMIT_AUTH = os.getenv("MEOWLLM_RATE_LIMIT_AUTH", "5/minute")
RATE_LIMIT_INFERENCE = os.getenv("MEOWLLM_RATE_LIMIT_INFERENCE", "10/minute")

# ===== Role / Guest Limits =====
GUEST_MAX_PROJECTS = int(os.getenv("MEOWLLM_GUEST_MAX_PROJECTS", "3"))
GUEST_CLEANUP_DAYS = int(os.getenv("MEOWLLM_GUEST_CLEANUP_DAYS", "7"))

# ===== Upload Rate Limit =====
RATE_LIMIT_UPLOAD = os.getenv("MEOWLLM_RATE_LIMIT_UPLOAD", "10/minute")

# ===== Logging =====
LOG_LEVEL = os.getenv("MEOWLLM_LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("MEOWLLM_LOG_FORMAT", "text")  # "text" or "json"

# ===== Email (optional — for email verification & password reset) =====
SMTP_HOST = os.getenv("MEOWLLM_SMTP_HOST", "")
SMTP_PORT = int(os.getenv("MEOWLLM_SMTP_PORT", "587"))
SMTP_USER = os.getenv("MEOWLLM_SMTP_USER", "")
SMTP_PASSWORD = os.getenv("MEOWLLM_SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("MEOWLLM_SMTP_FROM", "noreply@meowtrain.local")
SMTP_ENABLED = bool(SMTP_HOST and SMTP_USER)

# ===== OAuth (optional) =====
OAUTH_GOOGLE_CLIENT_ID = os.getenv("MEOWLLM_OAUTH_GOOGLE_CLIENT_ID", "")
OAUTH_GOOGLE_CLIENT_SECRET = os.getenv("MEOWLLM_OAUTH_GOOGLE_CLIENT_SECRET", "")
OAUTH_GITHUB_CLIENT_ID = os.getenv("MEOWLLM_OAUTH_GITHUB_CLIENT_ID", "")
OAUTH_GITHUB_CLIENT_SECRET = os.getenv("MEOWLLM_OAUTH_GITHUB_CLIENT_SECRET", "")
OAUTH_REDIRECT_BASE = os.getenv("MEOWLLM_OAUTH_REDIRECT_BASE", "http://localhost:5173")

# ===== Frontend URL (for email links) =====
FRONTEND_URL = os.getenv("MEOWLLM_FRONTEND_URL", "http://localhost:5173")

# ===== Training Defaults =====
DEFAULT_TRAINING_CONFIG = {
    "method": "lora",
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "max_tokens": 512,
    "train_split": 0.9,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "warmup_steps": 10,
    "gradient_accumulation_steps": 4,
    # Best-practice additions
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.01,
    "gradient_checkpointing": True,
    "fp16": False,   # auto-detect at runtime
    "bf16": False,   # auto-detect at runtime
    "eval_steps": 50,
}

# ===== Model Catalog (recommendations — any HuggingFace model can be used) =====
MODEL_CATALOG = [
    {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "name": "TinyLlama 1.1B Chat",
        "description": "Compact but capable. Great for learning and testing. Fast training on consumer GPUs.",
        "parameters": "1.1B",
        "size_gb": 2.2,
        "ram_required_gb": 6,
        "vram_required_gb": 4,
        "recommended_hardware": "4GB+ VRAM or 8GB+ RAM",
        "estimated_train_minutes": 15,
        "icon": "🦙",
    },
    {
        "model_id": "microsoft/phi-2",
        "name": "Microsoft Phi-2",
        "description": "Small but mighty reasoning model. Excellent for code and logical tasks.",
        "parameters": "2.7B",
        "size_gb": 5.5,
        "ram_required_gb": 12,
        "vram_required_gb": 6,
        "recommended_hardware": "6GB+ VRAM or 12GB+ RAM",
        "estimated_train_minutes": 30,
        "icon": "🔬",
    },
    {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "name": "Mistral 7B Instruct",
        "description": "Powerful instruction-following model. Great balance of size and quality.",
        "parameters": "7B",
        "size_gb": 14.0,
        "ram_required_gb": 20,
        "vram_required_gb": 8,
        "recommended_hardware": "8GB+ VRAM recommended",
        "estimated_train_minutes": 60,
        "icon": "🌪️",
    },
    {
        "model_id": "meta-llama/Llama-3.2-3B",
        "name": "Llama 3.2 3B",
        "description": "Meta's latest compact model. Strong performance across tasks. Requires HF token.",
        "parameters": "3B",
        "size_gb": 6.4,
        "ram_required_gb": 12,
        "vram_required_gb": 6,
        "recommended_hardware": "6GB+ VRAM or 12GB+ RAM",
        "estimated_train_minutes": 25,
        "icon": "🦙",
    },
]
