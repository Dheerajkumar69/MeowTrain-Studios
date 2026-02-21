# 🐱 MeowTrain

**Fine-tune LLMs in your browser — no PhD required.**

MeowTrain is a self-hosted platform that lets you upload data, pick a model, click "Train", and chat with your fine-tuned AI. Built for hobbyists, researchers, and teams who want an easy on-ramp to LLM fine-tuning without wrestling with CLI tools.

---

## ✨ Features

| Area | Capability |
|------|-----------|
| **Data** | Upload `.txt`, `.pdf`, `.md`, `.csv`, `.json`, `.docx` — auto tokenized |
| **Models** | Llama 3, Mistral, Phi-3, GPT-2, Qwen — compatibility-checked against your hardware |
| **Training** | LoRA, QLoRA, full fine-tune with real-time loss charts + WebSocket metrics |
| **Inference** | Built-in playground with SSE streaming, or connect LM Studio |
| **Security** | JWT auth with token revocation, account lockout, rate limiting, audit logging, HSTS |
| **Database** | SQLite (dev default) or PostgreSQL via `DATABASE_URL` |

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**, **Node.js 18+**, **npm**
- *Optional*: NVIDIA GPU + CUDA drivers for GPU-accelerated training

### One-Command Setup (Recommended)

The setup script auto-detects your CPU/GPU hardware and installs exactly the right dependencies:

```bash
# Create a virtual environment
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# python -m venv .venv && .venv\Scripts\activate     # Windows

# Auto-detect hardware & install everything
python setup.py
```

This will:
1. 🔍 Detect your CPU, RAM, and GPU (NVIDIA CUDA, Apple Silicon MPS, or CPU-only)
2. 📦 Install the correct PyTorch build (CUDA 11.8/12.1/12.4, MPS, or CPU-only)
3. 📦 Install GPU-specific packages (bitsandbytes, deepspeed) only when a GPU is found
4. ✅ Validate the installation with a quick smoke test
5. 💾 Save hardware config for the backend to use at startup

**Setup flags:**
```bash
python setup.py --info       # Just show detected hardware
python setup.py --dry-run    # Show what would be installed
python setup.py --cpu        # Force CPU-only mode
python setup.py --gpu        # Force GPU mode
```

### Backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev              # http://localhost:5173
```

### Docker (Recommended for deployment)

```bash
# GPU mode (auto-detects at container startup)
docker compose up --build

# CPU-only mode (no GPU) — standalone compose file
docker compose -f docker-compose.cpu.yml up --build
```

## 🛠️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEOWLLM_DATABASE_URL` | `sqlite:///data/meowllm.db` | Database connection string |
| `MEOWLLM_JWT_SECRET` | *(change in prod)* | Secret key for JWT signing |
| `MEOWLLM_JWT_EXPIRY_HOURS` | `24` | Token expiry duration |
| `MEOWLLM_TRUST_REMOTE_CODE` | `false` | Allow HuggingFace `trust_remote_code` |
| `MEOWLLM_RATE_LIMIT_INFERENCE` | `10/minute` | Rate limit for the inference endpoint |
| `MEOWLLM_ENFORCE_HTTPS` | `false` | Add HSTS header (enable behind TLS proxy) |
| `MEOWLLM_ACCOUNT_LOCKOUT_ATTEMPTS` | `5` | Failed logins before lockout |
| `MEOWLLM_ACCOUNT_LOCKOUT_MINUTES` | `15` | Lockout duration in minutes |
| `MEOWLLM_RATE_LIMIT_PASSWORD_RESET` | `3/hour` | Password reset rate limit |
| `HF_TOKEN` | — | HuggingFace token for gated models |

## � Security Notes

MeowTrain is designed for **local / trusted-network** use. Before exposing it to the internet:

| Risk | Mitigation |
|------|-----------|
| **No HTTPS** | Place behind a TLS reverse proxy (Caddy, nginx, Traefik). Set `MEOWLLM_ENFORCE_HTTPS=true` to enable HSTS headers. |
| **No CSRF tokens** | All state-changing routes rely on the `Authorization: Bearer` header, which mitigates classic CSRF. Keep `MEOWLLM_CORS_ORIGINS` locked to your frontend origin. |
| **Default JWT secret** | Change `MEOWLLM_JWT_SECRET` in your `.env` before going live. The server logs a warning on startup if you haven't. |
| **Token revocation** | Password changes and `/auth/logout-all` invalidate all existing tokens via `token_version`. |
| **Account lockout** | After 5 failed logins (configurable), accounts are locked for 15 minutes. |
| **Audit logging** | Security events (login, register, lockout, admin actions) are logged to `meowllm.security` logger. |
| **Guest accounts** | Guests are capped at 3 projects by default (`MEOWLLM_GUEST_MAX_PROJECTS`). |
| **`trust_remote_code`** | Keep `MEOWLLM_TRUST_REMOTE_CODE=false` (default). Models with `auto_map` entries are flagged as supply-chain risks. |
| **Display names** | HTML tags are stripped from display names to prevent stored XSS. |

## �📁 Project Structure

```
Meow-Train/
├── backend/
│   ├── app/
│   │   ├── ml/          # Training engine (trainer, data loader, callbacks)
│   │   ├── routes/      # FastAPI routes (auth, datasets, training, inference)
│   │   ├── services/    # Business logic (auth, inference, LM Studio)
│   │   ├── models/      # SQLAlchemy models
│   │   └── utils/       # Text extraction, prompt templates
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/  # React UI (dataset, training, playground, models)
│   │   ├── pages/       # Dashboard, Project, Auth pages
│   │   ├── services/    # API client, WebSocket service
│   │   └── contexts/    # Auth context provider
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🚀 Production Deployment Guide

### Docker (Recommended)

```bash
# 1. Clone and configure
git clone https://github.com/your-org/MeowTrain-Studios.git
cd MeowTrain-Studios

# 2. Set up environment variables
cp backend/.env.example backend/.env
# Edit backend/.env with your production values (see below)

# 3. Build and run
docker-compose up -d --build

# 4. Verify
curl http://localhost:8000/api/health
```

### Critical Production Settings

Edit `backend/.env` with these values **before deploying**:

```bash
# REQUIRED — generate a unique secret:
#   python -c "import secrets; print(secrets.token_urlsafe(32))"
MEOWLLM_JWT_SECRET=your-unique-secret-here

# Lock CORS to your domain only
MEOWLLM_CORS_ORIGINS=https://your-domain.com

# Enable HTTPS security headers
MEOWLLM_ENFORCE_HTTPS=true

# Use PostgreSQL in production (not SQLite)
MEOWLLM_DATABASE_URL=postgresql://user:pass@db:5432/meowtrain

# Optional: Email for password reset & verification
MEOWLLM_SMTP_HOST=smtp.gmail.com
MEOWLLM_SMTP_PORT=587
MEOWLLM_SMTP_USER=your-email@gmail.com
MEOWLLM_SMTP_PASSWORD=your-app-password
MEOWLLM_SMTP_FROM=noreply@your-domain.com
```

### HTTPS with Reverse Proxy

MeowTrain does NOT handle TLS directly. Place it behind a reverse proxy:

**Caddy (simplest):**
```
your-domain.com {
    reverse_proxy localhost:5173
    handle_path /api/* {
        reverse_proxy localhost:8000
    }
}
```

**Nginx:**
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    ssl_certificate /etc/ssl/cert.pem;
    ssl_certificate_key /etc/ssl/key.pem;

    location / { proxy_pass http://localhost:5173; }
    location /api/ { proxy_pass http://localhost:8000; }
    location /ws/ {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Security Checklist

- [ ] Change `MEOWLLM_JWT_SECRET` from default
- [ ] Set `MEOWLLM_ENFORCE_HTTPS=true`
- [ ] Lock `MEOWLLM_CORS_ORIGINS` to your domain
- [ ] Use PostgreSQL (not SQLite) for multi-user
- [ ] Set `MEOWLLM_TRUST_REMOTE_CODE=false` (default)
- [ ] Place behind TLS reverse proxy
- [ ] Set up log monitoring (`meowllm.security` logger)
- [ ] Configure rate limits for your expected traffic

## 📄 License

MIT
