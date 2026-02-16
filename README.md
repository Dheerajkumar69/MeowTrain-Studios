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
| **Security** | JWT auth with refresh tokens, rate limiting, configurable `trust_remote_code` |
| **Database** | SQLite (dev default) or PostgreSQL via `DATABASE_URL` |

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**, **Node.js 18+**, **npm**
- *Optional*: NVIDIA GPU + CUDA 12 for training (CPU mode works for inference)

### Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # edit JWT_SECRET, HF_TOKEN etc.
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev              # http://localhost:5173
```

### Docker (Recommended)

```bash
# GPU mode
docker compose up --build

# CPU-only mode (no GPU)
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

## 🛠️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEOWLLM_DATABASE_URL` | `sqlite:///data/meowllm.db` | Database connection string |
| `MEOWLLM_JWT_SECRET` | *(change in prod)* | Secret key for JWT signing |
| `MEOWLLM_JWT_EXPIRY_HOURS` | `24` | Token expiry duration |
| `MEOWLLM_TRUST_REMOTE_CODE` | `false` | Allow HuggingFace `trust_remote_code` |
| `MEOWLLM_RATE_LIMIT_INFERENCE` | `10/minute` | Rate limit for the inference endpoint |
| `HF_TOKEN` | — | HuggingFace token for gated models |

## � Security Notes

MeowTrain is designed for **local / trusted-network** use. Before exposing it to the internet:

| Risk | Mitigation |
|------|-----------|
| **No HTTPS** | Place behind a TLS reverse proxy (Caddy, nginx, Traefik). Docker Compose examples use plain HTTP on port 8000/5173. |
| **No CSRF tokens** | All state-changing routes rely on the `Authorization: Bearer` header, which mitigates classic CSRF. For extra safety, keep `MEOWLLM_CORS_ORIGINS` locked to your frontend origin. |
| **Default JWT secret** | Change `MEOWLLM_JWT_SECRET` in your `.env` before going live. The server logs a warning on startup if you haven't. |
| **Guest accounts** | Guests are capped at 3 projects by default (`MEOWLLM_GUEST_MAX_PROJECTS`). Disable guest login entirely by removing the `/auth/guest` route or rate-limiting further. |
| **`trust_remote_code`** | Keep `MEOWLLM_TRUST_REMOTE_CODE=false` (default) unless you audit every model repo you load. |

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

## 📄 License

MIT
