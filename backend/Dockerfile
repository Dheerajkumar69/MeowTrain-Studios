# ── Build stage ──
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system deps for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements-base.txt requirements-gpu.txt requirements-cpu.txt requirements.txt ./

# Install base requirements first (cached layer)
RUN pip install --no-cache-dir --prefix=/install -r requirements-base.txt

# ── Runtime stage ──
FROM python:3.11-slim

WORKDIR /app

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/projects data/model_cache

# Run auto-detection at container build time
# This detects GPU availability and installs the right PyTorch variant.
# When running with `docker compose` (nvidia runtime), GPU is available at
# runtime but NOT at build time, so we install base torch here and the
# entrypoint re-checks at startup.
ARG DEVICE_MODE=auto
RUN if [ "$DEVICE_MODE" = "gpu" ]; then \
        pip install --no-cache-dir "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cu121 && \
        pip install --no-cache-dir -r requirements-gpu.txt; \
    elif [ "$DEVICE_MODE" = "cpu" ]; then \
        pip install --no-cache-dir "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir -r requirements-cpu.txt; \
    else \
        pip install --no-cache-dir "torch>=2.4.0" && \
        pip install --no-cache-dir -r requirements-cpu.txt; \
    fi

# Copy the entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
