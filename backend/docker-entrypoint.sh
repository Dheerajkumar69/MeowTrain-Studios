#!/bin/bash
set -e

echo "🐱 MeowTrain — Runtime Device Detection"
echo ""

# ── Retry helper ──────────────────────────────────────────────────
pip_install_retry() {
    local max_attempts=3
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if pip install --no-cache-dir "$@" 2>/dev/null; then
            return 0
        fi
        echo "  ⚠ pip install attempt $attempt/$max_attempts failed, retrying..."
        attempt=$((attempt + 1))
        sleep $((attempt * 2))
    done
    echo "  ✗ pip install failed after $max_attempts attempts"
    return 1
}

# ── GPU Detection ─────────────────────────────────────────────────
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected at runtime"

    # Check if CUDA torch is already installed
    GPU_READY=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

    if [ "$GPU_READY" = "False" ]; then
        echo "⚠️  GPU available but PyTorch CUDA not installed — installing now..."
        pip_install_retry "torch>=2.4.0" --index-url https://download.pytorch.org/whl/cu121
        pip_install_retry "bitsandbytes>=0.44.0" "deepspeed>=0.14.0" || true
        echo "✅ GPU dependencies installed"
    fi

    # Write runtime device config
    python -c "
import json, torch, time
config = {
    'mode': 'cuda',
    'training_device': 'cuda',
    'cuda_available': True,
    'gpu_name': None,
    'vram_gb': None,
    'torch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'gpu_count': torch.cuda.device_count(),
    'setup_version': '2.0',
    'detected_at': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
    '_source': 'docker_entrypoint',
}
if torch.cuda.is_available():
    config['gpu_name'] = torch.cuda.get_device_name(0)
    try:
        props = torch.cuda.get_device_properties(0)
        config['vram_gb'] = round(props.total_memory / (1024**3), 1)
    except Exception:
        pass
with open('.device_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f'  GPU: {config[\"gpu_name\"]} ({config[\"vram_gb\"]} GB VRAM)')
"
else
    echo "ℹ️  No NVIDIA GPU detected — running in CPU mode"

    # Write CPU config
    python -c "
import json, time
try:
    import torch
    tv = torch.__version__
except ImportError:
    tv = 'not installed'
config = {
    'mode': 'cpu',
    'training_device': 'cpu',
    'cuda_available': False,
    'torch_version': tv,
    'setup_version': '2.0',
    'detected_at': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
    '_source': 'docker_entrypoint',
}
with open('.device_config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
fi

# ── Health smoke test ─────────────────────────────────────────────
echo ""
echo "🔍 Running smoke test..."
python -c "
import torch
x = torch.tensor([1.0, 2.0, 3.0])
assert x.sum().item() == 6.0, 'Tensor ops broken!'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'  ✓ PyTorch {torch.__version__} ({dev}) — tensor ops OK')
" || {
    echo "  ✗ Smoke test failed! PyTorch may not be properly installed."
    echo "    The server will start but training may not work."
}

echo ""

# Execute the main command
exec "$@"
