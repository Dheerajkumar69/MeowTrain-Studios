#!/bin/bash
#SBATCH --job-name=meow-train          # Job name shown in the queue
#SBATCH --output=logs/meow_%j.log      # stdout log (%j = job ID)
#SBATCH --error=logs/meow_%j.err       # stderr log
#SBATCH --partition=gpu                # Use GPU partition (change if your HPC calls it differently)
#SBATCH --gres=gpu:1                   # Request 1 GPU (SLURM can't do %; we cap at 60% below)
#SBATCH --cpus-per-task=18            # CPU cores (for DataLoader workers etc.)
#SBATCH --mem=64G                      # RAM
#SBATCH --time=24:00:00               # Max wall time — increase if cluster allows more
#SBATCH --nodes=1                      # Run on a single node

# ── Setup ─────────────────────────────────────────────────────────────
echo "=== MeowTrain Job Starting ==="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPU(s)   : $CUDA_VISIBLE_DEVICES"
echo "Date     : $(date)"
echo "=========================================="

# Absolute path to project root (works from symlink in ~/bin too)
PROJECT_DIR="$HOME/Meow-Train"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Go to project directory
cd "$PROJECT_DIR"

# Activate virtual environment (absolute path — safe from any working dir)
source "$PROJECT_DIR/.venv/bin/activate"

# ── Cap GPU usage to 60% ──────────────────────────────────────────────
# SLURM only allocates whole GPU cards — % capping is done at app level:
#
# 1) Limit compute threads to 60% (works if HPC has CUDA MPS enabled)
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=60
#
# 2) Hard-cap VRAM to 60% via PyTorch memory fraction
#    (runs before uvicorn so the limit is in effect from the start)
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.6, device=0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU VRAM cap set: 60% of {total:.1f} GB = {total*0.6:.1f} GB')
"

# Print GPU info for the log
nvidia-smi

# ── Run the backend ────────────────────────────────────────────────────
echo ""
echo "Starting MeowTrain backend on port 8000..."
echo "Frontend will be served from: frontend/dist/"
echo ""

cd "$PROJECT_DIR/backend"
uvicorn app.main:app --host 0.0.0.0 --port 8000

echo "=== MeowTrain Job Done ==="
