#!/bin/bash
set -euo pipefail

# ============================================
#  MeowTrain — Bulletproof Local Dev Runner
#  Opens backend & frontend in separate terminals
# ============================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[✔]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✘]${NC} $1"; }
log_step()  { echo -e "${CYAN}[→]${NC} $1"; }

# ── Banner ──
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║   🐱 MeowTrain Local Dev Runner 🐱  ║${NC}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════╝${NC}"
echo ""

# ============================================
#  DETECT TERMINAL EMULATOR
# ============================================
TERM_CMD=""
if command -v gnome-terminal &>/dev/null; then
    TERM_CMD="gnome-terminal"
elif command -v konsole &>/dev/null; then
    TERM_CMD="konsole"
elif command -v xfce4-terminal &>/dev/null; then
    TERM_CMD="xfce4-terminal"
elif command -v mate-terminal &>/dev/null; then
    TERM_CMD="mate-terminal"
elif command -v kitty &>/dev/null; then
    TERM_CMD="kitty"
elif command -v alacritty &>/dev/null; then
    TERM_CMD="alacritty"
elif command -v xterm &>/dev/null; then
    TERM_CMD="xterm"
else
    log_error "No supported terminal emulator found!"
    log_error "Install one of: gnome-terminal, konsole, xfce4-terminal, kitty, alacritty, xterm"
    exit 1
fi
log_info "Terminal emulator: ${TERM_CMD}"

# ============================================
#  PRE-FLIGHT CHECKS
# ============================================
log_step "Running pre-flight checks..."

ERRORS=0

# 1. Check Python
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    log_error "Python not found. Install Python 3.8+ first."
    ERRORS=$((ERRORS + 1))
fi

if [[ -n "$PYTHON_CMD" ]]; then
    PY_VERSION=$($PYTHON_CMD --version 2>&1)
    log_info "Python found: ${PY_VERSION}"
fi

# 2. Check Node/npm
if command -v node &>/dev/null && command -v npm &>/dev/null; then
    NODE_VERSION=$(node --version 2>&1)
    NPM_VERSION=$(npm --version 2>&1)
    log_info "Node found: ${NODE_VERSION} / npm ${NPM_VERSION}"
else
    log_error "Node.js or npm not found. Install Node.js 18+ first."
    ERRORS=$((ERRORS + 1))
fi

# 3. Check backend venv
VENV_ACTIVATE=""
if [[ -f "$PROJECT_DIR/backend/venv/bin/activate" ]]; then
    VENV_ACTIVATE="$PROJECT_DIR/backend/venv/bin/activate"
    log_info "Python venv found: backend/venv/"
elif [[ -f "$PROJECT_DIR/.venv/bin/activate" ]]; then
    VENV_ACTIVATE="$PROJECT_DIR/.venv/bin/activate"
    log_info "Python venv found: .venv/"
else
    log_warn "No Python virtualenv found. Using system Python."
fi

# 4. Check backend .env
if [[ -f "$PROJECT_DIR/backend/.env" ]]; then
    log_info "Backend .env found"
else
    log_warn "No backend/.env file. Using defaults."
    [[ -f "$PROJECT_DIR/backend/.env.example" ]] && log_warn "Tip: cp .env.example .env"
fi

# 5. Check node_modules
if [[ -d "$PROJECT_DIR/frontend/node_modules" ]]; then
    log_info "Frontend node_modules found"
else
    log_warn "Frontend node_modules missing. Installing..."
    (cd "$PROJECT_DIR/frontend" && npm install) || {
        log_error "npm install failed."
        ERRORS=$((ERRORS + 1))
    }
fi

# 6. Check critical files
[[ ! -f "$PROJECT_DIR/backend/run.py" ]] && log_error "backend/run.py not found!" && ERRORS=$((ERRORS + 1))
[[ ! -f "$PROJECT_DIR/frontend/package.json" ]] && log_error "frontend/package.json not found!" && ERRORS=$((ERRORS + 1))

# 8. GPU / PRIME check (informational, never blocks)
if command -v lspci &>/dev/null && lspci | grep -qi "nvidia\|radeon\|amd.*vga"; then
    if command -v prime-select &>/dev/null; then
        PRIME_MODE=$(prime-select query 2>/dev/null || echo "unknown")
        if [[ "$PRIME_MODE" == "intel" ]]; then
            log_warn "NVIDIA GPU found but PRIME is set to 'intel' — dGPU is powered off!"
            log_warn "  GPU will NOT be used for training. To enable it:"
            log_warn "    sudo prime-select on-demand   # then reboot"
        elif [[ "$PRIME_MODE" == "on-demand" ]]; then
            log_info "GPU: PRIME on-demand mode — NVIDIA dGPU available for CUDA"
        elif [[ "$PRIME_MODE" == "nvidia" ]]; then
            log_info "GPU: PRIME nvidia mode — NVIDIA dGPU active"
        fi
    elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        log_info "GPU: NVIDIA detected and driver is active"
    fi
elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    log_info "GPU: NVIDIA driver active"
fi

# 7. Check ports — auto-kill anything holding our dev ports
free_port() {
    local port=$1 name=$2
    local pids=""
    if command -v lsof &>/dev/null; then
        pids=$(lsof -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null)
    elif command -v ss &>/dev/null; then
        pids=$(ss -tlnp 2>/dev/null | awk -F'pid=' "/\:${port} /{print \$2}" | cut -d, -f1)
    fi
    if [[ -n "$pids" ]]; then
        log_warn "Port $port ($name) in use by PID(s) $pids — killing..."
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
        sleep 1
        # Verify it's free now
        local still=""
        if command -v lsof &>/dev/null; then
            still=$(lsof -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null)
        fi
        if [[ -n "$still" ]]; then
            log_error "Could not free port $port — try: sudo kill -9 $still"
            return 1
        fi
        log_info "Port $port ($name) freed"
    fi
    return 0
}

free_port 8000 "Backend"  || ERRORS=$((ERRORS + 1))
free_port 5173 "Frontend" || ERRORS=$((ERRORS + 1))

# Bail on errors
if [[ $ERRORS -gt 0 ]]; then
    echo ""
    log_error "Pre-flight failed with $ERRORS error(s). Fix the issues above and retry."
    exit 1
fi

echo ""
log_info "All pre-flight checks passed!"
echo ""

# ============================================
#  BUILD LAUNCH COMMANDS
# ============================================

# Backend command: activate venv (if exists) + run
BACKEND_CMD="cd '${PROJECT_DIR}/backend'"
if [[ -n "$VENV_ACTIVATE" ]]; then
    BACKEND_CMD="${BACKEND_CMD} && source '${VENV_ACTIVATE}'"
fi
BACKEND_CMD="${BACKEND_CMD} && echo -e '\\033[0;36m══════════════════════════════════════\\033[0m' && echo -e '\\033[1;32m  🐍 MeowTrain Backend (port 8000)\\033[0m' && echo -e '\\033[0;36m══════════════════════════════════════\\033[0m' && echo '' && ${PYTHON_CMD} run.py; echo ''; echo 'Backend exited. Press Enter to close.'; read"

# Frontend command
FRONTEND_CMD="cd '${PROJECT_DIR}/frontend' && echo -e '\\033[0;36m══════════════════════════════════════\\033[0m' && echo -e '\\033[1;32m  ⚡ MeowTrain Frontend (port 5173)\\033[0m' && echo -e '\\033[0;36m══════════════════════════════════════\\033[0m' && echo '' && npm run dev; echo ''; echo 'Frontend exited. Press Enter to close.'; read"

# ============================================
#  LAUNCH IN SEPARATE TERMINALS
# ============================================
log_step "Opening backend terminal..."

case "$TERM_CMD" in
    gnome-terminal)
        gnome-terminal --title="🐍 MeowTrain Backend" -- bash -c "$BACKEND_CMD"
        ;;
    konsole)
        konsole --new-tab -p tabtitle="🐍 MeowTrain Backend" -e bash -c "$BACKEND_CMD" &
        ;;
    xfce4-terminal)
        xfce4-terminal --title="🐍 MeowTrain Backend" -e "bash -c \"$BACKEND_CMD\"" &
        ;;
    mate-terminal)
        mate-terminal --title="🐍 MeowTrain Backend" -e "bash -c \"$BACKEND_CMD\"" &
        ;;
    kitty)
        kitty --title "🐍 MeowTrain Backend" bash -c "$BACKEND_CMD" &
        ;;
    alacritty)
        alacritty --title "🐍 MeowTrain Backend" -e bash -c "$BACKEND_CMD" &
        ;;
    xterm)
        xterm -T "🐍 MeowTrain Backend" -e "bash -c \"$BACKEND_CMD\"" &
        ;;
esac

sleep 1
log_info "Backend terminal opened"

log_step "Opening frontend terminal..."

case "$TERM_CMD" in
    gnome-terminal)
        gnome-terminal --title="⚡ MeowTrain Frontend" -- bash -c "$FRONTEND_CMD"
        ;;
    konsole)
        konsole --new-tab -p tabtitle="⚡ MeowTrain Frontend" -e bash -c "$FRONTEND_CMD" &
        ;;
    xfce4-terminal)
        xfce4-terminal --title="⚡ MeowTrain Frontend" -e "bash -c \"$FRONTEND_CMD\"" &
        ;;
    mate-terminal)
        mate-terminal --title="⚡ MeowTrain Frontend" -e "bash -c \"$FRONTEND_CMD\"" &
        ;;
    kitty)
        kitty --title "⚡ MeowTrain Frontend" bash -c "$FRONTEND_CMD" &
        ;;
    alacritty)
        alacritty --title "⚡ MeowTrain Frontend" -e bash -c "$FRONTEND_CMD" &
        ;;
    xterm)
        xterm -T "⚡ MeowTrain Frontend" -e "bash -c \"$FRONTEND_CMD\"" &
        ;;
esac

sleep 1
log_info "Frontend terminal opened"

# ============================================
#  DONE
# ============================================
echo ""
echo -e "${CYAN}${BOLD}╔══════════════════════════════════════╗${NC}"
echo -e "${CYAN}${BOLD}║      🚀 Both Terminals Launched 🚀  ║${NC}"
echo -e "${CYAN}${BOLD}╠══════════════════════════════════════╣${NC}"
echo -e "${CYAN}${BOLD}║${NC}  Backend  → ${GREEN}http://localhost:8000${NC}   ${CYAN}${BOLD}║${NC}"
echo -e "${CYAN}${BOLD}║${NC}  Frontend → ${GREEN}http://localhost:5173${NC}   ${CYAN}${BOLD}║${NC}"
echo -e "${CYAN}${BOLD}║${NC}  Health   → ${GREEN}http://localhost:8000/api/health${NC} ${CYAN}${BOLD}║${NC}"
echo -e "${CYAN}${BOLD}╚══════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Close each terminal window individually to stop its server.${NC}"
echo ""
