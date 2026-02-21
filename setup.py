#!/usr/bin/env python3
"""
MeowTrain — Automatic Hardware Detection & Dependency Installer

Detects your CPU/GPU hardware and installs exactly the right PyTorch
build and ML dependencies so training runs on the best available device.

Usage:
    python setup.py              # auto-detect and install
    python setup.py --cpu        # force CPU-only mode
    python setup.py --gpu        # force GPU mode (fails if no CUDA)
    python setup.py --info       # just print detected hardware, don't install
    python setup.py --dry-run    # show what would be installed without doing it
    python setup.py --reinstall  # force reinstall even if already configured

What it does:
    1. Detects NVIDIA GPU via nvidia-smi / NVML
    2. Detects CUDA version from the driver
    3. Picks the matching PyTorch wheel (cu118, cu121, cu124, or cpu)
    4. Installs base requirements + device-specific packages
    5. Validates the installation with a quick torch smoke test
    6. Writes .device_config.json so the app knows what's available
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# ── Minimum Python version check ────────────────────────────────────
if sys.version_info < (3, 10):
    print(f"ERROR: MeowTrain requires Python 3.10+, you have {sys.version}")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR / "backend" if (SCRIPT_DIR / "backend").exists() else SCRIPT_DIR
CONFIG_FILE = BACKEND_DIR / ".device_config.json"
LOCKFILE = BACKEND_DIR / ".setup_lock"

# Map CUDA driver major.minor → PyTorch index URL suffix
# PyTorch typically supports a few CUDA versions; pick closest match.
# Newer CUDA drivers are backward-compatible with older toolkit wheels.
CUDA_TO_TORCH_INDEX: dict[str, str] = {}
# Build the map dynamically for future-proofing
for _v in ("11.6", "11.7", "11.8"):
    CUDA_TO_TORCH_INDEX[_v] = "cu118"
for _v in ("12.0", "12.1", "12.2", "12.3"):
    CUDA_TO_TORCH_INDEX[_v] = "cu121"
for _minor in range(4, 20):          # 12.4 → 12.19
    CUDA_TO_TORCH_INDEX[f"12.{_minor}"] = "cu124"
for _major in range(13, 20):          # 13.x → 19.x  (future-proof)
    for _minor in range(0, 20):
        CUDA_TO_TORCH_INDEX[f"{_major}.{_minor}"] = "cu124"

TORCH_INDEX_URLS = {
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cpu":   "https://download.pytorch.org/whl/cpu",
}

# ── Terminal Colors ──────────────────────────────────────────────────

class C:
    """ANSI color codes — auto-disabled when stdout is not a TTY."""
    _enabled: bool = hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
    if sys.platform == "win32" and _enabled:
        try:  # Enable ANSI on Windows 10+
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            _enabled = False

    HEADER = "\033[95m" if _enabled else ""
    BLUE   = "\033[94m" if _enabled else ""
    GREEN  = "\033[92m" if _enabled else ""
    YELLOW = "\033[93m" if _enabled else ""
    RED    = "\033[91m" if _enabled else ""
    BOLD   = "\033[1m"  if _enabled else ""
    DIM    = "\033[2m"  if _enabled else ""
    RESET  = "\033[0m"  if _enabled else ""


def banner():
    print(f"""
{C.BOLD}{C.HEADER}╔══════════════════════════════════════════════════════════════╗
║        🐱  MeowTrain — Hardware Auto-Setup  🐱               ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
""")


# ── Locking (prevent parallel setup runs) ────────────────────────────

def _acquire_lock() -> bool:
    """Simple file-lock to prevent parallel setup.py runs."""
    if LOCKFILE.exists():
        try:
            age = time.time() - LOCKFILE.stat().st_mtime
            if age < 600:  # 10 minutes
                return False
            LOCKFILE.unlink(missing_ok=True)  # stale lock
        except OSError:
            pass
    try:
        LOCKFILE.write_text(str(os.getpid()))
        return True
    except OSError:
        return True  # can't write lock — proceed anyway


def _release_lock():
    try:
        LOCKFILE.unlink(missing_ok=True)
    except OSError:
        pass


# ── Pre-flight Checks ────────────────────────────────────────────────

def _check_pip() -> bool:
    """Verify pip is available."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _check_venv() -> bool:
    """Check if we're inside a virtual environment."""
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("VIRTUAL_ENV") is not None
        or os.environ.get("CONDA_DEFAULT_ENV") is not None
    )


def _check_internet(timeout: int = 5) -> bool:
    """Quick internet connectivity check."""
    try:
        import urllib.request
        urllib.request.urlopen("https://pypi.org/simple/", timeout=timeout)
        return True
    except Exception:
        return False


# ── Hardware Detection ───────────────────────────────────────────────

def detect_cpu_info() -> dict:
    """Gather CPU information."""
    cpu_name = "Unknown CPU"
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.split(":")[1].strip()
                        break
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                cpu_name = result.stdout.strip()
        elif platform.system() == "Windows":
            cpu_name = platform.processor() or "Unknown CPU"
    except Exception:
        cpu_name = platform.processor() or "Unknown CPU"

    try:
        import psutil
        core_count = psutil.cpu_count(logical=False) or 0
        thread_count = psutil.cpu_count(logical=True) or 0
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        try:
            import multiprocessing
            thread_count = multiprocessing.cpu_count()
            core_count = thread_count
        except Exception:
            thread_count = core_count = 1
        ram_gb = 0.0
        # Try to read RAM without psutil
        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            ram_gb = round(int(line.split()[1]) / (1024**2), 1)
                            break
            elif platform.system() == "Darwin":
                r = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    ram_gb = round(int(r.stdout.strip()) / (1024**3), 1)
        except Exception:
            pass

    return {
        "name": cpu_name,
        "cores": core_count,
        "threads": thread_count,
        "ram_gb": ram_gb,
        "arch": platform.machine(),
        "os": platform.system(),
    }


def detect_nvidia_gpu() -> dict | None:
    """
    Detect NVIDIA GPU via nvidia-smi.
    Returns None if no NVIDIA GPU found.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None

        gpus = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                vram_mb = int(float(parts[1]))
            except (ValueError, IndexError):
                vram_mb = 0
            gpus.append({
                "name": parts[0],
                "vram_mb": vram_mb,
                "vram_gb": round(vram_mb / 1024, 1),
                "driver_version": parts[2],
            })

        if not gpus:
            return None

        return {
            "available": True,
            "count": len(gpus),
            "gpus": gpus,
            "driver_version": gpus[0]["driver_version"],
            "primary_gpu": gpus[0]["name"],
            "primary_vram_gb": gpus[0]["vram_gb"],
        }
    except FileNotFoundError:
        return None  # nvidia-smi not installed
    except subprocess.TimeoutExpired:
        print(f"  {C.YELLOW}⚠ nvidia-smi timed out{C.RESET}")
        return None
    except Exception:
        return None


def detect_cuda_version() -> str | None:
    """
    Detect CUDA version supported by the installed NVIDIA driver.
    Returns version string like '12.4' or None.
    """
    # Method 1: nvidia-smi header (most reliable — shows max CUDA the driver supports)
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except Exception:
        pass

    # Method 2: nvcc --version
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            match = re.search(r"release\s+(\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except Exception:
        pass

    return None


def detect_amd_gpu() -> dict | None:
    """Detect AMD GPU via rocm-smi (ROCm)."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return {
                "available": True,
                "type": "amd_rocm",
                "info": result.stdout.strip(),
            }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass
    return None


def detect_apple_silicon() -> dict | None:
    """Detect Apple Silicon (MPS backend for PyTorch)."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    return {
        "available": True,
        "type": "apple_mps",
        "chip": platform.processor() or "Apple Silicon",
    }


def get_torch_index_url(cuda_version: str | None) -> tuple[str, str]:
    """
    Given a CUDA version string, return (torch_tag, index_url).
    Falls back to CPU if CUDA version is unknown.
    """
    if cuda_version is None:
        return "cpu", TORCH_INDEX_URLS["cpu"]

    # Try exact match first (e.g., "12.4")
    major_minor = ".".join(cuda_version.split(".")[:2])
    tag = CUDA_TO_TORCH_INDEX.get(major_minor)
    if tag:
        return tag, TORCH_INDEX_URLS[tag]

    # Try just major version — pick the highest compatible
    try:
        major = int(cuda_version.split(".")[0])
    except (ValueError, IndexError):
        return "cpu", TORCH_INDEX_URLS["cpu"]

    if major >= 12:
        return "cu124", TORCH_INDEX_URLS["cu124"]
    elif major == 11:
        return "cu118", TORCH_INDEX_URLS["cu118"]

    # CUDA < 11 is too old for modern PyTorch
    print(f"  {C.YELLOW}⚠ CUDA {cuda_version} is too old for current PyTorch, using CPU mode{C.RESET}")
    return "cpu", TORCH_INDEX_URLS["cpu"]


# ── Installation ─────────────────────────────────────────────────────

def run_pip(args: list[str], desc: str = "", dry_run: bool = False, retries: int = 2) -> bool:
    """Run a pip command with automatic retry. Returns True on success."""
    cmd = [sys.executable, "-m", "pip"] + args
    if dry_run:
        print(f"  {C.YELLOW}[DRY RUN]{C.RESET} {' '.join(cmd)}")
        return True

    print(f"  {C.BLUE}→{C.RESET} {desc or ' '.join(args)}")
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=900,  # 15 min
            )
            if result.returncode == 0:
                return True

            stderr_lines = [l for l in result.stderr.strip().split("\n") if l.strip()]
            last_line = stderr_lines[-1] if stderr_lines else "Unknown error"
            if attempt < retries:
                print(f"    {C.YELLOW}⚠ Attempt {attempt} failed, retrying...{C.RESET}")
                time.sleep(2 * attempt)  # backoff
            else:
                print(f"    {C.RED}✗ {last_line}{C.RESET}")
                return False

        except subprocess.TimeoutExpired:
            print(f"    {C.RED}✗ Timed out after 15 minutes{C.RESET}")
            return False
        except Exception as e:
            print(f"    {C.RED}✗ {e}{C.RESET}")
            return False
    return False


def install_torch(torch_tag: str, index_url: str, dry_run: bool = False) -> bool:
    """Install PyTorch with the correct CUDA/CPU variant."""
    desc = f"Installing PyTorch ({torch_tag})..."
    return run_pip(
        ["install", "torch>=2.4.0", "--index-url", index_url],
        desc=desc,
        dry_run=dry_run,
    )


def install_requirements(req_file: str, dry_run: bool = False) -> bool:
    """Install packages from a requirements file."""
    path = BACKEND_DIR / req_file
    if not path.exists():
        print(f"  {C.YELLOW}⚠ {req_file} not found, skipping{C.RESET}")
        return True
    desc = f"Installing {req_file}..."
    return run_pip(
        ["install", "-r", str(path)],
        desc=desc,
        dry_run=dry_run,
    )


# ── Validation ───────────────────────────────────────────────────────

def validate_installation(expected_device: str) -> dict:
    """
    Run a multi-step smoke test to verify the ML stack is working.
    Runs in a subprocess so freshly-installed packages are importable.
    Returns validation results dict. Never raises.
    """
    print(f"\n{C.BOLD}🔍 Validating installation...{C.RESET}")

    results: dict = {"valid": False, "torch_version": None, "device": "unknown", "errors": [], "warnings": []}

    validation_code = '''
import json, sys
result = {"valid": False, "torch_version": None, "device": "unknown",
          "errors": [], "warnings": [], "checks": {}}
try:
    import torch
    result["torch_version"] = torch.__version__
    result["checks"]["torch_import"] = True
    result["cuda_available"] = torch.cuda.is_available()
    result["cuda_built"] = torch.version.cuda is not None
    expected = sys.argv[1]
    if expected == "cuda":
        if torch.cuda.is_available():
            result["device"] = "cuda"
            result["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            result["vram_gb"] = round(props.total_memory / (1024**3), 1)
            result["gpu_arch"] = f"sm_{props.major}{props.minor}"
            result["checks"]["cuda_device"] = True
        else:
            result["device"] = "cpu"
            if torch.version.cuda is not None:
                result["warnings"].append("CUDA PyTorch installed but no GPU visible. Check nvidia-smi.")
            else:
                result["errors"].append("CPU-only PyTorch installed but GPU expected. Re-run: python setup.py --gpu")
    elif expected == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result["device"] = "mps"
            result["checks"]["mps_device"] = True
        else:
            result["device"] = "cpu"
            result["warnings"].append("MPS not available, using CPU")
    else:
        result["device"] = "cpu"
        result["checks"]["cpu_mode"] = True
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.sum().item() == 6.0
    result["checks"]["tensor_ops"] = True
    if result["device"] in ("cuda", "mps"):
        try:
            y = torch.tensor([1.0, 2.0], device=torch.device(result["device"]))
            assert y.sum().item() == 3.0
            result["checks"]["device_tensor"] = True
        except Exception as e:
            result["warnings"].append(f"Device tensor test failed: {e}")
    result["valid"] = len(result["errors"]) == 0
except ImportError as e:
    result["errors"].append(f"Cannot import torch: {e}")
except Exception as e:
    result["errors"].append(f"Validation error: {e}")
try:
    import transformers
    result["transformers_version"] = transformers.__version__
    result["checks"]["transformers_import"] = True
except ImportError:
    result["warnings"].append("transformers not importable yet")
except Exception:
    pass
try:
    import peft
    result["peft_version"] = peft.__version__
    result["checks"]["peft_import"] = True
except ImportError:
    result["warnings"].append("peft not importable yet")
except Exception:
    pass
print(json.dumps(result))
'''

    try:
        proc = subprocess.run(
            [sys.executable, "-c", validation_code, expected_device],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            results = json.loads(proc.stdout.strip())
        else:
            stderr = proc.stderr.strip()
            results["errors"].append(f"Validation subprocess failed: {stderr[:300]}")
    except subprocess.TimeoutExpired:
        results["errors"].append("Validation timed out after 60s")
    except json.JSONDecodeError:
        results["errors"].append("Validation produced invalid output")
    except Exception as e:
        results["errors"].append(f"Validation failed: {e}")

    # Pretty-print check results
    for check_name, passed in results.get("checks", {}).items():
        label = check_name.replace("_", " ").title()
        icon = f"{C.GREEN}✓" if passed else f"{C.RED}✗"
        print(f"  {icon} {label}{C.RESET}")

    if results.get("torch_version"):
        device = results.get("device", "cpu").upper()
        extra = ""
        if results.get("gpu_name"):
            extra = f" — {results['gpu_name']} ({results.get('vram_gb', '?')} GB)"
        print(f"  {C.GREEN}✓ PyTorch {results['torch_version']} ({device}{extra}){C.RESET}")

    if results.get("transformers_version"):
        print(f"  {C.GREEN}✓ Transformers {results['transformers_version']}{C.RESET}")

    for w in results.get("warnings", []):
        print(f"  {C.YELLOW}⚠ {w}{C.RESET}")
    for e in results.get("errors", []):
        print(f"  {C.RED}✗ {e}{C.RESET}")

    return results


# ── Config Persistence ───────────────────────────────────────────────

def save_device_config(config: dict):
    """Save detected hardware config to .device_config.json atomically."""
    config["detected_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    config["setup_version"] = "2.0"
    config["python_version"] = platform.python_version()
    config["python_executable"] = sys.executable
    config["platform"] = platform.platform()

    # Atomic write: temp file → rename (prevents half-written configs)
    tmp_file = CONFIG_FILE.with_suffix(".tmp")
    try:
        with open(tmp_file, "w") as f:
            json.dump(config, f, indent=2, default=str)
        tmp_file.replace(CONFIG_FILE)
        print(f"\n  {C.GREEN}✓ Device config saved to {CONFIG_FILE.name}{C.RESET}")
    except OSError as e:
        print(f"\n  {C.YELLOW}⚠ Could not save config: {e}{C.RESET}")
        try:  # Fallback to direct write
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2, default=str)
        except OSError:
            pass


def load_device_config() -> dict | None:
    """Load existing device config if present and valid."""
    if not CONFIG_FILE.exists():
        return None
    try:
        with open(CONFIG_FILE) as f:
            config = json.load(f)
        if not isinstance(config, dict) or "mode" not in config:
            return None
        return config
    except (json.JSONDecodeError, OSError):
        return None


# ── Main Logic ───────────────────────────────────────────────────────

def print_hardware_info(cpu_info: dict, gpu_info: dict | None, cuda_version: str | None,
                        apple_info: dict | None, amd_info: dict | None):
    """Pretty-print detected hardware."""
    print(f"{C.BOLD}📋 Detected Hardware{C.RESET}")
    print(f"  ├─ OS:       {cpu_info['os']} ({cpu_info['arch']})")
    print(f"  ├─ CPU:      {cpu_info['name']}")
    print(f"  ├─ Cores:    {cpu_info['cores']} physical / {cpu_info['threads']} logical")
    print(f"  ├─ RAM:      {cpu_info['ram_gb']} GB")

    if gpu_info:
        for i, gpu in enumerate(gpu_info.get("gpus", [])):
            prefix = "├" if i < gpu_info["count"] - 1 else "└"
            print(f"  {prefix}─ GPU {i}:    {C.GREEN}{gpu['name']}{C.RESET} — {gpu['vram_gb']} GB VRAM")
        print(f"  ├─ Driver:   {gpu_info['driver_version']}")
        print(f"  └─ CUDA:     {cuda_version or 'not detected'}")
    elif apple_info:
        print(f"  └─ GPU:      {C.GREEN}Apple Silicon (MPS){C.RESET}")
    elif amd_info:
        print(f"  └─ GPU:      {C.YELLOW}AMD ROCm detected (experimental){C.RESET}")
    else:
        print(f"  └─ GPU:      {C.YELLOW}None detected (CPU-only mode){C.RESET}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="🐱 MeowTrain — Auto-detect hardware & install dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python setup.py              Auto-detect and install
  python setup.py --info       Show hardware info only
  python setup.py --cpu        Force CPU-only (skip GPU detection)
  python setup.py --dry-run    Preview what would be installed
""",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only mode")
    parser.add_argument("--gpu", action="store_true", help="Force GPU mode (will fail if no CUDA)")
    parser.add_argument("--info", action="store_true", help="Print hardware info and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be installed without doing it")
    parser.add_argument("--skip-validation", action="store_true", help="Skip post-install validation")
    parser.add_argument("--reinstall", action="store_true", help="Force reinstall even if previously configured")
    args = parser.parse_args()

    banner()

    # ── Pre-flight checks ────────────────────────────────────────
    if not args.info and not args.dry_run:
        if not _check_pip():
            print(f"{C.RED}✗ pip is not available. Install it with: python -m ensurepip --upgrade{C.RESET}")
            return 1

        if not _check_venv():
            print(f"{C.YELLOW}⚠ WARNING: You are not in a virtual environment!{C.RESET}")
            print(f"  This will install packages system-wide, which may cause conflicts.")
            print(f"  Recommended: python -m venv .venv && source .venv/bin/activate\n")
            try:
                answer = input(f"  Continue anyway? [y/N] ").strip().lower()
                if answer not in ("y", "yes"):
                    print("Aborted.")
                    return 0
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return 0
            print()

        if not args.reinstall:
            existing = load_device_config()
            if existing and existing.get("setup_version"):
                mode = existing.get("mode", "unknown")
                when = existing.get("detected_at", "unknown")
                print(f"{C.GREEN}✓ Already configured:{C.RESET} mode={mode}, set up at {when}")
                print(f"  Run with --reinstall to reconfigure, or --info to see hardware.\n")
                return 0

        if not _acquire_lock():
            print(f"{C.YELLOW}⚠ Another setup.py is already running. Wait or delete {LOCKFILE}{C.RESET}")
            return 1

        print(f"{C.DIM}Checking internet connectivity...{C.RESET}")
        if not _check_internet():
            print(f"{C.RED}✗ Cannot reach PyPI. Check your internet connection.{C.RESET}")
            _release_lock()
            return 1

    try:
        return _main_inner(args)
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Setup interrupted by user.{C.RESET}")
        return 130
    except Exception as e:
        print(f"\n{C.RED}✗ Unexpected error: {e}{C.RESET}")
        traceback.print_exc()
        return 1
    finally:
        _release_lock()


def _main_inner(args: argparse.Namespace) -> int:
    """Core setup logic, separated for clean error handling."""

    # ── Step 1: Detect Hardware ──────────────────────────────────
    print(f"{C.BOLD}🔎 Detecting hardware...{C.RESET}\n")

    cpu_info = detect_cpu_info()
    gpu_info = detect_nvidia_gpu()
    cuda_version = detect_cuda_version() if gpu_info else None
    apple_info = detect_apple_silicon()
    amd_info = detect_amd_gpu() if not gpu_info and not apple_info else None

    print_hardware_info(cpu_info, gpu_info, cuda_version, apple_info, amd_info)

    if args.info:
        # Also show current torch status if installed
        try:
            import torch  # type: ignore
            print(f"{C.BOLD}🔧 Current PyTorch{C.RESET}")
            print(f"  ├─ Version:  {torch.__version__}")
            print(f"  ├─ CUDA:     {torch.version.cuda or 'N/A'}")
            cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'
            print(f"  ├─ cuDNN:    {cudnn}")
            dev = 'cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
            print(f"  └─ Device:   {dev}")
        except ImportError:
            print(f"{C.YELLOW}  PyTorch not installed yet.{C.RESET}")
        print()
        return 0

    # ── Step 2: Determine Install Mode ───────────────────────────
    if args.cpu and args.gpu:
        print(f"{C.RED}✗ Cannot specify both --cpu and --gpu{C.RESET}")
        return 1

    if args.cpu:
        mode = "cpu"
        reason = "forced by --cpu flag"
    elif args.gpu:
        if not gpu_info:
            print(f"{C.RED}✗ --gpu flag specified but no NVIDIA GPU detected!{C.RESET}")
            print(f"  Check that nvidia-smi works and GPU drivers are installed.")
            return 1
        if not cuda_version:
            print(f"{C.RED}✗ NVIDIA GPU found but CUDA version could not be detected.{C.RESET}")
            return 1
        mode = "cuda"
        reason = "forced by --gpu flag"
    elif gpu_info and cuda_version:
        mode = "cuda"
        reason = f"NVIDIA GPU detected ({gpu_info['primary_gpu']})"
    elif apple_info:
        mode = "mps"
        reason = "Apple Silicon detected"
    elif amd_info:
        mode = "rocm"
        reason = "AMD ROCm detected (experimental — installing CPU PyTorch)"
    else:
        mode = "cpu"
        reason = "no GPU detected"

    torch_tag, index_url = get_torch_index_url(cuda_version if mode == "cuda" else None)

    # Special case for Apple Silicon — use default PyTorch (ships with MPS)
    if mode == "mps":
        torch_tag = "mps"
        index_url = ""  # Use default PyPI, PyTorch ships with MPS on macOS arm64

    print(f"{C.BOLD}⚙️  Install Mode: {C.GREEN}{mode.upper()}{C.RESET}")
    print(f"   Reason: {reason}")
    if mode == "cuda":
        print(f"   CUDA:   {cuda_version} → PyTorch {torch_tag}")
        print(f"   Index:  {index_url}")
    elif mode == "mps":
        print(f"   Index:  PyPI default (MPS included)")
    else:
        print(f"   Index:  {index_url}")
    print()

    # ── Step 3: Install Dependencies ─────────────────────────────
    print(f"{C.BOLD}📦 Installing dependencies...{C.RESET}\n")
    t0 = time.time()
    errors: list[str] = []

    # 3a: Install PyTorch first (everything else depends on it)
    print(f"  {C.BOLD}[1/3] PyTorch{C.RESET}")
    if mode == "mps" or mode == "rocm":
        # macOS / ROCm: just install from PyPI
        ok = run_pip(
            ["install", "torch>=2.4.0"],
            desc=f"Installing PyTorch ({torch_tag})...",
            dry_run=args.dry_run,
        )
    else:
        ok = install_torch(torch_tag, index_url, dry_run=args.dry_run)
    if not ok:
        print(f"\n  {C.RED}✗ PyTorch installation failed.{C.RESET}")
        print(f"    • Check your internet connection")
        print(f"    • Try: pip install torch --index-url {index_url}")
        errors.append("PyTorch installation failed")

    # 3b: Install base requirements
    print(f"\n  {C.BOLD}[2/3] Base Requirements{C.RESET}")
    ok = install_requirements("requirements-base.txt", dry_run=args.dry_run)
    if not ok:
        print(f"  {C.YELLOW}⚠ Some base packages failed to install{C.RESET}")
        errors.append("Some base requirements failed")

    # 3c: Install device-specific requirements
    print(f"\n  {C.BOLD}[3/3] Device-Specific Packages{C.RESET}")
    if mode == "cuda":
        ok = install_requirements("requirements-gpu.txt", dry_run=args.dry_run)
        if not ok:
            print(f"  {C.YELLOW}⚠ Some GPU packages failed (bitsandbytes/deepspeed are optional){C.RESET}")
            errors.append("Some GPU packages failed (non-critical)")
    else:
        ok = install_requirements("requirements-cpu.txt", dry_run=args.dry_run)
        if not ok:
            print(f"  {C.YELLOW}⚠ Some CPU packages failed{C.RESET}")
            errors.append("Some CPU packages failed")

    elapsed = time.time() - t0
    print(f"\n  {C.GREEN}✓ Installation phase completed in {elapsed:.0f}s{C.RESET}")
    if errors:
        print(f"  {C.YELLOW}  ({len(errors)} non-fatal issue(s) — see above){C.RESET}")

    # ── Step 4: Validate ─────────────────────────────────────────
    validation: dict = {}
    if not args.skip_validation and not args.dry_run:
        expected_device = mode if mode in ("cuda", "mps") else "cpu"
        validation = validate_installation(expected_device)

        if mode == "cuda" and validation.get("device") == "cpu":
            print(f"\n  {C.YELLOW}⚠ GPU mode was selected but CUDA is not available to PyTorch.{C.RESET}")
            print(f"    Training will fall back to CPU.")
            mode = "cpu"

    # ── Step 5: Save Config ──────────────────────────────────────
    device_config = {
        "mode": mode,
        "torch_tag": torch_tag,
        "cpu": cpu_info,
        "gpu": gpu_info,
        "cuda_version": cuda_version,
        "apple_silicon": apple_info is not None,
        "amd_rocm": amd_info is not None,
        "validation": validation,
        "training_device": validation.get("device", mode),
        "install_errors": errors if errors else None,
    }
    if not args.dry_run:
        save_device_config(device_config)

    # ── Done ─────────────────────────────────────────────────────
    final_device = device_config["training_device"]
    if validation.get("valid", True) and not [e for e in errors if "PyTorch" in e]:
        print(f"""
{C.BOLD}{C.GREEN}╔══════════════════════════════════════════════════════════════╗
║                    ✅  Setup Complete!                        ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}

  Training device:  {C.BOLD}{final_device.upper()}{C.RESET}
""")
    else:
        print(f"""
{C.BOLD}{C.YELLOW}╔══════════════════════════════════════════════════════════════╗
║              ⚠️  Setup completed with warnings                ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}

  Training device:  {C.BOLD}{final_device.upper()}{C.RESET}
""")

    if mode == "cuda" and gpu_info:
        print(f"  {C.GREEN}🚀 GPU training enabled!{C.RESET}")
        print(f"     Your {gpu_info['primary_gpu']} with {gpu_info['primary_vram_gb']} GB VRAM is ready.")
        print(f"     LoRA, QLoRA, and full fine-tuning are all available.")
    elif mode == "mps":
        print(f"  {C.GREEN}🍎 Apple Silicon training enabled!{C.RESET}")
        print(f"     MPS acceleration will be used for training.")
        print(f"     Note: QLoRA (bitsandbytes) is not supported on MPS.")
    else:
        print(f"  {C.YELLOW}🐌 CPU-only mode.{C.RESET}")
        print(f"     Training will work but will be significantly slower.")
        print(f"     Consider using a machine with an NVIDIA GPU for faster training.")
        print(f"     TIP: Use LoRA with a small model (TinyLlama 1.1B) for CPU training.")

    print(f"""
  {C.BOLD}Next steps:{C.RESET}
    cd backend
    uvicorn app.main:app --reload --port 8000
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
