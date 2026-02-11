"""
LM Studio Integration Service

Communicates with LM Studio's OpenAI-compatible API for:
- Listing loaded models
- Running chat completions
- Health checking the connection

LM Studio's API mirrors OpenAI's format:
  GET  /v1/models           → list loaded models
  POST /v1/chat/completions → chat completion
"""

import httpx
import asyncio
import re
from typing import Optional
from urllib.parse import urlparse

# In-memory connection settings (persisted per-session)
_lmstudio_config = {
    "host": "http://localhost",
    "port": 1234,
    "enabled": False,
    "last_status": "disconnected",
}


def get_lmstudio_config() -> dict:
    return {**_lmstudio_config}


# Safe host patterns (localhost, private IPs only)
_SAFE_HOST_PATTERNS = re.compile(
    r'^(localhost|127\.\d{1,3}\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|'
    r'192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|'
    r'\[::1\])$'
)


def update_lmstudio_config(host: str = None, port: int = None, enabled: bool = None):
    if host is not None:
        # Normalize: ensure no trailing slash, add http if missing
        host = host.rstrip("/")
        if not host.startswith("http"):
            host = f"http://{host}"
        # SSRF protection: only allow localhost/private IPs
        parsed = urlparse(host)
        hostname = parsed.hostname or ""
        if not _SAFE_HOST_PATTERNS.match(hostname):
            raise ValueError(
                f"Invalid LM Studio host: {hostname}. "
                "Only localhost and private IP addresses are allowed."
            )
        _lmstudio_config["host"] = host
    if port is not None:
        if not (1 <= port <= 65535):
            raise ValueError("Port must be between 1 and 65535.")
        _lmstudio_config["port"] = port
    if enabled is not None:
        _lmstudio_config["enabled"] = enabled


def _base_url() -> str:
    return f"{_lmstudio_config['host']}:{_lmstudio_config['port']}/v1"


def check_connection() -> dict:
    """Check if LM Studio server is reachable."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{_base_url()}/models")
            if resp.status_code == 200:
                data = resp.json()
                model_count = len(data.get("data", []))
                _lmstudio_config["last_status"] = "connected"
                return {
                    "connected": True,
                    "model_count": model_count,
                    "url": _base_url(),
                }
            else:
                _lmstudio_config["last_status"] = "error"
                return {"connected": False, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        _lmstudio_config["last_status"] = "disconnected"
        return {"connected": False, "error": str(e)}


def list_models() -> list[dict]:
    """List models currently loaded in LM Studio."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{_base_url()}/models")
            resp.raise_for_status()
            data = resp.json()
            models = []
            for item in data.get("data", []):
                models.append({
                    "model_id": item.get("id", "unknown"),
                    "name": item.get("id", "Unknown Model").split("/")[-1],
                    "owned_by": item.get("owned_by", "lmstudio"),
                    "source": "lmstudio",
                })
            return models
    except Exception:
        return []


def chat_completion(
    model_id: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 512,
    stream: bool = False,
) -> dict:
    """
    Send a chat completion request to LM Studio.
    
    messages format: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{_base_url()}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Parse OpenAI-compatible response
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        return {
            "response": message.get("content", ""),
            "model": data.get("model", model_id),
            "tokens_used": usage.get("total_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "finish_reason": choice.get("finish_reason", "stop"),
        }


def text_completion(
    model_id: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> dict:
    """Wrapper that converts a simple prompt into chat format for LM Studio."""
    messages = [{"role": "user", "content": prompt}]
    return chat_completion(model_id, messages, temperature, max_tokens)
