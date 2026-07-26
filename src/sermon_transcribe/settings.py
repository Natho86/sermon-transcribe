import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

_lock = threading.Lock()
_config_path: Optional[Path] = None


def get_config_path() -> Path:
    global _config_path
    if _config_path is None:
        config_dir = Path(os.environ.get("CONFIG_DIR", "/app/config"))
        config_dir.mkdir(parents=True, exist_ok=True)
        _config_path = config_dir / "api_keys.json"
    return _config_path


def load_keys() -> dict:
    path = get_config_path()
    with _lock:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"keys": []}


def save_keys(data: dict) -> None:
    path = get_config_path()
    with _lock:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_active_provider() -> Optional[dict]:
    """Return info about the active key: {key, provider, model} or None.

    Falls back to env var ANTHROPIC_API_KEY with provider='anthropic'.
    """
    data = load_keys()
    for entry in data.get("keys", []):
        if entry.get("enabled"):
            return {
                "key": entry.get("key", ""),
                "provider": entry.get("provider", "anthropic"),
                "model": entry.get("model") or None,
            }
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return {
            "key": env_key,
            "provider": "anthropic",
            "model": os.environ.get("CLAUDE_MODEL") or None,
        }
    return None


def verify_anthropic_key(api_key: str) -> dict:
    """Verify an Anthropic API key via /v1/models."""
    url = "https://api.anthropic.com/v1/models"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            models = [m.get("id") for m in body.get("data", [])]
            return {"valid": True, "models": models}
    except urllib.error.HTTPError as exc:
        error_body = ""
        if exc.fp:
            error_body = exc.read().decode("utf-8")
        if exc.code == 401:
            return {"valid": False, "error": "Invalid API key"}
        if exc.code == 400 and "credit balance is too low" in error_body.lower():
            return {"valid": True, "error": "Credits exhausted", "credits_exhausted": True}
        return {"valid": False, "error": f"API error {exc.code}"}
    except urllib.error.URLError as exc:
        return {"valid": False, "error": f"Connection error: {exc}"}


def verify_openrouter_key(api_key: str) -> dict:
    """Verify an OpenRouter API key via /api/v1/models and return available models."""
    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            # Each model has id, name, context_length, pricing
            models = []
            for m in body.get("data", []):
                models.append({
                    "id": m.get("id", ""),
                    "name": m.get("name") or m.get("id", ""),
                })
            # Sort by id for consistent display
            models.sort(key=lambda m: m["id"])
            return {"valid": True, "models": models}
    except urllib.error.HTTPError as exc:
        error_body = ""
        if exc.fp:
            error_body = exc.read().decode("utf-8")
        if exc.code in {401, 403}:
            return {"valid": False, "error": "Invalid API key"}
        return {"valid": False, "error": f"API error {exc.code}: {error_body[:200]}"}
    except urllib.error.URLError as exc:
        return {"valid": False, "error": f"Connection error: {exc}"}


def verify_openai_key(api_key: str) -> dict:
    """Verify an OpenAI API key via /v1/models and return available models."""
    url = "https://api.openai.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            models = []
            for m in body.get("data", []):
                model_id = m.get("id", "")
                # Surface only chat-capable models
                if any(prefix in model_id for prefix in ("gpt-4", "gpt-3.5", "o1", "o3")):
                    models.append({"id": model_id, "name": model_id})
            models.sort(key=lambda m: m["id"])
            return {"valid": True, "models": models}
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            return {"valid": False, "error": "Invalid API key"}
        error_body = exc.read().decode("utf-8") if exc.fp else ""
        return {"valid": False, "error": f"API error {exc.code}: {error_body[:200]}"}
    except urllib.error.URLError as exc:
        return {"valid": False, "error": f"Connection error: {exc}"}


def verify_key(provider: str, api_key: str) -> dict:
    """Dispatch key verification to the correct provider."""
    if provider == "openrouter":
        return verify_openrouter_key(api_key)
    if provider == "openai":
        return verify_openai_key(api_key)
    return verify_anthropic_key(api_key)
