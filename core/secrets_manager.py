"""
Secure secret storage via the OS keyring (Windows Credential Manager).

Priority order for every secret:
  1. Windows Credential Manager  (keyring)
  2. Environment variable / .env  (fallback)

This means the app works with plain .env during development,
and switches to keyring automatically once secrets are migrated.
"""
from __future__ import annotations

import os

import keyring
from keyring.errors import PasswordDeleteError

SERVICE = "simple-rag-v5"

# Secrets that are managed (sensitive — should live in keyring, not .env)
MANAGED_KEYS = [
    "AZURE_OPENAI_API_KEY",
    "LANGCHAIN_API_KEY",
    "REDIS_URL",
]


# ── Core operations ───────────────────────────────────────────────────────────

def get_secret(key: str, fallback: str | None = None) -> str | None:
    """
    Return the secret value.
    Checks keyring first; falls back to env var, then to `fallback`.
    """
    try:
        value = keyring.get_password(SERVICE, key)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(key, fallback)


def set_secret(key: str, value: str) -> None:
    """Store a secret in Windows Credential Manager."""
    keyring.set_password(SERVICE, key, value)


def delete_secret(key: str) -> None:
    """Remove a secret from Windows Credential Manager."""
    try:
        keyring.delete_password(SERVICE, key)
    except PasswordDeleteError:
        pass


def secret_status() -> dict[str, str]:
    """
    Return the storage location of each managed key.
    Used by the admin dashboard Secrets tab.
    """
    result = {}
    for key in MANAGED_KEYS:
        try:
            in_kr = bool(keyring.get_password(SERVICE, key))
        except Exception:
            in_kr = False
        in_env = bool(os.getenv(key))
        if in_kr:
            result[key] = "keyring"
        elif in_env:
            result[key] = "env"
        else:
            result[key] = "missing"
    return result
