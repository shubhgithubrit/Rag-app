"""
Authentication — PBKDF2 password hashing, users stored in users.json.
No external dependencies beyond Python stdlib.
"""
import hashlib
import json
import os
import secrets

# users.json lives at project root (one level up from core/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_FILE = os.path.join(_ROOT, "users.json")


def _load() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}


def _save(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_pw(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}:{h.hex()}"


def _verify_pw(password: str, stored: str) -> bool:
    try:
        salt, h_hex = stored.split(":")
        h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return h.hex() == h_hex
    except Exception:
        return False


def verify_login(username: str, password: str) -> bool:
    users = _load()
    if username not in users:
        return False
    return _verify_pw(password, users[username]["password_hash"])


def get_user(username: str) -> dict | None:
    return _load().get(username)


def is_admin(username: str) -> bool:
    u = get_user(username)
    return bool(u and u.get("role") == "admin")


def create_user(username: str, password: str, role: str = "user", email: str = "") -> bool:
    users = _load()
    if username in users:
        return False
    users[username] = {"password_hash": _hash_pw(password), "role": role, "email": email}
    _save(users)
    return True


def delete_user(username: str) -> bool:
    users = _load()
    if username not in users or users[username].get("role") == "admin":
        return False
    del users[username]
    _save(users)
    return True


def list_users() -> list:
    return [
        {"username": u, "role": d["role"], "email": d.get("email", "")}
        for u, d in _load().items()
    ]


def init_default_users():
    """Create admin + demo users on first run."""
    if not os.path.exists(USERS_FILE):
        create_user("admin", "admin123", "admin", "admin@company.com")
        create_user("demo",  "demo123",  "user",  "demo@company.com")
