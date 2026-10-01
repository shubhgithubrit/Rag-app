"""
One-time setup: migrate secrets from .env → Windows Credential Manager.

Run once:
    python setup_secrets.py

After running, the sensitive keys in .env can be removed or left as
fallback — keyring values always take priority.
"""
import os
from dotenv import load_dotenv
from core.secrets_manager import MANAGED_KEYS, SERVICE, set_secret, get_secret

load_dotenv()

print(f"\n{'='*55}")
print(f"  Migrating secrets → Windows Credential Manager")
print(f"  Service name: {SERVICE}")
print(f"{'='*55}\n")

for key in MANAGED_KEYS:
    env_val = os.getenv(key)
    if env_val:
        set_secret(key, env_val)
        masked = env_val[:6] + "..." + env_val[-4:] if len(env_val) > 10 else "***"
        print(f"  ✅  {key:<35} → stored  ({masked})")
    else:
        print(f"  ⚠️   {key:<35} → not found in .env, skipped")

print(f"\n{'='*55}")
print("  Done! Secrets are now in Windows Credential Manager.")
print("  You can view them in:")
print("  Control Panel → Credential Manager → Windows Credentials")
print(f"{'='*55}\n")
