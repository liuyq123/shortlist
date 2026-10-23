"""Settings from the environment. A local .env file is read for development; hosted visits get theirs from the platform."""

import os
import tempfile
from pathlib import Path


def _load_env_file(path: Path) -> None:
    """Set KEY=value lines from a .env file, without overriding the real environment."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


HOSTED = os.getenv("WORKSHOP_RUNTIME") == "hosted"
if not HOSTED:
    _load_env_file(Path.cwd() / ".env")

# The class gateway: a development token locally, the visit's run token when hosted
if HOSTED:
    GATEWAY_URL = os.getenv("WORKSHOP_BROKER_URL", "")
    GATEWAY_TOKEN = os.getenv("WORKSHOP_RUN_TOKEN", "")
else:
    GATEWAY_URL = os.getenv("WORKSHOP_DEVELOPMENT_URL", "")
    GATEWAY_TOKEN = os.getenv("WORKSHOP_DEVELOPMENT_TOKEN", "")
RUN_ID = os.getenv("WORKSHOP_RUN_ID", "")
READINESS = os.getenv("WORKSHOP_READINESS") == "1"

# Hosted visits may only write to the temporary directory
DATA_DIR = Path(
    os.getenv("SHORTLIST_DATA_DIR")
    or (Path(tempfile.gettempdir()) / "shortlist" if HOSTED else Path.cwd() / ".local")
)
DB_PATH = DATA_DIR / "shortlist.db"
PROFILE_PATH = DATA_DIR / "profile.json"

HOST = os.getenv("HOST") or ("0.0.0.0" if HOSTED else "127.0.0.1")
PORT = int(os.getenv("PORT", "8080"))

# How many days of new arXiv papers the feed covers
WINDOW_DAYS = int(os.getenv("WINDOW_DAYS", "3"))
