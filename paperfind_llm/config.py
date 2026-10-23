"""Settings from the environment, with .env files loaded from the working directory and ~/.paperfind-llm."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")
load_dotenv(Path.home() / ".paperfind-llm" / ".env")

DATA_DIR = Path(os.getenv("PAPERFIND_LLM_DATA_DIR") or Path.home() / ".paperfind-llm")
DB_PATH = DATA_DIR / "paperfind.db"
PROFILE_PATH = DATA_DIR / "profile.json"

# Small, fast model for screening hundreds of papers; stronger model for the
# profile builder and the assistant.
SCREEN_MODEL = os.getenv("SCREEN_MODEL", "claude-haiku-5-5")
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-opus-5-5")

# How many days of new arXiv papers the feed covers
WINDOW_DAYS = int(os.getenv("WINDOW_DAYS", "3"))
