"""Settings shared by the local and online versions.

The online version (Streamlit Community Cloud) starts from streamlit_app.py, which sets CP_ONLINE=1.
Online there is no Ollama, so only the free cloud providers are offered and fastembed does the embeddings.
"""
import os
from dataclasses import dataclass

ONLINE = os.environ.get("CP_ONLINE") == "1"

APP_NAME = "Content Planner"
VERSION = "0.1.0"


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    default_model: str
    key_name: str | None  # name in .streamlit/secrets.toml / env; None = no key needed
    key_help: str
    local_only: bool = False
    # Tried in order when the chosen model answers "not found" (free model names get retired now and then).
    backup_models: tuple[str, ...] = ()
    min_interval: float = 0.0  # seconds between calls, to stay under a free per-minute limit


PROVIDERS: dict[str, Provider] = {
    "Local Ollama": Provider(
        "Local Ollama", os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/v1", "qwen2.5:7b",
        None, "Runs on this computer. Start Ollama first.", local_only=True),
    "Gemini": Provider(
        # Flash-Lite: the free tier allows about 15 requests/minute and 500/day, enough for ~12 full plans a day.
        # Full Flash allows only ~20/day (a full plan needs ~35 calls), so it's a backup only.
        # (2.5 Flash was retired for new keys in 2026; free limits change, see aistudio.google.com/rate-limit.)
        "Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.5-flash-lite",
        "GEMINI_API_KEY", "Free key: aistudio.google.com/apikey",
        backup_models=("gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"),
        min_interval=4.5),  # stay under 15 requests a minute
    "Groq": Provider(
        "Groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile",
        "GROQ_API_KEY", "Free key: console.groq.com/keys"),
    "OpenRouter": Provider(
        "OpenRouter", "https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free",
        "OPENROUTER_API_KEY", "Free key: openrouter.ai/keys (use models ending in :free)"),
}

DEFAULT_PROVIDER = "Gemini" if ONLINE else "Local Ollama"

# Embeddings: local = nomic-embed-text through Ollama, online = fastembed inside the app.
OLLAMA_EMBED_MODEL = "nomic-embed-text"
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Online free-run limit on the owner's key (option C).
FREE_RUNS_PER_DAY = 3
OWNER_KEY_DAILY_CAP = int(os.environ.get("CP_DAILY_CAP", "150"))  # all visitors together

# Fan-out size
DEFAULT_QUERIES_PER_ANGLE = 30
MAX_QUERIES_PER_ANGLE = 60
MAX_RETRIES_PER_ANGLE = 2


def available_providers() -> list[str]:
    return [n for n, p in PROVIDERS.items() if not (ONLINE and p.local_only)]
