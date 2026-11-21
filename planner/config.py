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

# ---------- A different model for each stage ----------
# Picked from the "Free AI Model Reference Guide" (21 Nov 2026). Each stage tries its models in order and moves to
# the next when one hits its daily limit, doesn't exist for the key, or rejects the request. Models whose provider
# has no key are skipped. The OpenCode Free models in the guide only work inside OpenCode, so they aren't here.
# Free daily limits (approx.): Gemini Flash ~20/day EACH, Flash-Lite ~500/day, Gemma generous, Groq ~1,000/day,
# OpenRouter ":free" models ~50/day SHARED by all of them, so they are late backups in stages with few calls.
STAGES: dict[str, str] = {  # key -> what the stage does (shown in the app)
    "angles": "Stage 2 · choose the angles for this topic",
    "fanout": "Stage 2 · write the search queries",
    "tagging": "Stage 2 · tag intent and funnel stage",
    "filter": "Stage 3 · remove generic terms",
    "naming": "Stage 3 · name and check clusters",
    "pillars": "Stage 4 · pillars and gap check",
    "outline": "Stage 4 · H2 subtopics",
    "types": "Stage 5 · choose content types",
    "ideas": "Stage 5 · content ideas and titles",
}
STAGE_MODELS: dict[str, list[tuple[str, str]]] = {
    # First choices are the models that did best in a full test run (dental clinic, Kolkata, 22 Nov 2026: all
    # 6 stages in 5.6 min, 62 calls): GPT OSS on Groq (fast, ~30 calls/min) and Gemini Flash-Lite (~15/min).
    # Not first choices: Gemini Flash (~5 calls/min, ~20/day: 10+ min for Stage 2), Gemma 4 (~2 min per batch, its
    # per-minute token limit) and Qwen3.8 27B on Groq (free tier allows only 1,000 output tokens a MINUTE).
    "angles": [("Groq", "openai/gpt-oss-120b"), ("Gemini", "gemini-3.8-flash"), ("Gemini", "gemini-3.7-flash"),
               ("OpenRouter", "nvidia/nemotron-3-ultra-550b-a55b:free"), ("Gemini", "gemini-3.5-flash-lite")],
    "fanout": [("Groq", "openai/gpt-oss-120b"), ("Gemini", "gemini-3.5-flash-lite"), ("Gemini", "gemini-3.7-flash"),
               ("Gemini", "gemini-3.6-flash"), ("OpenRouter", "inclusionai/ling-3.2-flash:free"),
               ("Gemini", "gemini-3.1-flash-lite"), ("Groq", "qwen/qwen3.8-27b")],
    "tagging": [("Groq", "openai/gpt-oss-20b"), ("Gemini", "gemini-3.5-flash-lite"),
                ("Gemini", "gemini-3.1-flash-lite"), ("OpenRouter", "nvidia/nemotron-3.5-lightning:free")],
    "filter": [("Gemini", "gemini-3.1-flash-lite"), ("Groq", "openai/gpt-oss-20b"),
               ("Gemini", "gemini-3.5-flash-lite"), ("OpenRouter", "nvidia/nemotron-3.5-lightning:free")],
    "naming": [("Gemini", "gemini-3.1-flash-lite"), ("Groq", "openai/gpt-oss-20b"),
               ("Gemini", "gemini-3.5-flash-lite")],
    "pillars": [("Groq", "openai/gpt-oss-120b"), ("OpenRouter", "nvidia/nemotron-3-ultra-550b-a55b:free"),
                ("Gemini", "gemini-3.6-flash"), ("Gemini", "gemini-3.5-flash-lite"), ("Groq", "qwen/qwen3.8-27b")],
    "types": [("Groq", "openai/gpt-oss-120b"), ("OpenRouter", "nvidia/nemotron-3-ultra-550b-a55b:free"),
              ("Gemini", "gemini-3.5-flash-lite"), ("Groq", "qwen/qwen3.8-27b")],
    "outline": [("Gemini", "gemini-3.5-flash-lite"), ("Groq", "openai/gpt-oss-120b"), ("Gemini", "gemini-3.1-flash-lite"),
                ("Gemini", "gemma-4-31b-it"), ("OpenRouter", "google/gemma-4-31b-it:free"), ("Groq", "qwen/qwen3.8-27b")],
    "ideas": [("Gemini", "gemini-3.5-flash-lite"), ("Groq", "openai/gpt-oss-120b"), ("Gemini", "gemini-3.5-flash"),
              ("Gemini", "gemma-4-31b-it"), ("OpenRouter", "google/gemma-4-31b-it:free"), ("Groq", "qwen/qwen3.8-27b")],
}


def min_interval_for(provider: str, model: str) -> float:
    """Seconds between calls to stay under each model's free per-minute limit."""
    m = model.lower()
    if provider == "Gemini":
        if "gemma" in m:
            return 2.5    # ~30 a minute
        if "lite" in m:
            return 4.5    # ~15 a minute
        return 12.5       # Flash: ~5 a minute
    if provider == "Groq":
        return 2.1        # ~30 a minute
    if provider == "OpenRouter":
        return 3.2        # ~20 a minute
    return 0.0

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

# When an AI step fails or skips items, ask again (instead of falling back to rules)
AI_TRIES = 3            # tries for choosing angles and removing generic terms; extra rounds for intent tagging
AI_RETRY_PAUSE = 5.0    # seconds between tries (a busy model often answers a few seconds later)


def available_providers() -> list[str]:
    return [n for n, p in PROVIDERS.items() if not (ONLINE and p.local_only)]
