"""One AI client for every provider (standard library HTTP, no SDK).

Ollama, Gemini, Groq and OpenRouter all accept the OpenAI chat format, so switching provider only
changes the server address, key and model name. Every stage asks for JSON and gets a dict back.
"""
from __future__ import annotations

import json
import re
import time

from .config import PROVIDERS


MAX_ANSWER_TOKENS = 1800


class LLMError(Exception):
    """The AI call failed (bad key, server down, unreadable answer)."""


class RateLimited(LLMError):
    """The free tier said 'too many requests'."""


def parse_json(text: str) -> dict:
    """Read a JSON object from a model answer, even when it is wrapped in ```json fences or extra text."""
    if not text:
        raise LLMError("The AI returned an empty answer.")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("The AI answer was not JSON.") from None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise LLMError(f"The AI answer was not valid JSON ({e.msg}).") from None
    if isinstance(data, list):
        data = {"items": data}
    if not isinstance(data, dict):
        raise LLMError("The AI answer was not a JSON object.")
    return data


class HTTPError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def post_json(url: str, payload: dict, api_key: str | None, timeout: float) -> dict:
    """POST JSON with the standard library only (no compiled dependencies)."""
    import urllib.error
    import urllib.request

    headers = {"Content-Type": "application/json", "User-Agent": "content-planner"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            err = json.loads(body)
            err = err[0] if isinstance(err, list) and err else err
            msg = (err.get("error") or {}).get("message") if isinstance(err.get("error"), dict) else err.get("error")
        except Exception:
            msg = None
        raise HTTPError(e.code, str(msg or body or e.reason)) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise HTTPError(0, str(getattr(e, "reason", e))) from None


class LLM:
    def __init__(self, provider: str, model: str | None = None, api_key: str | None = None, timeout: float = 240):
        p = PROVIDERS[provider]
        self.provider = provider
        self.model = model or p.default_model
        if p.key_name and not api_key:
            raise LLMError(f"{provider} needs an API key. {p.key_help}")
        self._url = p.base_url.rstrip("/") + "/chat/completions"
        self._key = api_key
        self._timeout = timeout
        self._json_mode = True
        self.calls = 0

    @property
    def label(self) -> str:
        return f"{self.provider} · {self.model}"

    def json(self, system: str, user: str, temperature: float = 0.7) -> dict:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(3):
            # max_tokens stops a small model that gets stuck repeating itself (seen once: 12 minutes on one call).
            # Normal answers here are well under 1,200 tokens; a cut-off answer is unreadable JSON and is re-asked.
            payload = {"model": self.model, "messages": msgs, "temperature": temperature,
                       "max_tokens": MAX_ANSWER_TOKENS}
            if self._json_mode:
                payload["response_format"] = {"type": "json_object"}
            try:
                self.calls += 1
                r = post_json(self._url, payload, self._key, self._timeout)
                content = (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                return parse_json(content)
            except HTTPError as e:
                if e.status == 429:
                    if attempt == 0:
                        time.sleep(8)  # one short wait, then let the caller fall back
                        continue
                    raise RateLimited(f"{self.provider} free limit reached: {_short(e)}") from None
                if e.status == 400 and self._json_mode:  # some models reject JSON mode: ask in plain text
                    self._json_mode = False
                    continue
                if e.status in (401, 403):
                    raise LLMError(f"{self.provider} did not accept the API key.") from None
                if e.status == 404:
                    raise LLMError(f"Model '{self.model}' was not found on {self.provider}.") from None
                if e.status == 0:
                    if "timed out" in e.message and attempt < 2:
                        continue
                    where = "Is Ollama running?" if self.provider == "Local Ollama" else "Check the internet connection."
                    raise LLMError(f"Could not reach {self.provider}. {where}") from None
                if e.status >= 500 and attempt < 2:
                    time.sleep(3)
                    continue
                raise LLMError(f"{self.provider} error {e.status}: {_short(e)}") from None
            except LLMError:
                if attempt < 2:  # unreadable answer: ask again
                    continue
                raise
        raise LLMError(f"{self.provider} did not give a usable answer.")


class FallbackLLM:
    """Use the first AI; if it hits its free limit, switch to the next one for the rest of the run."""

    def __init__(self, chain: list, on_switch=None):
        if not chain:
            raise LLMError("No AI provider is set up.")
        self.chain = list(chain)
        self.on_switch = on_switch or (lambda msg: None)

    @property
    def label(self) -> str:
        return self.chain[0].label

    @property
    def calls(self) -> int:
        return sum(getattr(c, "calls", 0) for c in self.chain)

    def json(self, system: str, user: str, temperature: float = 0.7) -> dict:
        while True:
            try:
                return self.chain[0].json(system, user, temperature)
            except RateLimited as e:
                if len(self.chain) == 1:
                    raise
                old = self.chain.pop(0)
                self.on_switch(f"{old.provider} hit its free limit ({e}). Switched to {self.chain[0].label}.")


def ollama_running(base: str | None = None, timeout: float = 2) -> tuple[bool, list[str]]:
    """Return (is Ollama up, installed model names)."""
    import urllib.request

    base = base or PROVIDERS["Local Ollama"].base_url.removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout) as r:
            models = [m["name"] for m in json.load(r).get("models", [])]
        return True, models
    except Exception:
        return False, []


def _short(e: Exception, n: int = 160) -> str:
    msg = getattr(e, "message", None) or str(e)
    return msg if len(msg) <= n else msg[:n] + "…"
