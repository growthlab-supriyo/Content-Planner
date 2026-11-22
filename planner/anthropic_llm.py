"""Claude (Anthropic) for a visitor's paid key, through the official Anthropic SDK.

Same interface as llm.LLM (json(), calls, ok, seconds, notes, exhausted, label), so a paid Claude key runs every
stage. Claude Opus 5.5 always thinks; its effort defaults to medium, so it is set explicitly. On a policy decline
the API re-runs the request on a fallback model by itself (server-side fallbacks, beta), so a run isn't cut short.
"""
from __future__ import annotations

import time

import anthropic

from .llm import LLMError, ModelUnavailable, RateLimited, _short, parse_json

# Models that take server-side fallbacks ("default" routes by refusal category) and the effort setting
FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16000  # non-streaming requests: enough for the longest answers here, well under SDK time limits


class AnthropicLLM:
    provider = "Anthropic"

    def __init__(self, model: str, api_key: str, timeout: float = 240, effort: str = "medium"):
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)
        self._effort = effort
        self.calls = 0
        self.ok = 0
        self.seconds = 0.0
        self.notes: list[str] = []
        self.exhausted = False

    @property
    def label(self) -> str:
        return f"{self.provider} · {self.model}"

    def _create(self, system: str, user: str):
        params = dict(model=self.model, max_tokens=MAX_TOKENS, system=system,
                      messages=[{"role": "user", "content": user}])
        parts = self.model.split("-")  # e.g. claude-opus-5-5 -> ["claude", "opus", "5", "5"]
        if len(parts) > 2 and parts[2].isdigit() and int(parts[2]) >= 5:
            params["output_config"] = {"effort": self._effort}  # Claude 5.x: set effort explicitly
        if self.model in FALLBACK_MODELS:
            return self._client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **params)
        return self._client.messages.create(**params)

    def json(self, system: str, user: str, temperature: float = 0.7) -> dict:
        # temperature is not sent: current Claude models reject sampling parameters
        if self.exhausted:
            raise ModelUnavailable(f"{self.provider} did not accept the API key.")
        for attempt in range(3):
            self.calls += 1
            started = time.monotonic()
            try:
                resp = self._create(system, user)
            except anthropic.AuthenticationError:
                self.exhausted = True
                raise ModelUnavailable(f"{self.provider} did not accept the API key.") from None
            except anthropic.PermissionDeniedError as e:
                raise ModelUnavailable(f"{self.label} isn't allowed for this key: {_short(e)}") from None
            except anthropic.NotFoundError:
                raise ModelUnavailable(f"Model '{self.model}' was not found on {self.provider}.") from None
            except anthropic.RateLimitError as e:
                raise RateLimited(f"{self.label} limit reached: {_short(e)}") from None
            except anthropic.BadRequestError as e:
                raise ModelUnavailable(f"{self.label} rejected the request: {_short(e)}") from None
            except anthropic.APIConnectionError:
                raise LLMError(f"Could not reach {self.provider}. Check the internet connection.") from None
            except anthropic.APIStatusError as e:  # 5xx after the SDK's own retries
                raise LLMError(f"{self.provider} error {e.status_code}: {_short(e)}") from None
            finally:
                self.seconds += time.monotonic() - started
            if resp.stop_reason == "refusal":
                raise LLMError(f"{self.label} declined this request.")
            text = "".join(b.text for b in resp.content if b.type == "text")
            try:
                data = parse_json(text)
            except LLMError:
                if attempt < 2:  # unreadable answer: ask again
                    continue
                raise
            self.ok += 1
            return data
        raise LLMError(f"{self.provider} did not give a usable answer.")
