"""One AI client for every provider (standard library HTTP, no SDK).

Ollama, Gemini, Groq and OpenRouter all accept the OpenAI chat format, so switching provider only
changes the server address, key and model name. Every stage asks for JSON and gets a dict back.
"""
from __future__ import annotations

import json
import re
import time

from .config import PROVIDERS


# Enough for the longest answers (6 clusters x 4 titles in Stage 5); at 1,800 those got cut off and a whole batch
# was lost. A model whose free tier allows less per minute gets a smaller limit automatically (see "too large").
MAX_ANSWER_TOKENS = 3000


class LLMError(Exception):
    """The AI call failed (bad key, server down, unreadable answer)."""


class RateLimited(LLMError):
    """The free tier said 'too many requests'."""


class ModelUnavailable(LLMError):
    """This model can't be used (doesn't exist for this key, or rejects the request format): try another model."""


# Models that think before answering. Their thinking counts toward the answer's token limit, so they get a larger
# limit and are asked to think briefly (JSON answers here don't need long reasoning).
THINKING = re.compile(r"gemini-[\d.]+-flash(?!-lite)|gemini-flash-latest|gpt-oss|qwen3|deepseek-r|nemotron", re.I)


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
    def __init__(self, provider: str, model: str | None = None, api_key: str | None = None, timeout: float = 240,
                 backups: bool = True, min_interval: float | None = None):
        """backups=False: on 'model not found' raise ModelUnavailable instead of switching to another model of the
        same provider (used when each stage has its own list of models, so the next model in that list is used)."""
        p = PROVIDERS[provider]
        self.provider = provider
        self.model = model or p.default_model
        if p.key_name and not api_key:
            raise LLMError(f"{provider} needs an API key. {p.key_help}")
        self._url = p.base_url.rstrip("/") + "/chat/completions"
        self._key = api_key
        self._timeout = timeout
        self._json_mode = True
        self._system_ok = True
        self._thinking = bool(THINKING.search(self.model))
        self._max_tokens = MAX_ANSWER_TOKENS * (3 if self._thinking else 1)
        self._backups = [m for m in p.backup_models if m != self.model] if backups else []
        self._list_tried = not backups
        self._min_interval = p.min_interval if min_interval is None else min_interval
        self.exhausted = False  # daily limit reached: skip this model for the rest of the run
        self._max_param = "max_tokens"   # becomes max_completion_tokens if the model asks for it (OpenAI)
        self._send_temperature = True    # dropped if the model takes only its default temperature
        self._last_call = 0.0
        self.calls = 0
        self.ok = 0          # calls that returned usable JSON
        self.seconds = 0.0   # time spent waiting for answers (not counting the spacing between calls)
        self.notes: list[str] = []  # e.g. "model X not found, switched to Y"

    def _next_model(self) -> bool:
        """After a 'model not found': switch to the next backup, else to a model the provider lists. False if none."""
        if not self._backups and not self._list_tried:
            self._list_tried = True
            self._backups = self._listed_models()
        if not self._backups:
            return False
        old, self.model = self.model, self._backups.pop(0)
        self.notes.append(f"Model '{old}' isn't available on {self.provider}; using '{self.model}' instead.")
        return True

    def _listed_models(self) -> list[str]:
        """Text models this key can use, best guess first (from the provider's /models list)."""
        import urllib.request

        if not self._key:  # Ollama: keep the clear "ollama pull <model>" advice instead of guessing
            return []
        req = urllib.request.Request(self._url.removesuffix("/chat/completions") + "/models",
                                     headers={"Authorization": f"Bearer {self._key}", "User-Agent": "content-planner"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                ids = [str(m.get("id", "")).removeprefix("models/") for m in json.loads(r.read()).get("data", [])]
        except Exception:
            return []
        return pick_text_models(self.provider, ids, exclude={self.model})

    @property
    def label(self) -> str:
        return f"{self.provider} · {self.model}"

    # (provider, key) pairs the provider refused in this process: every model on that key is skipped at once,
    # instead of each stage trying it again (seen online: a wrong Groq key was tried by every stage).
    _rejected_keys: set[tuple[str, str]] = set()

    def json(self, system: str, user: str, temperature: float = 0.7) -> dict:
        if self._key and (self.provider, self._key) in LLM._rejected_keys:
            self.exhausted = True
            raise ModelUnavailable(f"{self.provider} did not accept the API key.")
        for attempt in range(6):  # room to drop a few unsupported options one at a time
            # Some models (Gemma) take no system message: then it goes at the top of the user message.
            if self._system_ok:
                msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            else:
                msgs = [{"role": "user", "content": f"{system}\n\n{user}"}]
            # max_tokens stops a small model that gets stuck repeating itself (seen once: 12 minutes on one call).
            # Normal answers here are well under 1,200 tokens; a cut-off answer is unreadable JSON and is re-asked.
            payload = {"model": self.model, "messages": msgs, self._max_param: self._max_tokens}
            if self._send_temperature:
                payload["temperature"] = temperature
            if self._thinking:
                payload["reasoning_effort"] = "low"
            if self._json_mode:
                payload["response_format"] = {"type": "json_object"}
            try:
                wait = self._last_call + self._min_interval - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last_call = time.monotonic()
                self.calls += 1
                started = time.monotonic()
                try:
                    r = post_json(self._url, payload, self._key, self._timeout)
                finally:
                    self.seconds += time.monotonic() - started
                content = (r.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                data = parse_json(content)
                self.ok += 1
                return data
            except HTTPError as e:
                if e.status == 429:
                    # A per-minute limit clears in seconds: wait (as long as the provider asks, if it says) and
                    # try again, instead of letting a whole batch of pages fall back to empty answers.
                    # A daily limit won't clear today: hand over to the next provider straight away.
                    if "too large" in e.message.lower():
                        # the answer limit we asked for is above this model's free per-minute token allowance
                        # (Groq: "... output tokens per minute (OTPM): Limit 1000, Requested 1987"): ask for less
                        m = re.search(r"Limit (\d+)", e.message)
                        cap = int(int(m.group(1)) * 0.9) if m else self._max_tokens // 2
                        if 300 <= cap < self._max_tokens:
                            self._max_tokens = cap
                            continue
                    daily = bool(re.search(r"per ?day|daily", e.message, re.I))
                    if attempt < 2 and not daily:
                        m = re.search(r"retry in ([\d.]+)\s*s", e.message, re.I)
                        time.sleep(min(45.0, float(m.group(1)) + 1) if m else (10 if attempt == 0 else 25))
                        continue
                    self.exhausted = daily
                    raise RateLimited(f"{self.label} free limit reached: {_short(e)}") from None
                if e.status == 400 and re.search(r"api key|api_key", e.message, re.I):
                    # Gemini answers a bad key with 400 ("Please pass a valid API key"), not 401
                    if self._key:
                        LLM._rejected_keys.add((self.provider, self._key))
                    self.exhausted = True
                    raise ModelUnavailable(f"{self.provider} did not accept the API key.") from None
                if e.status == 400:  # an option this model doesn't take: drop options one by one
                    # OpenAI's newer models want max_completion_tokens and only their default temperature
                    if "max_completion_tokens" in e.message and self._max_param == "max_tokens":
                        self._max_param = "max_completion_tokens"
                        continue
                    if "temperature" in e.message.lower() and self._send_temperature:
                        self._send_temperature = False
                        continue
                    if self._thinking and "reason" in e.message.lower():
                        self._thinking = False
                        continue
                    if self._system_ok and re.search(r"system|developer instruction", e.message, re.I):
                        self._system_ok = False
                        continue
                    if self._json_mode:  # some models reject JSON mode: ask in plain text
                        self._json_mode = False
                        continue
                    if self._thinking:
                        self._thinking = False
                        continue
                    raise ModelUnavailable(f"{self.label} rejected the request: {_short(e)}") from None
                if e.status == 401 or (e.status == 403 and re.search(r"key|auth|credential", e.message, re.I)):
                    if self._key:
                        LLM._rejected_keys.add((self.provider, self._key))
                    self.exhausted = True
                    raise ModelUnavailable(f"{self.provider} did not accept the API key.") from None
                if e.status == 403:  # this key may not use this model (e.g. a gated model): try the next one
                    raise ModelUnavailable(f"{self.label} isn't allowed for this key: {_short(e)}") from None
                if e.status == 404:
                    if self._next_model():
                        return self.json(system, user, temperature)
                    raise ModelUnavailable(f"Model '{self.model}' was not found on {self.provider}. Type a current "
                                           f"model name in the sidebar's Model box.") from None
                if e.status == 0:
                    if "timed out" in e.message and attempt < 3:
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


_SKIP = ("tts", "audio", "image", "live", "embed", "vision", "guard", "whisper", "veo", "imagen", "robotics",
         "computer-use", "aqa")


def pick_text_models(provider: str, ids: list[str], exclude: set[str] = frozenset()) -> list[str]:
    """Chat models from a provider's model list, the most suitable first."""
    ids = [i for i in dict.fromkeys(ids) if i and i not in exclude and not any(s in i.lower() for s in _SKIP)]
    if provider == "Gemini":
        ids = [i for i in ids if i.startswith("gemini") and "flash" in i]
        # stable before preview, full Flash before Flash-Lite, newest version first
        def ver(i):
            m = re.search(r"gemini-(\d+(?:\.\d+)?)", i)
            return float(m.group(1)) if m else 0.0
        return sorted(ids, key=lambda i: ("preview" in i or "exp" in i, "lite" in i, -ver(i)))
    if provider == "OpenRouter":
        return [i for i in ids if i.endswith(":free")]
    return ids


class FallbackLLM:
    """Use the first AI; if it hits its free limit or can't be used, switch to the next one for the rest of the run."""

    def __init__(self, chain: list, on_switch=None, name: str = ""):
        if not chain:
            raise LLMError("No AI provider is set up.")
        self.chain = list(chain)
        self.all = list(chain)  # including models dropped after a limit, so their calls still count
        self.on_switch = on_switch or (lambda msg: None)
        self.name = name  # the stage this chain serves, for messages

    @property
    def label(self) -> str:
        return self.chain[0].label

    @property
    def calls(self) -> int:
        return sum(getattr(c, "calls", 0) for c in self.all)

    @property
    def notes(self) -> list[str]:
        return [n for c in self.all for n in getattr(c, "notes", [])]

    def json(self, system: str, user: str, temperature: float = 0.7) -> dict:
        while len(self.chain) > 1 and getattr(self.chain[0], "exhausted", False):
            self.chain.pop(0)  # another stage already used up this model's daily limit
        while True:
            try:
                return self.chain[0].json(system, user, temperature)
            except (RateLimited, ModelUnavailable) as e:
                if len(self.chain) == 1:
                    raise
                old = self.chain.pop(0)
                why = "hit its free limit" if isinstance(e, RateLimited) else "isn't available"
                where = f" for {self.name}" if self.name else ""
                self.on_switch(f"{old.label} {why}{where} ({e}). Switched to {self.chain[0].label}.")


class StageModels:
    """A different AI model chain for each stage (see config.STAGE_MODELS), e.g. a strong model to choose the
    angles and a fast high-quota model for bulk tagging. Stage code asks `for_stage(name)`; anything that isn't a
    StageModels (a single LLM, a test fake) is used for every stage."""

    def __init__(self, chains: dict[str, FallbackLLM], default: FallbackLLM):
        self.chains, self.default = chains, default

    def for_stage(self, stage: str):
        return self.chains.get(stage) or self.default

    def _models(self) -> list:
        seen: dict[int, object] = {}
        for ch in [*self.chains.values(), self.default]:
            for m in ch.all:
                seen[id(m)] = m
        return list(seen.values())

    @property
    def calls(self) -> int:
        return sum(getattr(m, "calls", 0) for m in self._models())

    @property
    def notes(self) -> list[str]:
        return list(dict.fromkeys(n for m in self._models() for n in getattr(m, "notes", [])))

    @property
    def label(self) -> str:
        return "a different model for each stage"

    def stats(self) -> list[dict]:
        """Per model: calls, usable answers, failures and average seconds per call (to pick the best primaries)."""
        out = []
        for m in self._models():
            calls = getattr(m, "calls", 0)
            if calls:
                ok = getattr(m, "ok", 0)
                out.append({"model": m.label, "calls": calls, "ok": ok, "failed": calls - ok,
                            "avg_seconds": round(getattr(m, "seconds", 0.0) / calls, 1),
                            "used_up": getattr(m, "exhausted", False)})
        return sorted(out, key=lambda x: -x["calls"])

    def used(self) -> dict[str, str]:
        """Stage -> the model it is using now (after any switches)."""
        return {s: ch.label for s, ch in self.chains.items()}

    def json(self, system: str, user: str, temperature: float = 0.7) -> dict:
        return self.default.json(system, user, temperature)


def build_stage_models(keys: dict[str, str], default: FallbackLLM, on_switch=None,
                       stage_models: dict[str, list[tuple[str, str]]] | None = None) -> StageModels:
    """One chain per stage from config.STAGE_MODELS, using only providers we have a key for. The models of
    `default` (the model chosen in the sidebar, plus its backups) are added at the end of every chain, so a stage
    still runs when all its own models are used up. A model shared by several stages is one LLM object, so its
    spacing between calls and its 'used up today' flag are shared too."""
    from .config import STAGE_MODELS, STAGES, min_interval_for

    cache: dict[tuple[str, str], LLM] = {}
    chains: dict[str, FallbackLLM] = {}
    for stage, entries in (stage_models or STAGE_MODELS).items():
        chain = []
        for provider, model in entries:
            if PROVIDERS[provider].key_name and not keys.get(provider):
                continue
            if (provider, model) not in cache:
                cache[(provider, model)] = LLM(provider, model, keys.get(provider), backups=False,
                                               min_interval=min_interval_for(provider, model))
            chain.append(cache[(provider, model)])
        chain += [m for m in default.chain if m not in chain]
        chains[stage] = FallbackLLM(chain, on_switch, name=STAGES.get(stage, stage))
    return StageModels(chains, default)


class WithBrief:
    """Adds the business brief (goal, offerings, customers, who the plan is for) to the top of every AI request,
    in every stage. Everything else (calls, notes, used(), stats()) comes from the wrapped model(s)."""

    def __init__(self, inner, brief: str):
        self.inner, self.brief = inner, brief

    def json(self, system: str, user: str, temperature: float = 0.7) -> dict:
        return self.inner.json(system, f"{self.brief}\n\n{user}" if self.brief else user, temperature)

    def for_stage(self, stage: str):
        return WithBrief(for_stage(self.inner, stage), self.brief)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def with_brief(llm, brief: str):
    return WithBrief(llm, brief) if brief else llm


def get_list(data: dict, key: str) -> list:
    """The list an answer gives under `key`. Models sometimes return the bare list instead of {"key": [...]}
    (parse_json wraps that as {"items": [...]}), or use another name for the only list; both are accepted, so a
    whole batch isn't lost over the answer's shape (seen in testing: 1 batch in 8 of Stage 5 ideas)."""
    v = data.get(key) if isinstance(data, dict) else None
    if isinstance(v, list):
        return v
    lists = [x for x in data.values() if isinstance(x, list)] if isinstance(data, dict) else []
    return lists[0] if len(lists) == 1 else []


def is_fatal(e: Exception) -> bool:
    """Errors that asking again can't fix (bad key, no connection): stop the run instead of retrying."""
    msg = str(e)
    return any(s in msg for s in ("API key", "did not accept", "Could not reach"))


def for_stage(llm, stage: str):
    """The model for one stage: from a StageModels, or the same single model everywhere."""
    return llm.for_stage(stage) if hasattr(llm, "for_stage") else llm


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
