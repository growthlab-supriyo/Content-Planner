"""Retired model names: the client switches to a working model instead of failing the run."""
import planner.llm as L
from planner.llm import LLM, pick_text_models
from planner.quota import Quota


def test_not_found_model_switches_to_backup(monkeypatch):
    seen = []

    def fake_post(url, payload, key, timeout):
        seen.append(payload["model"])
        if payload["model"] in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"):
            raise L.HTTPError(404, "models/x is not found")
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    monkeypatch.setattr(L, "post_json", fake_post)
    llm = LLM("Gemini", api_key="k")
    llm._min_interval = 0
    assert llm.json("s", "u") == {"ok": True}
    assert seen == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest"]
    assert llm.model == "gemini-flash-lite-latest" and len(llm.notes) == 2


def test_falls_back_to_listed_models_then_gives_clear_error(monkeypatch):
    monkeypatch.setattr(L, "post_json", lambda *a: (_ for _ in ()).throw(L.HTTPError(404, "not found")))
    llm = LLM("Gemini", "my-old-model", api_key="k")
    llm._backups, llm._min_interval = [], 0
    monkeypatch.setattr(LLM, "_listed_models", lambda self: [])
    try:
        llm.json("s", "u")
        assert False
    except L.LLMError as e:
        assert "Model box" in str(e)


def test_pick_text_models_orders_gemini_sensibly():
    ids = ["gemini-2.0-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3-flash-preview",
           "gemini-3.8-flash-tts", "text-embedding-004", "gemini-4-argon", "gemini-3.6-flash"]
    assert pick_text_models("Gemini", ids) == ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.0-flash",
                                               "gemini-3.5-flash-lite", "gemini-3-flash-preview"]
    assert pick_text_models("OpenRouter", ["a/b:free", "a/c", "x/whisper:free"]) == ["a/b:free"]


def test_quota_refund():
    q = Quota(per_visitor=3, daily_cap=10)
    assert q.use("v") and q.remaining("v") == 2
    q.refund("v")
    assert q.remaining("v") == 3
    q.refund("v")  # never below zero
    assert q.remaining("v") == 3


def test_min_interval_spaces_calls(monkeypatch):
    import time
    monkeypatch.setattr(L, "post_json", lambda *a: {"choices": [{"message": {"content": "{}"}}]})
    llm = LLM("Gemini", api_key="k")
    llm._min_interval = 0.3
    t = time.monotonic()
    for _ in range(3):
        llm.json("s", "u")
    assert time.monotonic() - t >= 0.5  # 2 waits of 0.3 s (Windows timers are coarse)
