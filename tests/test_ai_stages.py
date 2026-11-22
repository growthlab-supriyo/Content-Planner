"""AI-planned angles, AI intent/funnel tagging, AI generic-term filter and a different model per stage."""
import pytest

from planner import llm as L
from planner.cluster import filter_keywords
from planner.fanout import plan_angles, run_fanout, tag_intents
from planner.llm import FallbackLLM, LLMError, ModelUnavailable, RateLimited, StageModels, build_stage_models

from conftest import FakeLLM


JOURNEY = [{"label": l, "kind": k, "instruction": f"Write {l} searches for this topic."} for l, k in (
    ("Setup Help", "questions"), ("Common Problems", "problems"), ("Plans", "costs"), ("Alternatives", "comparisons"),
    ("Worries", "fears"))]


def test_ai_plans_the_angles_and_bad_entries_are_dropped():
    answer = {"angles": JOURNEY + [
        {"label": "Integrations", "kind": "other", "instruction": "Searches about connecting the tool to others.",
         "examples": ["connect crm to gmail"]},
        {"label": "x", "kind": "other", "instruction": "too short a label"},             # dropped
        {"label": "Plans", "kind": "costs", "instruction": "A duplicate label again."},    # dropped
        {"label": "Odd Kind", "kind": "nonsense", "instruction": "Unknown kinds become other."}]}
    angles = plan_angles(FakeLLM([answer]), "CRM software")
    assert [a["label"] for a in angles] == [a["label"] for a in JOURNEY] + ["Integrations", "Odd Kind"]
    assert [a["check"] for a in angles][:5] == ["questions", None, "costs", "comparisons", None]
    assert angles[-1]["kind"] == "other" and all(a["planned"] for a in angles)
    # a failed try is asked again: here the second try works
    llm = FakeLLM([LLMError("busy"), answer])
    assert len(plan_angles(llm, "CRM software")) == 7 and llm.calls == 2
    llm = FakeLLM([LLMError("down")] * 3)
    assert plan_angles(llm, "CRM software") == [] and llm.calls == 3      # 3 tries, then the caller decides


def test_angles_must_cover_the_whole_journey():
    """A plan with two cost angles and no comparisons/fears angle is sent back with feedback."""
    partial = {"angles": [{"label": l, "kind": k, "instruction": f"Write {l} searches for this topic."} for l, k in (
        ("Questions", "questions"), ("Symptoms", "problems"), ("Prices", "costs"), ("Insurance", "costs"))]}
    llm = FakeLLM([partial, {"angles": JOURNEY}])
    angles = plan_angles(llm, "Dental Clinic")
    assert [a["label"] for a in angles] == [a["label"] for a in JOURNEY] and llm.calls == 2
    assert "no angle of kind: comparisons, fears" in llm.prompts[1]
    assert "more than one angle of kind: costs" in llm.prompts[1]
    # still incomplete after every try: one angle per single-use kind, plus the standard angles it left out
    angles = plan_angles(FakeLLM([partial] * 3), "Dental Clinic", "Kolkata")
    assert [a["label"] for a in angles] == ["Questions", "Symptoms", "Prices", "Comparisons", "Fears & Myths", "Local"]


def test_fanout_uses_planned_angles_then_ai_tags(embedder):
    def answer(user):
        if "search ANGLES" in user:
            return {"angles": JOURNEY}
        if "Angle: Common Problems" in user or "Angle: Worries" in user:
            return {"queries": []}
        if "Angle: Setup Help" in user:
            return {"queries": ["how to set up crm", "how to import contacts"]}
        if "Angle: Plans" in user:
            return {"queries": ["crm price per user", "cheapest crm plan"]}
        if "Angle: Alternatives" in user:
            return {"queries": ["hubspot vs zoho crm"]}
        if "Tag each search query" in user:
            return {"tags": [{"q": "cheapest crm plan", "intent": "Commercial", "funnel": "MOFU"},
                             {"q": "how to set up crm", "intent": "informational", "funnel": "tofu"}]}
        raise AssertionError(user[:60])

    rows, report = run_fanout(FakeLLM([answer] * 30), embedder, "CRM software", n=2)
    assert report["angles_planned_by_ai"] and [a["label"] for a in report["angle_plan"]] == \
        [a["label"] for a in JOURNEY]
    by = {r["keyword"]: r for r in rows}
    assert by["cheapest crm plan"]["tagged_by"] == "AI" and by["cheapest crm plan"]["funnel"] == "MOFU"
    assert by["how to set up crm"]["intent"] == "Informational" and by["how to set up crm"]["funnel"] == "TOFU"
    assert by["hubspot vs zoho crm"]["tagged_by"] == "Rules" and by["hubspot vs zoho crm"]["intent"] == "Commercial"
    assert report["tagged_by_ai"] == 2


def test_tagging_ignores_queries_it_was_not_asked_about():
    rows = [{"keyword": "root canal cost", "intent": "Commercial", "funnel": "MOFU"}]
    bad = {"tags": [{"q": "something else", "intent": "Local", "funnel": "BOFU"},
                    {"q": "root canal cost", "intent": "Nonsense", "funnel": "BOFU"}]}
    llm = FakeLLM([bad] * 4)
    assert tag_intents(llm, rows, "Dental Clinic") == 0 and llm.calls == 4   # 1 + 3 more tries
    assert rows[0]["intent"] == "Commercial" and rows[0]["tagged_by"] == "Rules"


def test_skipped_keywords_are_asked_again():
    rows = [{"keyword": k, "intent": "Informational", "funnel": "TOFU"} for k in ("kids dentist near me", "braces cost")]
    first = {"tags": [{"q": "braces cost", "intent": "Commercial", "funnel": "MOFU"}]}          # skipped one
    second = {"tags": [{"q": "kids dentist near me", "intent": "Local", "funnel": "BOFU"}]}
    llm = FakeLLM([LLMError("busy"), first, second])  # the first batch fails outright, then two rounds
    assert tag_intents(llm, rows, "Dental Clinic") == 2
    assert [(r["intent"], r["tagged_by"]) for r in rows] == [("Local", "AI"), ("Commercial", "AI")]


def test_ai_decides_which_two_word_terms_are_generic():
    rows = [{"keyword": k} for k in ("dental clinic", "kids dentist", "best clinic", "dentist", "root canal cost")]
    llm = FakeLLM([{"generic": [{"term": "dental clinic", "reason": "just the business name"},
                                {"term": "best clinic", "reason": "filler"}, {"term": "not asked", "reason": "x"}]}])
    kept, removed = filter_keywords(rows, "Dental Clinic", llm)
    assert [r["keyword"] for r in kept] == ["kids dentist", "root canal cost"]
    assert {r["keyword"]: r["reason"].split(":")[0] for r in removed} == {
        "dental clinic": "Generic two-word term (AI)", "best clinic": "Generic two-word term (AI)",
        "dentist": "Single word"}
    # first try fails, second works
    llm = FakeLLM([LLMError("busy"), {"generic": [{"term": "best clinic", "reason": "filler"}]}])
    kept, removed = filter_keywords(rows, "Dental Clinic", llm)
    assert {r["keyword"] for r in removed} == {"best clinic", "dentist"} and llm.calls == 2
    # every try fails: no two-word term is removed (no word-list guessing), only single words
    llm = FakeLLM([LLMError("down")] * 3)
    kept, removed = filter_keywords(rows, "Dental Clinic", llm)
    assert {r["keyword"] for r in removed} == {"dentist"} and llm.calls == 3


class Named(FakeLLM):
    def __init__(self, label, answers=None, error=None):
        super().__init__(answers or [{"ok": label}] * 10)
        self.label, self.provider, self.error, self.exhausted = label, label, error, False

    def json(self, *a, **k):
        if self.error:
            self.calls += 1
            raise self.error
        return super().json(*a, **k)


def test_each_stage_uses_its_own_chain_and_moves_on_when_a_model_is_unavailable():
    a, b = Named("A"), Named("B", error=ModelUnavailable("404"))
    fallback = Named("C")
    switches = []
    models = StageModels({"tagging": FallbackLLM([b, fallback], switches.append, name="tagging"),
                          "angles": FallbackLLM([a], name="angles")}, FallbackLLM([fallback]))
    assert models.for_stage("angles").json("s", "u") == {"ok": "A"}
    assert models.for_stage("tagging").json("s", "u") == {"ok": "C"}    # B doesn't exist: next model
    assert "isn't available for tagging" in switches[0]
    assert models.for_stage("ideas").json("s", "u") == {"ok": "C"}      # stage without a chain: default
    assert models.calls == 4


def test_a_model_used_up_by_one_stage_is_skipped_by_the_next():
    shared, other = Named("Flash"), Named("Lite")
    shared.exhausted = True
    ch = FallbackLLM([shared, other])
    assert ch.json("s", "u") == {"ok": "Lite"} and shared.calls == 0


def test_build_stage_models_skips_providers_without_a_key():
    default = FallbackLLM([Named("Chosen")])
    sm = build_stage_models({"Groq": "gsk_test"}, default,
                            stage_models={"pillars": [("Gemini", "gemini-x"), ("Groq", "openai/gpt-oss-120b")],
                                          "naming": [("Gemini", "gemini-y")]})
    assert [m.label for m in sm.for_stage("pillars").chain] == ["Groq · openai/gpt-oss-120b", "Chosen"]
    assert [m.label for m in sm.for_stage("naming").chain] == ["Chosen"]   # only the sidebar model is left
    one = build_stage_models({"Groq": "k"}, default, stage_models={"a": [("Groq", "m")], "b": [("Groq", "m")]})
    assert one.for_stage("a").chain[0] is one.for_stage("b").chain[0]    # one object: shared spacing and limits


def test_llm_drops_options_a_model_rejects(monkeypatch):
    """Gemma takes no system message; a thinking model may reject reasoning_effort; 404 -> ModelUnavailable."""
    sent = []

    def fake_post(url, payload, key, timeout):
        sent.append(payload)
        if payload["messages"][0]["role"] == "system":
            raise L.HTTPError(400, "Developer instruction is not enabled for models/gemma-4-31b-it")
        return {"choices": [{"message": {"content": '{"ok": 1}'}}]}

    monkeypatch.setattr(L, "post_json", fake_post)
    m = L.LLM("Gemini", "gemma-4-31b-it", "key", backups=False, min_interval=0)
    assert m.json("sys", "user") == {"ok": 1}
    assert sent[-1]["messages"] == [{"role": "user", "content": "sys\n\nuser"}]

    monkeypatch.setattr(L, "post_json", lambda *a: (_ for _ in ()).throw(L.HTTPError(404, "model not found")))
    with pytest.raises(ModelUnavailable):
        L.LLM("Groq", "qwen/qwen3.8-27b", "key", backups=False, min_interval=0).json("s", "u")

    daily = L.LLM("Gemini", "gemini-3.8-flash", "key", backups=False, min_interval=0)
    assert daily._thinking  # Flash thinks: bigger token limit, brief reasoning asked
    monkeypatch.setattr(L, "post_json", lambda *a: (_ for _ in ()).throw(
        L.HTTPError(429, "Quota exceeded for metric generate_content_free_tier_requests PerDay")))
    with pytest.raises(RateLimited):
        daily.json("s", "u")
    assert daily.exhausted


def test_answer_limit_shrinks_to_the_models_per_minute_cap(monkeypatch):
    """Groq's free Qwen allows 1,000 output tokens a minute; asking for more must shrink the limit, not switch."""
    sent = []

    def fake_post(url, payload, key, timeout):
        sent.append(payload["max_tokens"])
        if payload["max_tokens"] > 1000:
            raise L.HTTPError(429, "Request too large for model `qwen/qwen3.8-27b` on output tokens per minute "
                                   "(OTPM): Limit 1000, Requested 5400.")
        return {"choices": [{"message": {"content": '{"ok": 1}'}}]}

    monkeypatch.setattr(L, "post_json", fake_post)
    m = L.LLM("Groq", "qwen/qwen3.8-27b", "key", backups=False, min_interval=0)
    assert m.json("s", "u") == {"ok": 1}
    assert sent == [9000, 900]


def test_a_bare_list_answer_is_not_lost():
    """Seen in testing: the AI returned [...] instead of {"clusters": [...]} and a whole batch of 6 was skipped."""
    from planner.llm import get_list, parse_json

    bare = parse_json('[{"cluster": 1, "ideas": []}]')
    assert get_list(bare, "clusters") == [{"cluster": 1, "ideas": []}]
    assert get_list({"clusters": [1]}, "clusters") == [1]
    assert get_list({"pages": [1], "other": [2]}, "clusters") == []   # two lists: can't tell which one is meant
    assert get_list({"note": "x"}, "clusters") == []


def test_a_rejected_key_is_not_tried_again_by_other_models(monkeypatch):
    calls = []

    def fake_post(url, payload, key, timeout):
        calls.append(payload["model"])
        raise L.HTTPError(401, "Invalid API Key")

    monkeypatch.setattr(L, "post_json", fake_post)
    L.LLM._rejected_keys.clear()
    big = L.LLM("Groq", "openai/gpt-oss-120b", "bad-key", backups=False, min_interval=0)
    small = L.LLM("Groq", "openai/gpt-oss-20b", "bad-key", backups=False, min_interval=0)
    with pytest.raises(ModelUnavailable):
        big.json("s", "u")
    with pytest.raises(ModelUnavailable):
        small.json("s", "u")
    assert calls == ["openai/gpt-oss-120b"] and small.exhausted   # the second model never called Groq
    L.LLM._rejected_keys.clear()


def test_gemini_bad_key_400_counts_as_a_rejected_key(monkeypatch):
    calls = []

    def fake_post(url, payload, key, timeout):
        calls.append(payload["model"])
        raise L.HTTPError(400, "API key not valid. Please pass a valid API key.")

    monkeypatch.setattr(L, "post_json", fake_post)
    L.LLM._rejected_keys.clear()
    for m in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"):
        with pytest.raises(ModelUnavailable, match="did not accept the API key"):
            L.LLM("Gemini", m, "bad", backups=False, min_interval=0).json("s", "u")
    assert calls == ["gemini-3.5-flash-lite"]
    L.LLM._rejected_keys.clear()


def test_brief_goes_into_every_ai_request():
    from planner.brief import brief_text, clean, summary
    from planner.llm import with_brief

    brief = {"goal": "Get enquiries and bookings", "offerings": "root canal,  implants", "customers": " ",
             "plan_for": "My own clinic", "extra": "ignored"}
    assert clean(brief) == {"goal": "Get enquiries and bookings", "offerings": "root canal, implants",
                            "plan_for": "My own clinic"}
    text = brief_text(brief)
    assert "Goal of the content: Get enquiries and bookings" in text and "root canal, implants" in text
    assert "customers" not in text.lower().split("services")[0]  # empty answers are left out
    assert summary(brief).startswith("Goal: Get enquiries and bookings")
    assert brief_text({}) == "" and with_brief("model", "") == "model"   # no brief: nothing changes

    seen = []

    class Rec:
        calls, label = 3, "rec"

        def json(self, system, user, temperature=0.7):
            seen.append(user)
            return {}

    llm = with_brief(StageModels({"angles": FallbackLLM([Rec()])}, FallbackLLM([Rec()])), text)
    llm.json("s", "default stage prompt")
    from planner.llm import for_stage
    for_stage(llm, "angles").json("s", "angles prompt")
    assert all(u.startswith("About this business") for u in seen) and seen[1].endswith("angles prompt")
    assert llm.label == "a different model for each stage"  # other attributes come from the wrapped models
