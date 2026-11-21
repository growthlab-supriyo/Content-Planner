import pytest

from planner import prompts
from planner.cluster import cluster_keywords, filter_keywords
from planner.fanout import run_angle, run_fanout
from planner.llm import FallbackLLM, LLMError, RateLimited, parse_json

from conftest import FakeLLM


def test_parse_json_handles_fences_and_extra_text():
    assert parse_json('```json\n{"queries": ["a b"]}\n```') == {"queries": ["a b"]}
    assert parse_json('Sure! {"queries": ["a b"]} hope this helps') == {"queries": ["a b"]}
    assert parse_json('["x y"]') == {"items": ["x y"]}
    with pytest.raises(LLMError):
        parse_json("no json here")


def test_prompt_location_rules():
    p = prompts.build_prompt("local", "Dental Clinic", "Newtown", 20)
    assert "You may mention Newtown" in p and "it's not required" in p  # a place is optional
    p = prompts.build_prompt("questions", "Dental Clinic", "Newtown", 20)
    assert "do not add the location" in p
    p = prompts.build_prompt("local", "Wooden Wall Decor", "", 20)
    assert "where to buy" in p and "do not add any city" in p
    p = prompts.build_prompt("costs", "X", "", 20, instructions={"costs": "CUSTOM TEXT"})
    assert "CUSTOM TEXT" in p


def test_run_angle_retries_with_feedback_when_repetitive():
    repetitive = {"queries": [f"dental clinic newtown {w}" for w in
                              "cost price fees charges rates insurance emi package discount budget".split()]}
    good = {"queries": ["root canal cost in newtown", "how much do braces cost", "dental implant price kolkata",
                        "teeth cleaning charges", "is dental insurance worth it", "cheapest way to fix a tooth",
                        "tooth extraction cost", "emi for dental implants"]}
    llm = FakeLLM([repetitive, good])
    kept, rep = run_angle(llm, "costs", "Dental Clinic", "Newtown", 10, max_retries=2)
    assert rep["attempts"] == 2
    assert "Too many queries started with" in llm.prompts[1]
    assert sum(q.startswith("dental clinic") for q in kept) <= 3
    assert "root canal cost in newtown" in kept


def test_run_angle_asks_again_when_off_angle():
    not_questions = {"queries": ["braces for adults", "root canal newtown", "teeth whitening", "dental implants",
                                 "gum treatment", "kids dentist"]}
    questions = {"queries": ["how long do braces take", "is root canal painful", "what to eat after extraction",
                             "can braces fix gaps", "why do gums bleed", "should i remove wisdom teeth"]}
    llm = FakeLLM([not_questions, questions])
    kept, rep = run_angle(llm, "questions", "Dental Clinic", "", 6, max_retries=1)
    assert "real question" in llm.prompts[1]
    assert all(q.split()[0] in ("how", "is", "what", "can", "why", "should") for q in kept)


def test_run_fanout_dedupes_across_angles_and_tags(embedder):
    def answer(user):
        if "Angle: Costs" in user:
            return {"queries": ["root canal cost", "root canal price", "braces cost kolkata"]}
        if "Angle: Questions" in user:
            return {"queries": ["is root canal painful", "how long do braces take", "root canal cost"]}
        return {"queries": []}

    llm = FakeLLM([answer] * 50)
    rows, report = run_fanout(llm, embedder, "Dental Clinic", "Kolkata", n=3, angles=["costs", "questions"])
    kws = [r["keyword"] for r in rows]
    assert kws.count("root canal cost") == 1          # exact duplicate across angles removed
    assert "root canal price" not in kws               # near-duplicate of "root canal cost" (fake: price == cost)
    assert report["near_duplicates_removed"] == 1
    assert all({"intent", "funnel", "angle"} <= set(r) for r in rows)


def test_run_fanout_stops_on_bad_key(embedder):
    llm = FakeLLM([LLMError("Gemini needs an API key.")])
    with pytest.raises(LLMError):
        run_fanout(llm, embedder, "Dental Clinic", "", n=5)


def test_filter_keywords():
    rows = [{"keyword": k} for k in ("dental clinic", "dentist", "best clinics", "invisible braces",
                                     "dentist newtown", "best dental clinic", "root canal cost")]
    kept, removed = filter_keywords(rows, "Dental Clinic")
    assert [r["keyword"] for r in kept] == ["invisible braces", "dentist newtown", "best dental clinic",
                                            "root canal cost"]
    assert {r["keyword"]: r["reason"] for r in removed} == {
        "dental clinic": "Generic two-word term", "dentist": "Single word", "best clinics": "Generic two-word term"}


def test_cluster_one_intent_per_cluster_and_names(embedder):
    rows = [
        {"keyword": "root canal cost newtown", "intent": "Commercial"},
        {"keyword": "root canal cost with crown", "intent": "Commercial"},
        {"keyword": "is root canal painful", "intent": "Informational"},
        {"keyword": "root canal recovery time", "intent": "Informational"},
        {"keyword": "bleeding gums causes", "intent": "Informational"},
        {"keyword": "bleeding gums while brushing", "intent": "Informational"},
    ]
    llm = FakeLLM([{"clusters": [{"id": 1, "name": "Root Canal Cost"}, {"id": 2, "name": "Bleeding Gums"},
                                 {"id": 3, "name": "Root Canal Questions"}]}])
    clusters = cluster_keywords(rows, embedder, llm, "Dental Clinic", "Newtown")
    by_kw = {k: c for c in clusters for k in c["keywords"]}
    assert by_kw["root canal cost newtown"] is by_kw["root canal cost with crown"]
    assert by_kw["root canal cost newtown"] is not by_kw["is root canal painful"]   # different intent
    assert by_kw["bleeding gums causes"] is by_kw["bleeding gums while brushing"]
    assert sum(c["size"] for c in clusters) == len(rows)
    assert len({c["name"] for c in clusters}) == len(clusters)
    assert all(len({next(r["intent"] for r in rows if r["keyword"] == k) for k in c["keywords"]}) == 1
               for c in clusters)


def test_fallback_switches_on_rate_limit():
    class Limited:
        provider, label, calls = "Gemini", "Gemini · x", 0

        def json(self, *a, **k):
            raise RateLimited("429")

    second = FakeLLM([{"ok": 1}])
    second.provider = "Local Ollama"
    msgs = []
    f = FallbackLLM([Limited(), second], on_switch=msgs.append)
    assert f.json("s", "u") == {"ok": 1}
    assert msgs and "Switched to" in msgs[0]


def test_ai_flagged_outlier_moves_out(embedder):
    rows = [{"keyword": k, "intent": "Informational"} for k in (
        "bleeding gums causes", "bleeding gums while brushing", "bleeding gums treatment", "gums at night pain",
        "tooth pain at night", "tooth pain relief", "tooth pain after filling")]
    import re

    def answer(user):
        out = []
        for line in user.splitlines():
            m = re.match(r"(\d+)\. (.*)", line)
            if m:
                kws = m.group(2).split("; ")
                gums = "bleeding gums causes" in kws
                out.append({"id": int(m.group(1)), "name": "Bleeding Gums" if gums else "Tooth Pain",
                            "outliers": ["gums at night pain"] if gums and "gums at night pain" in kws else []})
        return {"clusters": out}

    clusters = cluster_keywords(rows, embedder, FakeLLM([answer]), "Dental Clinic", "")
    by_kw = {k: c for c in clusters for k in c["keywords"]}
    assert by_kw["bleeding gums causes"]["name"] == "Bleeding Gums"
    assert by_kw["gums at night pain"] is not by_kw["bleeding gums causes"]
    assert sum(c["size"] for c in clusters) == len(rows)
    assert all("_members" not in c for c in clusters)


def test_near_duplicates_need_same_meaningful_words():
    import numpy as np
    from planner import quality as Q

    texts = ["braces cost in kolkata", "kolkata braces price", "best braces clinic kolkata"]
    v = np.ones((3, 4), dtype=np.float32) / 2  # all identical vectors: only the word rule can tell them apart
    assert Q.near_duplicates(v * 0.99, 0.9, texts, very_high=1.5) == [(0, 1)]
