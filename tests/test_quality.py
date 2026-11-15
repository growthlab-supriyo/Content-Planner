import numpy as np

from planner import quality as Q


def test_clean_strips_numbering_quotes_and_punctuation():
    assert Q.clean('1. "Is Root Canal Painful?"') == "is root canal painful"
    assert Q.clean("- Braces  vs  Aligners!") == "braces vs aligners"
    assert Q.clean("• tooth pain at night.") == "tooth pain at night"


def test_is_valid():
    assert Q.is_valid("tooth pain at night")
    assert not Q.is_valid("dentist")                 # one word
    assert not Q.is_valid("a b c d e f g h i j k l m")  # 13 words
    assert not Q.is_valid("123 456")                 # not words


def test_cap_repeats_stops_topic_template():
    topic = "Dental Clinic"
    qs = [f"dental clinic newtown {w}" for w in ("cost", "price", "reviews", "address", "hours", "insurance",
                                                 "timings", "phone")]
    qs += ["is root canal painful", "tooth pain at night", "braces vs aligners", "bleeding gums causes"]
    kept, dropped, worst = Q.cap_repeats(qs, topic)
    templ = [q for q in kept if q.startswith("dental clinic")]
    assert len(templ) == 3
    assert len(dropped) == 5
    assert worst == "dental clinic"
    assert "is root canal painful" in kept


def test_coverage_questions_and_local():
    ok, share, fb = Q.coverage("questions", ["how long do braces take", "braces cost"], [])
    assert not ok and share == 0.5 and "question" in fb
    loc = Q.location_tokens("Newtown, Kolkata")
    assert "newtown" in loc and "kolkata" in loc
    ok, _, _ = Q.coverage("local", ["dentist near me", "best dentist newtown", "dentist in kolkata"], loc)
    assert ok
    ok, _, _ = Q.coverage("local", ["best dentist", "dentist reviews", "dentist newtown"], loc)
    assert not ok


def test_coverage_costs_and_comparisons():
    assert Q.coverage("costs", ["root canal cost", "is braces worth it"], [])[0]
    assert Q.coverage("comparisons", ["braces vs aligners", "implant or bridge"], [])[0]
    assert not Q.coverage("comparisons", ["braces for adults", "dental implants"], [])[0]


def test_near_duplicates_keep_first():
    v = np.array([[1, 0], [0.999, 0.04], [0, 1]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    assert Q.near_duplicates(v, 0.95) == [(0, 1)]


def test_classify():
    loc = Q.location_tokens("Newtown")
    assert Q.classify("is root canal painful", "questions", loc) == ("Informational", "TOFU")
    assert Q.classify("root canal cost in newtown", "costs", loc) == ("Commercial", "MOFU")
    assert Q.classify("braces vs aligners", "comparisons", loc) == ("Commercial", "MOFU")
    assert Q.classify("best dentist in newtown", "local", loc) == ("Local", "BOFU")
    assert Q.classify("book dental appointment online", "services", loc) == ("Transactional", "BOFU")
    assert Q.classify("teeth whitening treatment", "services", loc) == ("Commercial", "MOFU")
