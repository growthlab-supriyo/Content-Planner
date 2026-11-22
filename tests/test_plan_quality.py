"""Regressions from the 'dental clinic, Kolkata' sample plan (Nov 2026): mislabeled clusters, one thin page per
neighbourhood, wrong intent on local queries, pages without H2s, catch-all pillars."""
from planner import quality as Q
from planner.cluster import cluster_keywords, name_clusters
from planner.llm import LLMError
from planner.plan import fallback_title, local_ok, map_keywords
from planner.topicmap import make_pillars, write_subtopics

from conftest import FakeLLM

LOCAL = ["best dentist near me in park street", "good dentist in lake town", "kids dentist near me in lake town",
         "pediatric dentist near alipore", "dental clinic open now in salt lake", "dentist open now in park street",
         "dentures specialist in entally", "best dentist ingaria", "kids dentist ingaria"]


def test_area_names_are_found_and_glued_words_split():
    places = Q.area_tokens(LOCAL, "Kolkata")
    for area in ("park street", "lake town", "alipore", "salt lake", "entally", "garia", "kolkata"):
        assert area in places
    assert "me" not in places and "now" not in places
    assert Q.split_glued("best dentist ingaria", places) == "best dentist in garia"
    assert Q.split_glued("implants cost in behalakolkata", ["behala", "kolkata"]) == "implants cost in behala kolkata"
    assert Q.split_glued("invisalign cost", places) == "invisalign cost"
    assert "behalakolkata" not in Q.area_tokens(["cost in behalakolkata", "dentist in behala"], "kolkata")
    assert Q.area_tokens(["pain in molars"], "") == ["molars"]     # why only Local-angle queries are read:
    assert Q.local_area_tokens([{"keyword": "pain in molars", "angle": "Questions"}], "") == []


def test_local_angle_queries_are_local_intent():
    places = Q.area_tokens(LOCAL, "Kolkata")
    assert Q.classify("toothache urgent care in shyambazar", "local", [])[0] == "Local"
    assert Q.classify("best crown and bridge clinic in tollygunge", "local", [])[0] == "Local"
    assert Q.classify("book a dentist appointment in salt lake", "local", places)[0] == "Transactional"
    assert Q.classify("root canal cost in lake town", "services", places)[0] == "Local"
    assert Q.classify("is root canal painful", "questions", places)[0] == "Informational"


def test_local_pages_group_by_service_not_by_area(embedder):
    rows = [{"keyword": k, "angle": "Local", "intent": "Local"} for k in (
        "kids dentist in park street", "kids dentist in lake town", "kids dentist near alipore",
        "dentures specialist in park street", "dentures specialist in lake town")]
    clusters = cluster_keywords(rows, embedder, None, "Dental Clinic", "Kolkata")
    by_kw = {k: c for c in clusters for k in c["keywords"]}
    assert by_kw["kids dentist in park street"] is by_kw["kids dentist near alipore"]
    assert by_kw["dentures specialist in park street"] is by_kw["dentures specialist in lake town"]
    assert by_kw["kids dentist in park street"] is not by_kw["dentures specialist in park street"]


def _clusters(embedder, names_kws):
    import numpy as np

    kws = [k for _, ks in names_kws for k in ks]
    vecs = embedder.embed(kws)
    out, i = [], 0
    for n, (name, ks) in enumerate(names_kws, 1):
        out.append({"id": n, "name": name, "intent": "Informational", "keywords": ks, "main_keyword": ks[0],
                    "size": len(ks), "_members": list(range(i, i + len(ks)))})
        i += len(ks)
    return out, np.asarray(vecs)


def test_cluster_names_cannot_shift_to_the_wrong_cluster(embedder):
    """The AI skipped cluster 1 and numbered the rest from 1: every name landed one cluster too early."""
    clusters, vecs = _clusters(embedder, [("Lost Crown", ["lost a crown during dinner"]),
                                          ("Laser Gum", ["laser gum treatment"]),
                                          ("Night Guard", ["night guard for teeth grinding"])])
    shifted = {"clusters": [{"id": 1, "name": "Laser Gum Treatment"}, {"id": 2, "name": "Night Guard Grinding"}]}
    name_clusters(clusters, FakeLLM([shifted]), "Dental Clinic", "", embedder=embedder, vecs=vecs)
    assert [c["name"] for c in clusters] == ["Lost Crown", "Laser Gum", "Night Guard"]  # rejected, kept

    echoed = {"clusters": [{"id": 1, "first": "laser gum treatment", "name": "Laser Gum Treatment"},
                           {"id": 2, "first": "night guard for teeth grinding", "name": "Night Guard Grinding"}]}
    name_clusters(clusters, FakeLLM([echoed]), "Dental Clinic", "", embedder=embedder, vecs=vecs)
    assert [c["name"] for c in clusters] == ["Lost Crown", "Laser Gum Treatment", "Night Guard Grinding"]


def test_lone_keyword_joins_a_cluster_only_when_it_shares_the_subject(embedder):
    rows = [{"keyword": k, "intent": "Informational"} for k in (
        "jaw clicking when chewing", "jaw clicking and popping", "jaw pain radiating to ear",
        "can cavities heal on their own")]
    clusters = cluster_keywords(rows, embedder, None, "Dental Clinic", "")
    by_kw = {k: c for c in clusters for k in c["keywords"]}
    assert by_kw["jaw pain radiating to ear"] is by_kw["jaw clicking when chewing"]
    assert by_kw["can cavities heal on their own"] is not by_kw["jaw clicking when chewing"]


def test_pillars_fall_back_to_broad_names_when_the_ai_fails(embedder):
    clusters = [{"id": i, "name": n, "keywords": ks, "intent": "Informational"} for i, (n, ks) in enumerate([
        ("Bleeding Gums", ["bleeding gums causes"]), ("Swollen Gums", ["swollen gums pain"]),
        ("Gum Recession", ["receding gums treatment"]), ("Braces Cost", ["braces cost"]),
        ("Metal Braces", ["metal braces for adults"]), ("Clear Braces", ["clear braces or aligners"])], 1)]
    from planner.topicmap import cluster_vectors

    vecs = cluster_vectors(clusters, embedder, "")
    pillars = make_pillars(clusters, vecs, FakeLLM([LLMError("down")]), embedder, "Dental Clinic", "")
    placed = sorted(cid for p in pillars for cid in p["cluster_ids"])
    assert placed == [1, 2, 3, 4, 5, 6]
    names = {c["name"] for c in clusters}
    assert all(p["name"] not in names for p in pillars if len(p["cluster_ids"]) > 1)  # not one page's name


def test_pages_missed_by_a_failed_batch_get_a_second_try():
    clusters = [{"id": 10 + i, "name": f"Page {i}", "keywords": [f"keyword {i}"]} for i in range(6)]

    def ok(user):
        import re
        pages = re.findall(r"^Page (\d+)\. ([^:]+):", user, flags=re.M)
        return {"pages": [{"page": int(n), "name": name, "subtopics": [f"About {name}"]} for n, name in pages]}

    llm = FakeLLM([LLMError("429"), ok, ok, ok])  # first batch of 4 fails, retried in the second pass
    out = write_subtopics(clusters, llm, "T", "", lambda m, f: None, batch=4)
    assert all(out[c["id"]] == [f"About {c['name']}"] for c in clusters)


def test_cluster_keeps_its_main_keyword_when_the_pillar_wants_it_too(embedder):
    clusters = [{"id": 1, "name": "Aligners or Braces", "intent": "Commercial", "pillar": "Braces",
                 "keywords": ["aligners or braces for adults", "aligners vs braces cost"],
                 "main_keyword": "aligners or braces for adults"}]
    pillars = [{"name": "Braces", "pillar_keyword": "aligners or braces for adults", "title": "Braces Guide",
                "cluster_ids": [1]}]
    ideas = [{"cluster_id": 1, "type": "Comparison", "title": "Aligners or Braces for Adults?", "ai_primary": ""}]
    rows, _ = map_keywords(ideas, clusters, pillars, embedder)
    by_role = {r["role"]: r for r in rows}
    assert by_role["Cluster Page"]["primary"] == "aligners or braces for adults"
    assert by_role["Pillar Page"]["primary"] == "braces"


def test_location_only_on_local_or_price_pages():
    places = ["kolkata", "salt lake"]
    assert local_ok({"intent": "Commercial", "keywords": ["root canal cost"]}, places)
    assert not local_ok({"intent": "Commercial", "keywords": ["electric vs manual toothbrush"]}, places)
    assert local_ok({"intent": "Commercial", "keywords": ["best braces clinic in salt lake"]}, places)


def test_fallback_titles_are_headlines_not_the_cluster_name():
    c = {"name": "Burning Tongue", "keywords": ["burning sensation on tongue"], "main_keyword": "burning sensation on tongue"}
    assert fallback_title(c, "Help Article") == "Burning Tongue: Causes and What to Do"
    q = {"name": "Cold Sensitivity", "keywords": ["why do my teeth hurt with cold water"],
         "main_keyword": "why do my teeth hurt with cold water"}
    assert fallback_title(q, "FAQs") == "Why do my teeth hurt with cold water?"


def test_ai_review_moves_clear_misfits_between_pillars():
    from planner.topicmap import review_pillars

    clusters = [{"id": i, "name": n} for i, n in enumerate(["Teeth Whitening", "Veneers", "Best Dentist in Kolkata",
                                                            "Find a Dentist Near You"], 1)]
    pillars = [{"name": "Cosmetic Dentistry", "cluster_ids": [1, 2, 3]}, {"name": "Choosing a Dentist", "cluster_ids": [4]}]
    answer = {"moves": [{"page": "best dentist in kolkata", "to": "Choosing a Dentist"},
                        {"page": "Unknown Page", "to": "Choosing a Dentist"},        # ignored
                        {"page": "Veneers", "to": "Cosmetic Dentistry"}]}           # already there: ignored
    assert review_pillars(pillars, clusters, FakeLLM([answer]), "Dental Clinic", "Kolkata") == 1
    assert pillars[0]["cluster_ids"] == [1, 2] and pillars[1]["cluster_ids"] == [4, 3]
    # a move that would empty a pillar is refused
    two = [{"name": "A", "cluster_ids": [1]}, {"name": "B", "cluster_ids": [2]}]
    assert review_pillars(two, clusters, FakeLLM([{"moves": [{"page": "Teeth Whitening", "to": "B"}]}]), "x", "") == 0
    assert two[0]["cluster_ids"] == [1]


def test_ai_joins_single_keyword_pages_of_the_same_intent(embedder):
    rows = [{"keyword": k, "intent": "Informational"} for k in (
        "bleeding gums causes", "bleeding gums while brushing", "periodontal disease warning signs",
        "can cavities heal")]

    def answer(user):
        import re
        if "SINGLE" not in user:
            return {"clusters": []}  # the naming step: keep fallback names
        nums = {kw: int(n) for n, kw in re.findall(r"^(\d+)\. (?:SINGLE )?(.+)$", user, flags=re.M)}
        perio = next(n for kw, n in nums.items() if "periodontal" in kw)
        gums = next(n for kw, n in nums.items() if "bleeding gums" in kw)
        return {"joins": [{"single": perio, "join": gums}]}

    clusters = cluster_keywords(rows, embedder, None, "Dental Clinic", "")  # no AI: what the grouping alone gives
    assert any(c["keywords"] == ["periodontal disease warning signs"] for c in clusters)
    clusters = cluster_keywords(rows, embedder, FakeLLM([answer] * 5), "Dental Clinic", "")
    by_kw = {k: c for c in clusters for k in c["keywords"]}
    assert by_kw["periodontal disease warning signs"] is by_kw["bleeding gums causes"]
    assert by_kw["can cavities heal"] is not by_kw["bleeding gums causes"]


def test_repeated_cluster_names_get_a_real_name():
    from planner.cluster import _unique_names

    cs = [{"name": "Emergency Dental Care", "intent": "Local", "main_keyword": "emergency dentist near me"},
          {"name": "Emergency Dental Care", "intent": "Informational", "main_keyword": "what counts as a dental emergency"}]
    _unique_names(cs)
    assert [c["name"] for c in cs] == ["Emergency Dental Care", "What Counts As A Dental Emergency"]


def test_pillar_page_takes_its_pages_intent_and_stage(embedder):
    clusters = [{"id": i, "name": f"P{i}", "intent": "Informational", "funnel": "TOFU", "pillar": "Care",
                 "keywords": [f"keyword {i}"], "main_keyword": f"keyword {i}"} for i in (1, 2)]
    pillars = [{"name": "Care", "pillar_keyword": "dental care kolkata", "title": "Dental Care", "cluster_ids": [1, 2]}]
    ideas = [{"cluster_id": i, "type": "Guide", "title": f"Guide {i}", "ai_primary": ""} for i in (1, 2)]
    rows, _ = map_keywords(ideas, clusters, pillars, embedder, "Kolkata")
    hub = next(r for r in rows if r["role"] == "Pillar Page")
    assert hub["intent"] == "Informational" and hub["funnel"] == "TOFU"


def test_enquiry_goal_requires_a_ready_to_act_angle():
    from planner.fanout import angle_problems, complete_angles, required_kinds

    assert "action" in required_kinds("Kalyani", "Get enquiries and bookings")
    assert "action" in required_kinds("", "Sell products online")
    assert "action" not in required_kinds("", "Build traffic and authority")
    angles = [{"label": k.title(), "kind": k} for k in ("questions", "problems", "costs", "comparisons", "fears")]
    assert "action" in angle_problems(angles, "", "Get enquiries and bookings")
    assert complete_angles(angles, "", "Get enquiries and bookings")[-1]["label"] == "Ready to Act"
    assert Q.classify("hotel management admission form kalyani", "action", [])[0] == "Transactional"


def test_place_stays_in_titles_of_pages_whose_searches_name_it():
    places = ["kalyani"]
    hostel = {"intent": "Informational", "keywords": ["hostel fee hotel management college kalyani",
                                                      "hotel management hostel cost kalyani", "hostel rules"]}
    general = {"intent": "Informational", "keywords": ["what is hotel management", "hotel management syllabus"]}
    assert local_ok(hostel, places) and not local_ok(general, places)


def test_news_and_interviews_are_never_planned():
    from planner.plan import decide_types

    answer = {"types": [{"type": "News Article", "decision": "Used", "reason": "x"},
                        {"type": "Interview", "decision": "Used", "reason": "x"},
                        {"type": "Guide", "decision": "Used", "reason": "x"}]}
    types = {t["type"]: t for t in decide_types([], [], FakeLLM([answer]), "Hotel management college", "")}
    assert types["News Article"]["decision"] == types["Interview"]["decision"] == "Skipped"
    assert "real" in types["Interview"]["reason"] and types["Guide"]["decision"] == "Used"


def test_old_years_in_titles_become_the_current_year():
    y = Q.current_year()
    assert Q.fix_years("Salary Trends for Hotel Management Graduates 2019") == \
        f"Salary Trends for Hotel Management Graduates {y}"
    assert Q.fix_years(f"Plans for {y + 1}") == f"Plans for {y + 1}" and Q.fix_years("Room 12019") == "Room 12019"


def test_a_pillar_with_too_few_pages_joins_the_closest_pillar():
    import numpy as np
    from planner.topicmap import merge_small_pillars

    v = {1: np.array([1.0, 0, 0]), 2: np.array([1.0, 0.1, 0]), 3: np.array([0.9, 0.2, 0]),
         4: np.array([0, 1.0, 0]), 5: np.array([0, 1.0, 0.1]), 6: np.array([0, 0.9, 0.2]),
         7: np.array([0.95, 0.05, 0])}
    pillars = [{"name": "Fees", "cluster_ids": [1, 2, 3]}, {"name": "Careers", "cluster_ids": [4, 5, 6]},
               {"name": "Comparison & Decision Guides", "cluster_ids": [7]}]
    assert merge_small_pillars(pillars, v) == 1
    assert [p["name"] for p in pillars] == ["Fees", "Careers"] and 7 in pillars[0]["cluster_ids"]


def test_tagging_shows_the_ai_each_querys_angle():
    from planner.fanout import tag_intents

    rows = [{"keyword": "download hotel management application form", "angle": "Enroll Now", "kind": "action",
             "intent": "Informational", "funnel": "TOFU"}]
    llm = FakeLLM([{"tags": [{"q": "download hotel management application form [Enroll Now · ready to act]",
                              "intent": "Transactional", "funnel": "BOFU"}]}])
    assert tag_intents(llm, rows, "Hotel management college") == 1
    assert "[Enroll Now · ready to act]" in llm.prompts[0] and rows[0]["funnel"] == "BOFU"
