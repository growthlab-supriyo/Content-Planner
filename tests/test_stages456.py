import io
import re

from openpyxl import load_workbook

from planner.content_types import TYPE_NAMES, normalise_type
from planner.export import to_excel
from planner.mapviz import graph_data, map_html
from planner.plan import build_content_plan, map_keywords
from planner.project import new_project
from planner.topicmap import all_clusters, build_topical_map

from conftest import FakeLLM


def make_project():
    p = new_project("Dental Clinic", "Newtown")
    kws = {
        1: ("Root Canal Cost", "Commercial", ["root canal cost newtown", "root canal price with crown",
                                              "is root canal covered by insurance"]),
        2: ("Root Canal Pain", "Informational", ["is root canal painful", "root canal recovery time",
                                                 "what to eat after root canal", "root canal swelling"]),
        3: ("Bleeding Gums", "Informational", ["bleeding gums causes", "bleeding gums while brushing"]),
        4: ("Best Dentist Newtown", "Local", ["best dentist in newtown", "dentist near me newtown"]),
    }
    p["stage2"] = {"keywords": [{"keyword": k, "angle": "Services", "intent": i, "funnel": "TOFU"}
                                for _, i, ks in kws.values() for k in ks], "report": {"model": "fake"}}
    p["stage3"] = {"clusters": [{"id": cid, "name": n, "intent": i, "keywords": ks, "main_keyword": ks[0],
                                 "size": len(ks)} for cid, (n, i, ks) in kws.items()], "filtered_out": []}
    return p


def stage4_answers(user):
    if "grouped into" in user:
        groups = re.findall(r"^Group (\d+): (.*)$", user, flags=re.M)
        # the second group is left unnamed on purpose: code must fall back to its most typical page name
        return {"pillars": [{"group": int(n), "name": f"Pillar {n}", "pillar_keyword": f"pillar {n} keyword",
                             "pillar_title": f"Pillar {n} hub"} for n, _ in groups[:1]]}
    if "subtopics: the H2" in user:
        pages = re.findall(r"^Page (\d+)\. ([^:]+):", user, flags=re.M)
        return {"pages": [{"page": int(n), "subtopics": [f"About {name}", "How it works", "Costs"]}
                          for n, name in pages]}
    if "MISSING" in user:
        return {"missing": [
            {"pillar": "Pillar 1", "name": "Root Canal Cost", "intent": "Commercial",  # duplicate of existing
             "keywords": ["x y"], "subtopics": ["a"]},
            {"pillar": "Nowhere", "name": "Kids Dental Care", "intent": "Informational",
             "keywords": ["when should a child first see a dentist", "bleeding gums causes"], "subtopics": ["Age"]},
        ]}
    raise AssertionError(user[:80])


def stage56_answers(user):
    if "For EACH of the" in user:
        return {"types": [{"type": "guide", "decision": "Used", "reason": "core"},
                          {"type": "FAQ", "decision": "Used", "reason": "questions"},
                          {"type": "Cost Guide", "decision": "Used", "reason": "prices"},
                          {"type": "Whitepaper", "decision": "Skipped", "reason": "not for patients"},
                          {"type": "Pillar Page", "decision": "Used", "reason": "hubs"},
                          {"type": "Testimonials", "decision": "Used", "reason": "trust"}]}
    if "suggest content pieces" in user:
        nums = [int(x) for x in re.findall(r"^Cluster (\d+)\. ", user, flags=re.M)]
        out = []
        for i in nums:
            out.append({"cluster": i, "ideas": [
                {"type": "Cost Guide" if i == 1 else "Guide", "title": f"Main page {i}",
                 "primary": "root canal cost newtown" if i == 1 else ""},
                {"type": "Whitepaper", "title": f"Extra {i}", "primary": ""}]})  # skipped type → remapped
        return {"clusters": out}
    if "have no page yet" in user:
        return {"ideas": [{"type": "Testimonials", "cluster_id": 4, "title": "Patient reviews"}]}
    raise AssertionError(user[:80])


def run_all(embedder):
    p = make_project()
    p["stage4"] = build_topical_map(p, FakeLLM([stage4_answers] * 10), embedder)
    p["stage5"], p["stage6"] = build_content_plan(p, FakeLLM([stage56_answers] * 10), embedder)
    return p


def test_topical_map_assigns_every_cluster_once_and_filters_gaps(embedder):
    p = run_all(embedder)
    s4 = p["stage4"]
    placed = [cid for pl in s4["pillars"] for cid in pl["cluster_ids"]]
    assert set(placed) == {c["id"] for c in all_clusters(p)} and len(placed) == len(set(placed))
    assert len(s4["pillars"]) >= 2
    assert s4["pillars"][0]["name"] == "Pillar 1"                # named by the AI
    assert s4["pillars"][1]["name"] in {c["name"] for c in p["stage3"]["clusters"]}  # fallback name
    names = [g["name"] for g in s4["gap_clusters"]]
    assert names == ["Kids Dental Care"]                       # duplicate "Root Canal Cost" dropped
    gap = s4["gap_clusters"][0]
    assert "bleeding gums causes" not in gap["keywords"]       # existing keyword not re-added
    assert all(pl["source"] == "Core" for pl in s4["pillars"])  # gap pages join existing pillars
    for c in all_clusters(p):                                   # subtopics belong to their own page
        if c["source"] == "Core":
            assert c["subtopics"][0] == f"About {c['name']}"


def test_batches_restarting_numbers_map_to_the_right_pages(embedder):
    """Regression: the model numbered every batch 1..n, which overwrote the first pages' subtopics and ideas."""
    from planner.topicmap import write_subtopics
    from planner.plan import generate_ideas

    clusters = [{"id": 100 + i, "name": f"Page {i}", "intent": "Informational", "keywords": [f"keyword {i} a"],
                 "main_keyword": f"keyword {i} a", "size": 1} for i in range(10)]

    def subs(user):
        pages = re.findall(r"^Page (\d+)\. ([^:]+):", user, flags=re.M)
        return {"pages": [{"page": int(n), "subtopics": [f"H2 for {name}"]} for n, name in pages]}

    out = write_subtopics(clusters, FakeLLM([subs] * 5), "T", "", lambda m, f: None, batch=4)
    assert all(out[c["id"]] == [f"H2 for {c['name']}"] for c in clusters)

    def copies(user):  # headings that only repeat the page name or a keyword are dropped
        pages = re.findall(r"^Page (\d+)\. ([^:]+):", user, flags=re.M)
        return {"pages": [{"page": int(n), "subtopics": [name, f"keyword {name.split()[-1]} a", "Real heading"]}
                          for n, name in pages]}

    out = write_subtopics(clusters, FakeLLM([copies] * 5), "T", "", lambda m, f: None, batch=4)
    assert all(out[c["id"]] == ["Real heading"] for c in clusters)

    def ideas(user):
        rows = re.findall(r"^Cluster (\d+)\. (\S+ \d+)", user, flags=re.M)
        return {"clusters": [{"cluster": int(n), "ideas": [{"type": "Guide", "title": f"Idea for {name}"}]}
                             for n, name in rows]}

    types = [{"type": "Guide", "decision": "Used"}]
    got = generate_ideas(clusters, types, FakeLLM([ideas] * 5), "T", "", lambda m, f: None, batch=4)
    assert {(i["cluster_id"], i["title"]) for i in got} == {(c["id"], f"Idea for {c['name']}") for c in clusters}


def test_content_plan_rules(embedder):
    p = run_all(embedder)
    rows = p["stage6"]["rows"]
    prim = [r["primary"] for r in rows if r["primary"]]
    assert len(prim) == len(set(prim))                         # never primary twice
    clusters = {c["id"]: c for c in all_clusters(p)}
    for r in rows:
        if r["cluster_id"] is not None:
            c = clusters[r["cluster_id"]]
            assert not r["primary"] or r["primary"] in c["keywords"]
            assert set(r["secondary"]) <= set(c["keywords"]) and len(r["secondary"]) <= 8
    assert all(r["type"] in TYPE_NAMES for r in rows)
    assert not any(r["type"] == "Whitepaper" for r in rows)   # skipped type never used
    assert sum(r["role"] == "Pillar Page" for r in rows) == len(p["stage4"]["pillars"])
    assert any(r["type"] == "Testimonials" for r in rows)     # filled in by the extra call
    types = {t["type"]: t for t in p["stage5"]["types"]}
    assert len(types) == len(TYPE_NAMES) == 23
    assert types["Whitepaper"]["decision"] == "Skipped"
    # every keyword is primary, secondary or leftover
    used = set(prim) | {k for r in rows for k in r["secondary"]} | {x["keyword"] for x in p["stage6"]["leftover"]}
    assert {k for c in clusters.values() for k in c["keywords"]} <= used
    assert [r["n"] for r in rows] == list(range(1, len(rows) + 1))


def test_map_keywords_no_free_keyword_left(embedder):
    clusters = [{"id": 1, "name": "A", "intent": "Commercial", "keywords": ["only one keyword"],
                 "main_keyword": "only one keyword", "pillar": "P"}]
    ideas = [{"cluster_id": 1, "type": "Guide", "title": "First", "ai_primary": ""},
             {"cluster_id": 1, "type": "FAQs", "title": "Second", "ai_primary": "only one keyword"}]
    rows, leftover = map_keywords(ideas, clusters, [], embedder)
    assert rows[0]["primary"] == "only one keyword" and rows[0]["role"] == "Cluster Page"
    assert rows[1]["primary"] == "" and rows[1]["note"]


def test_normalise_type():
    assert normalise_type("FAQ") == "FAQs"
    assert normalise_type("myths vs facts") == "Myth vs Fact"
    assert normalise_type("Comparison post") == "Comparison"
    assert normalise_type("something else") is None


def test_excel_and_map_with_all_stages(embedder):
    p = run_all(embedder)
    wb = load_workbook(io.BytesIO(to_excel(p)))
    assert wb.sheetnames == ["Clusters & Keywords", "Topical Map", "Content Plan & Mapping"]
    tm, cp, kw = wb["Topical Map"], wb["Content Plan & Mapping"], wb["Clusters & Keywords"]
    assert tm["A6"].value == "Dental Clinic" and str(tm["F6"].value).startswith("=COUNTIF('Clusters & Keywords'")
    assert cp["B6"].value == "Cluster (Page)" and cp["A7"].value == 1
    # the COUNTIF ranges must cover the data rows exactly
    m = re.search(r"\$B\$(\d+):\$B\$(\d+)", tm["F6"].value)
    first, last = int(m.group(1)), int(m.group(2))
    assert kw.cell(row=first - 1, column=1).value == "Pillar" and kw.cell(row=last, column=3).value
    assert kw.cell(row=last + 1, column=3).value in (None, "")
    used_as = {kw.cell(row=r, column=7).value for r in range(first, last + 1)}
    assert "Primary" in used_as
    nodes, edges = graph_data(p)
    assert sum(n["group"] == "pillar" for n in nodes) == len(p["stage4"]["pillars"])
    assert any(n["group"] == "ai" for n in nodes)
    html = map_html(p, "dark")
    assert "vis-network@9.1.9" in html and "</script><" not in html.split("const DATA")[1].split("const groups")[0]


def test_idea_count_and_location_rules():
    from planner.plan import ideas_for, local_ok, strip_location

    assert [ideas_for({"size": s, "keywords": []}) for s in (1, 2, 3, 5, 6, 11, 12, 30)] == [1, 1, 2, 2, 3, 3, 4, 4]
    assert local_ok({"intent": "Local"}) and not local_ok({"intent": "Informational"})
    loc = "Newtown, Kolkata"
    assert strip_location("Can Dental Plaque Really Harm Teeth: Debunking Myths in Newtown, Kolkata", loc) == \
        "Can Dental Plaque Really Harm Teeth: Debunking Myths"
    assert strip_location("Will Root Canal Be Painful? Tips from a Newtown, Kolkata Dentist", loc) == \
        "Will Root Canal Be Painful? Tips from a Dentist"
    assert strip_location("Overcoming Dental Fear in Kolkata", loc) == "Overcoming Dental Fear"
    assert strip_location("Root Canal Recovery Guide", loc) == "Root Canal Recovery Guide"
