"""Stage 5 (content ideas) and Stage 6 (keyword mapping).

Stage 5
  a. The AI decides, for every one of the 23 content types, whether it is needed to cover the topic and build
     topical authority (Used / Skipped + reason).
  b. Per cluster, the AI suggests 1–4 pieces using only Used types. The first piece is the cluster's main page.
  c. Every pillar gets a Pillar Page. Used types that got no page are filled in with one more AI call.
Stage 6 (rules in code, not left to the AI)
  - one primary keyword per page, taken from its own cluster, never the primary of two pages
  - 3–8 secondary keywords per page from its cluster (fewer only if the cluster is small)
  - every other keyword goes to the closest page of its cluster; anything left is listed as leftover
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import quality as Q
from .content_types import DEFAULT_USED, TYPE_NAMES, normalise_type, types_prompt_list
from .llm import LLMError, for_stage, get_list
from .topicmap import all_clusters

Progress = Callable[[str, float], None]
SYSTEM = "You are an SEO content strategist planning content for topical authority. You answer with JSON only."

TYPES_PROMPT = """Business / topic: {topic}{loc}
Pillars and pages planned so far:
{summary}

Content types:
{types}

For EACH of the {n} content types, decide whether this business needs it to cover the topic completely and build
topical authority. "Used" if it fits this business and its customers' searches, "Skipped" if not. Give a short reason
(under 15 words) for each.

Return JSON only: {{"types": [{{"type": "Guide", "decision": "Used", "reason": "..."}}, ...]}}"""

IDEAS_PROMPT = """Business / topic: {topic}{loc}
Content types you may use: {used}

For each page cluster below (use the cluster numbers exactly as given), suggest content pieces: never more than
the "max" shown for that cluster. The FIRST piece is the main page for that cluster. Add more only when the
cluster's keywords need a different format. Choose the type that fits the keywords: prices → Cost Guide;
"x vs y" → Comparison; symptoms or problems → Help Article; fears or myths → Myth vs Fact or FAQs; many questions →
FAQs; how-to → Tutorial; "best ..." → Listicle; reviews → Testimonials. Vary types across the plan.
Titles: specific and natural, like a real article headline, not just the cluster name, and never the keyword pasted
in as it is (bad: "Hostel Fee Hotel Management College and Accommodation Details"; good: "Hostel Fees and Rooms at
a Hotel Management College: What to Expect"). Mention {place} ONLY for
clusters marked [local]; never on how-to, question, myth, symptom or comparison pieces.
If a title needs a year, use {year}, never an earlier year.
Never invent people, doctor names, clinic names, statistics or rankings (no "Dr. Sharma", no "Why We Are the
Best ..."): the business will fill in real details.
For each piece copy one keyword EXACTLY from that cluster as "primary". Answer for every cluster.

{clusters}

Return JSON only:
{{"clusters": [{{"cluster": 1, "ideas": [{{"type": "Guide", "title": "...", "primary": "..."}}]}}]}}"""

FILL_PROMPT = """Business / topic: {topic}{loc}
These content types are needed but have no page yet: {missing}

Page clusters:
{clusters}

Suggest exactly one piece for each of those content types and attach it to the best-fitting cluster id.

Return JSON only: {{"ideas": [{{"type": "...", "cluster_id": 1, "title": "..."}}]}}"""


def _loc(location: str) -> str:
    return f"\nLocation: {location}" if location else ""


def _list(data: dict, key: str) -> list:
    return get_list(data, key)


# ---------------- Stage 5a: content type decisions ----------------
def decide_types(clusters: list[dict], pillars: list[dict], llm, topic: str, location: str) -> list[dict]:
    names = {c["id"]: c["name"] for c in clusters}
    summary = "\n".join(f"{p['name']}: " + "; ".join(names[i] for i in p["cluster_ids"][:10]) for p in pillars)
    decided: dict[str, dict] = {}
    try:
        data = llm.json(SYSTEM, TYPES_PROMPT.format(topic=topic, loc=_loc(location), summary=summary,
                                                    types=types_prompt_list(), n=len(TYPE_NAMES)), temperature=0.3)
        for t in _list(data, "types"):
            if not isinstance(t, dict):
                continue
            name = normalise_type(t.get("type"))
            if not name or name in decided:
                continue
            d = str(t.get("decision") or "").strip().lower()
            decision = "Skipped" if d.startswith("skip") or d in ("no", "false", "not needed") else "Used"
            decided[name] = {"type": name, "decision": decision,
                             "reason": str(t.get("reason") or "").strip()[:160] or "—"}
    except LLMError:
        pass
    out = []
    for name in TYPE_NAMES:
        out.append(decided.get(name) or {"type": name, "decision": "Used" if name in DEFAULT_USED else "Skipped",
                                         "reason": "Default (the AI gave no decision)"})
    if not any(t["decision"] == "Used" for t in out):
        for t in out:
            if t["type"] in DEFAULT_USED:
                t["decision"] = "Used"
    for t in out:  # a plan can't invent these: they need something real (seen: "Interview on Affordable Diplomas")
        if t["type"] in NEEDS_REAL:
            t["decision"], t["reason"] = "Skipped", NEEDS_REAL[t["type"]]
    return out


NEEDS_REAL = {
    "News Article": "Needs real news (a new service, event or rule): add one when there is something to announce.",
    "Interview": "Needs a real person to interview: add one when an expert or the owner is available.",
}


# ---------------- Stage 5b/c: ideas ----------------
def _pick_type(raw, used: list[str], fallback: str = "Guide") -> str:
    t = normalise_type(raw)
    if t in used:
        return t
    return fallback if fallback in used else used[0]


def ideas_for(c: dict, max_ideas: int = 4) -> int:
    """How many pieces a cluster can carry: small clusters get one page, so their keywords become secondaries
    instead of each turning into its own thin page."""
    size = c.get("size") or len(c["keywords"])
    return min(max_ideas, 1 if size <= 2 else 2 if size <= 5 else 3 if size <= 11 else 4)


def local_ok(c: dict, places: list[str] | None = None) -> bool:
    """The location belongs in titles of local and transactional pages, and of commercial pages about prices or
    that name a place (prices differ by city). Not on comparisons like 'electric vs manual toothbrush'."""
    if c.get("intent") in ("Local", "Transactional"):
        return True
    kws = c.get("keywords", [])
    if places and kws and sum(Q.mentions_location(k, places) for k in kws) * 2 >= len(kws):
        # most of its searches name the place ("hostel fee hotel management college kalyani"): the place belongs
        # in the title, and cutting it out of a title written around it left broken headlines (seen in testing)
        return True
    if c.get("intent") != "Commercial":
        return False
    return any(any(w in k.split() for w in Q.COST_WORDS) for k in kws) or \
        bool(places and any(Q.mentions_location(k, places) for k in kws))


TITLE_PATTERNS = {"Cost Guide": "{name}: Prices and What Affects Them", "Comparison": "{name}: Which Is Better?",
                  "FAQs": "{name}: Common Questions Answered", "Help Article": "{name}: Causes and What to Do",
                  "Myth vs Fact": "{name}: Myths and Facts", "Troubleshooting": "{name}: Causes and Fixes",
                  "Tutorial": "{name}: Step by Step", "Checklist": "{name}: A Checklist"}


def fallback_title(c: dict, ctype: str) -> str:
    """A headline when the AI gave none, so the plan never just repeats the cluster name."""
    main = c.get("main_keyword") or c["keywords"][0]
    if main.split()[0] in Q.QUESTION_WORDS:
        return main[0].upper() + main[1:] + "?"
    return TITLE_PATTERNS.get(ctype, "{name}: A Complete Guide").format(name=c["name"])


def strip_location(title: str, location: str) -> str:
    """'Is Root Canal Painful? Tips from a Newtown, Kolkata Dentist' -> '... Tips from a Dentist'."""
    import re

    parts = [location] + [p.strip() for p in location.split(",") if p.strip()]
    for p in sorted(set(parts), key=len, reverse=True):
        esc = re.escape(p)
        title = re.sub(rf"\s*(?:,\s*|\b(?:in|for|from|near|around|at)\s+){esc}\b", "", title, flags=re.I)
        title = re.sub(rf"\b{esc}(?:'s)?\s*", "", title, flags=re.I)
    title = re.sub(r"\s{2,}", " ", title)
    title = re.sub(r"\s+([?:!,.])", r"\1", title)
    return re.sub(r"[\s:,–-]+$", "", title).strip() or title


def default_type(c: dict, used: list[str]) -> str:
    """A sensible type for a cluster when the AI gave no idea for it."""
    kws = c["keywords"]

    def share(test) -> float:
        return sum(1 for k in kws if test(k)) / max(1, len(kws))

    wants = []
    if share(lambda k: any(w in k.split() for w in Q.COST_WORDS)) >= 0.5:
        wants.append("Cost Guide")
    if share(lambda k: " vs " in f" {k} " or " or " in f" {k} " or "difference" in k) >= 0.5:
        wants.append("Comparison")
    if share(lambda k: k.split()[0] in Q.QUESTION_WORDS) >= 0.5:
        wants.append("FAQs")
    if c.get("intent") == "Informational":
        wants.append("Help Article")
    wants.append("Guide")
    return next((t for t in wants if t in used), used[0])


def generate_ideas(clusters: list[dict], types: list[dict], llm, topic: str, location: str, progress: Progress,
                   batch: int = 6, max_ideas: int = 4, places: list[str] | None = None) -> list[dict]:
    used = [t["type"] for t in types if t["decision"] == "Used" and t["type"] != "Pillar Page"]
    used = used or ["Guide"]
    places = places or (Q.location_tokens(location) if location else [])

    def ask(part: list[dict], got: dict[int, list]) -> None:
        # numbered 1..n inside the batch; answers are matched back by their primary keyword, else by number
        text = "\n".join(
            f"Cluster {n}. {c['name']} [{c['intent']}]{' [local]' if location and local_ok(c, places) else ''} "
            f"(max {ideas_for(c, max_ideas)}) keywords: " + "; ".join(c["keywords"][:14])
            for n, c in enumerate(part, 1))
        owner = {k: c["id"] for c in part for k in c["keywords"]}
        try:
            data = llm.json(SYSTEM, IDEAS_PROMPT.format(
                topic=topic, loc=_loc(location), used=", ".join(used), max_ideas=max_ideas,
                place=location or "a place name", clusters=text, year=Q.current_year()), temperature=0.6)
        except LLMError:
            return
        for item in _list(data, "clusters"):
            if not isinstance(item, dict):
                continue
            for x in item.get("ideas") or []:
                if not isinstance(x, dict) or not str(x.get("title") or "").strip():
                    continue
                cid = owner.get(Q.clean(x.get("primary") or ""))
                if cid is None:
                    try:
                        n = int(item.get("cluster", item.get("id")))
                    except (TypeError, ValueError):
                        continue
                    if not 1 <= n <= len(part):
                        continue
                    cid = part[n - 1]["id"]
                got.setdefault(cid, []).append(x)

    got: dict[int, list] = {}
    for s in range(0, len(clusters), batch):
        part = clusters[s:s + batch]
        progress(f"Stage 5 · content ideas for clusters {s + 1}–{s + len(part)} of {len(clusters)}",
                 0.1 + 0.6 * s / max(1, len(clusters)))
        ask(part, got)
    missing = [c for c in clusters if c["id"] not in got][:30]  # second try, at most 10 extra calls
    for s in range(0, len(missing), 3):
        progress(f"Stage 5 · content ideas: second try for {len(missing)} clusters", 0.7)
        ask(missing[s:s + 3], got)

    ideas: list[dict] = []
    for c in clusters:
        items = got.get(c["id"], [])
        if not items:  # AI gave nothing for this cluster: one main page, type chosen from its keywords
            t = default_type(c, used)
            items = [{"type": t, "title": fallback_title(c, t), "primary": c["main_keyword"]}]
        seen_titles = set()
        for x in items[:ideas_for(c, max_ideas)]:
            ctype = _pick_type(x.get("type"), used)
            title = Q.fix_years(str(x["title"]).strip()[:140])
            if location and not local_ok(c, places):
                title = strip_location(title, location)
            if title.lower() == c["name"].lower():  # a bare cluster name is a topic, not a headline
                title = fallback_title(c, ctype)
            if title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            ideas.append({"cluster_id": c["id"], "type": ctype, "title": title,
                          "ai_primary": Q.clean(x.get("primary") or "")})
    return ideas


def fill_missing_types(ideas: list[dict], clusters: list[dict], types: list[dict], llm, topic: str,
                       location: str) -> None:
    have = {i["type"] for i in ideas}
    missing = [t["type"] for t in types if t["decision"] == "Used" and t["type"] not in have
               and t["type"] != "Pillar Page"]
    if not missing:
        return
    # only clusters with a keyword to spare, so the extra piece gets its own primary keyword
    pieces = {c["id"]: sum(1 for i in ideas if i["cluster_id"] == c["id"]) for c in clusters}
    spare = [c for c in clusters if len(c["keywords"]) > pieces[c["id"]]]
    if not spare:
        return
    clusters = spare
    text = "\n".join(f"{c['id']}. {c['name']} ({c['main_keyword']})" for c in clusters)
    try:
        data = llm.json(SYSTEM, FILL_PROMPT.format(topic=topic, loc=_loc(location), missing=", ".join(missing),
                                                   clusters=text), temperature=0.6)
    except LLMError:
        return
    ids = {c["id"] for c in clusters}
    for x in _list(data, "ideas"):
        if not isinstance(x, dict):
            continue
        t = normalise_type(x.get("type"))
        try:
            cid = int(x.get("cluster_id"))
        except (TypeError, ValueError):
            continue
        title = Q.fix_years(str(x.get("title") or "").strip()[:140])
        if t in missing and cid in ids and title:
            c = next(c for c in clusters if c["id"] == cid)
            if location and not local_ok(c):
                title = strip_location(title, location)
            ideas.append({"cluster_id": cid, "type": t, "title": title, "ai_primary": ""})
            missing.remove(t)


# ---------------- Stage 6: keyword mapping ----------------
def map_keywords(ideas: list[dict], clusters: list[dict], pillars: list[dict], embedder,
                 location: str = "", pillar_pages: bool = True) -> tuple[list[dict], list[dict]]:
    by_id = {c["id"]: c for c in clusters}
    loc = Q.location_tokens(location) if location else []
    all_kw = list(dict.fromkeys(k for c in clusters for k in c["keywords"]))
    kv = dict(zip(all_kw, embedder.embed(all_kw))) if all_kw else {}
    titles = [i["title"] for i in ideas]
    tv = embedder.embed(titles) if titles else np.zeros((0, 1))
    primary_used: set[str] = set()
    rows: list[dict] = []
    # Primary keywords, in order: the cluster's first (main) piece picks first
    order = sorted(range(len(ideas)), key=lambda i: (ideas[i]["cluster_id"], i))
    first_of_cluster: set[int] = set()
    plan_rows: dict[int, dict] = {}
    for i in order:
        idea = ideas[i]
        c = by_id[idea["cluster_id"]]
        role = "Cluster Page" if c["id"] not in first_of_cluster else "Supporting"
        first_of_cluster.add(c["id"])
        free = [k for k in c["keywords"] if k not in primary_used]
        if idea.get("ai_primary") in free:
            prim = idea["ai_primary"]
        elif free:
            sims = np.array([float(kv[k] @ tv[i]) for k in free])
            prim = c["main_keyword"] if role == "Cluster Page" and c["main_keyword"] in free \
                else free[int(np.argmax(sims))]
        else:
            prim = ""
        if not prim and role == "Supporting":
            continue  # an extra piece with no keyword of its own would compete with its cluster page: drop it
        if prim:
            primary_used.add(prim)
        plan_rows[i] = {"cluster_id": c["id"], "cluster": c["name"], "pillar": c.get("pillar", ""),
                        "type": idea["type"], "title": idea["title"], "primary": prim, "secondary": [],
                        "role": role, "source": c.get("source", "Core"),
                        "note": "" if prim else "No unused keyword left in this cluster"}
    # Pillar pages: the pillar's broad keyword (it is the hub, not in any one cluster). They pick after the
    # cluster pages, so a cluster whose main keyword happens to equal the pillar keyword keeps it.
    for p in pillars if pillar_pages else []:
        kw = next((k for k in (p["pillar_keyword"], Q.clean(p["name"]), Q.clean(p["title"]))
                   if k and k not in primary_used), f"{p['pillar_keyword']} guide")
        primary_used.add(kw)
        rows.append({"cluster_id": None, "cluster": "—", "pillar": p["name"], "type": "Pillar Page",
                     "title": p["title"], "primary": kw, "secondary": [], "role": "Pillar Page",
                     "source": p.get("source", "Core")})
    # Secondary keywords: each remaining keyword goes to the closest page of its own cluster
    by_cluster: dict[int, list[int]] = {}
    for i in plan_rows:
        by_cluster.setdefault(plan_rows[i]["cluster_id"], []).append(i)
    leftover = []
    for cid, idx in by_cluster.items():
        c = by_id[cid]
        rest = [k for k in c["keywords"] if k not in primary_used]
        for k in rest:
            ranked = sorted(idx, key=lambda i: -float(kv[k] @ tv[i]))
            for i in ranked:
                if len(plan_rows[i]["secondary"]) < 8:
                    plan_rows[i]["secondary"].append(k)
                    break
            else:
                leftover.append({"keyword": k, "cluster": c["name"],
                                 "suggestion": "Add as an FAQ or a new supporting piece on this cluster"})
    for c in clusters:  # clusters that got no page at all
        if c["id"] not in by_cluster:
            for k in c["keywords"]:
                if k not in primary_used:
                    leftover.append({"keyword": k, "cluster": c["name"], "suggestion": "Cluster has no page yet"})
    ordered = [plan_rows[i] for i in order if i in plan_rows]
    pillar_rank = {p["name"]: n for n, p in enumerate(pillars)}
    ordered.sort(key=lambda r: (pillar_rank.get(r["pillar"], 99), r["cluster_id"], r["role"] != "Cluster Page"))
    # put each pillar page right before its pillar's pages
    final = []
    for p in pillars:
        final += [r for r in rows if r["pillar"] == p["name"]]
        final += [r for r in ordered if r["pillar"] == p["name"]]
    final += [r for r in ordered if r["pillar"] not in pillar_rank]
    for n, r in enumerate(final, 1):
        r["n"] = n
        cl = by_id.get(r["cluster_id"])
        if cl:
            r["intent"] = cl["intent"]
            r["funnel"] = cl.get("funnel") or Q.FUNNEL[cl["intent"]]  # the stage most of its keywords are at
        else:  # pillar page: a hub for all its pillar's pages, so it takes their most common intent and stage
            mine = [by_id[x["cluster_id"]] for x in ordered if x["pillar"] == r["pillar"]]
            intents = [c["intent"] for c in mine]
            funnels = [c.get("funnel") or Q.FUNNEL[c["intent"]] for c in mine]
            r["intent"] = max(sorted(set(intents)), key=intents.count) if intents else "Informational"
            r["funnel"] = max(sorted(set(funnels)), key=funnels.count) if funnels else "TOFU"
        r.setdefault("funnel", Q.FUNNEL[r["intent"]])
        r.setdefault("note", "")
    return final, leftover


def build_content_plan(project: dict, llm, embedder, progress: Progress | None = None) -> tuple[dict, dict]:
    progress = progress or (lambda m, f: None)
    topic, location = project["topic"], project.get("location", "")
    pillars = project["stage4"]["pillars"]
    clusters = all_clusters(project)
    progress("Stage 5 · choosing content types", 0.02)
    types = decide_types(clusters, pillars, for_stage(llm, "types"), topic, location)
    places = Q.local_area_tokens(project["stage2"]["keywords"], location) if location else []
    ideas = generate_ideas(clusters, types, for_stage(llm, "ideas"), topic, location, progress, places=places)
    progress("Stage 5 · filling content types that have no page yet", 0.75)
    fill_missing_types(ideas, clusters, types, for_stage(llm, "ideas"), topic, location)
    pillar_pages = any(t["type"] == "Pillar Page" and t["decision"] == "Used" for t in types)
    progress("Stage 6 · mapping keywords to pages", 0.85)
    rows, leftover = map_keywords(ideas, clusters, pillars, embedder, location, pillar_pages)
    counts = {t: sum(1 for r in rows if r["type"] == t) for t in TYPE_NAMES}
    for t in types:
        t["pages"] = counts.get(t["type"], 0)
    progress("Stage 6 · done", 1.0)
    return {"types": types}, {"rows": rows, "leftover": leftover}
