"""Stage 4: topical map. Topic → Pillar → Cluster (page) → Subtopics (H2 headings).

1. Pillars: the AI proposes the business's main areas (name + the subjects each covers); the code then places every
   page in the closest one by meaning. Small models are unreliable at sorting 100+ pages, but good at naming a
   business's main service areas. Without a usable AI answer, pages are grouped by k-means and named by their
   most shared words.
2. The AI writes 3–6 subtopics (H2s) per page, in small batches; pages a batch missed get a second try.
3. Gap check: the AI lists important pages that are missing. They're added as "AI-suggested" pages in the closest
   pillar, after dropping anything that repeats an existing page.

Every batch shows the AI its items numbered 1…n and asks it to copy each item's name back; answers are matched by
that name first and the number second, so a model that skips one item can't shift answers onto the wrong page.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import quality as Q
from .llm import LLMError, for_stage, get_list

Progress = Callable[[str, float], None]
SYSTEM = "You are an SEO content strategist who builds topical maps for topical authority. You answer with JSON only."

PILLAR_PROMPT = """Business / topic: {topic}{loc}

These are the {n} pages planned for the site:
{pages}

Split the site into {k_min} to {k_max} pillars: the main areas of this business that a customer would recognise,
such as a service line or a customer need (for a dental clinic, e.g. "Gum Care", "Braces & Aligners",
"Dental Implants & Dentures", "Emergency Dental Care", "Cosmetic Dentistry", "Costs & Payment").
Rules:
- Every page above must fit one pillar. Together the pillars cover all the pages.
- A pillar is a broad area, never one narrow page, one symptom or one neighbourhood.
- Names are different from each other and don't overlap in meaning.{loc_pillar}
For each pillar give:
- name: 2 to 5 words, Title Case.
- covers: 6 to 12 short phrases naming the subjects it holds (taken from the pages above).
- pillar_keyword: the broad search term for the pillar's hub page (lowercase){loc_rule}.
- pillar_title: the hub page title.

Return JSON only:
{{"pillars": [{{"name": "...", "covers": ["...", "..."], "pillar_keyword": "...", "pillar_title": "..."}}]}}"""

SUBTOPIC_PROMPT = """Business / topic: {topic}{loc}

For each page below, list 3 to 6 subtopics: the H2 headings that page should cover so it fully answers ITS OWN
keywords. Write headings a reader would see in a good article (e.g. "What happens during the procedure",
"How long recovery takes", "Signs you need to see a dentist"), not the keywords copied as they are.
Short (3 to 8 words), Sentence case, no numbering. Use the page numbers exactly as given, copy each page's name
as "name", and answer for every page without skipping any.

{pages}

Return JSON only: {{"pages": [{{"page": 1, "name": "...", "subtopics": ["...", "..."]}}]}}"""

GAP_PROMPT = """Business / topic: {topic}{loc}

Current topical map (pillar: pages):
{map}

To build topical authority, the site must cover this topic completely. List important page topics that are
MISSING from the map: things customers search for that no page above covers. At most {max_gaps}.
Do not repeat or rename existing pages. For each missing page give:
- pillar: the existing pillar it belongs to
- name: page topic (2 to 6 words, Title Case)
- intent: Informational, Commercial, Transactional or Local
- keywords: 3 to 6 realistic search queries for it (lowercase)
- subtopics: 3 to 5 H2 headings

Return JSON only: {{"missing": [{{"pillar": "...", "name": "...", "intent": "...", "keywords": ["..."], "subtopics": ["..."]}}]}}"""


def _loc(location: str) -> str:
    return f"\nLocation: {location}" if location else ""


def _as_list(data: dict, key: str) -> list:
    return get_list(data, key)


def pillar_count(n_clusters: int) -> int:
    return int(min(8, max(2 if n_clusters < 8 else 4, round((n_clusters / 1.5) ** 0.5))))


def cluster_vectors(clusters: list[dict], embedder, location: str, places: list[str] | None = None) -> np.ndarray:
    """One vector per page: the mean meaning of its keywords (location, area names and question words removed)."""
    from .cluster import topical_text

    loc = places or (Q.location_tokens(location) if location else [])
    texts = list(dict.fromkeys(topical_text(k, loc) for c in clusters for k in c["keywords"][:20]))
    vec = dict(zip(texts, embedder.embed(texts)))
    out = []
    for c in clusters:
        v = np.mean([vec[topical_text(k, loc)] for k in c["keywords"][:20]], axis=0)
        out.append(v / (np.linalg.norm(v) or 1))
    return np.array(out, dtype=np.float32)


def group_into_pillars(vecs: np.ndarray, k: int) -> list[list[int]]:
    """k-means on page vectors; returns groups of page indexes, biggest first, none empty."""
    k = max(1, min(k, len(vecs)))
    if k == 1:
        return [list(range(len(vecs)))]
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vecs)
    groups = [[i for i, lab in enumerate(labels) if lab == g] for g in range(k)]
    return sorted((g for g in groups if g), key=len, reverse=True)


def propose_pillars(clusters: list[dict], llm, topic: str, location: str, k: int) -> list[dict]:
    """Ask the AI for the business's main areas (pillars). It only proposes them; the code assigns the pages."""
    loc_rule = f"; include {location} only for a local pillar" if location else ""
    loc_pillar = (f"\n- If several pages are about finding or choosing a provider in an area, put them in one pillar "
                  f"about choosing a provider in {location}.") if location else ""
    pages = "; ".join(c["name"] for c in clusters[:150])
    try:
        data = llm.json(SYSTEM, PILLAR_PROMPT.format(
            topic=topic, loc=_loc(location), n=len(clusters), pages=pages, k_min=max(2, k - 2), k_max=k + 1,
            loc_rule=loc_rule, loc_pillar=loc_pillar), temperature=0.3)
    except LLMError:
        return []
    out, seen = [], set()
    for p in _as_list(data, "pillars"):
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip()[:60]
        covers = [str(x).strip() for x in (p.get("covers") or []) if str(x).strip()][:15]
        if len(name) < 3 or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append({"name": name, "covers": covers, "pillar_keyword": Q.clean(p.get("pillar_keyword") or name),
                    "title": str(p.get("pillar_title") or name).strip()[:120]})
    return out if len(out) >= 2 else []


def assign_to_pillars(proposed: list[dict], vecs: np.ndarray, embedder, location: str) -> list[list[int]]:
    """Each page goes to the pillar whose subjects it is closest to in meaning.

    A pillar is represented by its name and each subject it covers; a page's score is its best match among them,
    so a pillar covering "gum disease; bleeding gums; receding gums" catches every gum page.
    """
    from .cluster import topical_text

    loc = Q.location_tokens(location) if location else []
    texts, owner = [], []
    for pi, p in enumerate(proposed):
        for t in [p["name"]] + p["covers"]:
            texts.append(topical_text(t.lower(), loc) or t.lower())
            owner.append(pi)
    tv = embedder.embed(texts)
    sims = vecs @ tv.T  # pages x subjects
    owner_arr = np.array(owner)
    best = np.full((len(vecs), len(proposed)), -1.0, dtype=np.float32)
    for pi in range(len(proposed)):
        best[:, pi] = sims[:, owner_arr == pi].max(axis=1)
    pick = best.argmax(axis=1)
    return [[i for i in range(len(vecs)) if pick[i] == pi] for pi in range(len(proposed))]


def fallback_pillar_name(g: list[int], clusters: list[dict], topic: str = "", places: list[str] = ()) -> str:
    """A broad name for a group when the AI gave none: the two words most of its pages share ("Braces & Aligners"),
    leaving out filler, the topic's own words and place names."""
    from .cluster import CLUSTER_STOP, GENERIC_WORDS, NOT_SUBJECT

    skip = CLUSTER_STOP | NOT_SUBJECT | GENERIC_WORDS | set(Q.clean(topic).split()) | \
        {w for p in places for w in p.split()}
    seen_in: dict[str, int] = {}  # words on a large share of ALL pages ("teeth", "dentist") name nothing
    for c in clusters:
        for w in {w for k in c["keywords"] for w in k.split()}:
            seen_in[w] = seen_in.get(w, 0) + 1
    skip |= {w for w, n in seen_in.items() if n > max(3, 0.08 * len(clusters))}
    count: dict[str, int] = {}
    for i in g:
        for w in {w for k in clusters[i]["keywords"] for w in k.split() if w not in skip and len(w) > 3}:
            count[w] = count.get(w, 0) + 1
    top = [w for w, _ in sorted(count.items(), key=lambda x: (-x[1], x[0]))[:2]]
    return " & ".join(w.title() for w in top) or clusters[g[0]]["name"]


def make_pillars(clusters: list[dict], vecs: np.ndarray, llm, embedder, topic: str, location: str,
                 places: list[str] | None = None) -> list[dict]:
    """Pillars = the business's main areas. The AI proposes them (it knows what a dental clinic's main services
    are); the code places every page in the closest one by meaning. If the AI gives no usable answer, pages are
    grouped by meaning (k-means) and each group gets a name from its most shared words."""
    k = pillar_count(len(clusters))
    proposed = propose_pillars(clusters, llm, topic, location, k)
    if proposed:
        groups = assign_to_pillars(proposed, vecs, embedder, location)
        pairs = [(p, g) for p, g in zip(proposed, groups) if g]
    else:
        pairs = []
        places = places or (Q.location_tokens(location) if location else [])
        for g in group_into_pillars(vecs, k):
            name = fallback_pillar_name(g, clusters, topic, places)
            pairs.append(({"name": name, "pillar_keyword": Q.clean(name), "title": name}, g))
    pairs.sort(key=lambda x: -len(x[1]))
    pillars, seen = [], set()
    for p, g in pairs:
        name = p["name"]
        if name.lower() in seen:
            name = f"{name} {len(pillars) + 1}"
        seen.add(name.lower())
        pillars.append({"name": name, "pillar_keyword": p["pillar_keyword"] or Q.clean(name), "title": p["title"],
                        "cluster_ids": [clusters[i]["id"] for i in g], "source": "Core"})
    return pillars


def _subtopic_batch(part: list[dict], out: dict[int, list[str]], llm, topic: str, location: str) -> None:
    from .cluster import match_answer

    pages = "\n".join(f"Page {n}. {c['name']}: " + "; ".join(c["keywords"][:10]) for n, c in enumerate(part, 1))
    try:
        data = llm.json(SYSTEM, SUBTOPIC_PROMPT.format(topic=topic, loc=_loc(location), pages=pages),
                        temperature=0.4)
    except LLMError:
        return
    for item in _as_list(data, "pages"):
        if not isinstance(item, dict):
            continue
        page = match_answer(item, part, "name", ("page", "id"))
        if page is None or page["id"] in out:
            continue
        subs = [str(x).strip().rstrip(".:") for x in (item.get("subtopics") or []) if str(x).strip()]
        copies = {page["name"].lower()} | set(page["keywords"])  # a heading that only repeats the page is no help
        subs = list(dict.fromkeys(x[0].upper() + x[1:] for x in subs
                                  if 2 <= len(x) <= 80 and x.lower() not in copies))[:6]
        if subs:
            out[page["id"]] = subs


def write_subtopics(clusters: list[dict], llm, topic: str, location: str, progress: Progress,
                    batch: int = 8) -> dict[int, list[str]]:
    """H2s for every page, in batches. Pages a batch missed (a failed call, a skipped answer) are asked again in
    small batches, so only a page the AI never answers for falls back to its own keywords as headings."""
    out: dict[int, list[str]] = {}
    for s in range(0, len(clusters), batch):
        part = clusters[s:s + batch]
        progress(f"Stage 4 · subtopics for pages {s + 1}–{s + len(part)} of {len(clusters)}",
                 0.25 + 0.5 * s / max(1, len(clusters)))
        _subtopic_batch(part, out, llm, topic, location)
    missing = [c for c in clusters if c["id"] not in out][:40]  # at most 10 extra calls
    for s in range(0, len(missing), 4):
        progress(f"Stage 4 · subtopics: second try for {len(missing)} pages", 0.75)
        _subtopic_batch(missing[s:s + 4], out, llm, topic, location)
    for c in clusters:  # last resort: the page's own keywords as headings
        out.setdefault(c["id"], [k[0].upper() + k[1:] for k in c["keywords"][1:5]] or [c["name"]])
    return out


def find_gaps(pillars: list[dict], clusters: list[dict], all_keywords: list[str], llm, embedder, topic: str,
              location: str, max_gaps: int = 12) -> list[dict]:
    by_id = {c["id"]: c for c in clusters}
    text = "\n".join(f"{p['name']}: " + "; ".join(by_id[i]["name"] for i in p["cluster_ids"]) for p in pillars)
    try:
        data = llm.json(SYSTEM, GAP_PROMPT.format(topic=topic, loc=_loc(location), map=text, max_gaps=max_gaps),
                        temperature=0.5)
    except LLMError:
        return []
    existing_names = [c["name"] for c in clusters]
    existing_v = embedder.embed(existing_names)
    taken = set(all_keywords)
    next_id = max(by_id) + 1
    gaps: list[dict] = []
    for g in _as_list(data, "missing")[:max_gaps * 2]:
        if not isinstance(g, dict):
            continue
        name = str(g.get("name") or "").strip()[:60]
        if len(name) < 3 or any(x["name"].lower() == name.lower() for x in gaps):
            continue
        v = embedder.embed([name])[0]
        if float(np.max(existing_v @ v)) >= 0.9:
            continue  # repeats an existing page
        kws = []
        for k in g.get("keywords") or []:
            k = Q.clean(k)
            if Q.is_valid(k) and k not in taken and k not in kws:
                kws.append(k)
        if not kws:
            kws = [Q.clean(name)]
        taken.update(kws)
        intent = str(g.get("intent") or "").title()
        intent = intent if intent in Q.INTENTS else "Informational"
        subs = [str(x).strip() for x in (g.get("subtopics") or []) if str(x).strip()][:5]
        gaps.append({"id": next_id, "name": name, "intent": intent, "keywords": kws, "main_keyword": kws[0],
                     "size": len(kws), "source": "AI-suggested", "pillar": str(g.get("pillar") or "").strip(),
                     "subtopics": subs or [k[0].upper() + k[1:] for k in kws[1:5]] or [name]})
        existing_v = np.vstack([existing_v, v[None, :]])
        next_id += 1
        if len(gaps) >= max_gaps:
            break
    return gaps


REVIEW_PROMPT = """Business / topic: {topic}{loc}

Pages were placed into pillars (the main areas of the site) by meaning. Check every page: does it clearly belong to
a DIFFERENT pillar? Move only clear misfits (e.g. "Best Dentist in Kolkata" under "Cosmetic Dentistry" belongs under
a pillar about choosing a dentist). Leave pages that fit reasonably well. Copy page and pillar names exactly.

{pillars}

Return JSON only: {{"moves": [{{"page": "...", "to": "..."}}]}}  (an empty list when every page fits)"""


def review_pillars(pillars: list[dict], clusters: list[dict], llm, topic: str, location: str) -> int:
    """The AI checks the placement of every page and moves clear misfits to the pillar they belong to.

    Pages are placed by meaning first (fast and complete), but similar words can mislead it ("sedation dentistry"
    near "routine checkups"). Asked up to config.AI_TRIES times; returns how many pages moved."""
    import time

    from . import config as C
    from .llm import is_fatal

    if len(pillars) < 2:
        return 0
    by_id = {c["id"]: c for c in clusters}
    text = "\n\n".join(f"Pillar: {p['name']}\n" + "\n".join(f"- {by_id[i]['name']}" for i in p["cluster_ids"]
                                                             if i in by_id) for p in pillars)
    data = None
    for attempt in range(C.AI_TRIES):
        if attempt:
            time.sleep(C.AI_RETRY_PAUSE)
        try:
            data = llm.json(SYSTEM, REVIEW_PROMPT.format(topic=topic, loc=_loc(location), pillars=text),
                            temperature=0.2)
        except LLMError as e:
            if is_fatal(e):
                raise
            continue
        if isinstance(data.get("moves"), list) or get_list(data, "moves"):
            break
        data = None
    if data is None:
        return 0
    pillar_by_name = {p["name"].lower(): p for p in pillars}
    page_by_name = {c["name"].lower(): c["id"] for c in clusters}
    moved = 0
    for m in get_list(data, "moves"):
        if not isinstance(m, dict):
            continue
        cid = page_by_name.get(str(m.get("page") or "").strip().lower())
        to = pillar_by_name.get(str(m.get("to") or "").strip().lower())
        src = next((p for p in pillars if cid in p["cluster_ids"]), None)
        if cid is None or to is None or src is None or src is to or len(src["cluster_ids"]) < 2:
            continue  # unknown names, no real move, or it would empty a pillar
        src["cluster_ids"].remove(cid)
        to["cluster_ids"].append(cid)
        moved += 1
    return moved


def build_topical_map(project: dict, llm, embedder, progress: Progress | None = None) -> dict:
    progress = progress or (lambda m, f: None)
    topic, location = project["topic"], project.get("location", "")
    clusters = [dict(c, source="Core") for c in project["stage3"]["clusters"]]
    if not clusters:
        raise LLMError("There are no clusters to map. Run Stages 2–3 first.")
    progress("Stage 4 · grouping pages into pillars", 0.05)
    places = Q.local_area_tokens(project["stage2"]["keywords"], location) if location else []
    vecs = cluster_vectors(clusters, embedder, location, places)
    pillars = make_pillars(clusters, vecs, for_stage(llm, "pillars"), embedder, topic, location, places)
    index = {c["id"]: i for i, c in enumerate(clusters)}
    groups = [[index[cid] for cid in p["cluster_ids"]] for p in pillars]
    subtopics = write_subtopics(clusters, for_stage(llm, "outline"), topic, location, progress)
    progress("Stage 4 · gap check: looking for missing pages", 0.8)
    all_kw = [k["keyword"] for k in project["stage2"]["keywords"]]
    gaps = find_gaps(pillars, clusters, all_kw, for_stage(llm, "pillars"), embedder, topic, location)
    if gaps:
        cen = np.array([vecs[g].mean(axis=0) for g in groups])
        cen /= np.linalg.norm(cen, axis=1, keepdims=True)
        gv = cluster_vectors(gaps, embedder, location, places)
        for g, v in zip(gaps, gv):
            named = next((p for p in pillars if p["name"].lower() == g["pillar"].lower()), None)
            p = named or pillars[int(np.argmax(cen @ v))]
            p["cluster_ids"].append(g["id"])
            subtopics[g["id"]] = g.pop("subtopics")
            g.pop("pillar")
    progress("Stage 4 · checking every page sits in the right pillar", 0.9)
    review_pillars(pillars, clusters + gaps, for_stage(llm, "pillars"), topic, location)
    pillars = [p for p in pillars if p["cluster_ids"]]
    for i, p in enumerate(pillars, 1):
        p["id"] = i
    progress("Stage 4 · done", 1.0)
    return {"pillars": pillars, "gap_clusters": gaps, "subtopics": {str(k): v for k, v in subtopics.items()}}


def all_clusters(project: dict) -> list[dict]:
    """Stage 3 clusters plus the AI-suggested ones from the gap check, each with its pillar name."""
    s4 = project.get("stage4") or {}
    core = [dict(c, source="Core") for c in (project.get("stage3") or {}).get("clusters", [])]
    allc = core + [dict(c) for c in s4.get("gap_clusters", [])]
    pillar_of = {cid: p["name"] for p in s4.get("pillars", []) for cid in p["cluster_ids"]}
    subs = s4.get("subtopics", {})
    for c in allc:
        c["pillar"] = pillar_of.get(c["id"], "")
        c["subtopics"] = subs.get(str(c["id"]), [])
    return allc
