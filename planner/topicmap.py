"""Stage 4: topical map. Topic → Pillar → Cluster (page) → Subtopics (H2 headings).

1. Pillars: the code groups pages by meaning (embeddings of each page's keywords, k-means into 4–8 groups), then
   the AI only names each group and writes its pillar keyword and hub page title. Small models are unreliable at
   sorting 40+ items, but good at naming a group.
2. The AI writes 3–6 subtopics (H2s) per page, in small batches.
3. Gap check: the AI lists important pages that are missing. They're added as "AI-suggested" pages in the closest
   pillar, after dropping anything that repeats an existing page.

Every batch shows the AI its items numbered 1…n and the code maps those numbers back, so a model that restarts
numbering in each batch can't attach answers to the wrong page.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import quality as Q
from .llm import LLMError

Progress = Callable[[str, float], None]
SYSTEM = "You are an SEO content strategist who builds topical maps for topical authority. You answer with JSON only."

PILLAR_PROMPT = """Business / topic: {topic}{loc}

The planned pages have been grouped into {k} pillars (main areas of the business). For each group give:
- name: short pillar name (2 to 5 words, Title Case), e.g. "Dental Treatments", "Costs & Payment", "Kids' Dental Care".
  Names must be different from each other and describe the whole group.
- pillar_keyword: the broad search term for the pillar's hub page (lowercase){loc_rule}.
- pillar_title: the hub page title.

{groups}

Return JSON only:
{{"pillars": [{{"group": 1, "name": "...", "pillar_keyword": "...", "pillar_title": "..."}}]}}"""

SUBTOPIC_PROMPT = """Business / topic: {topic}{loc}

For each page below, list 3 to 6 subtopics: the H2 headings that page should cover so it fully answers ITS OWN
keywords. Write headings a reader would see in a good article (e.g. "What happens during the procedure",
"How long recovery takes", "Signs you need to see a dentist"), not the keywords copied as they are.
Short (3 to 8 words), Sentence case, no numbering. Use the page numbers exactly as given.

{pages}

Return JSON only: {{"pages": [{{"page": 1, "subtopics": ["...", "..."]}}]}}"""

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
    v = data.get(key)
    return v if isinstance(v, list) else []


def _num(x) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def pillar_count(n_clusters: int) -> int:
    return int(min(8, max(2 if n_clusters < 8 else 4, round((n_clusters / 1.5) ** 0.5))))


def cluster_vectors(clusters: list[dict], embedder, location: str) -> np.ndarray:
    """One vector per page: the mean meaning of its keywords (location and question words removed)."""
    from .cluster import topical_text

    loc = Q.location_tokens(location) if location else []
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


def name_pillars(groups: list[list[int]], clusters: list[dict], vecs: np.ndarray, llm, topic: str,
                 location: str) -> list[dict]:
    lines = []
    for n, g in enumerate(groups, 1):
        cen = vecs[g].mean(axis=0)
        central = sorted(g, key=lambda i: -float(vecs[i] @ cen))  # most typical pages first
        lines.append(f"Group {n}: " + "; ".join(clusters[i]["name"] for i in central[:12]))
    loc_rule = f"; include {location} only for a local pillar" if location else ""
    named: dict[int, dict] = {}
    try:
        data = llm.json(SYSTEM, PILLAR_PROMPT.format(topic=topic, loc=_loc(location), k=len(groups),
                                                     loc_rule=loc_rule, groups="\n".join(lines)), temperature=0.3)
        for p in _as_list(data, "pillars"):
            n = _num(p.get("group")) if isinstance(p, dict) else None
            if n and 1 <= n <= len(groups) and n not in named and str(p.get("name") or "").strip():
                named[n] = p
    except LLMError:
        pass
    pillars, seen = [], set()
    for n, g in enumerate(groups, 1):
        cen = vecs[g].mean(axis=0)
        top = clusters[max(g, key=lambda i: float(vecs[i] @ cen))]
        p = named.get(n, {})
        name = str(p.get("name") or top["name"]).strip()[:60]
        if name.lower() in seen:
            name = f"{name} ({top['name']})"[:60]
        seen.add(name.lower())
        pillars.append({"name": name, "pillar_keyword": Q.clean(p.get("pillar_keyword") or name),
                        "title": str(p.get("pillar_title") or name).strip()[:120],
                        "cluster_ids": [clusters[i]["id"] for i in g], "source": "Core"})
    return pillars


def write_subtopics(clusters: list[dict], llm, topic: str, location: str, progress: Progress,
                    batch: int = 8) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for s in range(0, len(clusters), batch):
        part = clusters[s:s + batch]
        progress(f"Stage 4 · subtopics for pages {s + 1}–{s + len(part)} of {len(clusters)}",
                 0.25 + 0.5 * s / max(1, len(clusters)))
        pages = "\n".join(f"Page {n}. {c['name']}: " + "; ".join(c["keywords"][:10]) for n, c in enumerate(part, 1))
        try:
            data = llm.json(SYSTEM, SUBTOPIC_PROMPT.format(topic=topic, loc=_loc(location), pages=pages),
                            temperature=0.4)
        except LLMError:
            continue
        for item in _as_list(data, "pages"):
            n = _num(item.get("page", item.get("id"))) if isinstance(item, dict) else None
            if not n or not 1 <= n <= len(part):
                continue
            subs = [str(x).strip().rstrip(".:") for x in (item.get("subtopics") or []) if str(x).strip()]
            page = part[n - 1]
            copies = {page["name"].lower()} | set(page["keywords"])  # a heading that only repeats the page is no help
            subs = list(dict.fromkeys(x[0].upper() + x[1:] for x in subs
                                      if 2 <= len(x) <= 80 and x.lower() not in copies))[:6]
            if subs:
                out.setdefault(part[n - 1]["id"], subs)
    for c in clusters:  # fallback: the page's own keywords as headings
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


def build_topical_map(project: dict, llm, embedder, progress: Progress | None = None) -> dict:
    progress = progress or (lambda m, f: None)
    topic, location = project["topic"], project.get("location", "")
    clusters = [dict(c, source="Core") for c in project["stage3"]["clusters"]]
    if not clusters:
        raise LLMError("There are no clusters to map. Run Stages 2–3 first.")
    progress("Stage 4 · grouping pages into pillars", 0.05)
    vecs = cluster_vectors(clusters, embedder, location)
    groups = group_into_pillars(vecs, pillar_count(len(clusters)))
    pillars = name_pillars(groups, clusters, vecs, llm, topic, location)
    subtopics = write_subtopics(clusters, llm, topic, location, progress)
    progress("Stage 4 · gap check: looking for missing pages", 0.8)
    all_kw = [k["keyword"] for k in project["stage2"]["keywords"]]
    gaps = find_gaps(pillars, clusters, all_kw, llm, embedder, topic, location)
    if gaps:
        cen = np.array([vecs[g].mean(axis=0) for g in groups])
        cen /= np.linalg.norm(cen, axis=1, keepdims=True)
        gv = cluster_vectors(gaps, embedder, location)
        for g, v in zip(gaps, gv):
            named = next((p for p in pillars if p["name"].lower() == g["pillar"].lower()), None)
            p = named or pillars[int(np.argmax(cen @ v))]
            p["cluster_ids"].append(g["id"])
            subtopics[g["id"]] = g.pop("subtopics")
            g.pop("pillar")
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
