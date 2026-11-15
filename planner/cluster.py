"""Stage 3: remove meaningless / generic keywords, then group the rest into clusters.

Each cluster holds one kind of keyword: one theme and one intent. There is no minimum or maximum
size; a cluster can have 2 keywords or 40. Each cluster later becomes one page.

How: embeddings of what each keyword is about (question words and location removed) are grouped per intent,
then the AI names each cluster and flags keywords that don't belong, which move to a better cluster.
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np

from . import quality as Q
from .llm import LLMError

GENERIC_WORDS = {
    "best", "top", "good", "near", "me", "service", "services", "online", "cheap", "price", "cost", "info",
    "company", "companies", "shop", "store", "list", "near-me", "nearby", "local", "agency", "agencies",
    "provider", "providers", "expert", "experts", "professional", "the", "a", "in", "for", "of", "and", "my",
}

Progress = Callable[[str, float], None]


def _stem(w: str) -> str:
    return re.sub(r"(ies|es|s)$", "", w) if len(w) > 4 else w


def filter_keywords(rows: list[dict], topic: str) -> tuple[list[dict], list[dict]]:
    """Drop single words and generic two-word terms ("dental clinic", "wall decor").

    Two-word terms that are specific ("invisible braces", "dentist newtown") are kept.
    """
    topic_words = {_stem(w) for w in Q.clean(topic).split()}
    generic = topic_words | {_stem(w) for w in GENERIC_WORDS}
    kept, removed = [], []
    for r in rows:
        words = r["keyword"].split()
        if len(words) < 2:
            removed.append({**r, "reason": "Single word"})
        elif len(words) == 2 and all(_stem(w) in generic for w in words):
            removed.append({**r, "reason": "Generic two-word term"})
        else:
            kept.append(r)
    return kept, removed


def _groups(vecs: np.ndarray, distance: float) -> list[list[int]]:
    if len(vecs) == 1:
        return [[0]]
    from sklearn.cluster import AgglomerativeClustering

    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=distance).fit_predict(vecs)
    out: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        out.setdefault(int(lab), []).append(i)
    return list(out.values())


def _closest(i: int, groups: list[list[int]], vecs: np.ndarray) -> tuple[int, float]:
    """Index of the group whose centre is closest to item i, and that distance."""
    best, best_d = -1, 2.0
    for gi, g in enumerate(groups):
        if not g:
            continue
        cen = vecs[g].mean(axis=0)
        d = 1 - float(vecs[i] @ (cen / (np.linalg.norm(cen) or 1)))
        if d < best_d:
            best, best_d = gi, d
    return best, best_d


def _attach_singletons(groups: list[list[int]], vecs: np.ndarray, distance: float) -> list[list[int]]:
    """A lone keyword joins the closest cluster if it is reasonably close; otherwise it stays its own page."""
    big = [g for g in groups if len(g) > 1]
    lone = [g[0] for g in groups if len(g) == 1]
    if not big:
        return groups
    rest = []
    for i in lone:
        j, d = _closest(i, big, vecs)
        if d <= distance * 1.15:
            big[j].append(i)
        else:
            rest.append([i])
    return big + rest


CLUSTER_STOP = set(Q.QUESTION_WORDS) | Q.SOFT_WORDS | {"i", "you", "your", "it", "do", "get", "should", "much",
                                                       "many", "long", "often", "best", "near", "me"}


def topical_text(kw: str, loc_tokens: list[str]) -> str:
    """What a keyword is about, without question words or the location, so phrasing doesn't drive the clusters."""
    for t in loc_tokens:
        kw = re.sub(rf"\b{re.escape(t)}\b", " ", kw)
    words = [w for w in kw.split() if w not in CLUSTER_STOP]
    return " ".join(words) or kw.strip()


def _main_keyword(idx: list[int], vecs: np.ndarray, kws: list[str]) -> str:
    c = vecs[idx].mean(axis=0)
    sims = vecs[idx] @ c
    top = [idx[i] for i in np.argsort(-sims)[:3]]
    return min((kws[i] for i in top), key=lambda k: len(k.split()))


def _fill(c: dict, vecs: np.ndarray, kws: list[str]) -> dict:
    """(Re)build a cluster's keyword list from its members; keeps an existing name."""
    m = c["_members"]
    main = _main_keyword(m, vecs, kws)
    c["main_keyword"] = main
    c["keywords"] = [main] + sorted((kws[i] for i in m if kws[i] != main), key=lambda k: (len(k), k))
    c["size"] = len(m)
    c.setdefault("name", main.title())
    return c


NAME_SYSTEM = "You are an SEO content strategist. You answer with JSON only."
NAME_PROMPT = """Topic: {topic}{loc}
Below are keyword clusters. Each cluster will become one web page.
For each cluster give:
- "name": a short, specific page topic (2 to 6 words, Title Case), e.g. "Root Canal Cost", "Tooth Pain at Night".
  Names must be different from each other.
- "outliers": keywords copied exactly from that cluster that are about a different subject and would not
  belong on the same page. Usually there are none: use [].

{clusters}

Return JSON only: {{"clusters": [{{"id": 1, "name": "...", "outliers": []}}, ...]}}"""


def name_clusters(clusters: list[dict], llm, topic: str, location: str, progress: Progress | None = None,
                  batch: int = 15) -> dict[int, list[str]]:
    """Name clusters in place; return {cluster id: keywords the AI says don't belong}."""
    progress = progress or (lambda m, f: None)
    outliers: dict[int, list[str]] = {}
    for start in range(0, len(clusters), batch):
        part = clusters[start:start + batch]
        progress(f"Stage 3 · naming and checking clusters {start + 1}–{start + len(part)} of {len(clusters)}",
                 0.5 + 0.4 * start / max(1, len(clusters)))
        text = "\n".join(f"{n}. " + "; ".join(c["keywords"][:15]) for n, c in enumerate(part, 1))
        try:
            data = llm.json(NAME_SYSTEM, NAME_PROMPT.format(
                topic=topic, loc=f"\nLocation: {location}" if location else "", clusters=text), temperature=0.3)
        except LLMError:
            continue  # keep fallback names
        items = data.get("clusters") if isinstance(data.get("clusters"), list) else []
        by_num = {n: c for n, c in enumerate(part, 1)}  # numbered 1..n inside the batch
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                c = by_num.get(int(item.get("id")))
            except (TypeError, ValueError):
                continue
            if c is None:
                continue
            name = str(item.get("name") or "").strip()
            if 2 <= len(name) <= 60:
                c["name"] = name
            out = item.get("outliers")
            if isinstance(out, list) and c["size"] >= 3:
                shown = c["keywords"][:15]
                bad = list(dict.fromkeys(k for k in (str(x).strip().lower() for x in out) if k in shown))
                if 0 < len(bad) < c["size"] / 2:  # never empty a cluster on the AI's word
                    outliers[c["id"]] = bad
    return outliers


def _move_outliers(clusters: list[dict], outliers: dict[int, list[str]], vecs: np.ndarray, kws: list[str],
                   distance: float) -> int:
    """Move each flagged keyword to the closest other cluster of the same intent, or give it its own page."""
    index = {k: i for i, k in enumerate(kws)}
    by_id = {c["id"]: c for c in clusters}
    new_id = max(by_id) + 1
    moved = 0
    for cid, bad in outliers.items():
        src = by_id[cid]
        for kw in bad:
            i = index[kw]
            if i not in src["_members"]:
                continue
            src["_members"].remove(i)
            others = [c for c in clusters if c["intent"] == src["intent"] and c is not src and c["_members"]]
            j, d = _closest(i, [c["_members"] for c in others], vecs)
            if j >= 0 and d <= distance * 1.15:
                others[j]["_members"].append(i)
            else:
                clusters.append({"id": new_id, "intent": src["intent"], "_members": [i], "name": kw.title()})
                new_id += 1
            moved += 1
    for c in clusters:
        if c["_members"]:
            _fill(c, vecs, kws)
    return moved


def _unique_names(clusters: list[dict]) -> None:
    seen: dict[str, int] = {}
    for c in clusters:
        k = c["name"].lower()
        if k in seen:
            seen[k] += 1
            c["name"] = f"{c['name']} ({c['intent']})" if seen[k] == 2 else f"{c['name']} {seen[k]}"
        else:
            seen[k] = 1


def cluster_keywords(rows: list[dict], embedder, llm=None, topic: str = "", location: str = "",
                     progress: Progress | None = None) -> list[dict]:
    progress = progress or (lambda m, f: None)
    if not rows:
        return []
    progress("Stage 3 · grouping keywords", 0.1)
    kws = [r["keyword"] for r in rows]
    loc = Q.location_tokens(location) if location else []
    vecs = embedder.embed([topical_text(k, loc) for k in kws])
    d = embedder.cluster_distance
    clusters = []
    for intent in Q.INTENTS:  # one intent per cluster
        idx = [i for i, r in enumerate(rows) if r.get("intent", "Informational") == intent]
        if not idx:
            continue
        sub = vecs[idx]
        for g in _attach_singletons(_groups(sub, d), sub, d):
            clusters.append(_fill({"intent": intent, "_members": [idx[i] for i in g]}, vecs, kws))
    clusters.sort(key=lambda c: (-c["size"], c["name"]))
    for i, c in enumerate(clusters, 1):
        c["id"] = i
    if llm is not None:
        outliers = name_clusters(clusters, llm, topic, location, progress)
        if outliers:
            progress(f"Stage 3 · moving {sum(map(len, outliers.values()))} keywords to better clusters", 0.95)
            _move_outliers(clusters, outliers, vecs, kws, d)
            clusters = [c for c in clusters if c["_members"]]
    _unique_names(clusters)
    for c in clusters:
        c.pop("_members", None)
    clusters.sort(key=lambda c: (-c["size"], c["name"]))
    progress("Stage 3 · done", 1.0)
    return clusters
