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
from .llm import LLMError, for_stage, get_list, is_fatal

GENERIC_WORDS = {
    "best", "top", "good", "near", "me", "service", "services", "online", "cheap", "price", "cost", "info",
    "company", "companies", "shop", "store", "list", "near-me", "nearby", "local", "agency", "agencies",
    "provider", "providers", "expert", "experts", "professional", "the", "a", "in", "for", "of", "and", "my",
}

Progress = Callable[[str, float], None]


def _stem(w: str) -> str:
    return re.sub(r"(ies|es|s)$", "", w) if len(w) > 4 else w


GENERIC_PROMPT = """Business / topic: {topic}{loc}

Below are two-word search terms. Each kept term will be planned into a web page. Mark a term GENERIC when it is
too broad to plan a page around: just the business or category name, or that name with filler ("dental clinic",
"best dentist", "clinic near"). Keep terms that name something specific (a service, problem, product, audience or
place), e.g. "invisible braces", "dentist newtown", "kids dentist". When unsure, keep it.
Copy generic terms exactly, each with a short reason (under 10 words).

{terms}

Return JSON only: {{"generic": [{{"term": "...", "reason": "..."}}]}}"""


def _generic_by_rule(kw: str, topic: str) -> bool:
    generic = {_stem(w) for w in Q.clean(topic).split()} | {_stem(w) for w in GENERIC_WORDS}
    return all(_stem(w) in generic for w in kw.split())


def generic_by_ai(terms: list[str], llm, topic: str, location: str = "") -> dict[str, str] | None:
    """{term: reason} for the two-word terms the AI judges too generic, asked up to config.AI_TRIES times; None if
    no try gave a usable answer."""
    import time

    from . import config as C

    if not terms:
        return {}
    for attempt in range(C.AI_TRIES):
        if attempt:
            time.sleep(C.AI_RETRY_PAUSE)
        out = _ask_generic(terms, llm, topic, location)
        if out is not None:
            return out
    return None


def _ask_generic(terms: list[str], llm, topic: str, location: str) -> dict[str, str] | None:
    try:
        data = llm.json(NAME_SYSTEM, GENERIC_PROMPT.format(
            topic=topic, loc=f"\nLocation: {location}" if location else "", terms="\n".join(terms)), temperature=0.1)
    except LLMError as e:
        if is_fatal(e):
            raise
        return None
    if not isinstance(data.get("generic"), list) and not get_list(data, "generic"):
        return None
    out = {}
    for g in get_list(data, "generic"):
        term = Q.clean(g.get("term") if isinstance(g, dict) else g or "")
        if term in terms:
            reason = str(g.get("reason") or "").strip() if isinstance(g, dict) else ""
            out[term] = reason[:80] or "Too generic"
    return out


def filter_keywords(rows: list[dict], topic: str, llm=None, location: str = "") -> tuple[list[dict], list[dict]]:
    """Drop single words, and two-word terms too generic to plan a page around ("dental clinic", "wall decor").

    The AI judges every two-word term (so "kids dentist" is kept but "best clinic" goes), asked up to 3 times. If
    every try fails, no two-word term is removed: keeping a term is safer than a word list guessing wrong. Without
    an AI at all (offline tests, tools), the word list decides.
    """
    two = list(dict.fromkeys(r["keyword"] for r in rows if len(r["keyword"].split()) == 2))
    judged = generic_by_ai(two, for_stage(llm, "filter"), topic, location) if llm is not None else None
    kept, removed = [], []
    for r in rows:
        words = r["keyword"].split()
        if len(words) < 2:
            removed.append({**r, "reason": "Single word"})
        elif len(words) == 2 and judged is not None and r["keyword"] in judged:
            removed.append({**r, "reason": f"Generic two-word term (AI): {judged[r['keyword']]}"})
        elif len(words) == 2 and llm is None and _generic_by_rule(r["keyword"], topic):
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


SINGLETON_REACH = 1.15  # a lone keyword joins a cluster up to this many times the normal cluster distance
SUBJECT_REACH = 1.5     # ...or up to this far when it shares the cluster's subject word ("jaw", "dentures")
NOT_SUBJECT = set(Q.COST_WORDS) | {"vs", "versus", "or", "better", "difference", "between", "pain", "painful",
                                   "hurt", "hurts", "treatment", "treatments", "procedure", "safe", "bad", "after",
                                   "before", "during", "without", "always", "really", "very", "lot", "new", "like",
                                   "general", "normal", "common", "type", "types", "option", "options"}


def subject_words(texts: list[str]) -> list[set[str]]:
    """The subject words of each keyword: not filler, and not so common in this list that they say nothing
    (for a dental clinic "teeth" and "dental" are in every other keyword, so they can't tell pages apart)."""
    words = [{_stem(w) for w in t.split() if w not in CLUSTER_STOP and w not in NOT_SUBJECT
              and w not in GENERIC_WORDS and len(w) > 2} for t in texts]
    freq: dict[str, int] = {}
    for ws in words:
        for w in ws:
            freq[w] = freq.get(w, 0) + 1
    common = {w for w, n in freq.items() if n > max(4, 0.08 * len(texts))}
    return [ws - common for ws in words]


def _shares_subject(i: int, group: list[int], subj: list[set[str]]) -> bool:
    """Item i shares a subject word with at least half of the group."""
    if not subj[i]:
        return False
    hits = sum(1 for g in group if subj[i] & subj[g])
    return hits * 2 >= len(group)


def _attach_singletons(groups: list[list[int]], vecs: np.ndarray, distance: float,
                       subj: list[set[str]] | None = None) -> list[list[int]]:
    """A lone keyword joins the closest cluster if it is reasonably close. With subject words, a lone keyword
    that is a bit further away still joins (or pairs up with another lone keyword) when they share a subject
    word, so "jaw clicking when i chew" and "pain radiating from jaw to ear" become one page, but "can
    cavities heal" doesn't join them just because both are dental problems."""
    big = [g for g in groups if len(g) > 1]
    lone = [g[0] for g in groups if len(g) == 1]
    rest = []
    for i in lone:
        j, d = _closest(i, big, vecs) if big else (-1, 2.0)
        if j >= 0 and d <= distance * SINGLETON_REACH:
            big[j].append(i)
        else:
            rest.append(i)
    if subj is None:
        return big + [[i] for i in rest]
    out = big
    for i in rest:
        best, best_d = None, distance * SUBJECT_REACH
        for g in out:
            cen = vecs[g].mean(axis=0)
            d = 1 - float(vecs[i] @ (cen / (np.linalg.norm(cen) or 1)))
            if d <= best_d and _shares_subject(i, g, subj):
                best, best_d = g, d
        if best is None:
            out.append([i])
        else:
            best.append(i)
    return out


CLUSTER_STOP = set(Q.QUESTION_WORDS) | Q.SOFT_WORDS | {
    "i", "you", "your", "it", "do", "get", "should", "much", "many", "long", "often", "best", "near", "me",
    # "choose a provider" filler: "top rated dentist in salt lake" and "good dentist in lake town" are one page
    "top", "rated", "good", "reliable", "trusted", "reputed", "famous", "nearby", "around", "open", "now", "today"}


def topical_text(kw: str, loc_tokens: list[str]) -> str:
    """What a keyword is about, without question words, the location or area names, so phrasing and place
    don't drive the clusters ("dentist in park street" and "dentist in lake town" are about the same thing)."""
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
- "id": the cluster number, exactly as given.
- "first": the cluster's FIRST keyword, copied exactly (this is how your answer is matched to the cluster).
- "name": a short, specific page topic (2 to 6 words, Title Case) that describes THAT cluster's keywords,
  e.g. "Root Canal Cost", "Tooth Pain at Night". Names must be different from each other.
- "outliers": keywords copied exactly from that cluster that are about a different subject and would not
  belong on the same page. Usually there are none: use [].
Answer for every cluster, in order, without skipping any.

{clusters}

Return JSON only: {{"clusters": [{{"id": 1, "first": "...", "name": "...", "outliers": []}}, ...]}}"""


def match_answer(item: dict, part: list[dict], key: str, num_keys: tuple[str, ...]) -> dict | None:
    """The item of `part` an AI answer belongs to: by the keyword it copied back (key), else by its number.

    Small models sometimes skip one item and number the rest 1..n anyway, which shifts every later answer onto
    the wrong page. A copied keyword can't shift, so it wins over the number.
    """
    echo = Q.clean(item.get(key) or "")
    if echo:
        for c in part:
            if echo in c["keywords"][:15] or echo == Q.clean(c.get("name", "")):
                return c
    for k in num_keys:
        try:
            n = int(item.get(k))
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(part):
            return part[n - 1]
    return None


def name_clusters(clusters: list[dict], llm, topic: str, location: str, progress: Progress | None = None,
                  batch: int = 15, embedder=None, vecs: np.ndarray | None = None,
                  loc_tokens: list[str] | None = None) -> dict[int, list[str]]:
    """Name clusters in place; return {cluster id: keywords the AI says don't belong}.

    With an embedder, a name that fits another cluster in the batch better than its own is rejected (the
    cluster keeps its fallback name, its main keyword), so a shifted answer can't mislabel a page.
    """
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
        items = get_list(data, "clusters")
        centres = None
        if embedder is not None and vecs is not None and "_members" in part[0]:
            centres = np.array([vecs[c["_members"]].mean(axis=0) for c in part])
            centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        answers: list[tuple[dict, dict, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            c = match_answer(item, part, "first", ("id",))
            if c is None or any(c is a for a, _, _ in answers):
                continue
            answers.append((c, item, str(item.get("name") or "").strip()))
        fits = [True] * len(answers)
        if centres is not None and answers:
            nv = embedder.embed([topical_text(n.lower(), loc_tokens or []) or n.lower() for _, _, n in answers])
            sims = nv @ centres.T
            fits = [float(sims[i, part.index(c)]) >= float(sims[i].max()) - 0.02
                    for i, (c, _, _) in enumerate(answers)]
        for (c, item, name), ok in zip(answers, fits):
            if 2 <= len(name) <= 60 and ok:
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


INTENT_WORDS = {"Informational": "Guide", "Commercial": "Options and Costs", "Transactional": "Booking",
                "Local": "Near You"}


def _unique_names(clusters: list[dict]) -> None:
    """Two pages can't share a name. A repeat is renamed after its own main keyword ("Emergency Tooth Pain At
    Night"), or, if that's taken too, gets a word for what the page is for ("... Near You"), never "(Local)"."""
    seen: set[str] = set()
    for c in clusters:
        if c["name"].lower() in seen:
            options = [c["main_keyword"].title(), f"{c['name']} {INTENT_WORDS.get(c['intent'], 'Guide')}"]
            c["name"] = next((o for o in options if o.lower() not in seen), f"{c['name']} {len(seen)}")
        seen.add(c["name"].lower())


MERGE_PROMPT = """Topic: {topic}{loc}
Each line below will become its own web page. Pages marked SINGLE have only one search query.
For each SINGLE page, decide: would a reader expect that query answered on one of the OTHER pages in this list (the
same subject, so one page can cover both)? If yes, give that page's number as "join". If it is a different subject,
leave it out. Don't join pages just because both are about the same business.

{pages}

Return JSON only: {{"joins": [{{"single": 3, "join": 7}}]}}  (an empty list when none should be joined)"""


def merge_singletons(clusters: list[dict], llm, topic: str, location: str, vecs: np.ndarray,
                     kws: list[str]) -> int:
    """The AI joins single-keyword clusters to the page of the same intent that should answer them, so the plan
    doesn't become dozens of thin one-keyword pages. Asked up to config.AI_TRIES times per intent; returns how many
    clusters were joined. The clusters keep one intent each (only same-intent pages are shown together)."""
    import time

    from . import config as C

    joined = 0
    for intent in Q.INTENTS:
        group = [c for c in clusters if c["intent"] == intent and c["_members"]]
        if len(group) < 2 or not any(c["size"] == 1 for c in group):
            continue
        group = group[:200]
        lines = "\n".join(f"{n}. {'SINGLE ' if c['size'] == 1 else ''}" + "; ".join(c["keywords"][:4])
                          for n, c in enumerate(group, 1))
        data = None
        for attempt in range(C.AI_TRIES):
            if attempt:
                time.sleep(C.AI_RETRY_PAUSE)
            try:
                data = llm.json(NAME_SYSTEM, MERGE_PROMPT.format(
                    topic=topic, loc=f"\nLocation: {location}" if location else "", pages=lines), temperature=0.1)
            except LLMError as e:
                if is_fatal(e):
                    raise
                continue
            if isinstance(data.get("joins"), list) or get_list(data, "joins"):
                break
            data = None
        if data is None:
            continue
        for j in get_list(data, "joins"):
            if not isinstance(j, dict):
                continue
            try:
                s, t = int(j.get("single")), int(j.get("join"))
            except (TypeError, ValueError):
                continue
            if not (1 <= s <= len(group) and 1 <= t <= len(group)) or s == t:
                continue
            src, dst = group[s - 1], group[t - 1]
            if src["size"] != 1 or not src["_members"] or not dst["_members"]:
                continue  # only single pages move, and not into a page that has itself moved away
            dst["_members"] += src["_members"]
            src["_members"] = []
            _fill(dst, vecs, kws)
            joined += 1
    return joined


def cluster_keywords(rows: list[dict], embedder, llm=None, topic: str = "", location: str = "",
                     progress: Progress | None = None) -> list[dict]:
    progress = progress or (lambda m, f: None)
    if not rows:
        return []
    progress("Stage 3 · grouping keywords", 0.1)
    kws = [r["keyword"] for r in rows]
    loc = Q.local_area_tokens(rows, location) if location else []  # the location plus areas the queries name
    texts = [topical_text(k, loc) for k in kws]
    vecs = embedder.embed(texts)
    subj = subject_words(texts)
    d = embedder.cluster_distance
    clusters = []
    for intent in Q.INTENTS:  # one intent per cluster
        idx = [i for i, r in enumerate(rows) if r.get("intent", "Informational") == intent]
        if not idx:
            continue
        sub = vecs[idx]
        for g in _attach_singletons(_groups(sub, d), sub, d, [subj[i] for i in idx]):
            clusters.append(_fill({"intent": intent, "_members": [idx[i] for i in g]}, vecs, kws))
    if llm is not None and any(c["size"] == 1 for c in clusters):
        progress("Stage 3 · joining one-keyword pages to the page that should answer them", 0.3)
        merge_singletons(clusters, for_stage(llm, "naming"), topic, location, vecs, kws)
        clusters = [c for c in clusters if c["_members"]]
    clusters.sort(key=lambda c: (-c["size"], c["name"]))
    for i, c in enumerate(clusters, 1):
        c["id"] = i
    if llm is not None:
        outliers = name_clusters(clusters, for_stage(llm, "naming"), topic, location, progress, embedder=embedder,
                                 vecs=vecs,
                                 loc_tokens=loc)
        if outliers:
            progress(f"Stage 3 · moving {sum(map(len, outliers.values()))} keywords to better clusters", 0.95)
            _move_outliers(clusters, outliers, vecs, kws, d)
            clusters = [c for c in clusters if c["_members"]]
    _unique_names(clusters)
    funnel_of = {r["keyword"]: r.get("funnel") for r in rows}
    for c in clusters:
        c.pop("_members", None)
        votes = [funnel_of.get(k) for k in c["keywords"] if funnel_of.get(k)]
        c["funnel"] = max(sorted(set(votes)), key=votes.count) if votes else Q.FUNNEL[c["intent"]]  # most keywords' stage
    clusters.sort(key=lambda c: (-c["size"], c["name"]))
    progress("Stage 3 · done", 1.0)
    return clusters
