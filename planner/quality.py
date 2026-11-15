"""Keyword quality checks. Pure functions (no AI calls), so they are easy to test.

- clean(): tidy one query (lowercase, strip numbering/quotes/punctuation)
- cap_repeats(): stop one wording pattern from taking over a list
- coverage(): does an angle's list actually match its angle (questions are questions, etc.)
- near_duplicates(): find pairs like "braces cost" / "braces price" using embeddings
- classify(): intent + funnel stage for a keyword
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

QUESTION_WORDS = ("how", "what", "why", "when", "where", "which", "who", "is", "are", "can", "could", "does",
                  "do", "did", "should", "will", "would", "am", "was", "were", "has", "have")
COST_WORDS = ("cost", "costs", "price", "prices", "pricing", "fee", "fees", "charge", "charges", "emi",
              "insurance", "cheap", "cheapest", "affordable", "worth", "package", "packages", "discount",
              "budget", "expensive", "rate", "rates", "payment", "installment", "covered", "afford")
COMPARE_PATTERNS = (r"\bvs\.?\b", r"\bversus\b", r"\bor\b", r"\bbetter\b", r"\bdifference\b", r"\bcompare",
                    r"\bbest\b", r"\balternative", r"\binstead of\b", r"\bcompared\b", r"\bwhich\b")
TRANSACTIONAL_WORDS = ("book", "booking", "book an appointment", "buy", "order", "hire", "contact", "consultation",
                       "schedule", "enquiry", "phone number", "whatsapp")
NEAR_ME = ("near me", "nearby", "near by", "open now", "closest", "around me")

INTENTS = ("Informational", "Commercial", "Transactional", "Local")
FUNNEL = {"Informational": "TOFU", "Commercial": "MOFU", "Transactional": "BOFU", "Local": "BOFU"}


def clean(q: str) -> str:
    q = str(q).strip().lower()
    q = re.sub(r"^\s*(?:\d+[.)\-:]|[-*•])\s*", "", q)       # "1. ", "- ", "• "
    q = q.replace("’", "'").replace("“", "").replace("”", "").replace('"', "")
    q = re.sub(r"[?!.,;:]+$", "", q)
    q = re.sub(r"[^a-z0-9'&/+\- ]", " ", q)
    q = re.sub(r"\s+", " ", q).strip(" -'")
    return q


def is_valid(q: str) -> bool:
    words = q.split()
    if not 2 <= len(words) <= 12:
        return False
    letters = sum(ch.isalpha() for ch in q)
    return letters >= 0.6 * len(q.replace(" ", ""))


def location_tokens(location: str) -> list[str]:
    parts = re.split(r"[,/]| and ", location.lower())
    toks = set()
    for p in parts:
        p = p.strip()
        if p:
            toks.add(p)
            toks.update(w for w in p.split() if len(w) > 3)
    return sorted(toks, key=len, reverse=True)


def mentions_location(q: str, loc_tokens: list[str]) -> bool:
    return any(t in q for t in loc_tokens) or any(n in q for n in NEAR_ME)


def prefix(q: str, n: int = 2) -> str:
    return " ".join(q.split()[:n])


def cap_repeats(queries: list[str], topic: str, max_share: float = 0.15, min_cap: int = 3):
    """Keep at most `cap` queries per opening two words and per 'topic + filler' template.

    Returns (kept, dropped, worst_prefix).
    """
    cap = max(min_cap, round(len(queries) * max_share))
    topic_l = clean(topic)
    seen: Counter = Counter()
    kept, dropped = [], []
    for q in queries:
        keys = [("p", prefix(q))]
        if topic_l and q.startswith(topic_l + " "):
            keys.append(("t", topic_l))  # "<topic> ..." template, however it continues
        if any(seen[k] >= cap for k in keys):
            dropped.append(q)
            continue
        for k in keys:
            seen[k] += 1
        kept.append(q)
    worst = None
    if dropped:
        c = Counter(prefix(q) for q in queries)
        worst = c.most_common(1)[0][0]
    return kept, dropped, worst


def coverage(check: str | None, queries: list[str], loc_tokens: list[str]) -> tuple[bool, float, str]:
    """Return (passes, share, feedback to send back to the AI if it fails)."""
    if not check or not queries:
        return True, 1.0, ""
    if check == "questions":
        hits = [q for q in queries if q.split()[0] in QUESTION_WORDS]
        need, fb = 0.6, "Every query must be a real question starting with how, what, why, when, is, can, does or should."
    elif check == "costs":
        hits = [q for q in queries if any(w in q.split() for w in COST_WORDS)]
        need, fb = 0.5, "Every query must be about price, cost, fees, payment, insurance or value for money."
    elif check == "comparisons":
        hits = [q for q in queries if any(re.search(p, q) for p in COMPARE_PATTERNS)]
        need, fb = 0.5, "Every query must compare options ('x vs y', 'x or y', 'difference between', 'which is better')."
    elif check == "local":
        if not loc_tokens:
            return True, 1.0, ""
        hits = [q for q in queries if mentions_location(q, loc_tokens)]
        need, fb = 0.7, "Every query must mention the location, a nearby area or 'near me'."
    else:
        return True, 1.0, ""
    share = len(hits) / len(queries)
    return share >= need, share, fb


SYNONYMS = {"price": "cost", "prices": "cost", "pricing": "cost", "costs": "cost", "fee": "cost", "fees": "cost",
            "charge": "cost", "charges": "cost", "rate": "cost", "rates": "cost", "top": "best", "dentists": "dentist",
            "clinics": "clinic", "centre": "center", "nearby": "near", "versus": "vs", "teeth": "tooth"}
SOFT_WORDS = {"in", "for", "the", "a", "an", "of", "at", "to", "and", "my", "me", "near", "is", "are", "with", "on"}


def canonical(q: str) -> frozenset:
    """Words that carry meaning, with synonyms merged: 'braces price in kolkata' == 'kolkata braces cost'."""
    return frozenset(SYNONYMS.get(w, w) for w in q.split() if SYNONYMS.get(w, w) not in SOFT_WORDS)


def near_duplicates(vectors: np.ndarray, threshold: float, texts: list[str] | None = None,
                    very_high: float = 0.975) -> list[tuple[int, int]]:
    """Pairs (keep, drop) where drop is the same search as an earlier, kept item.

    With texts: the same meaningful words after merging synonyms and similar meaning (>= threshold), or almost
    identical meaning (>= very_high). Without texts: similarity >= threshold only.
    """
    if len(vectors) < 2:
        return []
    sims = vectors @ vectors.T
    canon = [canonical(t) for t in texts] if texts is not None else None
    dropped = set()
    pairs = []
    for i in range(len(vectors)):
        if i in dropped:
            continue
        for j in np.nonzero(sims[i, i + 1:] >= threshold)[0] + i + 1:
            j = int(j)
            if j in dropped:
                continue
            if canon is not None and canon[i] != canon[j] and sims[i, j] < very_high:
                continue
            dropped.add(j)
            pairs.append((i, j))
    return pairs


def classify(q: str, angle: str, loc_tokens: list[str]) -> tuple[str, str]:
    words = q.split()
    has_loc = mentions_location(q, loc_tokens) if loc_tokens else any(n in q for n in NEAR_ME)
    question = words[0] in QUESTION_WORDS
    compares = angle == "comparisons" and any(re.search(p, q) for p in COMPARE_PATTERNS)
    if not question and any(re.search(rf"\b{re.escape(w)}\b", q) for w in TRANSACTIONAL_WORDS):
        intent = "Transactional"
    elif has_loc and angle in ("local", "services"):
        intent = "Local"
    elif any(w in words for w in COST_WORDS) or re.search(r"\b(vs|versus|best|top|review|reviews)\b", q) \
            or "which is better" in q or "difference between" in q or compares:
        intent = "Commercial"
    elif has_loc:
        intent = "Local"
    elif angle == "services" and words[0] not in QUESTION_WORDS:
        intent = "Commercial"
    else:
        intent = "Informational"
    return intent, FUNNEL[intent]
