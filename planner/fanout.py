"""Stage 2: query fan-out, one focused AI pass per angle, with quality checks after every pass."""
from __future__ import annotations

from typing import Callable

from . import quality as Q
from .config import MAX_RETRIES_PER_ANGLE
from .llm import LLMError
from .prompts import ANGLE_ORDER, ANGLES, SYSTEM, build_prompt

Progress = Callable[[str, float], None]


def _queries_from(data: dict) -> list[str]:
    for key in ("queries", "keywords", "items", "results"):
        if isinstance(data.get(key), list):
            return [q for q in data[key] if isinstance(q, str)]
    for v in data.values():  # any list of strings
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return v
    return []


def run_angle(llm, angle: str, topic: str, location: str, n: int, instructions: dict | None = None,
              seen: set[str] | None = None, max_retries: int = MAX_RETRIES_PER_ANGLE) -> tuple[list[str], dict]:
    """Ask for one angle, re-asking with feedback when the list is repetitive, off-angle or too short."""
    a = ANGLES[angle]
    loc = Q.location_tokens(location) if location else []
    seen = seen if seen is not None else set()
    collected: list[str] = []
    received = invalid = 0
    feedback = ""
    rep = {"angle": a["label"], "asked": n, "attempts": 0, "received": 0, "kept": 0,
           "dropped_repetitive": 0, "dropped_invalid": 0, "coverage": None, "notes": []}
    kept: list[str] = []
    repetitive_total = 0
    for attempt in range(max_retries + 1):
        rep["attempts"] = attempt + 1
        data = llm.json(SYSTEM, build_prompt(angle, topic, location, n, feedback, instructions))
        raw = _queries_from(data)
        received += len(raw)
        new = 0
        for q in raw:
            c = Q.clean(q)
            if not Q.is_valid(c):
                invalid += 1
                continue
            if c in seen or c in collected:
                continue
            collected.append(c)
            new += 1
        kept, dropped, worst = Q.cap_repeats(collected, topic)
        collected = list(kept)  # repeats are dropped for good; judge the next attempt on its own items
        repetitive_total += len(dropped)
        ok_cov, share, cov_fb = Q.coverage(a["check"], kept, loc)
        rep["coverage"] = round(share, 2)
        too_repetitive = len(dropped) > 0.25 * max(1, new)
        enough = len(kept) >= 0.7 * n
        if ok_cov and enough and not too_repetitive:
            break
        if attempt == max_retries:
            break
        parts = []
        if too_repetitive and worst:
            parts.append(f'Too many queries started with "{worst}". Start each query with different words.')
        if not ok_cov:
            parts.append(cov_fb)
        if not enough:
            parts.append(f"Give {n} new queries.")
        if kept:
            parts.append("Do not repeat these: " + "; ".join(kept[-12:]))
        feedback = " ".join(parts)
    if a["check"] in ("questions", "comparisons", "costs"):
        # the off-angle items are still real keywords, but the angle list should stay on-topic
        on = [q for q in kept if Q.coverage(a["check"], [q], loc)[0]]
        if len(on) >= 0.5 * len(kept):
            kept = on
    rep.update(received=received, kept=len(kept), dropped_invalid=invalid, dropped_repetitive=repetitive_total)
    if rep["coverage"] is not None and a["check"] and not Q.coverage(a["check"], kept, loc)[0]:
        rep["notes"].append("Off-angle: some queries don't match this angle well.")
    if len(kept) < 0.5 * n:
        rep["notes"].append("Thin: fewer queries than asked for.")
    return kept, rep


def run_fanout(llm, embedder, topic: str, location: str = "", n: int = 30, instructions: dict | None = None,
               angles: list[str] | None = None, progress: Progress | None = None) -> tuple[list[dict], dict]:
    progress = progress or (lambda msg, frac: None)
    topic, location = topic.strip(), location.strip()
    angles = angles or ANGLE_ORDER
    loc = Q.location_tokens(location) if location else []
    seen: set[str] = set()
    rows: list[dict] = []
    reports, errors = [], []
    for i, angle in enumerate(angles):
        label = ANGLES[angle]["label"]
        progress(f"Stage 2 · angle {i + 1}/{len(angles)}: {label}", i / (len(angles) + 1))
        try:
            kept, rep = run_angle(llm, angle, topic, location, n, instructions, seen)
        except LLMError as e:
            errors.append(f"{label}: {e}")
            reports.append({"angle": label, "asked": n, "kept": 0, "notes": [f"Failed: {e}"]})
            if "API key" in str(e) or "not found" in str(e) or "Could not reach" in str(e):
                raise  # same problem for every angle: stop now
            continue
        reports.append(rep)
        for q in kept:
            seen.add(q)
            rows.append({"keyword": q, "angle": label})
    if not rows:
        raise LLMError("No keywords came back. " + (errors[0] if errors else ""))

    progress("Stage 2 · removing near-duplicates", len(angles) / (len(angles) + 1))
    examples = []
    vecs = embedder.embed([r["keyword"] for r in rows])
    pairs = Q.near_duplicates(vecs, embedder.near_duplicate, [r["keyword"] for r in rows])
    drop = {j for _, j in pairs}
    for keep_i, drop_i in pairs[:15]:
        examples.append([rows[keep_i]["keyword"], rows[drop_i]["keyword"]])
    rows = [r for i, r in enumerate(rows) if i not in drop]
    for r in rows:
        r["intent"], r["funnel"] = Q.classify(r["keyword"], _angle_key(r["angle"]), loc)

    report = {
        "angles": reports,
        "errors": errors,
        "total_kept": len(rows),
        "near_duplicates_removed": len(drop),
        "near_duplicate_examples": examples,
        "questions": sum(1 for r in rows if r["keyword"].split()[0] in Q.QUESTION_WORDS),
        "with_location": sum(1 for r in rows if loc and Q.mentions_location(r["keyword"], loc)),
        "ai_calls": getattr(llm, "calls", None),
        "model": getattr(llm, "label", ""),
        "embedder": getattr(embedder, "name", ""),
    }
    progress("Stage 2 · done", 1.0)
    return rows, report


def _angle_key(label: str) -> str:
    return next((k for k, v in ANGLES.items() if v["label"] == label), "services")
