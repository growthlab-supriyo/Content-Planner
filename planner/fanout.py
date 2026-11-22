"""Stage 2: query fan-out.

1. The AI studies the topic and chooses the angles (who searches, at what stage, about what). It is asked up to
   3 times; only if every try fails are the 7 standard angles used, so the run can still finish.
2. One focused AI pass per angle, with quality checks after every pass (repetition, on-angle, enough queries).
3. Duplicates and near-duplicates are removed; area names the AI used are found and glued words fixed.
4. The AI tags every keyword's intent and funnel stage, in batches. Keywords it skips are asked again, up to 3 more
   rounds; only a keyword still untagged after that keeps a rule-based tag (counted in the quality report).
"""
from __future__ import annotations

import re
import time
from typing import Callable

from . import config as C
from . import quality as Q
from .config import MAX_RETRIES_PER_ANGLE
from .llm import LLMError, for_stage, get_list, is_fatal
from .prompts import (ACTION_GOALS, ANGLE_ORDER, ANGLE_PROMPT, ANGLES, KINDS, REQUIRED_KINDS, SINGLE_KINDS, SYSTEM,
                      build_prompt, get_angle)

Progress = Callable[[str, float], None]
CHECKED_KINDS = ("questions", "costs", "comparisons")  # kinds with a coverage check in quality.py


def _queries_from(data: dict) -> list[str]:
    for key in ("queries", "keywords", "items", "results"):
        if isinstance(data.get(key), list):
            return [q for q in data[key] if isinstance(q, str)]
    for v in data.values():  # any list of strings
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return v
    return []


def required_kinds(location: str, goal: str = "") -> tuple[str, ...]:
    """Kinds every angle plan must have: local with a location, and people ready to act when the goal is
    enquiries or sales (without them a 'get enquiries' plan is almost all TOFU, seen in testing)."""
    return REQUIRED_KINDS + (("local",) if location else ()) + (("action",) if goal in ACTION_GOALS else ())


def angle_problems(angles: list[dict], location: str, goal: str = "") -> str:
    """What a plan of angles is missing or repeats, as feedback for the AI ("" when it covers the whole journey)."""
    kinds = [a["kind"] for a in angles]
    missing = [k for k in required_kinds(location, goal) if k not in kinds]
    repeated = sorted({k for k in kinds if k in SINGLE_KINDS and kinds.count(k) > 1})
    parts = []
    if missing:
        parts.append("Your last answer had no angle of kind: " + ", ".join(missing) + ". Add them.")
    if repeated:
        parts.append("It had more than one angle of kind: " + ", ".join(repeated) + ". Merge each into one angle.")
    return " ".join(parts)


def complete_angles(angles: list[dict], location: str, goal: str = "") -> list[dict]:
    """Last resort after every try: keep the first angle of each single-use kind and add a standard angle for
    each required kind the AI still left out, so the plan always covers the whole journey."""
    out, used = [], set()
    for a in angles:
        if a["kind"] in SINGLE_KINDS and a["kind"] in used:
            continue
        used.add(a["kind"])
        out.append(a)
    return out + [ANGLES[k] for k in required_kinds(location, goal) if k not in used]


def plan_angles(llm, topic: str, location: str = "", k_min: int = 6, k_max: int = 9, goal: str = "") -> list[dict]:
    """The AI chooses the angles for this topic, asked up to C.AI_TRIES times. A plan that leaves out part of the
    customer journey (no comparisons, no fears...) or repeats a kind is sent back with feedback; the best plan is
    completed with standard angles only if every try still misses something. [] if no try gave any usable answer."""
    best: list[dict] = []
    feedback = ""
    for attempt in range(C.AI_TRIES):
        if attempt:
            time.sleep(C.AI_RETRY_PAUSE)
        angles = _ask_angles(llm, topic, location, k_min, k_max, feedback, goal)
        if not angles:
            continue
        feedback = angle_problems(angles, location, goal)
        if not feedback:
            return angles
        if not best or len(angle_problems(best, location, goal)) > len(feedback):
            best = angles
    return complete_angles(best, location, goal) if best else []


def _ask_angles(llm, topic: str, location: str, k_min: int, k_max: int, feedback: str = "",
                goal: str = "") -> list[dict]:
    try:
        data = llm.json(SYSTEM, ANGLE_PROMPT.format(
            topic=topic, location_text=location or "none (not a local search)", k_min=k_min, k_max=k_max,
            kinds=", ".join(KINDS), required=", ".join(required_kinds(location, goal)),
            feedback=f"\n- {feedback}" if feedback else ""), temperature=0.4)
    except LLMError as e:
        if is_fatal(e):
            raise
        return []
    angles, seen = [], set()
    for a in get_list(data, "angles"):
        if not isinstance(a, dict):
            continue
        label = re.sub(r"\s+", " ", str(a.get("label") or "")).strip()[:40]
        instruction = str(a.get("instruction") or "").strip()[:400]
        if len(label) < 3 or len(instruction) < 10 or label.lower() in seen:
            continue
        seen.add(label.lower())
        kind = str(a.get("kind") or "").strip().lower()
        kind = kind if kind in KINDS else "other"
        examples = [Q.clean(x) for x in (a.get("examples") or []) if Q.clean(x)][:4]
        angles.append({"key": "ai_" + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_"), "label": label,
                       "kind": kind, "instruction": instruction, "examples": examples,
                       "check": kind if kind in CHECKED_KINDS else None, "planned": True})
        if len(angles) >= k_max:
            break
    return angles if len(angles) >= 3 else []


TAG_PROMPT = """Business / topic: {topic}{loc}

Tag each search query below with:
- intent: Informational (wants to learn), Commercial (comparing options, prices, best, reviews before choosing),
  Transactional (ready to book, buy, call or order now) or Local (looking for a provider in a place or near them).
- funnel: TOFU (just noticed a need or learning), MOFU (comparing and considering), BOFU (ready to choose or act).
  BOFU includes the last steps before acting, even when phrased as a question: how to apply or book, admission or
  booking process, forms and documents needed, fees at a named place, timings, contact details, "near me" with a
  specific service, joining or ordering.
Each line shows the query, then in [brackets] the research angle it came from. Use the angle as a hint: queries
from a "ready to act" angle are usually Transactional and BOFU; from a comparisons or costs angle, usually MOFU.
Judge what the searcher wants, not single words. Copy each query exactly as "q", without the [angle].

{queries}

Return JSON only: {{"tags": [{{"q": "...", "intent": "...", "funnel": "..."}}]}}"""

FUNNELS = ("TOFU", "MOFU", "BOFU")


def tag_intents(llm, rows: list[dict], topic: str, location: str = "", batch: int = 60,
                progress: Progress | None = None) -> int:
    """The AI tags intent and funnel stage for every row, in place. Rows it leaves out (a failed call, a skipped
    query) are asked again in smaller batches, up to C.AI_TRIES more rounds. Only rows still untagged after that keep
    the rule-based tags they already have. Returns how many rows the AI tagged."""
    progress = progress or (lambda m, f: None)
    tagged: dict[str, tuple[str, str]] = {}

    def ask(part: list[dict]) -> None:
        def angle_of(r: dict) -> str:
            hint = {"action": "ready to act", "comparisons": "comparisons", "costs": "costs"}.get(r.get("kind", ""))
            label = r.get("angle") or ""
            return f" [{label}{' · ' + hint if hint and hint.lower() not in label.lower() else ''}]" if label else ""

        text = "\n".join(f"{i}. {r['keyword']}{angle_of(r)}" for i, r in enumerate(part, 1))
        try:
            data = llm.json(SYSTEM, TAG_PROMPT.format(topic=topic, loc=f"\nLocation: {location}" if location else "",
                                                      queries=text), temperature=0.1)
        except LLMError as e:
            if is_fatal(e):
                raise
            return
        mine = {r["keyword"] for r in part}
        for t in get_list(data, "tags"):
            if not isinstance(t, dict):
                continue
            q = Q.clean(re.sub(r"\[.*?\]", "", str(t.get("q") or "")))  # in case the [angle] was copied too
            intent = str(t.get("intent") or "").strip().title()
            funnel = str(t.get("funnel") or "").strip().upper()
            if q in mine and intent in Q.INTENTS:
                tagged[q] = (intent, funnel if funnel in FUNNELS else Q.FUNNEL[intent])

    for s in range(0, len(rows), batch):
        progress(f"Stage 2 · tagging intent and funnel stage {s + 1}–{min(len(rows), s + batch)} of {len(rows)}",
                 s / max(1, len(rows)))
        ask(rows[s:s + batch])
    for attempt in range(C.AI_TRIES):  # skipped or failed: ask again, in smaller batches
        missing = [r for r in rows if r["keyword"] not in tagged]
        if not missing:
            break
        progress(f"Stage 2 · tagging: try {attempt + 2} for {len(missing)} keywords the AI skipped", 1.0)
        time.sleep(C.AI_RETRY_PAUSE)
        for s in range(0, len(missing), 30):
            ask(missing[s:s + 30])
    for r in rows:
        if r["keyword"] in tagged:
            r["intent"], r["funnel"] = tagged[r["keyword"]]
            r["tagged_by"] = "AI"
        else:
            r["tagged_by"] = "Rules"
    return sum(1 for r in rows if r["tagged_by"] == "AI")


def run_angle(llm, angle, topic: str, location: str, n: int, instructions: dict | None = None,
              seen: set[str] | None = None, max_retries: int = MAX_RETRIES_PER_ANGLE) -> tuple[list[str], dict]:
    """Ask for one angle, re-asking with feedback when the list is repetitive, off-angle or too short."""
    a = get_angle(angle)
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
        data = llm.json(SYSTEM, build_prompt(a, topic, location, n, feedback, instructions))
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
    if a["check"] in CHECKED_KINDS:
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
               angles: list | None = None, progress: Progress | None = None,
               ai_tags: bool = True, goal: str = "") -> tuple[list[dict], dict]:
    """angles: built-in keys or angle dicts to use as given (tests, saved plans). Without them, the AI plans the
    angles; the 7 standard ones are used only if every try fails."""
    progress = progress or (lambda msg, frac: None)
    topic, location = topic.strip(), location.strip()
    planned = False
    if angles is None:
        progress("Stage 2 · studying the topic to choose the angles", 0.0)
        found = plan_angles(for_stage(llm, "angles"), topic, location, goal=goal)
        planned = bool(found)
        angles = found or None
    standard = ANGLE_ORDER + (["action"] if goal in ACTION_GOALS else [])  # only if the AI gave no angles at all
    angles = [get_angle(a) for a in (angles or standard)]
    loc = Q.location_tokens(location) if location else []
    seen: set[str] = set()
    rows: list[dict] = []
    reports, errors = [], []
    writer = for_stage(llm, "fanout")
    steps = len(angles) + 2
    for i, a in enumerate(angles):
        label = a["label"]
        progress(f"Stage 2 · angle {i + 1}/{len(angles)}: {label}", (i + 0.5) / steps)
        try:
            kept, rep = run_angle(writer, a, topic, location, n, instructions, seen)
        except LLMError as e:
            errors.append(f"{label}: {e}")
            reports.append({"angle": label, "asked": n, "kept": 0, "notes": [f"Failed: {e}"]})
            if "API key" in str(e) or "not found" in str(e) or "Could not reach" in str(e):
                raise  # same problem for every angle: stop now
            continue
        reports.append(rep)
        for q in kept:
            seen.add(q)
            rows.append({"keyword": q, "angle": label, "kind": a["kind"]})
    if not rows:
        raise LLMError("No keywords came back. " + (errors[0] if errors else ""))

    progress("Stage 2 · removing near-duplicates", (len(angles) + 0.5) / steps)
    examples = []
    vecs = embedder.embed([r["keyword"] for r in rows])
    pairs = Q.near_duplicates(vecs, embedder.near_duplicate, [r["keyword"] for r in rows])
    drop = {j for _, j in pairs}
    for keep_i, drop_i in pairs[:15]:
        examples.append([rows[keep_i]["keyword"], rows[drop_i]["keyword"]])
    rows = [r for i, r in enumerate(rows) if i not in drop]
    places = Q.local_area_tokens(rows, location) if location else []
    if places:  # "ingaria" -> "in garia", "behalakolkata" -> "behala kolkata"
        seen_kw: set[str] = set()
        fixed = []
        for r in rows:
            r["keyword"] = Q.split_glued(r["keyword"], places)
            if r["keyword"] not in seen_kw:
                seen_kw.add(r["keyword"])
                fixed.append(r)
        rows = fixed
    for r in rows:  # rule-based tags first: the fallback for anything the AI doesn't tag
        r["intent"], r["funnel"] = Q.classify(r["keyword"], r.get("kind", "services"), places or loc)
    ai_tagged = 0
    if ai_tags:
        ai_tagged = tag_intents(for_stage(llm, "tagging"), rows, topic, location,
                                progress=lambda m, f: progress(m, (len(angles) + 1 + f) / steps))
    else:
        for r in rows:
            r["tagged_by"] = "Rules"

    report = {
        "angles": reports,
        "angles_planned_by_ai": planned,
        "angle_plan": [{"label": a["label"], "kind": a["kind"], "instruction": a["instruction"]} for a in angles],
        "errors": errors,
        "total_kept": len(rows),
        "near_duplicates_removed": len(drop),
        "near_duplicate_examples": examples,
        "tagged_by_ai": ai_tagged,
        "questions": sum(1 for r in rows if r["keyword"].split()[0] in Q.QUESTION_WORDS),
        "with_location": sum(1 for r in rows if loc and Q.mentions_location(r["keyword"], places or loc)),
        "areas": [p for p in places if p not in loc],
        "ai_calls": getattr(llm, "calls", None),
        "model": getattr(llm, "label", ""),
        "stage_models": llm.used() if hasattr(llm, "used") else {},
        "embedder": getattr(embedder, "name", ""),
    }
    progress("Stage 2 · done", 1.0)
    return rows, report
