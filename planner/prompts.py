"""Fan-out angles and prompts.

Each angle is asked separately with a focused instruction, so small models don't fall into one
repeated pattern ("<topic> <location> cost", "<topic> <location> price", ...).
The examples are from an unrelated business on purpose, so the model copies the style, not the words.
In the local version these instructions can be edited in the app (Advanced panel).
"""

SYSTEM = (
    "You are an expert SEO keyword researcher. You think like real people searching on Google: "
    "short, natural, sometimes informal queries. You answer with JSON only."
)

# key -> angle settings. "check" names the coverage rule in quality.py.
ANGLES: dict[str, dict] = {
    "services": {
        "label": "Services",
        "instruction": "Specific services, treatments, products or offerings people search for "
                       "(each named specifically, not the generic business name).",
        "examples": ["prenatal yoga classes", "hot yoga for beginners", "private yoga session at home",
                     "yoga therapy for back pain"],
        "check": None,
    },
    "problems": {
        "label": "Problems & Symptoms",
        "instruction": "Problems, symptoms, situations or needs that make people look for this kind of business, "
                       "phrased the way people describe their problem (not the solution).",
        "examples": ["lower back pain after sitting all day", "stiff neck in the morning", "can't touch my toes",
                     "stress and trouble sleeping"],
        "check": None,
    },
    "costs": {
        "label": "Costs",
        "instruction": "Prices, fees, costs, packages, EMI or payment plans, insurance, discounts and "
                       "'is it worth it' searches — for specific services, not just the business name.",
        "examples": ["monthly yoga membership fees", "is a private yoga teacher worth it",
                     "cheapest way to learn yoga", "yoga teacher training cost"],
        "check": "costs",
    },
    "questions": {
        "label": "Questions",
        "instruction": "Real questions people ask (how, what, why, when, is, can, does, should, which). "
                       "Cover beginners, safety, duration, preparation, aftercare and results.",
        "examples": ["how often should i do yoga", "is yoga safe during pregnancy", "what to wear to a yoga class",
                     "can yoga help with anxiety"],
        "check": "questions",
    },
    "comparisons": {
        "label": "Comparisons",
        "instruction": "Comparisons and choices between options, methods, materials or alternatives "
                       "('x vs y', 'x or y', 'difference between', 'which is better', 'best ... for ...').",
        "examples": ["yoga vs pilates for back pain", "hatha or vinyasa for beginners",
                     "difference between yin and restorative yoga", "best yoga style for weight loss"],
        "check": "comparisons",
    },
    "local": {
        "label": "Local",
        "instruction": "Searches to find or choose a provider in the area: 'near me', area and landmark names, "
                       "open now, timings, reviews, best/top in the area.",
        "instruction_no_location": "Searches to find, buy or choose a provider: 'near me', online, where to buy, "
                                   "shops, delivery, reviews, best brands or sellers.",
        "examples": ["yoga studio near me open now", "best yoga classes in indiranagar", "yoga centre near hsr layout",
                     "weekend yoga classes bangalore"],
        "check": None,  # naming a place is natural for local searches, not required
    },
    "fears": {
        "label": "Fears & Myths",
        "instruction": "Worries, fears, doubts, myths and misconceptions people search about "
                       "(pain, safety, side effects, 'is it true', 'does it really').",
        "examples": ["is yoga bad for your knees", "does yoga really help lose weight", "yoga injuries myths",
                     "am i too old to start yoga"],
        "check": None,
    },
    "action": {  # required when the goal is enquiries/bookings or sales (see fanout.required_kinds)
        "label": "Ready to Act",
        "instruction": "Searches from people ready to take the next step: apply, enrol, book, buy, order, enquire, "
                       "call, visit, sign up, admission or booking forms, timings, contact details, offers.",
        "examples": ["book a trial yoga class", "yoga studio membership sign up", "join prenatal yoga classes",
                     "yoga class timings and contact"],
        "check": None,
    },
}

ANGLE_ORDER = [k for k in ANGLES if k != "action"]  # the standard 7; "action" is added when the goal needs it
for _k, _a in ANGLES.items():
    _a.setdefault("key", _k)
    _a.setdefault("kind", _k)

# The kinds an angle can be. The kind decides the quality check run on its list and how the location is used.
KINDS = ("questions", "costs", "comparisons", "local", "services", "problems", "fears", "action", "other")

ANGLE_PROMPT = """Business / topic: {topic}
Location: {location_text}

Before researching keywords, work out how people search around this topic. Think about who searches (different
customer types and situations) and the whole journey: first noticing a need or problem, learning, comparing
options, worrying (fears, myths, side effects, "is it safe"), checking prices, and choosing a provider or product.

List {k_min} to {k_max} search ANGLES for THIS topic: groups of searches that need their own research.
Rules:
- Cover the whole journey. These kinds must each appear: {required}.
- First list the main services or product lines of this business (for a dental clinic: checkups and cleaning, gum
  care, fillings and root canals, implants and dentures, braces and aligners, cosmetic, kids' dentistry,
  emergencies...). Every angle must span ALL of them, not one: a comparisons angle compares options across every
  service line (implant vs bridge, braces vs aligners, filling vs crown...), a fears angle covers worries about all
  of them, a costs angle prices all of them. Never narrow an angle to one service.
- Use each kind once, except "services" and "problems", which may appear more than once to cover different service
  lines (e.g. "Restorative Treatments" and "Orthodontics").
- Every angle must be something many real people search for. No niche or abstract angles that few would ever type
  (bad: "interpreting patient feedback reliability", "clinic pricing transparency policies").
- Add angles this topic really has beyond the required ones; don't force ones that don't fit.{feedback}
For each angle give:
- label: 1 to 4 words, Title Case.
- kind: one of {kinds}. "questions" = real how/what/why questions, "problems" = symptoms or situations,
  "services" = specific services or products, "costs" = prices and payment, "comparisons" = "x vs y" choices,
  "fears" = worries, myths and safety doubts, "local" = finding a provider nearby, "action" = people ready to
  act now (apply, enrol, book, buy, enquire, contact, forms, timings), "other" = none of these.
- instruction: 1 to 2 sentences telling a researcher exactly which searches to write for this angle.
- examples: 3 or 4 short example searches for this topic.

Return JSON only:
{{"angles": [{{"label": "...", "kind": "...", "instruction": "...", "examples": ["...", "..."]}}]}}"""

# Kinds every plan must have (local only when there is a location), and the kinds used at most once.
REQUIRED_KINDS = ("questions", "problems", "costs", "comparisons", "fears")
SINGLE_KINDS = ("questions", "costs", "comparisons", "fears", "local", "action")
# Goals that need people ready to act: the plan must then have an "action" angle (enquire, apply, book, buy).
ACTION_GOALS = ("Get enquiries and bookings", "Sell products online")

USER_TEMPLATE = """Business / topic: {topic}
Location: {location_text}

Angle: {label}
{instruction}

Write {n} different search queries real people type into Google for this angle.
Rules:
- Only searches many real people make, in the words they'd type. No abstract, academic or niche phrasing that
  hardly anyone searches (bad: "interpreting patient safety feedback", "testimonial reliability").
- Never make up brand, clinic, product or doctor names: use only names that are real and widely known.
- Natural English, all lowercase, 2 to 10 words each, spelled correctly, with a space between every word
  (write "in salt lake", never "insalt lake"; place names spelled the way locals write them).
- Vary the wording. Do not start more than 3 queries with the same two words.
- Do not just add one word to the business name (bad: "{topic_l} cost", "{topic_l} price", "{topic_l} reviews").
- {location_rule}
- If a query needs a year, use {year}, never an earlier year.
- No numbering, no quotes, no duplicates.{feedback}

{examples_intro}
{examples}

Return JSON only: {{"queries": ["...", "..."]}}"""


def location_rule(kind: str, location: str) -> str:
    if not location:
        return "There is no location: do not add any city or area name."
    if kind == "local":
        return (f"You may mention {location}, a nearby area or 'near me' where people naturally would, but it's not "
                f"required: write each query the way a local customer would really type it, with or without a place.")
    if kind in ("services", "costs"):
        return f"Mention {location} in about one third of the queries only; the rest are general."
    return "These are general searches: do not add the location."


def get_angle(angle) -> dict:
    """An angle as a dict: a built-in key ("costs") or an angle dict the AI planned."""
    return ANGLES[angle] if isinstance(angle, str) else angle


def build_prompt(angle, topic: str, location: str, n: int, feedback: str = "",
                 instructions: dict[str, str] | None = None) -> str:
    a = get_angle(angle)
    instruction = (instructions or {}).get(a["key"])
    if not instruction:
        instruction = a.get("instruction_no_location") if (a["kind"] == "local" and not location) else None
        instruction = instruction or a["instruction"]
    planned = a.get("planned", False)
    from .quality import current_year

    return USER_TEMPLATE.format(
        topic=topic, topic_l=topic.lower(), year=current_year(),
        location_text=location or "none (not a local search)",
        label=a["label"], instruction=instruction, n=n,
        location_rule=location_rule(a["kind"], location),
        feedback=f"\n- {feedback}" if feedback else "",
        examples_intro="Style examples, do not copy these:" if planned
        else "Style example for a different business (a yoga studio), do not copy these:",
        examples=", ".join(f'"{e}"' for e in a["examples"]),
    )


def default_instructions() -> dict[str, str]:
    return {k: v["instruction"] for k, v in ANGLES.items()}
