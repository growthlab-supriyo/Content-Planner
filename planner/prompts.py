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
        "check": "local",
    },
    "fears": {
        "label": "Fears & Myths",
        "instruction": "Worries, fears, doubts, myths and misconceptions people search about "
                       "(pain, safety, side effects, 'is it true', 'does it really').",
        "examples": ["is yoga bad for your knees", "does yoga really help lose weight", "yoga injuries myths",
                     "am i too old to start yoga"],
        "check": None,
    },
}

ANGLE_ORDER = list(ANGLES)

USER_TEMPLATE = """Business / topic: {topic}
Location: {location_text}

Angle: {label}
{instruction}

Write {n} different search queries real people type into Google for this angle.
Rules:
- Natural English, all lowercase, 2 to 10 words each.
- Vary the wording. Do not start more than 3 queries with the same two words.
- Do not just add one word to the business name (bad: "{topic_l} cost", "{topic_l} price", "{topic_l} reviews").
- {location_rule}
- No numbering, no quotes, no duplicates.{feedback}

Style example for a different business (a yoga studio), do not copy these:
{examples}

Return JSON only: {{"queries": ["...", "..."]}}"""


def location_rule(angle: str, location: str) -> str:
    if not location:
        return "There is no location: do not add any city or area name."
    if angle == "local":
        return f"Every query must mention {location} (or a nearby area) or 'near me'."
    if angle in ("services", "costs"):
        return f"Mention {location} in about one third of the queries only; the rest are general."
    return "These are general searches: do not add the location."


def build_prompt(angle: str, topic: str, location: str, n: int, feedback: str = "",
                 instructions: dict[str, str] | None = None) -> str:
    a = ANGLES[angle]
    instruction = (instructions or {}).get(angle)
    if not instruction:
        instruction = a.get("instruction_no_location") if (angle == "local" and not location) else None
        instruction = instruction or a["instruction"]
    return USER_TEMPLATE.format(
        topic=topic, topic_l=topic.lower(),
        location_text=location or "none (not a local search)",
        label=a["label"], instruction=instruction, n=n,
        location_rule=location_rule(angle, location),
        feedback=f"\n- {feedback}" if feedback else "",
        examples=", ".join(f'"{e}"' for e in a["examples"]),
    )


def default_instructions() -> dict[str, str]:
    return {k: v["instruction"] for k, v in ANGLES.items()}
