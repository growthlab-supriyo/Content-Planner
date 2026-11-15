"""The content types from the planning sheet (duplicate Tutorial removed) plus Listicle, Myth vs Fact and Cost Guide."""
from __future__ import annotations

import re

# name, what it is (shown to the AI), fits for (hint shown to the AI)
CONTENT_TYPES: list[tuple[str, str, str]] = [
    ("Guide", "In-depth article that fully explains one topic", "treatments, services, 'how to choose'"),
    ("Tutorial", "Step-by-step how-to", "'how to' and preparation searches"),
    ("Case Study", "A real customer's problem, what was done and the result", "proof of results"),
    ("Video", "Short explainer or walkthrough video with a page around it", "procedures, demos"),
    ("FAQs", "Short answers to many related questions", "clusters full of questions"),
    ("Checklist", "Printable or quick list of steps to tick off", "preparation, aftercare, routines"),
    ("Interview", "Q&A with an expert or the business owner", "trust and expertise"),
    ("Vocabulary", "Glossary of terms in plain language", "jargon-heavy topics"),
    ("Testimonials", "Customer reviews and quotes", "local and 'reviews' searches"),
    ("Success Story", "Before/after story of one customer", "long treatments or projects"),
    ("Pillar Page", "Hub page for a pillar that links to all its cluster pages", "each pillar"),
    ("Report", "Original data, statistics or trends", "earning links and press"),
    ("News Article", "Something new: technology, service, event, regulation", "timely topics"),
    ("Expert Opinion", "Expert advice or point of view on a debated question", "fears, myths, choices"),
    ("Comparison", "X vs Y with a clear recommendation", "'vs', 'or', 'which is better' searches"),
    ("Whitepaper", "Long research document, often gated", "B2B and technical buyers"),
    ("Help Article", "Short practical fix for one urgent problem", "symptoms and problems"),
    ("Profile Page", "Page about a person, team or branch", "'best <provider> in <area>' searches"),
    ("Technical Documentation", "Specs, setup or API documentation", "software, products, technical services"),
    ("Troubleshooting", "Diagnose-and-fix guide for something not working", "products, devices, software"),
    ("Listicle", "Numbered list article ('7 reasons…', 'best 10…')", "'best', 'ideas', 'reasons' searches"),
    ("Myth vs Fact", "Corrects common myths and misconceptions", "fears and myths"),
    ("Cost Guide", "Prices, what changes the cost, payment options", "cost and price searches"),
]

TYPE_NAMES = [t[0] for t in CONTENT_TYPES]

# Used when the AI leaves a type out of its decision.
DEFAULT_USED = {"Guide", "FAQs", "Pillar Page", "Comparison", "Help Article", "Cost Guide", "Listicle",
                "Myth vs Fact", "Checklist", "Tutorial"}

_ALIASES = {
    "faq": "FAQs", "faqs": "FAQs", "faq page": "FAQs", "pillar": "Pillar Page", "hub page": "Pillar Page",
    "how-to": "Tutorial", "how to": "Tutorial", "how-to guide": "Tutorial", "step-by-step guide": "Tutorial",
    "list": "Listicle", "list article": "Listicle", "myth": "Myth vs Fact", "myths": "Myth vs Fact",
    "myth busting": "Myth vs Fact", "myths vs facts": "Myth vs Fact", "pricing guide": "Cost Guide",
    "price guide": "Cost Guide", "cost": "Cost Guide", "vs": "Comparison", "comparison article": "Comparison",
    "glossary": "Vocabulary", "reviews": "Testimonials", "review": "Testimonials", "testimonial": "Testimonials",
    "news": "News Article", "blog": "Guide", "blog post": "Guide", "article": "Guide", "service page": "Guide",
    "landing page": "Guide", "help": "Help Article", "help guide": "Help Article", "case": "Case Study",
    "story": "Success Story", "opinion": "Expert Opinion", "expert advice": "Expert Opinion",
    "profile": "Profile Page", "documentation": "Technical Documentation", "docs": "Technical Documentation",
    "troubleshooting guide": "Troubleshooting", "data report": "Report", "video guide": "Video",
}


def normalise_type(name: str) -> str | None:
    """Map what the AI wrote ("FAQ", "myths", "pillar") to one of the 23 type names."""
    if not name:
        return None
    k = re.sub(r"\s+", " ", str(name).strip().lower())
    for t in TYPE_NAMES:
        if t.lower() == k:
            return t
    if k in _ALIASES:
        return _ALIASES[k]
    for t in TYPE_NAMES:  # "Comparison post", "Guide (long form)"
        if k.startswith(t.lower()):
            return t
    return None


def types_prompt_list() -> str:
    return "\n".join(f"- {n}: {d} (fits: {f})" for n, d, f in CONTENT_TYPES)
