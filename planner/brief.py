"""The business brief: what the user told us about the business and why they want a content plan.

The four answers (goal, offerings, customers, who the plan is for) are added to the top of every AI request, so
every stage plans for this business: angles follow its customers, pillars its real service lines, content types and
titles its goal. With no answers, the planner works from the topic alone, as before.
"""
from __future__ import annotations

# Goal (shown in the drop-down) -> what it means for the plan
GOALS: dict[str, str] = {
    "Get enquiries and bookings":
        "Favour pages that lead to an enquiry or booking: service pages, prices and costs, comparisons, "
        "'near me' and location pages, and answers to the worries that stop people booking. Most pages MOFU/BOFU.",
    "Sell products online":
        "Favour pages that lead to a purchase: product and category pages, gift and buying guides, 'best ... for' "
        "lists, comparisons, occasions and seasons. No booking or 'near me' pages unless the business has a shop.",
    "Build traffic and authority":
        "Favour pages that answer what people want to learn: questions, how-to guides, explainers, myths, "
        "glossaries and in-depth guides that earn links. Most pages TOFU, with a few that lead to the business.",
    "Support existing customers":
        "Favour pages that help people who already bought or booked: how-to and aftercare guides, troubleshooting, "
        "FAQs, maintenance and 'what to expect' pages.",
}

# Grey examples shown inside the empty fields (they disappear as soon as the user types)
EXAMPLES = {
    "topic": "e.g. Travel agency · Personalised gifts · Dental clinic · Coaching institute",
    "location": "e.g. Kolkata · Newtown, Kolkata · (empty if you sell online across India)",
    "offerings": ("Travel agency: Kashmir, Kerala and Goa packages, honeymoon tours, visa help\n"
                  "Gift shop: personalised mugs and frames, gift hampers, corporate gifts\n"
                  "Dental clinic: root canal, implants, braces, kids' dentistry\n"
                  "Institute: JEE and NEET coaching, foundation courses, spoken English"),
    "customers": ("Travel agency: families, honeymoon couples, senior citizens\n"
                  "Gift shop: people buying for a partner or parents, companies buying in bulk\n"
                  "Dental clinic: families with children, working adults, seniors\n"
                  "Institute: students in classes 8–12 and their parents"),
    "plan_for": ("e.g. My own travel agency · My gift shop · A dental clinic, my client (I'm an agency) · "
                 "A coaching institute I work for"),
}

FIELDS = ("goal", "offerings", "customers", "plan_for")


def clean(brief: dict | None) -> dict:
    """Only the four known fields, trimmed; empty ones dropped."""
    brief = brief or {}
    return {k: " ".join(str(brief.get(k) or "").split())[:500] for k in FIELDS if str(brief.get(k) or "").strip()}


def brief_text(brief: dict | None) -> str:
    """The block added to the top of every AI request ("" when there is no brief)."""
    b = clean(brief)
    if not b:
        return ""
    lines = ["About this business (use it in every decision):"]
    if "goal" in b:
        lines.append(f"- Goal of the content: {b['goal']}. {GOALS.get(b['goal'], '')}".rstrip())
    if "offerings" in b:
        lines.append(f"- Services or products it offers: {b['offerings']}. Plan only for these and the topic around "
                     f"them; don't plan pages selling services or products it doesn't offer.")
    if "customers" in b:
        lines.append(f"- Its customers: {b['customers']}. Think and search like them.")
    if "plan_for" in b:
        lines.append(f"- Who the plan is for: {b['plan_for']}. If it is for a client, avoid first-person claims "
                     f"('we', 'our clinic') in titles.")
    return "\n".join(lines)


def summary(brief: dict | None) -> str:
    """One line for the Excel header and the results page."""
    b = clean(brief)
    names = {"goal": "Goal", "offerings": "Offers", "customers": "Customers", "plan_for": "For"}
    return "   |   ".join(f"{names[k]}: {v}" for k, v in b.items())
