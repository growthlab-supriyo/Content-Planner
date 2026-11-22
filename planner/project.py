"""A project = everything one run produced (the topic, the business brief and each stage's results)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .config import VERSION

FORMAT = "content-planner-project"


def new_project(topic: str, location: str) -> dict:
    return {
        "format": FORMAT, "app_version": VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "topic": topic.strip(), "location": location.strip(),
        "stage2": None,   # {"keywords": [...], "report": {...}}
        "stage3": None,   # {"clusters": [...], "filtered_out": [...]}
    }


def to_json(project: dict) -> bytes:
    return json.dumps(project, ensure_ascii=False, indent=2).encode("utf-8")


def from_json(data: bytes | str) -> dict:
    try:
        p = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("This file isn't a Content Planner project (not valid JSON).") from None
    if not isinstance(p, dict) or p.get("format") != FORMAT or not p.get("topic"):
        raise ValueError("This file isn't a Content Planner project.")
    s2 = p.get("stage2")
    if s2 is not None:
        kws = s2.get("keywords") if isinstance(s2, dict) else None
        if not isinstance(kws, list) or not all(isinstance(k, dict) and isinstance(k.get("keyword"), str) for k in kws):
            raise ValueError("The keyword list in this project file is damaged.")
    p.setdefault("location", "")
    p.setdefault("stage3", None)
    return p


def filename(project: dict, ext: str) -> str:
    import re

    base = "-".join(x for x in (project["topic"], project.get("location", "")) if x)
    base = re.sub(r"[^A-Za-z0-9]+", "-", base).strip("-").lower()[:60] or "project"
    return f"content-plan-{base}.{ext}"
