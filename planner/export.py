"""Excel export, 3 tabs (same layout as samples/sample-output-dental-clinic-newtown.xlsx):

1. Clusters & Keywords   – one row per keyword, grouped by pillar and cluster, with how it is used (primary/secondary)
2. Topical Map           – Topic → Pillar → Cluster (page) → Subtopics, with live keyword / content counts
3. Content Plan & Mapping – one row per page + content type decisions + leftover keywords

Tabs 2 and 3 fill in once Stages 4–6 have run.
"""
from __future__ import annotations

from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from .themes import THEMES
from .topicmap import all_clusters

FONT = "Arial"
INK, MUTED = "1F2937", "6B7280"
PILL = {"Used": "DCFCE7", "Skipped": "FEE2E2", "Core": "E0E7FF", "AI-suggested": "FEF3C7", "Primary": "DCFCE7",
        "Secondary": "E0E7FF", "Leftover": "FEE2E2", "Not mapped yet": "F3F4F6", "Pillar Page": "EDE9FE",
        "Cluster Page": "DCFCE7", "Supporting": "F3F4F6", "Informational": "E0E7FF", "Commercial": "FEF3C7",
        "Transactional": "DCFCE7", "Local": "FFE4E6"}
BAND = PatternFill("solid", fgColor="F9FAFB")
LINE = Border(bottom=Side(style="thin", color="E5E7EB"))
KW_HEADER_ROW, PLAN_HEADER_ROW, MAP_HEADER_ROW = 5, 6, 5


def _title(ws, title: str, project: dict, note: str) -> None:
    ws["A1"] = title
    ws["A1"].font = Font(name=FONT, size=16, bold=True, color=INK)
    ws["A2"] = (f"Topic: {project['topic']}   |   Location: {project.get('location') or '—'}   |   "
                f"Created: {project.get('created', '')[:10]}")
    ws["A2"].font = Font(name=FONT, size=10, color=MUTED)
    ws["A3"] = note
    ws["A3"].font = Font(name=FONT, size=9, italic=True, color=MUTED)


def _header(ws, row: int, cols: list[str], accent: str) -> None:
    for c, h in enumerate(cols, 1):
        if h is None:
            continue
        cell = ws.cell(row=row, column=c, value=h)
        cell.font = Font(name=FONT, size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=accent)
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    ws.row_dimensions[row].height = 22


def _cell(ws, row: int, col: int, value, bold=False, wrap=False, color=INK, center=False, band=False):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = Font(name=FONT, size=10, bold=bold, color=color)
    cell.alignment = Alignment(vertical="top", wrap_text=wrap, horizontal="center" if center else None)
    cell.border = LINE
    if band:
        cell.fill = BAND
    if isinstance(value, str) and value in PILL:
        cell.fill = PatternFill("solid", fgColor=PILL[value])
    return cell


def _widths(ws, widths: list[int]) -> None:
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _section(ws, row: int, text: str) -> None:
    ws.cell(row=row, column=1, value=text).font = Font(name=FONT, size=12, bold=True, color=INK)


def to_excel(project: dict, theme: str = "light") -> bytes:
    accent = THEMES.get(theme, THEMES["light"])["excel_accent"]
    s2 = project.get("stage2") or {}
    s6 = project.get("stage6") or {}
    model = (s2.get("report") or {}).get("model", "AI")
    clusters = all_clusters(project)
    pillars = (project.get("stage4") or {}).get("pillars", [])
    pillar_order = {p["name"]: n for n, p in enumerate(pillars)}
    clusters.sort(key=lambda c: (pillar_order.get(c["pillar"], 99), c["source"] != "Core", -c["size"]))
    rows = s6.get("rows", [])
    usage: dict[str, tuple[str, int]] = {}
    for r in rows:
        if r["primary"]:
            usage[r["primary"]] = ("Primary", r["n"])
        for k in r["secondary"]:
            usage.setdefault(k, ("Secondary", r["n"]))
    leftover = {x["keyword"] for x in s6.get("leftover", [])}
    info = {k["keyword"]: k for k in s2.get("keywords", [])}

    wb = Workbook()
    # ---- Tab 1 ----
    ws = wb.active
    ws.title = "Clusters & Keywords"
    n_ai = s2.get("report", {}).get("tagged_by_ai", 0)
    tags = (f"Intent and funnel stage: tagged by AI for {n_ai} of {len(info)} keywords, rules for the rest. "
            if n_ai else "Intent and funnel stage are rule-based tags. ")
    _title(ws, "Clusters & Keywords", project,
           f"Generated with {model}. {tags}'Used As' shows which page (Content #) each keyword is mapped to.")
    _header(ws, KW_HEADER_ROW, ["Pillar", "Cluster (Page)", "Keyword / Query", "Angle", "Intent", "Funnel Stage",
                                "Used As", "Content #"], accent)
    r = KW_HEADER_ROW + 1
    for n, c in enumerate(clusters):
        band = n % 2 == 0
        for kw in c["keywords"]:
            k = info.get(kw, {})
            used, num = usage.get(kw, ("Leftover" if kw in leftover else "Not mapped yet", None))
            if not rows:
                used, num = "", None
            angle = k.get("angle") or ("AI-suggested (gap check)" if c["source"] == "AI-suggested" else "")
            intent = k.get("intent") or c["intent"]
            vals = [c["pillar"] or "—", c["name"], kw, angle, intent,
                    k.get("funnel") or {"Informational": "TOFU", "Commercial": "MOFU"}.get(intent, "BOFU"), used, num]
            for col, v in enumerate(vals, 1):
                _cell(ws, r, col, v, bold=(col == 2), band=band, center=(col == 8))
            r += 1
    kw_last = max(r - 1, KW_HEADER_ROW + 1)
    _widths(ws, [24, 32, 44, 22, 14, 12, 14, 10])
    ws.freeze_panes = ws.cell(row=KW_HEADER_ROW + 1, column=1)
    ws.auto_filter.ref = f"A{KW_HEADER_ROW}:H{kw_last}"
    removed = (project.get("stage3") or {}).get("filtered_out", [])
    if removed:
        r = kw_last + 3
        _section(ws, r, "Filtered out in Stage 3")
        for x in removed:
            r += 1
            _cell(ws, r, 1, x["reason"], color=MUTED)
            _cell(ws, r, 3, x["keyword"], color=MUTED)

    # ---- Tab 2 ----
    wm = wb.create_sheet("Topical Map")
    _title(wm, "Topical Map", project, "Each cluster is one page; subtopics are its H2 headings. "
                                      "'AI-suggested' = added by the gap check.")
    plan_last = PLAN_HEADER_ROW + max(1, len(rows))
    if pillars:
        _header(wm, MAP_HEADER_ROW, ["Topic", "Pillar", "Cluster (Page)", "Subtopics (H2 headings on the page)",
                                     "Source", "Keywords", "Content Pieces"], accent)
        r = MAP_HEADER_ROW + 1
        first = True
        by_id = {c["id"]: c for c in clusters}
        for p in pillars:
            pfirst = True
            for cid in p["cluster_ids"]:
                c = by_id.get(cid)
                if not c:
                    continue
                _cell(wm, r, 1, project["topic"] if first else None, bold=True)
                _cell(wm, r, 2, p["name"] if pfirst else None, bold=True)
                _cell(wm, r, 3, c["name"])
                _cell(wm, r, 4, "\n".join(c["subtopics"]), wrap=True)
                _cell(wm, r, 5, c["source"])
                _cell(wm, r, 6, f"=COUNTIF('Clusters & Keywords'!$B${KW_HEADER_ROW + 1}:$B${kw_last},C{r})",
                      center=True)
                _cell(wm, r, 7, f"=COUNTIF('Content Plan & Mapping'!$B${PLAN_HEADER_ROW + 1}:$B${plan_last},C{r})",
                      center=True)
                first = pfirst = False
                r += 1
        map_last = r - 1
        for col, v in ((5, "Total"), (6, f"=SUM(F{MAP_HEADER_ROW + 1}:F{map_last})"),
                       (7, f"=SUM(G{MAP_HEADER_ROW + 1}:G{map_last})")):
            c = wm.cell(row=r, column=col, value=v)
            c.font = Font(name=FONT, size=10, bold=True, color=INK)
            c.alignment = Alignment(horizontal="center")
        r += 3
        _section(wm, r, "Pillar pages")
        _header(wm, r + 1, ["", "Pillar", "Pillar Page Title", "Pillar Keyword", "Source"], accent)
        r += 2
        for p in pillars:
            for col, v in ((2, p["name"]), (3, p["title"]), (4, p["pillar_keyword"]), (5, p.get("source", "Core"))):
                _cell(wm, r, col, v, bold=(col == 2), wrap=(col == 3))
            r += 1
        wm.freeze_panes = wm.cell(row=MAP_HEADER_ROW + 1, column=1)
    else:
        wm["A5"] = "Filled in by Stage 4 (run 'Build topical map and content plan')."
        wm["A5"].font = Font(name=FONT, size=10, italic=True, color=MUTED)
    _widths(wm, [16, 26, 34, 52, 14, 11, 15])

    # ---- Tab 3 ----
    wc = wb.create_sheet("Content Plan & Mapping")
    _title(wc, "Content Plan & Keyword Mapping", project,
           "One primary keyword per page, never used as primary twice. Secondary keywords come from the same cluster.")
    if rows:
        _section(wc, 5, "Content Plan")
        _header(wc, PLAN_HEADER_ROW, ["#", "Cluster (Page)", "Pillar", "Content Type", "Page Title",
                                      "Primary Keyword", "Secondary Keywords", "Intent", "Funnel Stage", "Role"],
                accent)
        r = PLAN_HEADER_ROW + 1
        for row in rows:
            vals = [row["n"], row["cluster"], row["pillar"], row["type"], row["title"], row["primary"] or "—",
                    "\n".join(row["secondary"]) or "—", row["intent"], row["funnel"], row["role"]]
            for col, v in enumerate(vals, 1):
                _cell(wc, r, col, v, bold=(col == 5), wrap=(col in (5, 7)), center=(col == 1))
            r += 1
        r += 2
        _section(wc, r, "Content Type Decisions")
        th = r + 1
        _header(wc, th, [None, "Content Type", "Decision", "Pages", "Reason"], accent)
        r = th + 1
        for t in (project.get("stage5") or {}).get("types", []):
            _cell(wc, r, 2, t["type"])
            _cell(wc, r, 3, t["decision"])
            _cell(wc, r, 4, f"=COUNTIF($D${PLAN_HEADER_ROW + 1}:$D${plan_last},B{r})", center=True)
            _cell(wc, r, 5, t["reason"], wrap=True)
            r += 1
        types_last = r - 1
        wc.cell(row=r, column=2, value="Total pages").font = Font(name=FONT, size=10, bold=True)
        c = wc.cell(row=r, column=4, value=f"=SUM(D{th + 1}:D{types_last})")
        c.font = Font(name=FONT, size=10, bold=True)
        c.alignment = Alignment(horizontal="center")
        r += 3
        _section(wc, r, "Leftover Keywords (not mapped to any page)")
        _header(wc, r + 1, [None, "Keyword", "Cluster", "Suggestion"], accent)
        r += 2
        if not s6.get("leftover"):
            wc.cell(row=r, column=2, value="None: every keyword is mapped.").font = Font(
                name=FONT, size=10, italic=True, color=MUTED)
        for x in s6.get("leftover", []):
            for col, v in ((2, x["keyword"]), (3, x["cluster"]), (4, x["suggestion"])):
                _cell(wc, r, col, v)
            r += 1
        wc.freeze_panes = wc.cell(row=PLAN_HEADER_ROW + 1, column=1)
    else:
        wc["A5"] = "Filled in by Stages 5 and 6 (run 'Build topical map and content plan')."
        wc["A5"].font = Font(name=FONT, size=10, italic=True, color=MUTED)
    _widths(wc, [5, 30, 24, 18, 52, 34, 36, 14, 12, 14])

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
