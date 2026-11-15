from pathlib import Path

from openpyxl import load_workbook
import io

import pytest

from planner.export import to_excel
from planner.project import filename, from_json, new_project, to_json
from planner.quota import Quota
from planner.themes import config_toml

ROOT = Path(__file__).resolve().parents[1]


def sample_project():
    p = new_project("Dental Clinic", "Newtown")
    p["stage2"] = {"keywords": [{"keyword": "root canal cost", "angle": "Costs", "intent": "Commercial",
                                 "funnel": "MOFU"}], "report": {"model": "fake"}}
    p["stage3"] = {"clusters": [{"id": 1, "name": "Root Canal Cost", "intent": "Commercial", "size": 1,
                                 "keywords": ["root canal cost"], "main_keyword": "root canal cost"}],
                   "filtered_out": [{"keyword": "dental clinic", "reason": "Generic two-word term"}]}
    return p


def test_project_round_trip_and_bad_files():
    p = sample_project()
    assert from_json(to_json(p)) == p
    for bad in (b"not json", b'{"format": "other"}', b'{"format": "content-planner-project", "topic": "x", '
                                                     b'"stage2": {"keywords": [1]}}'):
        with pytest.raises(ValueError):
            from_json(bad)
    assert filename(p, "xlsx") == "content-plan-dental-clinic-newtown.xlsx"


def test_excel_has_three_tabs_and_rows():
    for mode in ("light", "dark"):
        wb = load_workbook(io.BytesIO(to_excel(sample_project(), mode)))
        assert wb.sheetnames == ["Clusters & Keywords", "Topical Map", "Content Plan & Mapping"]
        ws = wb["Clusters & Keywords"]
        # before Stages 4–6: no pillar yet, keyword not mapped yet
        assert ws["A5"].value == "Pillar" and ws["B6"].value == "Root Canal Cost"
        assert ws["C6"].value == "root canal cost" and ws["G6"].value in (None, "")
        assert "Stage 4" in wb["Topical Map"]["A5"].value


def test_quota_limits_each_visitor_and_total():
    q = Quota(per_visitor=3, daily_cap=4)
    a, b = Quota.visitor_id("1.1.1.1"), Quota.visitor_id("2.2.2.2")
    assert [q.use(a) for _ in range(4)] == [True, True, True, False]
    assert q.remaining(a) == 0 and q.remaining(b) == 1
    assert q.use(b) and not q.use(b) and q.cap_reached()


def test_streamlit_config_matches_themes():
    assert (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8") == config_toml()
