import os

from streamlit.testing.v1 import AppTest


def _run(online):
    if online:
        os.environ["CP_ONLINE"] = "1"
    else:
        os.environ.pop("CP_ONLINE", None)
    import importlib
    import planner.config
    importlib.reload(planner.config)
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.run()
    return at


def test_local_app_starts():
    at = _run(False)
    assert not at.exception
    assert "Local Ollama" in at.sidebar.selectbox[0].options


def test_online_app_hides_ollama_and_prompt_editor():
    at = _run(True)
    assert not at.exception
    assert "Local Ollama" not in at.sidebar.selectbox[0].options
    assert not any("fan-out prompts" in e.label for e in at.sidebar.expander)
    os.environ.pop("CP_ONLINE", None)


def test_empty_topic_shows_error():
    at = _run(False)
    at.button[0].click().run()
    assert any("Enter a service" in e.value for e in at.error)


def test_results_view_renders_saved_project():
    from pathlib import Path
    from planner.project import from_json
    runs = sorted(Path(__file__).resolve().parents[1].glob("runs/*.json"))
    if not runs:
        import pytest
        pytest.skip("no saved run")
    at = _run(False)
    at.session_state["project"] = from_json(runs[0].read_bytes())
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Clusters (pages)" in labels and "Keywords" in labels
