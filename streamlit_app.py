"""Start file for the free online version on Streamlit Community Cloud.

Online mode offers only the free cloud AI providers (no Ollama), uses fastembed for embeddings, hides the prompt
editor and gives each visitor 3 free runs a day on the owner's key. To run it with Ollama, see README.md.
"""
import os
import runpy
import sys
from pathlib import Path


def fresh_planner() -> None:
    """After a push, Streamlit Cloud reruns app.py with the new code but keeps the old planner/*.py modules in
    memory, so a new import fails ("cannot import name ...") until the app is rebooted. When any planner file has
    changed since it was loaded, drop the cached modules so they load fresh."""
    stamp = max(p.stat().st_mtime for p in Path(__file__).with_name("planner").glob("*.py"))
    loaded = sys.modules.get("planner")
    if loaded is not None and getattr(loaded, "_loaded_stamp", None) != stamp:
        for name in [n for n in sys.modules if n == "planner" or n.startswith("planner.")]:
            del sys.modules[name]
    import planner

    planner._loaded_stamp = stamp


os.environ.setdefault("CP_ONLINE", "1")
fresh_planner()
runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
