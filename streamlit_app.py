"""Start file for the free online version on Streamlit Community Cloud.

Online mode offers only the free cloud AI providers (no Ollama), uses fastembed for embeddings, hides the prompt
editor and gives each visitor 3 free runs a day on the owner's key. To run it with Ollama, see README.md.
"""
import os
import runpy
from pathlib import Path

os.environ.setdefault("CP_ONLINE", "1")
runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
