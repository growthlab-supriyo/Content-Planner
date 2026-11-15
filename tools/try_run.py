"""Run Stages 2–3 from the command line (no Streamlit) and save the project + Excel.

Usage: python tools/try_run.py "Dental Clinic" "Newtown, Kolkata" [queries_per_angle] [provider] [model]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planner.cluster import cluster_keywords, filter_keywords  # noqa: E402
from planner.embed import OllamaEmbedder  # noqa: E402
from planner.export import to_excel  # noqa: E402
from planner.fanout import run_fanout  # noqa: E402
from planner.llm import LLM  # noqa: E402
from planner.project import filename, new_project, to_json  # noqa: E402


def main():
    topic, location = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    provider = sys.argv[4] if len(sys.argv) > 4 else "Local Ollama"
    model = sys.argv[5] if len(sys.argv) > 5 else None
    llm, emb = LLM(provider, model), OllamaEmbedder()
    t0 = time.time()
    log = lambda m, f: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    project = new_project(topic, location)
    rows, report = run_fanout(llm, emb, topic, location, n, progress=log)
    project["stage2"] = {"keywords": rows, "report": report}
    kept, removed = filter_keywords(rows, topic)
    project["stage3"] = {"clusters": cluster_keywords(kept, emb, llm, topic, location, log), "filtered_out": removed}
    out = Path(__file__).resolve().parents[1] / "runs"
    out.mkdir(exist_ok=True)
    (out / filename(project, "json")).write_bytes(to_json(project))
    (out / filename(project, "xlsx")).write_bytes(to_excel(project))
    print(json.dumps({k: v for k, v in report.items() if k != "angles"}, indent=1))
    for a in report["angles"]:
        print(a)
    print(f"TOTAL {time.time() - t0:.0f}s, keywords {len(rows)}, clusters {len(project['stage3']['clusters'])}, "
          f"filtered {len(removed)}")


if __name__ == "__main__":
    main()
