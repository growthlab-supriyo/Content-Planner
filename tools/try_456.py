"""Run Stages 4–6 on a saved project (from tools/try_run.py or the app) and save the result + Excel.

Usage: python tools/try_456.py runs/<project>.json [provider] [model]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planner.embed import OllamaEmbedder  # noqa: E402
from planner.export import to_excel  # noqa: E402
from planner.llm import LLM  # noqa: E402
from planner.plan import build_content_plan  # noqa: E402
from planner.project import from_json, to_json  # noqa: E402
from planner.topicmap import build_topical_map  # noqa: E402


def main():
    path = Path(sys.argv[1])
    provider = sys.argv[2] if len(sys.argv) > 2 else "Local Ollama"
    model = sys.argv[3] if len(sys.argv) > 3 else None
    project = from_json(path.read_bytes())
    llm, emb = LLM(provider, model), OllamaEmbedder()
    t0 = time.time()
    log = lambda m, f: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
    project["stage4"] = build_topical_map(project, llm, emb, log)
    project["stage5"], project["stage6"] = build_content_plan(project, llm, emb, log)
    path.write_bytes(to_json(project))
    path.with_suffix(".xlsx").write_bytes(to_excel(project))
    s4, s6 = project["stage4"], project["stage6"]
    print(f"TOTAL {time.time() - t0:.0f}s, AI calls {llm.calls}, pillars {len(s4['pillars'])}, "
          f"AI-suggested pages {len(s4['gap_clusters'])}, content pieces {len(s6['rows'])}, "
          f"leftover {len(s6['leftover'])}")


if __name__ == "__main__":
    main()
