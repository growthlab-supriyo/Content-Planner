"""Run all 6 stages from the command line with the keys in .streamlit/secrets.toml, the same way the online app
does (a different model for each stage, fastembed embeddings), and save the project + Excel to runs/.

Usage: python tools/full_run.py "Dental Clinic" "Kolkata" [queries_per_angle]
"""
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from planner import config as C  # noqa: E402
from planner.cluster import cluster_keywords, filter_keywords  # noqa: E402
from planner.embed import FastEmbedder  # noqa: E402
from planner.export import to_excel  # noqa: E402
from planner.fanout import run_fanout  # noqa: E402
from planner.llm import LLM, FallbackLLM, build_stage_models  # noqa: E402
from planner.plan import build_content_plan  # noqa: E402
from planner.project import filename, new_project, to_json  # noqa: E402
from planner.topicmap import build_topical_map  # noqa: E402


def load_keys() -> dict[str, str]:
    path = ROOT / ".streamlit" / "secrets.toml"
    if not path.exists():
        sys.exit("No .streamlit/secrets.toml yet: copy secrets.toml.example to secrets.toml and add your keys.")
    secrets = tomllib.loads(path.read_text(encoding="utf-8"))
    keys = {name: secrets[p.key_name] for name, p in C.PROVIDERS.items()
            if p.key_name and str(secrets.get(p.key_name) or "").strip()}
    if not keys:
        sys.exit("secrets.toml has no keys filled in.")
    return keys


def main():
    topic, location = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    keys = load_keys()
    print("Keys found for:", ", ".join(keys))
    notes: list[str] = []

    def switched(msg: str) -> None:
        notes.append(msg)
        print("  SWITCH:", msg, flush=True)

    first = next(iter(keys))
    default = FallbackLLM([LLM(first, C.PROVIDERS[first].default_model, keys[first])], on_switch=switched)
    llm = build_stage_models(keys, default, switched)
    emb = FastEmbedder()
    t0 = time.time()
    log = lambda m, f: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731

    project = new_project(topic, location)
    rows, report = run_fanout(llm, emb, topic, location, n, progress=log)
    project["stage2"] = {"keywords": rows, "report": report}
    kept, removed = filter_keywords(rows, topic, llm, location)
    project["stage3"] = {"clusters": cluster_keywords(kept, emb, llm, topic, location, log), "filtered_out": removed}
    project["stage4"] = build_topical_map(project, llm, emb, log)
    project["stage5"], project["stage6"] = build_content_plan(project, llm, emb, log)
    report["stage_models"] = llm.used()

    out = ROOT / "runs"
    out.mkdir(exist_ok=True)
    (out / filename(project, "json")).write_bytes(to_json(project))
    (out / filename(project, "xlsx")).write_bytes(to_excel(project))

    s3, s4, s6 = project["stage3"], project["stage4"], project["stage6"]
    print("\nAngles (" + ("chosen by AI" if report["angles_planned_by_ai"] else "STANDARD 7 - AI failed") + "):")
    for a in report["angle_plan"]:
        print(f"  - {a['label']} [{a['kind']}]: {a['instruction']}")
    print(f"\nKeywords {len(rows)} (AI-tagged {report['tagged_by_ai']}), removed {len(removed)}, "
          f"clusters {len(s3['clusters'])} ({sum(c['size'] == 1 for c in s3['clusters'])} single-keyword), "
          f"pillars {len(s4['pillars'])}: " + "; ".join(p["name"] for p in s4["pillars"]))
    print(f"Pieces {len(s6['rows'])}, no primary {sum(not r['primary'] for r in s6['rows'])}, "
          f"leftover {len(s6['leftover'])}, AI calls {llm.calls}, {time.time() - t0:.0f}s")
    print("\nModel each stage ended on:")
    for stage, label in llm.used().items():
        print(f"  {C.STAGES[stage]}: {label}")
    print("\nModel performance (calls · usable · failed · avg seconds per call):")
    for st in llm.stats():
        print(f"  {st['model']}: {st['calls']} · {st['ok']} · {st['failed']} · {st['avg_seconds']}s"
              + ("  [daily limit reached]" if st["used_up"] else ""))
    if notes or llm.notes:
        print("\nSwitches and notes:")
        for msg in dict.fromkeys(notes + llm.notes):
            print("  -", msg)
    print(f"\nSaved: runs/{filename(project, 'xlsx')}")


if __name__ == "__main__":
    main()
