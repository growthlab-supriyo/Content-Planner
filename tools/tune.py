"""Try clustering thresholds on a saved run. Usage: python tools/tune.py runs/<file>.json 0.12 0.15 0.18"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planner.embed import OllamaEmbedder
from planner.cluster import cluster_keywords
from planner import quality as Q

p = json.load(open(sys.argv[1], encoding="utf-8"))
loc = Q.location_tokens(p["location"])
key = {v["label"]: k for k, v in __import__("planner.prompts", fromlist=["ANGLES"]).ANGLES.items()}
rows = [dict(r, intent=Q.classify(r["keyword"], key.get(r["angle"], "services"), loc)[0]) for r in p["stage2"]["keywords"]]
emb = OllamaEmbedder()
cache = {}
class Cached:
    near_duplicate = emb.near_duplicate
    def __init__(self, d): self.cluster_distance = d
    def embed(self, t):
        miss = [x for x in t if x not in cache]
        if miss:
            cache.update(zip(miss, emb.embed(miss)))
        import numpy as np
        return np.array([cache[x] for x in t])
for d in map(float, sys.argv[2:]):
    cl = cluster_keywords(rows, Cached(d), location=p['location'])
    sizes = sorted((c["size"] for c in cl), reverse=True)
    print(f"\n===== distance {d}: {len(cl)} clusters, sizes {sizes[:12]} ... singles {sizes.count(1)}")
    for c in cl[:14]:
        print(f"  [{c['intent'][:4]}] ({c['size']}) " + "; ".join(c["keywords"][:7]))
