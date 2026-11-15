"""Interactive topical map (vis-network, loaded from a pinned CDN version) for st.components.v1.html."""
from __future__ import annotations

import html
import json

from .topicmap import all_clusters

COLORS = {
    "light": {"bg": "#FFFFFF", "text": "#111827", "muted": "#6B7280", "topic": "#4F46E5", "pillar": "#818CF8",
              "core": "#E0E7FF", "core_border": "#6366F1", "ai": "#FEF3C7", "ai_border": "#D97706",
              "sub": "#F3F4F6", "edge": "#CBD5E1", "button": "#F5F7FB", "border": "#E5E7EB"},
    "dark": {"bg": "#18181B", "text": "#FAFAFA", "muted": "#A1A1AA", "topic": "#F97316", "pillar": "#FB923C",
             "core": "#3F3F46", "core_border": "#FB923C", "ai": "#713F12", "ai_border": "#FACC15",
             "sub": "#27272A", "edge": "#52525B", "button": "#27272A", "border": "#3F3F46"},
}


def graph_data(project: dict) -> tuple[list[dict], list[dict]]:
    s4 = project.get("stage4") or {}
    clusters = {c["id"]: c for c in all_clusters(project)}
    nodes = [{"id": "topic", "label": project["topic"] + (f"\n{project['location']}" if project.get("location") else ""),
              "group": "topic", "title": "Topic"}]
    edges = []
    for p in s4.get("pillars", []):
        pid = f"p{p['id']}"
        nodes.append({"id": pid, "label": p["name"], "group": "pillar",
                      "title": f"Pillar page: {p['title']}\nKeyword: {p['pillar_keyword']}"})
        edges.append({"from": "topic", "to": pid})
        for cid in p["cluster_ids"]:
            c = clusters.get(cid)
            if not c:
                continue
            nid = f"c{cid}"
            tip = f"{c['name']} ({c['intent']}, {c['size']} keywords)\n" + "\n".join(c["keywords"][:8])
            nodes.append({"id": nid, "label": c["name"], "group": "ai" if c["source"] == "AI-suggested" else "core",
                          "title": tip})
            edges.append({"from": pid, "to": nid})
            for i, s in enumerate(c["subtopics"]):
                sid = f"s{cid}_{i}"
                nodes.append({"id": sid, "label": s, "group": "sub", "title": s})
                edges.append({"from": nid, "to": sid, "sub": True})
    return nodes, edges


def map_html(project: dict, theme: str = "light", height: int = 640) -> str:
    c = COLORS.get(theme, COLORS["light"])
    nodes, edges = graph_data(project)
    data = json.dumps({"nodes": nodes, "edges": edges}).replace("</", "<\\/")
    groups = {
        "topic": {"shape": "box", "color": {"background": c["topic"], "border": c["topic"]},
                  "font": {"color": "#FFFFFF", "size": 18, "face": "Plus Jakarta Sans, Arial"}, "margin": 12},
        "pillar": {"shape": "box", "color": {"background": c["pillar"], "border": c["pillar"]},
                   "font": {"color": "#FFFFFF", "size": 15, "face": "Plus Jakarta Sans, Arial"}, "margin": 10},
        "core": {"shape": "box", "color": {"background": c["core"], "border": c["core_border"]},
                 "font": {"color": c["text"], "size": 13, "face": "Plus Jakarta Sans, Arial"}, "margin": 8},
        "ai": {"shape": "box", "color": {"background": c["ai"], "border": c["ai_border"]},
               "shapeProperties": {"borderDashes": [5, 4]},
               "font": {"color": c["text"], "size": 13, "face": "Plus Jakarta Sans, Arial"}, "margin": 8},
        "sub": {"shape": "dot", "size": 5, "color": {"background": c["muted"], "border": c["muted"]},
                "font": {"color": c["muted"], "size": 11, "face": "Plus Jakarta Sans, Arial"}},
    }
    title = html.escape(project["topic"])
    return f"""<!doctype html><html><head><meta charset="utf-8">
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
 body{{margin:0;background:{c['bg']};font-family:'Plus Jakarta Sans',Arial,sans-serif;color:{c['text']}}}
 #bar{{display:flex;gap:8px;align-items:center;padding:6px 2px;font-size:13px;flex-wrap:wrap}}
 button,label{{background:{c['button']};border:1px solid {c['border']};color:{c['text']};border-radius:8px;
   padding:5px 10px;font-size:13px;cursor:pointer;font-family:inherit}}
 .k{{display:inline-block;width:12px;height:12px;border-radius:3px;margin:0 4px -2px 10px}}
 #net{{height:{height}px;border:1px solid {c['border']};border-radius:10px}}
</style></head><body>
<div id="bar">
 <label><input type="checkbox" id="subs"> Show subtopics</label>
 <button id="fit">Fit to screen</button>
 <button id="png">Download PNG</button>
 <span style="color:{c['muted']}">
  <span class="k" style="background:{c['pillar']}"></span>Pillar
  <span class="k" style="background:{c['core']};border:1px solid {c['core_border']}"></span>Cluster page
  <span class="k" style="background:{c['ai']};border:1px dashed {c['ai_border']}"></span>AI-suggested
 </span>
</div>
<div id="net"></div>
<script>
const DATA = {data};
const groups = {json.dumps(groups)};
const nodes = new vis.DataSet(DATA.nodes.filter(n => n.group !== "sub"));
const edges = new vis.DataSet(DATA.edges.filter(e => !e.sub));
const net = new vis.Network(document.getElementById("net"), {{nodes, edges}}, {{
  groups, nodes: {{widthConstraint: {{maximum: 190}}}},
  edges: {{color: "{c['edge']}", width: 1.2, smooth: {{type: "continuous"}}}},
  physics: {{solver: "forceAtlas2Based", forceAtlas2Based: {{gravitationalConstant: -90, springLength: 120}},
            stabilization: {{iterations: 300}}}},
  interaction: {{hover: true, tooltipDelay: 120}}
}});
net.once("stabilizationIterationsDone", () => net.setOptions({{physics: false}}));
document.getElementById("subs").onchange = (e) => {{
  net.setOptions({{physics: true}});
  if (e.target.checked) {{
    nodes.add(DATA.nodes.filter(n => n.group === "sub"));
    edges.add(DATA.edges.filter(x => x.sub));
  }} else {{
    nodes.remove(DATA.nodes.filter(n => n.group === "sub").map(n => n.id));
    edges.remove(edges.get({{filter: x => x.sub}}).map(x => x.id));
  }}
  net.once("stabilizationIterationsDone", () => net.setOptions({{physics: false}}));
  net.stabilize(250);
}};
document.getElementById("fit").onclick = () => net.fit({{animation: true}});
document.getElementById("png").onclick = () => {{
  const src = document.querySelector("#net canvas");
  const out = document.createElement("canvas");
  out.width = src.width; out.height = src.height;
  const ctx = out.getContext("2d");
  ctx.fillStyle = "{c['bg']}"; ctx.fillRect(0, 0, out.width, out.height); ctx.drawImage(src, 0, 0);
  const a = document.createElement("a");
  a.download = "topical-map-{title.replace('"', '')}.png".replace(/\\s+/g, "-").toLowerCase();
  a.href = out.toDataURL("image/png"); a.click();
}};
</script></body></html>"""
