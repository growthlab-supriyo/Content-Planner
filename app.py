"""Content Planner – Streamlit app.

Stages: 1 input · 2 query fan-out · 3 filter and cluster · 4 topical map · 5 content ideas · 6 keyword mapping.

Local:  streamlit run app.py            (Ollama + optional free cloud providers)
Online: streamlit_app.py sets CP_ONLINE=1 (free cloud providers only, fastembed embeddings)
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from planner import config as C
from planner import quality as Q
from planner.cluster import cluster_keywords, filter_keywords
from planner.embed import FastEmbedder, OllamaEmbedder
from planner.export import to_excel
from planner.fanout import run_fanout
from planner.llm import LLM, FallbackLLM, LLMError, ollama_running
from planner.mapviz import map_html
from planner.plan import build_content_plan
from planner.project import filename, from_json, new_project, to_json
from planner.prompts import ANGLES, default_instructions
from planner.quota import Quota
from planner.topicmap import all_clusters, build_topical_map

st.set_page_config(page_title=C.APP_NAME, page_icon=":material/hub:", layout="wide")


# ---------- cached resources ----------
@st.cache_resource(show_spinner="Loading the embedding model (first run only)…")
def get_embedder(kind: str):
    return OllamaEmbedder() if kind == "ollama" else FastEmbedder()


@st.cache_resource
def get_quota() -> Quota:
    return Quota()


def secret(name: str) -> str | None:
    try:
        v = st.secrets.get(name)
    except Exception:  # no secrets.toml
        v = None
    return v or os.environ.get(name) or None


def visitor_ip() -> str | None:
    try:
        ip = st.context.ip_address
    except Exception:
        ip = None
    if not ip:
        fwd = st.context.headers.get("X-Forwarded-For", "") if hasattr(st.context, "headers") else ""
        ip = fwd.split(",")[0].strip() or None
    return ip


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


ss = st.session_state
ss.setdefault("project", None)
ss.setdefault("instructions", default_instructions())
ss.setdefault("notices", [])
ss.setdefault("paid", set())  # projects (by created time) whose run already used a free run

# ---------- sidebar: AI settings ----------
with st.sidebar:
    st.markdown(f"### :material/hub: {C.APP_NAME}")
    st.caption("Online version · free cloud AI" if C.ONLINE else "Local version · runs on this computer")

    providers = C.available_providers()
    provider = st.selectbox("AI provider", providers, index=providers.index(C.DEFAULT_PROVIDER))
    p = C.PROVIDERS[provider]
    model = st.text_input("Model", value=secret(f"{(p.key_name or 'OLLAMA').split('_')[0]}_MODEL") or p.default_model,
                          key=f"model_{provider}", help="Free model names change now and then; you can type another.")

    owner_key = secret(p.key_name) if p.key_name else None
    user_key = None
    quota = get_quota()
    vid = Quota.visitor_id(visitor_ip())
    if p.key_name:
        if C.ONLINE:
            user_key = st.text_input("Your own API key (optional)", type="password", help=p.key_help,
                                     key=f"key_{provider}") or None
            if user_key:
                st.caption(":material/key: Using your key: no daily limit here. It's kept only for this session.")
            elif owner_key:
                left = quota.remaining(vid)
                st.caption(f":material/bolt: Free runs left today: **{left} of {C.FREE_RUNS_PER_DAY}**")
                if left == 0:
                    st.info(f"Add your own free key to keep going. {p.key_help}")
            else:
                st.warning(f"Add your free {provider} key to run. {p.key_help}")
        else:
            user_key = st.text_input(f"{provider} API key", type="password", value=owner_key or "",
                                     help=f"{p.key_help}. Or save it as {p.key_name} in .streamlit/secrets.toml",
                                     key=f"key_{provider}") or None

    ollama_ok, ollama_models = (False, []) if C.ONLINE else ollama_running()
    fallback_local = False
    if not C.ONLINE:
        if ollama_ok:
            st.caption(f":material/check_circle: Ollama is running ({len(ollama_models)} models)")
            if provider == "Local Ollama" and not any(m.split(":")[0] == model.split(":")[0] for m in ollama_models):
                st.warning(f"Model '{model}' isn't installed. Run: `ollama pull {model}`")
        else:
            st.caption(":material/error: Ollama isn't running. Start it, or use a cloud provider.")
        if provider != "Local Ollama":
            fallback_local = st.checkbox("If the free limit is reached, continue on local Ollama", value=ollama_ok)

    n_per_angle = st.slider("Queries per angle", 15, C.MAX_QUERIES_PER_ANGLE if not C.ONLINE else 40,
                            C.DEFAULT_QUERIES_PER_ANGLE, step=5,
                            help="7 angles are asked separately. More queries = longer run.")

    if not C.ONLINE:
        with st.expander(":material/tune: Advanced: fan-out prompts"):
            st.caption("What the AI is asked for each angle. Rules against repetition are always added.")
            for key, a in ANGLES.items():
                ss["instructions"][key] = st.text_area(a["label"], ss["instructions"][key], key=f"ins_{key}",
                                                       height=90)
            if st.button("Reset to defaults", icon=":material/restart_alt:"):
                ss["instructions"] = default_instructions()
                for key in ANGLES:
                    ss.pop(f"ins_{key}", None)
                st.rerun()


def embedder_kind() -> str:
    if C.ONLINE:
        return "fastembed"
    if ollama_ok and any(m.startswith(C.OLLAMA_EMBED_MODEL) for m in ollama_models):
        return "ollama"
    return "fastembed"


def build_llm(notice) -> FallbackLLM:
    key = user_key or owner_key
    chain = [LLM(provider, model, key)]
    if fallback_local and ollama_ok:
        chain.append(LLM("Local Ollama", C.PROVIDERS["Local Ollama"].default_model))
    if C.ONLINE and not user_key:  # owner's other free keys as backup
        for name in C.available_providers():
            other = C.PROVIDERS[name]
            if name != provider and other.key_name and secret(other.key_name):
                chain.append(LLM(name, secret(f"{name.upper()}_MODEL") or other.default_model, secret(other.key_name)))
    return FallbackLLM(chain, on_switch=notice)


def run_stage3(project: dict, rows: list[dict], llm, progress) -> None:
    kept, removed = filter_keywords(rows, project["topic"])
    clusters = cluster_keywords(kept, get_embedder(embedder_kind()), llm, project["topic"], project["location"],
                                progress)
    project["stage3"] = {"clusters": clusters, "filtered_out": removed}
    for k in ("stage4", "stage5", "stage6"):  # later stages depend on the clusters
        project[k] = None


def run_stages_456(project: dict, llm, progress) -> None:
    emb = get_embedder(embedder_kind())
    project["stage4"] = build_topical_map(project, llm, emb, lambda m, f: progress(m, 0.4 * f))
    project["stage5"], project["stage6"] = build_content_plan(project, llm, emb,
                                                              lambda m, f: progress(m, 0.4 + 0.6 * f))


# ---------- main ----------
st.title("Content planner")
st.caption("Type a service, category or topic. The planner finds the searches around it, removes the noise and "
           "groups them into clusters. Each cluster becomes one page.")

tab_new, tab_open = st.tabs([":material/add: New plan", ":material/upload: Continue a saved project"])
with tab_new:
    with st.form("input"):
        c1, c2 = st.columns([3, 2])
        topic = c1.text_input("Service / category / topic", placeholder="Dental clinic")
        location = c2.text_input("Location (optional)", placeholder="Newtown, Kolkata")
        full = st.checkbox("Also build the topical map, content plan and keyword mapping (Stages 4–6)", value=True,
                           help="Turn off to check the keyword clusters first; you can build the rest later.")
        go = st.form_submit_button("Build content plan", type="primary", icon=":material/play_arrow:")

with tab_open:
    up = st.file_uploader("Project file (.json) downloaded from this app", type=["json"])
    if up is not None and st.button("Open project", icon=":material/folder_open:"):
        try:
            ss["project"] = from_json(up.getvalue())
            st.success(f"Opened: {ss['project']['topic']}")
        except ValueError as e:
            st.error(str(e))

if go:
    problem = None
    if not topic.strip():
        problem = "Enter a service, category or topic first."
    elif len(topic) > 80 or len(location) > 80:
        problem = "Keep the topic and location under 80 characters."
    elif p.key_name and not (user_key or owner_key):
        problem = f"{provider} needs an API key. {p.key_help}"
    elif provider == "Local Ollama" and not ollama_ok:
        problem = "Ollama isn't running. Start the Ollama app, then try again."
    elif C.ONLINE and not user_key and not quota.use(vid):
        problem = (f"You've used today's {C.FREE_RUNS_PER_DAY} free runs. Add your own free key in the sidebar "
                   f"to keep going. {p.key_help}")
    if problem:
        st.error(problem)
    else:
        project = new_project(topic, location)
        ss["paid"].add(project["created"])
        ss["notices"] = []
        llm = None
        try:
            with st.status("Building your content plan…", expanded=True) as status:
                bar = st.progress(0.0, text="Starting")
                note = st.empty()

                def notice(msg):
                    ss["notices"].append(msg)
                    note.warning(msg)

                llm = build_llm(notice)
                st.write(f"AI: **{llm.label}** · embeddings: **{embedder_kind()}**")
                share = 0.55 if full else 0.8
                rows, report = run_fanout(llm, get_embedder(embedder_kind()), topic, location, n_per_angle,
                                          ss["instructions"] if not C.ONLINE else None,
                                          progress=lambda m, f: bar.progress(min(f, 1.0) * share, text=m))
                project["stage2"] = {"keywords": rows, "report": report}
                stop = 0.7 if full else 1.0
                run_stage3(project, rows, llm,
                           lambda m, f: bar.progress(share + (stop - share) * min(f, 1.0), text=m))
                ss["project"] = project  # keep Stages 1–3 even if a later stage fails
                if full:
                    run_stages_456(project, llm, lambda m, f: bar.progress(0.7 + 0.3 * min(f, 1.0), text=m))
                bar.progress(1.0, text="Done")
                status.update(label="Content plan ready" if full else "Keyword clusters ready", state="complete",
                              expanded=False)
        except LLMError as e:
            st.error(str(e))
            if C.ONLINE and not user_key and not project.get("stage2"):
                quota.refund(vid)  # nothing was made, so the free run isn't counted
                ss["paid"].discard(project["created"])
                st.caption("This run didn't count towards your free runs.")
        finally:
            ss["notices"] += [n for n in getattr(llm, "notes", []) if n not in ss["notices"]]

# ---------- results ----------
project = ss.get("project")
if project and project.get("stage2"):
    s2, s3 = project["stage2"], project.get("stage3") or {"clusters": [], "filtered_out": []}
    rep = s2.get("report", {})
    st.divider()
    loc = f" · {project['location']}" if project.get("location") else ""
    st.subheader(f"{project['topic']}{loc}")
    for n in ss.get("notices", []):
        st.warning(n)

    s4, s5, s6 = project.get("stage4"), project.get("stage5"), project.get("stage6")
    clusters = s3["clusters"]
    m = st.columns(6)
    m[0].metric("Keywords", len(s2["keywords"]))
    m[1].metric("Clusters (pages)", len(clusters))
    m[2].metric("Pillars", len(s4["pillars"]) if s4 else "—")
    m[3].metric("AI-suggested pages", len(s4["gap_clusters"]) if s4 else "—")
    m[4].metric("Content pieces", len(s6["rows"]) if s6 else "—")
    m[5].metric("Questions", rep.get("questions", 0))

    if clusters and not s6:
        with st.container(border=True):
            st.markdown("**Next: topical map, content plan and keyword mapping (Stages 4–6)**")
            st.caption("Groups the clusters into pillars, adds missing pages, picks content types and maps every "
                       "keyword to a page. On a local model this takes several minutes.")
            if st.button("Build topical map and content plan", type="primary", icon=":material/account_tree:"):
                problem = None
                if p.key_name and not (user_key or owner_key):
                    problem = f"{provider} needs an API key. {p.key_help}"
                elif provider == "Local Ollama" and not ollama_ok:
                    problem = "Ollama isn't running. Start the Ollama app, then try again."
                elif C.ONLINE and not user_key and project["created"] not in ss["paid"]:
                    if quota.use(vid):
                        ss["paid"].add(project["created"])
                    else:
                        problem = (f"You've used today's {C.FREE_RUNS_PER_DAY} free runs. Add your own free key in "
                                   f"the sidebar to keep going.")
                if problem:
                    st.error(problem)
                else:
                    try:
                        with st.status("Building topical map and content plan…", expanded=True) as status:
                            bar = st.progress(0.0, text="Starting")
                            run_stages_456(project, build_llm(lambda msg: st.warning(msg)),
                                           lambda m_, f: bar.progress(min(f, 1.0), text=m_))
                            status.update(label="Content plan ready", state="complete", expanded=False)
                        st.rerun()
                    except LLMError as e:
                        st.error(str(e))

    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("Download Excel", to_excel(project, theme_mode()), filename(project, "xlsx"),
                       icon=":material/download:", type="primary", use_container_width=True)
    d2.download_button("Save project (.json)", to_json(project), filename(project, "json"),
                       icon=":material/save:", use_container_width=True,
                       help="Upload this file later under 'Continue a saved project'.")

    t_map, t_plan, t_cl, t_kw, t_types, t_q, t_f = st.tabs([
        ":material/account_tree: Topical map", ":material/calendar_month: Content plan",
        ":material/workspaces: Clusters", ":material/edit_note: Keywords", ":material/category: Content types",
        ":material/verified: Quality report", ":material/filter_alt_off: Filtered out"])

    with t_map:
        if not s4:
            st.caption("Build the topical map (Stages 4–6) to see it here.")
        else:
            components.html(map_html(project, theme_mode()), height=700, scrolling=False)
            st.caption("Drag to move, scroll to zoom, hover a page to see its keywords. Dashed = AI-suggested by the "
                       "gap check.")
            allc = {c["id"]: c for c in all_clusters(project)}
            for pl in s4["pillars"]:
                with st.expander(f"**{pl['name']}** · {len(pl['cluster_ids'])} pages · pillar page: {pl['title']}"):
                    st.dataframe(pd.DataFrame([{
                        "Cluster (page)": allc[i]["name"], "Source": allc[i]["source"], "Intent": allc[i]["intent"],
                        "Subtopics (H2s)": " · ".join(allc[i]["subtopics"]), "Keywords": allc[i]["size"]}
                        for i in pl["cluster_ids"] if i in allc]), hide_index=True, use_container_width=True)

    with t_plan:
        if not s6:
            st.caption("Build the content plan (Stages 4–6) to see it here.")
        else:
            rows = s6["rows"]
            f1, f2 = st.columns([2, 3])
            q = f1.text_input("Search the plan", placeholder="title, keyword, cluster…", label_visibility="collapsed")
            roles = f2.pills("Role", ["Pillar Page", "Cluster Page", "Supporting"], selection_mode="multi",
                             default=["Pillar Page", "Cluster Page", "Supporting"], label_visibility="collapsed")
            view = [r for r in rows if r["role"] in (roles or []) and (not q or q.lower() in " ".join(
                [r["title"], r["primary"], r["cluster"], r["pillar"], " ".join(r["secondary"])]).lower())]
            st.caption(f"{len(view)} of {len(rows)} pieces · one primary keyword per page, never used twice")
            st.dataframe(pd.DataFrame([{
                "#": r["n"], "Pillar": r["pillar"], "Cluster (page)": r["cluster"], "Type": r["type"],
                "Title": r["title"], "Primary keyword": r["primary"], "Secondary keywords": ", ".join(r["secondary"]),
                "Intent": r["intent"], "Funnel": r["funnel"], "Role": r["role"]} for r in view]),
                hide_index=True, use_container_width=True, height=560,
                column_config={"Title": st.column_config.TextColumn(width="large"),
                               "Secondary keywords": st.column_config.TextColumn(width="large")})
            if s6["leftover"]:
                with st.expander(f"{len(s6['leftover'])} leftover keywords (not mapped to any page)"):
                    st.dataframe(pd.DataFrame(s6["leftover"]), hide_index=True, use_container_width=True)
            else:
                st.caption(":material/check_circle: Every keyword is mapped to a page.")

    with t_types:
        if not s5:
            st.caption("Build the content plan (Stages 4–6) to see which content types are used.")
        else:
            used = [t for t in s5["types"] if t["decision"] == "Used"]
            st.caption(f"{len(used)} of {len(s5['types'])} content types used to cover the topic.")
            st.dataframe(pd.DataFrame([{"Content type": t["type"], "Decision": t["decision"],
                                        "Pages": t.get("pages", 0), "Reason": t["reason"]} for t in s5["types"]]),
                         hide_index=True, use_container_width=True, height=600,
                         column_config={"Reason": st.column_config.TextColumn(width="large")})

    with t_cl:
        f1, f2 = st.columns([2, 3])
        search = f1.text_input("Search clusters", placeholder="cost, pain, kids…", label_visibility="collapsed")
        intents = f2.pills("Intent", Q.INTENTS, selection_mode="multi", default=list(Q.INTENTS),
                           label_visibility="collapsed")
        shown = [c for c in clusters if (c["intent"] in (intents or []))
                 and (not search or search.lower() in (c["name"] + " " + " ".join(c["keywords"])).lower())]
        st.caption(f"{len(shown)} of {len(clusters)} clusters")
        cols = st.columns(2)
        for i, c in enumerate(shown):
            with cols[i % 2].container(border=True):
                st.markdown(f"**{c['name']}**")
                st.caption(f"{c['intent']} · {c['size']} keywords · main keyword: {c['main_keyword']}")
                st.markdown("\n".join(f"- {k}" for k in c["keywords"][:8]))
                if len(c["keywords"]) > 8:
                    with st.expander(f"{len(c['keywords']) - 8} more"):
                        st.markdown("\n".join(f"- {k}" for k in c["keywords"][8:]))

    with t_kw:
        st.caption("Edit, delete or add keywords, then rebuild the clusters. Intent decides which cluster group a "
                   "keyword can join.")
        df = pd.DataFrame(s2["keywords"], columns=["keyword", "angle", "intent", "funnel"])
        edited = st.data_editor(
            df, num_rows="dynamic", use_container_width=True, hide_index=True, key="kw_editor",
            column_config={
                "keyword": st.column_config.TextColumn("Keyword", required=True, width="large"),
                "angle": st.column_config.TextColumn("Angle", disabled=True),
                "intent": st.column_config.SelectboxColumn("Intent", options=list(Q.INTENTS), required=True),
                "funnel": st.column_config.TextColumn("Funnel", disabled=True),
            })
        if s6:
            st.caption(":material/info: Rebuilding clusters clears the topical map and content plan; build them "
                       "again afterwards.")
        if st.button("Rebuild clusters with my edits", icon=":material/refresh:"):
            rows, seen = [], set()
            for r in edited.to_dict("records"):
                kw = Q.clean(r.get("keyword") or "")
                if not kw or kw in seen:
                    continue
                seen.add(kw)
                intent = r.get("intent") if r.get("intent") in Q.INTENTS else "Informational"
                rows.append({"keyword": kw, "angle": r.get("angle") or "Added by you", "intent": intent,
                             "funnel": Q.FUNNEL[intent]})
            try:
                with st.spinner("Rebuilding clusters…"):
                    llm = build_llm(lambda msg: None) if (not p.key_name or user_key or owner_key) else None
                    s2["keywords"] = rows
                    run_stage3(project, rows, llm, lambda m, f: None)
                st.rerun()
            except LLMError as e:
                st.error(str(e))

    with t_q:
        st.caption(f"Made with {rep.get('model', '')} · embeddings: {rep.get('embedder', '')} · "
                   f"{rep.get('ai_calls', '?')} AI calls")
        qa = pd.DataFrame([{
            "Angle": a.get("angle"), "Asked": a.get("asked"), "Kept": a.get("kept"),
            "Attempts": a.get("attempts", 0), "Repetitive removed": a.get("dropped_repetitive", 0),
            "On-angle share": a.get("coverage"), "Notes": " ".join(a.get("notes", [])) or "OK",
        } for a in rep.get("angles", [])])
        st.dataframe(qa, hide_index=True, use_container_width=True,
                     column_config={"On-angle share": st.column_config.ProgressColumn(
                         "On-angle share", min_value=0, max_value=1, format="percent")})
        if rep.get("near_duplicate_examples"):
            st.markdown("**Near-duplicates removed (kept → removed)**")
            st.dataframe(pd.DataFrame(rep["near_duplicate_examples"], columns=["Kept", "Removed"]),
                         hide_index=True, use_container_width=True)
        if rep.get("errors"):
            st.warning("Some angles failed: " + " · ".join(rep["errors"]))

    with t_f:
        if s3["filtered_out"]:
            st.dataframe(pd.DataFrame(s3["filtered_out"])[["keyword", "reason", "angle"]], hide_index=True,
                         use_container_width=True)
        else:
            st.caption("Nothing was filtered out.")
