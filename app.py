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
from planner.llm import LLM, FallbackLLM, LLMError, build_stage_models, ollama_running
from planner.mapviz import map_html
from planner.plan import build_content_plan
from planner.project import filename, from_json, new_project, to_json
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
    v = str(v or os.environ.get(name) or "").strip().strip('"').strip()  # pasted keys often carry spaces or quotes
    return v or None


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
ss.setdefault("notices", [])
ss.setdefault("paid", set())  # projects (by created time) whose run already used a free run

# ---------- sidebar: AI settings ----------
with st.sidebar:
    st.markdown(f"### :material/hub: {C.APP_NAME}")
    st.caption("Online version · free cloud AI" if C.ONLINE else "Local version · runs on this computer")

    quota = get_quota()
    vid = Quota.visitor_id(visitor_ip())
    cloud = [n for n in C.available_providers() if C.PROVIDERS[n].key_name]  # Gemini, Groq, OpenRouter

    # Visitors' own keys: one box per provider, folded away. Locally, keys saved in secrets.toml are used anyway.
    user_keys: dict[str, str] = {}
    with st.expander(":material/key: Use your own API keys"):
        st.caption("Optional. With your own keys there's no daily limit; they're kept only for this session. "
                   "Add any or all: the planner uses whichever you add." if C.ONLINE else
                   "Keys saved in .streamlit/secrets.toml are used automatically. Type one here to use a different "
                   "key in this session.")
        for name in cloud:
            k = st.text_input(f"{name} API key", type="password", key=f"key_{name}",
                              help=C.PROVIDERS[name].key_help) or ""
            if k.strip():
                user_keys[name] = k.strip()
    using_own = bool(user_keys)

    def active_keys() -> dict[str, str]:
        """Keys this run uses. Online with a visitor's own keys: only theirs (the owner's keys stay unused)."""
        if C.ONLINE and using_own:
            return dict(user_keys)
        keys = {n: secret(C.PROVIDERS[n].key_name) for n in cloud}
        keys = {n: k for n, k in keys.items() if k}
        keys.update(user_keys)
        return keys

    if C.ONLINE:
        if using_own:
            st.caption(":material/key: Using your keys: no daily limit here.")
        elif active_keys():
            left = quota.remaining(vid)
            st.caption(f":material/bolt: Free runs left today: **{left} of {C.FREE_RUNS_PER_DAY}**")
            if left == 0:
                st.info("Add your own free key under 'Use your own API keys' to keep going.")
        else:
            st.warning("Add a free Gemini, Groq or OpenRouter key under 'Use your own API keys' to run.")

    ollama_ok, ollama_models = (False, []) if C.ONLINE else ollama_running()
    if not C.ONLINE:
        st.caption(f":material/check_circle: Ollama is running ({len(ollama_models)} models)" if ollama_ok else
                   ":material/error: Ollama isn't running (only needed for the local model).")

    have = active_keys()
    singles = [m for m in C.SINGLE_MODELS if m[0] in have or (m[0] == "Local Ollama" and not C.ONLINE)]
    model_mode = st.radio(
        "Models", ["A different model for each stage", "One model for everything"],
        index=0 if have else 1, disabled=not have,
        help="A different model for each stage: each step uses the free model that does it best, and moves to its "
             "backups when one runs out. One model for everything: you pick the model, and every step uses it."
        if have else "Add a Gemini, Groq or OpenRouter key to use a different model for each stage.")
    per_stage = bool(have) and model_mode.startswith("A different")
    single = None
    if per_stage:
        st.caption(C.PER_STAGE_EFFECT)
        with st.expander(":material/model_training: Models used in each stage"):
            for stage, label in C.STAGES.items():
                models = [m for pv, m in C.STAGE_MODELS[stage] if pv in have]
                st.markdown(f"**{label}**  \n" + " → ".join(f"`{m}`" for m in models))
            st.caption("Each stage moves to the next model when one hits its free limit or isn't available.")
    else:
        labels = [m[2] for m in singles]
        pick = st.selectbox("Model for every stage", labels, index=0,
                            help="Every step (angles, queries, tagging, clusters, pillars, H2s, titles) uses this "
                                 "model. If it hits its free limit, the run moves to your other keys' models.")
        single = next(m for m in singles if m[2] == pick)
        st.caption(f"**What changes:** {single[3]}")
        if single[0] == "Local Ollama" and ollama_ok and not any(
                x.split(":")[0] == single[1].split(":")[0] for x in ollama_models):
            st.warning(f"Model '{single[1]}' isn't installed. Run: `ollama pull {single[1]}`")

    n_per_angle = st.slider("Queries per angle", 15, C.MAX_QUERIES_PER_ANGLE if not C.ONLINE else 40,
                            C.DEFAULT_QUERIES_PER_ANGLE, step=5,
                            help="The AI first studies the topic and chooses its search angles; each angle is then "
                                 "asked separately. More queries = longer run.")


def embedder_kind() -> str:
    if C.ONLINE:
        return "fastembed"
    if ollama_ok and any(m.startswith(C.OLLAMA_EMBED_MODEL) for m in ollama_models):
        return "ollama"
    return "fastembed"


def setup_problem() -> str | None:
    """Why a run can't start with the current settings (None when it can)."""
    if single and single[0] == "Local Ollama":
        return None if ollama_ok else "Ollama isn't running. Start the Ollama app, or choose a cloud model."
    if not have:
        return ("Add a free Gemini, Groq or OpenRouter key under 'Use your own API keys' in the sidebar."
                if C.ONLINE else "Add a Gemini, Groq or OpenRouter key, or start Ollama and choose the local model.")
    return None


def build_llm(notice):
    """The chosen model (or the per-stage chains), with every other model we have a key for as backup, so a run
    still finishes when one free limit is reached. Locally, Ollama is the last backup when it's running."""
    backups = [LLM(n, C.PROVIDERS[n].default_model, k) for n, k in have.items()]
    if not C.ONLINE and ollama_ok:
        backups.append(LLM("Local Ollama", C.PROVIDERS["Local Ollama"].default_model))
    if per_stage:
        return build_stage_models(have, FallbackLLM(backups, on_switch=notice), notice)
    first = LLM(single[0], single[1], have.get(single[0]))
    rest = [b for b in backups if (b.provider, b.model) != (first.provider, first.model)]
    return FallbackLLM([first] + rest, on_switch=notice)


def run_stage3(project: dict, rows: list[dict], llm, progress) -> None:
    progress("Stage 3 · removing generic terms", 0.02)
    kept, removed = filter_keywords(rows, project["topic"], llm, project["location"])
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
    elif setup_problem():
        problem = setup_problem()
    elif C.ONLINE and not using_own and not quota.use(vid):
        problem = (f"You've used today's {C.FREE_RUNS_PER_DAY} free runs. Add your own free key under 'Use your "
                   f"own API keys' in the sidebar to keep going.")
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

                def notice(msg):  # model switches are normal: recorded for the Quality report, not shown as warnings
                    ss["notices"].append(msg)

                llm = build_llm(notice)
                st.write(f"AI: **{llm.label}** · embeddings: **{embedder_kind()}**")
                share = 0.55 if full else 0.8
                rows, report = run_fanout(llm, get_embedder(embedder_kind()), topic, location, n_per_angle,
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
            if C.ONLINE and not using_own and not project.get("stage2"):
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
                problem = setup_problem()
                if problem:
                    pass
                elif C.ONLINE and not using_own and project["created"] not in ss["paid"]:
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
                            run_stages_456(project, build_llm(lambda msg: ss["notices"].append(msg)),
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
        df = pd.DataFrame(s2["keywords"], columns=["keyword", "angle", "intent", "funnel", "kind"])
        edited = st.data_editor(
            df, num_rows="dynamic", use_container_width=True, hide_index=True, key="kw_editor",
            column_config={
                "keyword": st.column_config.TextColumn("Keyword", required=True, width="large"),
                "angle": st.column_config.TextColumn("Angle", disabled=True),
                "intent": st.column_config.SelectboxColumn("Intent", options=list(Q.INTENTS), required=True),
                "funnel": st.column_config.SelectboxColumn("Funnel", options=["TOFU", "MOFU", "BOFU"]),
                "kind": None,  # hidden: the angle's kind, kept for the area-name rules
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
                funnel = r.get("funnel") if r.get("funnel") in ("TOFU", "MOFU", "BOFU") else Q.FUNNEL[intent]
                rows.append({"keyword": kw, "angle": r.get("angle") or "Added by you", "intent": intent,
                             "funnel": funnel, "kind": r.get("kind") or ""})
            try:
                with st.spinner("Rebuilding clusters…"):
                    llm = build_llm(lambda msg: None) if not setup_problem() else None
                    s2["keywords"] = rows
                    run_stage3(project, rows, llm, lambda m, f: None)
                st.rerun()
            except LLMError as e:
                st.error(str(e))

    with t_q:
        st.caption(f"Made with {rep.get('model', '')} · embeddings: {rep.get('embedder', '')} · "
                   f"{rep.get('ai_calls', '?')} AI calls in Stage 2 · intent and funnel tagged by AI for "
                   f"{rep.get('tagged_by_ai', 0)} of {len(s2['keywords'])} keywords")
        if ss.get("notices"):
            with st.expander(f"Model switches during this run ({len(ss['notices'])})"):
                st.caption("A stage moves to its next model when one hits a free limit, doesn't exist for the key, or "
                           "its key isn't accepted. The plan is still complete.")
                st.markdown("\n".join(f"- {n}" for n in ss["notices"]))
        if rep.get("stage_models"):
            with st.expander("Model used in each stage"):
                st.dataframe(pd.DataFrame([{"Stage": C.STAGES.get(k, k), "Model": v}
                                           for k, v in rep["stage_models"].items()]),
                             hide_index=True, use_container_width=True)
        if rep.get("angle_plan"):
            st.markdown("**Angles** · " + ("chosen by the AI for this topic" if rep.get("angles_planned_by_ai")
                                           else "the 7 standard angles"))
            st.dataframe(pd.DataFrame(rep["angle_plan"]).rename(columns={
                "label": "Angle", "kind": "Kind", "instruction": "What was asked"}),
                hide_index=True, use_container_width=True)
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
