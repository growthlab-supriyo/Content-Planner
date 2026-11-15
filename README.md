# Content Planner

Type a service, category or topic (plus an optional location). The planner finds the searches around it,
removes the noise and groups them into clusters, and each cluster becomes one page. Later stages add the topical
map, content ideas across 22 content types, and keyword-to-page mapping.

| Stage | What it does | Status |
|---|---|---|
| 1. Input | Topic (required) + location (optional) | Done |
| 2. Query fan-out | 7 focused AI passes (services, problems & symptoms, costs, questions, comparisons, local, fears & myths) with quality checks | Done |
| 3. Filter & cluster | Removes single words and generic two-word terms, then groups keywords into clusters (one theme and one intent each, no size limit) | Done |
| 4. Topical map | Topic → Pillar → Cluster → Subtopic (H2s); gap check adds missing pages as "AI-suggested"; interactive map | Done |
| 5. Content ideas | AI decides which of 23 content types are needed (Used / Skipped + reason), then 1–4 pieces per cluster plus a Pillar Page per pillar | Done |
| 6. Keyword mapping | One primary keyword per page from its own cluster, never primary twice; up to 8 secondary; leftovers listed | Done |

Stages 4–6 rules that are enforced in code (not left to the AI): every cluster sits in exactly one pillar; gap-check
pages that repeat an existing page are dropped; skipped content types are never used; Used types without a page get
one extra AI call; primary keywords are unique across the whole plan.

## Keyword quality checks (Stage 2)

- Each angle is asked separately, with its own instruction and examples from an unrelated business.
- **Repetition cap:** no more than ~15% of an angle's queries may start with the same two words or with the topic name.
  If the AI repeats itself, it is asked again with feedback ("Too many queries started with …").
- **On-angle check:** questions must be questions, comparisons must compare, costs must be about price, and local
  queries must mention the area or "near me". If not, the angle is asked again.
- **Near-duplicates** ("braces cost" / "braces price") are removed with embeddings.
- The location is used only where it fits: local, and about a third of service and cost queries.
- The **quality report** tab shows, per angle, how many queries were asked for, kept, removed as repetitive, and how
  many attempts it took.

## Two versions, one code base

| | Local (`streamlit run app.py`) | Online (`streamlit_app.py`, Streamlit Community Cloud) |
|---|---|---|
| AI | Ollama + optional Gemini / Groq / OpenRouter | Gemini (default), Groq, OpenRouter |
| Keys | `.streamlit/secrets.toml` or typed in the sidebar | Owner's key: 3 free runs per visitor per day, then the visitor's own key |
| Embeddings | `nomic-embed-text` through Ollama (fastembed if missing) | fastembed (`BAAI/bge-small-en-v1.5`) |
| Prompt editor | Advanced panel in the sidebar | Hidden (tested defaults) |
| Saving | Download Excel / project JSON, upload JSON to continue | Same |

AI calls use plain HTTPS from the Python standard library (every provider accepts the OpenAI chat format), so there's
no AI SDK to install.

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt pytest
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
.venv\Scripts\streamlit run app.py
```

Optional: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add free keys.

If Windows blocks `streamlit.exe`, start it with `.venv\Scripts\python -m streamlit run app.py`.

Command-line test runs (save JSON + Excel to `runs/`):

```bash
.venv\Scripts\python tools\try_run.py "Dental Clinic" "Newtown, Kolkata" 25
.venv\Scripts\python tools\try_456.py runs\content-plan-dental-clinic-newtown-kolkata.json
```

On a laptop CPU with `qwen2.5:7b`, Stages 1–3 take about 15 minutes and Stages 4–6 roughly as long again.
Online (Gemini/Groq) the whole plan takes a few minutes.

## Put it online (free)

1. Push this folder to a public GitHub repository (`secrets.toml` is git-ignored).
2. On share.streamlit.io, create an app from the repo with main file `streamlit_app.py`.
3. In the app's **Settings → Secrets**, paste your keys (same format as `secrets.toml.example`).
4. Keep-awake: in GitHub **Settings → Secrets and variables → Actions → Variables**, add `APP_URL` with the app's
   address. `.github/workflows/keep-awake.yml` then opens the app every 6 hours.

Free-run counts are kept in memory (reset when the app restarts) and keyed by a hash of the visitor's IP and the date.
`CP_DAILY_CAP` (default 150) caps all visitors together on the owner's key.

## Themes

Light = **Indigo** (indigo accents, green status colours). Dark = **Sunset** (orange on zinc). Visitors switch in
Streamlit's menu → Settings. Colours live in `planner/themes.py`; `.streamlit/config.toml` is generated from it
(a test keeps them in sync).

## Tests

```bash
.venv\Scripts\python -m pytest -q
```
