# CV Optimizer Agent

A multi-agent CV evaluation pipeline. Scores manually-generated CV variants against a job description and produces structured, actionable feedback per dimension. Outcome data from real applications feeds back over time to calibrate the scoring judge.

Built to demonstrate: **LangGraph multi-agent patterns**, **RAG**, **LLM-as-judge**, and **LangSmith observability**.

---

## How it works

```
Intake → Outcome Sync → JD Analyzer
                              ├── Gap Analyzer  ─┐
                              └── RAG Retrieve   ┘
                                        │
                                 Load Baselines
                                        │
                         Send(score_baseline) × N   ← parallel
                                        │
                                   Aggregate → Report
```

1. **JD Analyzer** extracts must-haves, keywords, seniority, and role category from the job description.
2. **Gap Analyzer** and **RAG Retrieve** run in parallel — gap analysis diffs the master CV against the JD; RAG retrieves outcome patterns from similar past runs.
3. **Load Baselines** discovers manually-created CV variants under `data/inputs/baselines/{jd_id}/`.
4. **Score Baseline** runs once per variant concurrently (LangGraph `Send()`), with the JD analysis, gap context, and RAG context all injected into the scorer prompt.
5. **Report** prints a comparison table + per-dimension feedback, then persists the run to the vector store and cache.

Past application outcomes (`contacted` / `rejected` / `no_response`) are synced from `career-ops/applications.md` at run start and used to weight RAG retrieval, so the scoring judge adapts to what has actually correlated with getting contacted.

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# or: .venv/bin/pip install -r requirements.txt  # macOS/Linux

cp .env.example .env
# Fill in GOOGLE_API_KEY (required)
# Fill in LANGCHAIN_API_KEY for LangSmith tracing (optional)
```

Add your master CV and job description:

```
data/inputs/CV.md
data/inputs/JobDescription.txt
```

---

## Adding baselines

Generate CV variants with any tool (Claude Code skills, direct prompts, etc.) and drop them into the structured folder:

```
data/inputs/baselines/
  {jd_id}/
    {skill_id}/
      {model}__{version}.txt
```

Example:

```
data/inputs/baselines/
  ml_eng_google/
    composio_tailored/
      gemini-2.5-flash__v1.txt
      claude-sonnet-4-6__v1.txt
```

- `{jd_id}` — your name for this application (e.g. `ml_eng_google`)
- `{skill_id}` — must be a key in `skills_registry.py`
- `{model}__{version}.txt` — model ID + prompt version, double-underscore separated

Files not matching the pattern are rejected at load time.

---

## Running

```bash
# Score all baselines for a job application
python graph.py --run ml_eng_google

# Ignore cache and re-evaluate
python graph.py --run ml_eng_google --force

# Score master CV only (no baselines)
python graph.py
```

Output: comparison table + per-dimension feedback report. Results cached to `data/cache/baselines.json`.

---

## Recording outcomes

When a real application produces a result, add a row to `career-ops/applications.md`:

```markdown
| jd_id          | outcome     | date       |
|----------------|-------------|------------|
| ml_eng_google  | contacted   | 2026-06-01 |
| backend_stripe | no_response | 2026-06-15 |
```

Valid outcomes: `contacted`, `rejected`, `no_response`. On the next run, `outcome_sync` picks up new rows and updates the vector store. Subsequent runs for similar roles will include this calibration data in the scorer prompt.

---

## Observability

Traces are sent to [LangSmith](https://smith.langchain.com) automatically when `LANGCHAIN_API_KEY` is set. Each run produces:

- Parallel spans for `gap_analyzer` / `rag_retrieve`
- N concurrent `score_baseline` spans (one per baseline)
- Full scorer prompt visible per span, including injected gap and RAG context

---

## Tests

```bash
python -m pytest tests/ -v
```

Covers: scorer JSON parse fallback, score clamping, fence stripping, cache key format, state reducer, RAG context summarization.

---

## Scoring dimensions

| Dimension | What it measures |
|---|---|
| `keyword_coverage` | JD must-have and nice-to-have keywords present naturally (70–80% optimal; 90%+ penalised as stuffing) |
| `achievement_specificity` | Ratio of quantified, impact-first bullets vs vague statements |
| `jd_alignment` | How well the CV summary and key bullets match JD requirements |
| `readability` | Sentence clarity, active voice, ATS-safe formatting |
| `voice` | Authentically human language, scored against the avoid-AI-writing pattern list |

Each dimension: 0–10. Total = average × 10 → 0–100.

---

## Project structure

```
graph.py              — LangGraph pipeline (all nodes and wiring)
rag_store.py          — ChromaDB vector store (upsert, retrieve, summarize)
skills_registry.py    — Valid skill IDs for baseline discovery
data/
  inputs/             — CV.md, JobDescription.txt, baselines/
  cache/              — baselines.json (cached scores)
  rag_store/          — ChromaDB persistence
tests/
  test_scorer.py      — score_baseline, _merge_scores, _cache_key
  test_rag_store.py   — summarize_context, threshold bounds
```
