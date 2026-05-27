# CV Optimization Evaluator

A multi-agent CV evaluation pipeline. Scores manually-generated CV variants against a job description and produces structured, actionable feedback per dimension. Outcome data from real applications feeds back over time to calibrate the scoring judge.

Built to demonstrate: **LangGraph multi-agent patterns**, **RAG**, **LLM-as-judge**, and **LLM observability**.

---

## How it works

```
Intake → Outcome Sync → JD Analyzer
                              ├── Gap Analyzer  ─┐
                              └── RAG Retrieve   ┘
                                        │
                                 Load Candidates
                                        │
                         Send(score_candidate) × N   ← parallel
                                        │
                                   Aggregate → Report
```

1. **JD Analyzer** extracts must-haves, keywords, seniority, and role category from the job description.
2. **Gap Analyzer** and **RAG Retrieve** run in parallel — gap analysis diffs the master CV against the JD; RAG retrieves outcome patterns from similar past runs.
3. **Load Candidates** discovers manually-created CV variants under `data/inputs/{jd_id}/`.
4. **Score Candidate** runs once per variant concurrently (LangGraph `Send()`), with the JD analysis, gap context, and RAG context all injected into the scorer prompt.
5. **Report** prints a comparison table + per-dimension feedback, then persists the run to the vector store and cache.

Past application outcomes (`contacted` / `rejected` / `no_response`) are synced from `data/applications.md` at run start and used to weight RAG retrieval, so the scoring judge adapts to what has actually correlated with getting contacted.

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# or: .venv/bin/pip install -r requirements.txt  # macOS/Linux

cp .env.example .env
# Fill in GOOGLE_API_KEY (required)
```

Add your master CV:

```
data/inputs/CV.md
```

---

## Adding candidates

Generate CV variants with any tool (Claude Code skills, direct prompts, etc.) and drop them into the job application folder:

```
data/inputs/
  {jd_id}/
    JobDescription.txt
    {generation_prompt_id}/
      {model}.txt
```

Example:

```
data/inputs/
  ml_eng_google/
    JobDescription.txt
    composio_tailored/
      gemini-2.5-flash.txt
      claude-sonnet-4-6.txt
```

- `{jd_id}` — your name for this application (e.g. `ml_eng_google`); also used as the `--run` argument
- `{generation_prompt_id}` — any name using alphanumeric characters, hyphens, and underscores; identifies the prompt used to generate the variants inside
- `{model}.txt` — the model used to generate the variant

Files not matching the pattern are rejected at load time.

---

## Running

```bash
# Score all candidates for a job application
python graph.py --run ml_eng_google

# Ignore cache and re-evaluate
python graph.py --run ml_eng_google --force

# Score master CV only (skip generation prompt folders)
python graph.py --run ml_eng_google --master-only
```

Output: comparison table + per-dimension feedback report. Results cached to `data/cache/candidates.json`.

---

## Recording outcomes

When a real application produces a result, add a row to `data/applications.md` in this repo. Create the file if it does not exist.

```markdown
| jd_id          | outcome     | date       |
|----------------|-------------|------------|
| ml_eng_google  | contacted   | 2026-06-01 |
| backend_stripe | no_response | 2026-06-15 |
```

Valid outcomes: `contacted`, `rejected`, `no_response`. On the next run, `outcome_sync` picks up new rows and updates the vector store. Subsequent runs for similar roles will include this calibration data in the scorer prompt.

> **Note:** [career-ops](https://github.com/santifer/career-ops) is an external AI job-search tool that tracks applications in its own format. Integrating directly with its tracker — rather than maintaining this separate file — is a natural next step to explore.

---

## Observability

Traces are sent to [Phoenix](https://phoenix.arize.com) (local, open-source). Start the server once before running the pipeline:

```bash
phoenix serve
# UI available at http://localhost:6006
```

Each run produces:

- Parallel spans for `gap_analyzer` / `rag_retrieve`
- N concurrent `score_candidate` spans (one per candidate)
- Full scorer prompt visible per span, including injected gap and RAG context

Phoenix is optional — if the server is not running, the pipeline continues without tracing.

---

## Tests

```bash
python -m pytest tests/ -v
```

Covers: scorer JSON parse fallback, score clamping, fence stripping, state reducer, RAG context summarization.

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

## Feedback format

Each dimension returns a score and a short actionable feedback string. Examples:

- *Keyword coverage*: "Missing 'distributed systems' and 'MLOps' which appear 4× in the JD. Uses 'machine learning' where JD consistently says 'ML pipelines' — align terminology."
- *Achievement specificity*: "3 of 7 bullets are vague ('contributed to', 'helped with'). Rewrite using: action verb + what + how/why + result."
- *Readability*: "Two bullets exceed 25 words and use passive voice. Split and rewrite as: '[Subject] [verb] [outcome]'."
- *Voice*: `"best practices"` — Tier 1 — replace with the specific practice (e.g. "zero-downtime deploy process"). `"streamlined"+"elevated"` in same section — Tier 2 cluster — replace with action + outcome (e.g. "cut deploy cycle from X to Y").

---

## Project structure

```
graph.py              — LangGraph pipeline (all nodes and wiring)
rag_store.py          — ChromaDB vector store (upsert, retrieve, summarize)
data/
  inputs/             — CV.md + {jd_id}/ folders (JD + candidate CVs)
  cache/              — candidates.json (cached scores)
  rag_store/          — ChromaDB persistence
tests/
  test_scorer.py      — score_candidate, _merge_scores
  test_rag_store.py   — summarize_context, threshold bounds
```
