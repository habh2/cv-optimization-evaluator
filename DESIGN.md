# CV Optimizer Agent — Design

## Goal

Score and compare CV baselines against a target JD, and produce specific, actionable feedback per scoring dimension per baseline. Baselines are generated manually outside the tool — the pipeline only evaluates them.

> Low-level details (state schema, agent interfaces, file structure) live in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## File Convention

Baseline CVs are generated manually (via Claude Code skills, direct LLM prompts, etc.) and dropped into a structured folder. The pipeline discovers, validates, and scores whatever it finds there.

### Directory structure

```
data/inputs/
  CV.md                          ← master CV (always scored as baseline)
  JobDescription.txt             ← current JD
  baselines/
    {jd_id}/                     ← one folder per job application (user-defined name)
      {skill_id}/                ← one subfolder per skill used
        {model}__{version}.txt   ← one file per model × version combination
```

**Example:**

```
data/inputs/baselines/
  ml_eng_google/
    composio_tailored/
      gemini-2.5-flash__v1.txt
      claude-sonnet-4-6__v1.txt
    my_custom_prompt/
      gemini-2.5-flash__v2.txt
  backend_swe_stripe/
    composio_tailored/
      gemini-2.5-flash__v1.txt
```

### Filename rules

The file inside each `{skill_id}/` folder must match: `{model}__{version}.txt`

- `model` — the model identifier as it appears in the provider's API (e.g. `gemini-2.5-flash`, `claude-sonnet-4-6`)
- `version` — the prompt/skill version used (e.g. `v1`, `v2`); must be bumped whenever the skill prompt changes
- Separator is double underscore (`__`); single underscores within each part are fine
- Files that don't match this pattern are rejected with an error at load time

Valid skill IDs and their display names are defined in `skills_registry.py`. An unknown `{skill_id}` folder logs a warning and is skipped.

### Invocation

```
python graph.py --run {jd_id}
```

The `{jd_id}` argument tells the pipeline which subfolder of `baselines/` to load. The JD and master CV are always read from their standard paths.

---

## High-Level Workflow

```
python graph.py --run {jd_id}
      │
      ▼
┌─────────────┐
│   Intake    │  Load JD + master CV; resolve baselines/{jd_id}/; check cache
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ JD Analyzer  │  Extract: must-haves, nice-to-haves, seniority, keywords, tone
└──────┬───────┘          (gives the scorer structured JD context)
       │
       ▼
┌──────────────┐
│ Gap Analyzer │  Diff master CV vs JD; query RAG for similar past runs
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│  Load Baselines                      │
│  Discover + validate files under     │
│  baselines/{jd_id}/                  │
│  master_cv always included           │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Score All                           │
│  For each baseline:                  │
│    LLM-as-judge → 5 dimension scores │
│    + per-dimension feedback text     │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────┐
│    Report    │  Comparison table + feedback report; save to cache
└──────────────┘
```

---

## Scoring Dimensions

Each baseline is scored on five dimensions (0–10 each). Total = average × 10 → 0–100.

| Dimension | What it measures |
|---|---|
| **Keyword coverage** | % of JD must-have and nice-to-have keywords present naturally |
| **Achievement specificity** | Ratio of quantified, impact-first bullets vs vague statements |
| **JD alignment** | How well the CV summary and key bullets match JD requirements |
| **Readability** | Sentence clarity, active voice, ATS-safe formatting |
| **AI / human voice** | Whether the language sounds authentically human vs AI-generated |

The AI / human voice dimension uses the [`avoid-ai-writing`](https://github.com/conorbronsdon/avoid-ai-writing) pattern — detect AI-isms and flag locations — adapted as a scorer sub-prompt. Detection is model-agnostic.

---

## Feedback Structure

Each dimension returns a score and a short actionable feedback string. Examples of expected tone:

- *Keyword coverage*: "Missing 'distributed systems' and 'MLOps' which appear 4× in the JD. Uses 'machine learning' where JD consistently says 'ML pipelines' — align terminology."
- *Achievement specificity*: "3 of 7 bullets are vague ('contributed to', 'helped with'). Rewrite using: action verb + what + how/why + result."
- *AI / human voice*: "'Results-driven professional' and 'passionate about' are common AI tells. Bullet 4 reads as templated — rephrase in the candidate's natural register."

---

## Scorer Design

### Prompt (version-locked)

One LLM-as-judge call per baseline returns all five dimension scores and feedback in a single structured JSON response. Prompt is version-locked (`SCORER_VERSION`); cached results store the version alongside scores so a prompt change invalidates the relevant cache entries.

### Cache schema

Cached per `{jd_id}_{scorer_version}`:

```json
{
  "ml_eng_google_v1.0": {
    "date": "<ISO date>",
    "baselines": {
      "master_cv": {
        "keyword_coverage":        { "score": 7, "feedback": "..." },
        "achievement_specificity": { "score": 6, "feedback": "..." },
        "jd_alignment":            { "score": 8, "feedback": "..." },
        "readability":             { "score": 9, "feedback": "..." },
        "voice":                   { "score": 8, "feedback": "..." },
        "total": 76.0
      },
      "composio_tailored/gemini-2.5-flash__v1": {
        "keyword_coverage":        { "score": 9, "feedback": "..." },
        ...
      }
    }
  }
}
```

---

## RAG Design

### What is stored

Each completed evaluation run persists a **run artifact**:

```
{
  jd_id:          "ml_eng_google",
  jd_embedding:   <vector>,
  role_category:  "ml-engineer" | "backend-swe" | ...,  // inferred by JD Analyzer
  scores:         { baseline_id: { per-dimension scores and totals } },
  best_baseline:  <baseline_id>,
  outcome:        "contacted" | "rejected" | "no_response" | null,
  date:           <ISO date>
}
```

`role_category` is inferred by the JD Analyzer. Unknown roles fall back to a generic category.

Outcome starts as `null` and is updated when career-ops syncs `applications.md`.

### Retrieval

At Gap Analysis, retrieve top-k most similar past runs by role category + cosine similarity. If best match falls below a similarity threshold, skip retrieval — Gap Analyzer runs on JD + master CV alone.

Retrieval weighted by outcome: `contacted` > `no_response` > `rejected`.

### How it is used

Retrieved artifacts annotate the report: "for similar ML Engineer roles, keyword coverage was the strongest differentiator between contacted and not contacted." This is display context only — RAG does not influence scores.

### Outcome ingestion

At run start, the tool checks `career-ops/applications.md` for new outcome records and upserts them into the vector store before retrieval runs.

---

## Observability

**Core question:** *What did each baseline get right and wrong, and how does it compare across skills and models?*

**Output:**

1. **Comparison table** — one row per baseline, one column per dimension + total
2. **Feedback report** — per baseline × dimension: score + actionable feedback text
3. **RAG annotation** (when available) — which dimensions were most predictive for similar roles

**Tooling:** LangSmith for trace capture. **Fallback:** structured JSON run log written locally on every run.

---

## Trade-offs & Rationale

| Decision | Alternative considered | Rationale |
|---|---|---|
| User generates baselines manually | Pipeline calls LLMs to generate | Simpler pipeline; user controls prompt wording, model, and timing directly |
| Skill × model encoded in file path | Single flat folder | Path structure is self-documenting and browsable; avoids filename collisions |
| Double-underscore separator in filename | Single underscore or dash | Allows single underscores within model IDs (e.g. `gemini-2.5-flash`) without ambiguity |
| Version in filename, not in a manifest | Separate metadata file | Filename is self-contained; no manifest to keep in sync |
| Pipeline rejects invalid filenames | Silent skip | Fail-fast prevents silently missing a baseline due to a typo |
| `avoid-ai-writing` pattern for voice dimension | Static blocklist | Pattern-based detection catches evolving AI-isms; model-agnostic |
| RAG annotates report, does not influence scores | RAG influences scoring weights | Keeps scoring model-agnostic and avoids feedback loops in evaluation |
| Scorer prompt co-cached with results | Separate version tracking | Ensures cached scores are always comparable to the current scoring prompt |

---

## Integration: career-ops

```
cv-optimizer-agent
  └── outputs ResultCV.txt (best-scoring baseline)
        └── career-ops /career-ops pdf → ATS-clean PDF

Shared state:
  - cv.md (master CV) read by both tools via symlink
  - career-ops applications.md → outcome sync at run start → vector store
```

Tools are decoupled — cv-optimizer-agent does not depend on career-ops at runtime.
