# CV Optimization Evaluator — Design

## Goal

Score and compare CV baselines against a target JD, and produce specific, actionable feedback per scoring dimension per baseline. Baselines are generated manually outside the tool — the pipeline only evaluates them. Outcome data from real applications feeds back into scoring context over time so the evaluator adapts to what actually gets candidates contacted.

---

## High-Level Workflow

```
python graph.py --run {jd_id}
      │
      ▼
┌─────────────┐
│   Intake    │  Load JD + master CV; check cache; pass force flag via state
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  Outcome Sync    │  Read career-ops/applications.md; upsert new outcomes into vector store
└──────┬───────────┘
       │
       ▼
┌──────────────┐
│ JD Analyzer  │  Extract: must-haves, nice-to-haves, seniority, keywords, tone, role_category
└──────┬───────┘
       │
       ├─────────────────────────┐
       ▼                         ▼
┌──────────────┐        ┌─────────────────┐
│ Gap Analyzer │        │  RAG Retrieve   │  Embed JD; query vector store filtered by role_category;
│ (LLM call)   │        │  (vector query) │  weight by outcome; summarize as scorer context string
└──────┬───────┘        └────────┬────────┘
       │                         │
       └──────────┬──────────────┘   (LangGraph waits for both)
                  ▼
       ┌──────────────────┐
       │  Load Candidates │  Discover + validate files; always includes master_cv
       └──────┬───────────┘
              │
              │  Send(score_candidate) × N    ← parallel fan-out, one per candidate
              ▼
       ┌──────────────────┐
       │  score_candidate │  LLM-as-judge → 5 dimension scores + feedback
       │  (× N, parallel) │  Scorer prompt includes RAG context when available
       └──────┬───────────┘
              │
              ▼
       ┌──────────────┐
       │  Aggregate   │  Collect scores; compute totals; identify best candidate
       └──────┬───────┘
              │
              ▼
       ┌──────────────┐
       │    Report    │  Comparison table + feedback; persist run artifact to vector store; save cache
       └──────────────┘
```

---

## Scorer Design

### Prompt (version-locked)

One LLM-as-judge call per baseline returns all five dimension scores and feedback in a single structured JSON response. Prompt is version-locked (`SCORER_VERSION`); cached results store the version alongside scores so a prompt change invalidates the relevant cache entries.

When RAG context is available, the scorer prompt includes a `{rag_context}` block with outcome patterns from similar past runs (e.g., *"For similar ML Engineer roles: contacted candidates averaged keyword_coverage=8.2, voice=7.8; rejected candidates averaged 5.1, 5.3."*). The LLM uses this to calibrate its scoring judgment. When no past runs exist, the block is omitted and scoring behaviour is identical to a run without RAG.

### Cache schema

Cached per `{jd_id}_{jd_hash[:8]}_{scorer_version}`:

```json
{
  "{jd_id}": {
    "jd_hash": "<first 8 chars of SHA-256 of JD content>",
    "scorer_version": "<SCORER_VERSION>",
    "date": "<ISO date>",
    "candidates": {
      "master_cv": {
        "keyword_coverage":        { "score": 7, "feedback": "..." },
        "achievement_specificity": { "score": 6, "feedback": "..." },
        "jd_alignment":            { "score": 8, "feedback": "..." },
        "readability":             { "score": 9, "feedback": "..." },
        "voice":                   { "score": 8, "feedback": "..." },
        "total": 76.0
      },
      "{generation_prompt_id}/{model}": {
        "keyword_coverage":        { "score": 9, "feedback": "..." }
      }
    }
  }
}
```

---

## RAG Design

### What is stored

Each completed evaluation run persists a **run artifact** to the vector store:

```
{
  jd_id:          "ml_eng_google",
  jd_embedding:   <vector>,            // embedding of the raw JD text
  role_category:  "ml-engineer" | "backend-swe" | ...,  // inferred by JD Analyzer
  best_scores:    { keyword_coverage: 8, ... },  // scores of the best-performing baseline
  outcome:        "contacted" | "rejected" | "no_response" | null,
  date:           <ISO date>
}
```

`role_category` is inferred by the JD Analyzer. Unknown roles fall back to a `"general"` category.

Outcome starts as `null` and is updated when career-ops syncs `applications.md`.

### Retrieval

At `rag_retrieve`, the current JD is embedded and the vector store is queried for top-k most similar past runs, filtered by `role_category`. Similarity uses cosine distance. If the best match falls below a similarity threshold, retrieval returns empty — scoring runs without RAG context.

Retrieved runs are weighted by outcome before summarizing: `contacted` > `no_response` > `rejected`.

### How it influences scoring

The retrieved artifacts are summarized into a scorer context string injected into the scorer prompt. This gives the LLM judge calibration data from real-world outcomes, so the scoring rubric adapts to what has actually correlated with getting contacted for similar roles — satisfying the non-negotiable that "agents change their judging approach."

This is soft influence: the LLM decides how to apply the context. It is not hard-coded weight adjustment, which would create evaluation feedback loops.

### Outcome ingestion

The `outcome_sync` node runs at the start of every pipeline run. It reads `career-ops/applications.md` and upserts any outcome records not yet in the vector store. It no-ops gracefully if the file does not exist.

---

## LangGraph Patterns Demonstrated

| Pattern | Where |
|---|---|
| **Parallel nodes** | `gap_analyzer` + `rag_retrieve` run concurrently after `jd_analyzer` |
| **Map-reduce fan-out** | `load_candidates` dispatches `Send(score_candidate)` per candidate; all run in parallel; `aggregate` collects |
| **Conditional skip** | `intake` short-circuits to `report` on cache hit |
| **State reducer** | `scores` field uses dict-merge reducer (`{**existing, **new}`) to collect parallel scorer outputs |

---

## Observability

**Tooling:** Phoenix (Arize) for local trace capture — open-source, no data leaves the machine. **Fallback:** structured JSON run log written locally on every run.

**What the traces show:**
- Parallel execution of `gap_analyzer` / `rag_retrieve` (visible as concurrent spans)
- Per-candidate scorer invocations (one span per candidate, all concurrent)
- RAG context string passed to each scorer (visible in span inputs)
- Score output per dimension per baseline

---

## Trade-offs & Rationale

| Decision | Alternative considered | Rationale |
|---|---|---|
| User generates candidates manually | Pipeline calls LLMs to generate | Simpler pipeline; user controls prompt wording, model, and timing directly |
| `generation_prompt_id` × model encoded in file path | Single flat folder | Path structure is self-documenting and browsable; groups variants by the prompt that produced them |
| Pipeline rejects invalid filenames | Silent skip | Fail-fast prevents silently missing a candidate due to a typo |
| `avoid-ai-writing` pattern for voice dimension | Static blocklist | Pattern-based detection catches evolving AI-isms; model-agnostic |
| RAG injects context into scorer prompt | RAG hard-codes scoring weights | Soft influence lets the LLM decide how to apply outcome data; avoids rigid feedback loops |
| ChromaDB local persistent store | Hosted vector DB | No server required; suitable for single-user portfolio tool; trivially swappable |
| Cache key includes JD hash | Cache key is jd_id only | JD file changes for the same application invalidate cache correctly |
| `force` flag threaded through state | Read from sys.argv in node | Nodes must be context-free; threading via state allows programmatic invocation and testing |

---

## Integration: career-ops

```
cv-optimization-evaluator
  └── outputs ResultCV.txt (best-scoring baseline)
        └── career-ops /career-ops pdf → ATS-clean PDF

Shared state:
  - cv.md (master CV) read by both tools via symlink
  - career-ops applications.md → outcome_sync at run start → vector store
```

Tools are decoupled — cv-optimization-evaluator does not depend on career-ops at runtime.
