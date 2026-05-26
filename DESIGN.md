# CV Optimizer Agent — Design

## Goals

Produce a tailored CV that demonstrates meaningful improvement over:
1. The unmodified master CV against the target JD
2. The best available single-LLM public skill run on the same input

"Meaningful improvement" means headroom exists on at least one scoring dimension — not raw supremacy over an unbounded competitor pool.

---

## Two Modes of Operation

### Evaluate mode
Runs the JD + master CV through a set of publicly available single-LLM resume skills and scores each output. Purpose: establish whether optimisation is worth running at all.

**Baseline pool (examples):**
- [Composio Tailored Resume Generator](https://github.com/ComposioHQ/awesome-claude-skills/blob/master/tailored-resume-generator/SKILL.md) — priority-mapped keyword alignment + ATS optimisation
- Additional public skills added as discovered

Each skill output is scored using the same Scorer rubric. Results are logged: which skill won, by how much, on which dimensions.

**Saturation warning:** If the best baseline already scores above a configurable threshold (default: 85/100) *and* all scoring dimensions show less than 10 points of headroom, the system warns:

> "Best public skill scored X/100. Dimensions with remaining headroom: [achievement_specificity: +8, readability: +3]. Running the optimisation pipeline is unlikely to add meaningful value. Proceed? [y/N]"

The warning surfaces *which dimensions* are saturated, not just the raw score. A user may choose to proceed specifically to close the remaining gap on one dimension.

**Baselines are cached per JD hash.** If the same JD is used for a second run, cached baselines are reused without re-calling the public skills. Re-evaluation can be forced explicitly.

### Optimise mode
Runs the full multi-agent pipeline. Requires either a prior evaluate run for this JD or acceptance of cached baselines. The pipeline's output must demonstrate headroom improvement over the best baseline — if it doesn't, the run is flagged as "no improvement over best public skill."

---

## High-Level Workflow (Optimise Mode)

```
JD + master CV
      │
      ▼
┌─────────────┐
│   Intake    │  Parse & normalise inputs; compute JD hash for cache lookup
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ JD Analyzer │  Extract: required skills, keywords, seniority signals, tone
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Gap Analyzer│  Diff CV vs JD; query RAG for similar past runs (see RAG Design)
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  ATS / Keyword      │  Add/naturalise JD keywords
├─────────────────────┤
│  Achievements       │  Quantify and rewrite bullets (operates on content,
├─────────────────────┤   not structure — safe before Relevance)
│  Relevance          │  Reorder and trim based on JD priority
├─────────────────────┤
│  Tone / Voice       │  Align register to JD tone; harmonise full document
└──────────┬──────────┘
           │
           ▼
  ┌─────────────────┐
  │  AI-Phrase Gate │  detect → rewrite in-place → re-detect (max 2 rewrites)
  └────────┬────────┘  if still failing: abort run, report flagged phrases
           │ pass
           ▼
  ┌─────────────────┐
  │     Scorer      │  Structured rubric score vs JD
  └────────┬────────┘
           │
    ┌──────┴──────────────────────┐
score ≥ threshold          score < threshold
    │                            │
    ▼                     score still improving?
 Output                yes → route to agent responsible
                              for weakest dimension (max 3 iterations)
                        no → abort, output best so far with plateau warning
```

---

## Agent Responsibilities & Ordering Rationale

| Agent | Responsibility | Why this position |
|---|---|---|
| **JD Analyzer** | Extracts structured requirements: must-haves, nice-to-haves, seniority, domain keywords, tone signals | Must precede everything — all downstream agents depend on its output |
| **Gap Analyzer** | Diffs CV vs JD requirements; queries RAG; outputs prioritised gap list | Needs JD analysis; its gap list is the agenda for refinement agents |
| **ATS / Keyword** | Ensures JD keywords appear naturally; avoids stuffing | Runs first in refinement — establishes vocabulary that Achievements and Relevance agents can build on |
| **Achievements** | Rewrites bullets to be specific, quantified, impact-first | Runs before Relevance — rewrites content *before* Relevance decides what to keep; wasted work if reversed |
| **Relevance** | Reorders and trims sections so JD-relevant experience leads | Runs after Achievements so it operates on already-improved content |
| **Tone / Voice** | Aligns register to JD tone (startup vs enterprise); fixes passive voice; harmonises language introduced by prior agents | Last in chain — prior agents may introduce strong action verbs or restructured bullets that don't yet match the target register; acts as a final harmonising pass over the whole document |
| **AI-Phrase Gate** | Uses [`avoid-ai-writing`](https://github.com/conorbronsdon/avoid-ai-writing) pattern: detect → rewrite in-place → re-detect. Hard abort after 2 failed rewrites | After all content agents, before scoring — gate is self-contained so Tone/Voice stays focused on register |
| **Scorer** | LLM-as-judge: returns structured score breakdown | Final evaluation; prompt is version-locked (see Scoring section) |

---

## RAG Design

### What is stored

Each completed run persists a **run artifact** to the vector store — not the full CV text, but the decisions the pipeline made and whether they worked:

```
{
  jd_embedding:       <vector>,
  role_category:      "ml-engineer" | "backend-swe" | ...,   // inferred by JD Analyzer
  cumulative_diff:    [ {agent, section, before, after}, ... ],
  score_breakdown:    {keyword, achievement, alignment, readability},
  outcome:            "contacted" | "rejected" | "no_response" | null,
  date:               <ISO date>
}
```

`role_category` is inferred by the JD Analyzer from the JD text — it is not user-provided. The taxonomy is a fixed list; unknown roles fall back to a generic category rather than blocking the run.

Outcome starts as `null` and is updated when career-ops syncs `applications.md`.

### What is retrieved

At Gap Analysis, given the current JD embedding, retrieve the top-k most similar past run artifacts filtered by:
- Same role category
- Outcome = `contacted` (primary), fallback to `no_response` if insufficient contacted examples

### How it is used

The retrieved artifacts answer: *for similar JDs that led to contact, what did the pipeline emphasise?* Concretely:
- Which scoring dimensions had the most headroom that got closed (e.g. achievement_specificity improved most)
- What types of changes appeared most in the diffs (e.g. "quantified metrics added to 4+ bullets", "section reordered to lead with relevant project")

This shapes the Gap Analyzer's output directly: the gap list it produces is prioritised according to the historical signal — "for this role type, achievement specificity gap ranks above keyword coverage gap." Refinement agents consume the same weighted gap list they always would; RAG reaches them indirectly through the Gap Analyzer's output, not through separate context injection. This keeps refinement agents single-purpose and decoupled from the RAG layer.

### Cold-start & low-signal handling

Retrieval is attempted on every run, but the trigger for using it is **similarity threshold**, not a count of past runs. If the best-matching artifact falls below a cosine similarity threshold, retrieval is skipped entirely — the Gap Analyzer runs on JD + CV alone, exactly as it would with an empty store.

This avoids the failure mode of having many past runs that are superficially similar (e.g. many "backend engineer" runs) but semantically wrong for the current JD (e.g. "ML platform engineer"). Low-confidence retrieval can steer the gap list toward the wrong emphasis patterns, which is worse than no signal.

When retrieval is skipped, the run summary states: "RAG not used — no sufficiently similar past runs found." The pipeline's behaviour is identical to a cold-start; the difference is invisible to downstream agents.

### Outcome ingestion

career-ops `applications.md` is the source of truth for outcomes. At the start of each optimise run, the tool checks for new outcome records in `applications.md` and upserts them into the vector store (keyed by JD hash + run ID) before retrieval runs. This means outcomes recorded since the last run are always available without a separate manual sync step.

---

## AI-Phrase Gate

The gate follows the detect → rewrite → re-detect pattern established by the [`avoid-ai-writing`](https://github.com/conorbronsdon/avoid-ai-writing) Claude Code skill. The skill is a reference implementation for the gate prompt — the pipeline calls the LLM API directly, so the gate is model-agnostic (Claude, Gemini, or any capable model can run it).

1. **Detect**: scan the draft for AI-isms and flag them with locations
2. If issues found → **Rewrite**: fix flagged phrases in-place
3. Re-run detection to confirm clean. Max 2 rewrite attempts.
4. If still failing: **run aborts** with a report of the flagged phrases

This makes the gate self-contained: it detects and fixes without delegating back to Tone/Voice. Tone/Voice agent is now purely responsible for tone/register alignment — cleaner separation of concerns.

Runs before scoring — scorer never evaluates AI-sounding text. Failing open is not acceptable.

---

## Observability

**Core question:** *What changes were applied at each stage, and where did the score land?*

**Approach:**
- Each agent appends its changes to a **cumulative diff** — a list of `{agent, section, before, after}` records representing all changes since master CV up to this point in the pipeline
- The Scorer returns a structured breakdown `{keyword_score, achievement_score, alignment_score, readability_score}` at each scored iteration
- Run summary: ordered list of per-agent cumulative diffs + score snapshot at each checkpoint

**What the observability layer answers:**
- How did the CV evolve through the pipeline, step by step?
- At what point did the score cross the threshold?
- Did the AI-Phrase Gate trigger, and what was replaced?
- Did the pipeline plateau? At what iteration?

**Tooling:** LangSmith for trace capture — each agent invocation is a named span with diffs and score breakdowns attached as metadata. **Fallback:** on LangSmith unavailability, a structured JSON run log is written locally so traces are never silently lost.

---

## Scoring

### Rubric

| Dimension | What it measures |
|---|---|
| Keyword coverage | % of JD must-have and nice-to-have keywords present |
| Achievement specificity | Ratio of quantified bullets vs vague statements |
| JD alignment | Semantic similarity of CV summary + top bullets to JD requirements |
| Readability | Sentence length, active voice ratio, ATS-safe formatting signals |

### Prompt versioning

The scorer's system prompt is a versioned artifact. The baseline cache stores scores alongside the prompt version used to produce them:

```
{
  jd_hash:               <full-text hash of JD>,
  scorer_prompt_version: "v1.2",
  scores:                { skill_A: {keyword, achievement, alignment, readability}, ... },
  date:                  <ISO date>
}
```

At the start of every optimise run, if the cached baselines were computed with a different scorer prompt version than the current one, the cache is stale and re-evaluation is forced before the pipeline runs. This ensures pipeline output and baselines are always scored by the same rubric.

The JD hash is a full-text hash. For the same role at two different companies with near-identical JDs, users can force re-evaluation explicitly — the default is to reuse the cache.

### Baselines

1. Master CV vs JD → raw baseline
2. Each public skill's output vs JD → skill baselines (cached with scorer prompt version)

Pipeline output must show headroom improvement over the best skill baseline on at least one dimension. If it doesn't, the run is flagged.

---

## Cost & Latency

A single optimise run makes approximately 6–8 LLM calls (one per agent + scorer per iteration). Evaluate mode adds ~2–4 calls per public skill, but these are cached per JD hash and only re-run on explicit request. A typical full workflow (evaluate + optimise) is 10–15 LLM calls — acceptable for a human-initiated, async tool.

---

## Trade-offs & Rationale

| Decision | Alternative considered | Rationale |
|---|---|---|
| Two-mode design with saturation warning | Always run pipeline | Honest about when the pipeline adds no value; the warning is itself a portfolio-worthy design decision |
| Store run artifacts, not full CV text | Store full CVs | Run artifacts are the actionable signal; full CV storage is unnecessary and privacy-awkward |
| RAG signal shapes Gap Analyzer output only | RAG informs individual refinement agents directly | Centralising signal at Gap Analysis keeps refinement agents single-purpose and decoupled from the RAG layer |
| Sequential refinement with justified ordering | Parallel with merge | Sequential allows compounding; ordering is non-arbitrary (see table above) |
| Feedback loop routes to weakest-dimension agent | Re-run full chain | Targeted retry avoids regressing already-good dimensions and reduces token cost per iteration |
| Gate uses `avoid-ai-writing` pattern (model-agnostic) | Static blocklist | Pattern-based detection catches evolving AI-isms beyond any fixed list; model-agnostic so gate node can run on Claude, Gemini, or any capable LLM |
| Gate aborts on persistent failure | Fails open | Failing open corrupts scores; abort with report is the correct failure mode |
| Scorer prompt is version-locked and co-cached with baselines | Ad hoc scoring prompt | Ensures pipeline output and baselines are always compared under the same rubric |
| LangSmith + local JSON fallback | LangSmith only | Trace data is load-bearing for observability claims; silent loss is unacceptable |
| Retrieval gated by similarity threshold | Gated by run count | Count-based gate fails when existing runs are for superficially similar but semantically wrong roles |
| Baselines cached per JD hash | Re-run baselines every time | Avoids redundant calls; re-evaluation available on demand |

---

## Integration: career-ops

```
cv-optimizer-agent
  └── outputs ResultCV.txt
        └── career-ops /career-ops pdf → ATS-clean PDF

Shared state:
  - cv.md (master CV) read by both tools via symlink
  - career-ops applications.md → outcome sync at run start → vector store (updates run artifact outcome field)
```

Tools are intentionally decoupled — cv-optimizer-agent does not depend on career-ops at runtime.
