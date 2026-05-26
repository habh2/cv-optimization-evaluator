import os
import re
import json
import sys
import hashlib
import argparse
from datetime import date
from typing import Dict, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from skills_registry import SKILLS

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SCORER_VERSION = "v1.0"
PROVIDER       = "gemini"

_DIR          = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH    = os.path.join(_DIR, "data", "cache", "baselines.json")
BASELINES_DIR = os.path.join(_DIR, "data", "inputs", "baselines")

# {model}__{version}.txt  — double underscore separates model from version
_FILENAME_RE = re.compile(r"^([a-zA-Z0-9_.\-]+)__([a-zA-Z0-9_\-]+)\.txt$")


# ── LLM factory ───────────────────────────────────────────────────────────────
def get_llm(provider: str = PROVIDER, model: str = None):
    if provider == "ollama":
        return ChatOllama(model=model or "qwen3:4b")
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(model=model or "gemini-2.5-flash", temperature=0.3)
    raise ValueError(f"Unknown provider: {provider}")


# ── Utilities ─────────────────────────────────────────────────────────────────
def loadfile(filename: str) -> str:
    path = os.path.join(_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(response) -> str:
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, list):
            return content[0]["text"]
        if isinstance(content, str):
            return content
    return str(response)


def strip_fences(raw: str) -> str:
    if raw.startswith("```"):
        parts = raw.split("```")
        return parts[1].lstrip("json").strip() if len(parts) > 1 else raw
    return raw


# ── Prompts (version-locked at SCORER_VERSION) ────────────────────────────────
JD_ANALYZER_PROMPT = """\
Analyze this job description and extract structured information.

Return JSON only, no prose, no markdown fences:
{{"must_haves": ["<requirement>", ...], "nice_to_haves": ["<requirement>", ...], "keywords": ["<keyword>", ...], "seniority": "<junior|mid|senior|lead|staff|principal>", "tone": "<formal|casual|technical|startup>"}}

JOB DESCRIPTION:
{jd}"""

GAP_ANALYZER_PROMPT = """\
Compare this resume against the job description analysis and identify gaps and strengths.

JOB DESCRIPTION ANALYSIS:
{jd_analysis}

MASTER CV:
{master_cv}

Provide a concise analysis (3-5 bullet points):
- Missing must-have skills or experience gaps
- Areas of weaker alignment
- Strengths that map well to the JD
"""

SCORER_PROMPT = """\
You are a brutal resume evaluator. Score this resume on five dimensions (0-10 each) using the JD analysis as ground truth. Be harsh: a 7 means genuinely strong, a 10 is nearly impossible. Most resumes should score 4-6 on most dimensions. Do not inflate scores to be encouraging.

Dimensions:
- keyword_coverage: how well JD must-haves and nice-to-haves appear in the CV — but non-linearly: coverage peaking around 70-80% scores highest (natural, targeted); 90-100% is a red flag for keyword stuffing or AI generation and should score lower than 75%, not higher
- achievement_specificity: ratio of quantified, impact-first bullets vs vague statements ("contributed to", "helped with")
- jd_alignment: how well the CV summary and key bullets match JD requirements
- readability: sentence clarity, active voice, ATS-safe formatting (no tables, special chars, or graphics)
- voice: how authentically human the language sounds, scored against the avoid-ai-writing pattern list below

AVOID-AI-WRITING PATTERN LIST (for the voice dimension):
Tier 1 — always penalise (each hit -1 pt): leverage, utilize, robust, comprehensive, cutting-edge, seamless, impactful, actionable, holistic, meticulous, pivotal, synergy, thought leader, best practices, spearhead, transformative, cornerstone, paramount, results-driven, passionate about, seeking to leverage, dynamic professional, serves as, boasts, showcasing, embark, endeavor, keen, commenced
Tier 2 — penalise in clusters (2+ per section): harness, foster, elevate, streamline, empower, revolutionize, facilitate, nuanced, multifaceted, overarching, instrumental, world-class, state-of-the-art, best-in-class, exceptional, remarkable, sophisticated
Structural tells (each hit -1 pt): formulaic summary opener ("Results-driven [role]", "In today's fast-paced", "Dynamic professional with X years"), hollow intensifiers (truly, genuinely, highly motivated), bullet list where every item has identical grammatical shape and similar word count, vague attributions without specifics ("collaborated with stakeholders", "drove impact"), copula evasion ("serves as", "features" instead of "is"/"has")

For each dimension provide a score and short actionable feedback (1-2 sentences, be specific about which phrases or bullets to fix).

Return JSON only, no prose, no markdown fences:
{{"keyword_coverage": {{"score": <int>, "feedback": "<text>"}}, "achievement_specificity": {{"score": <int>, "feedback": "<text>"}}, "jd_alignment": {{"score": <int>, "feedback": "<text>"}}, "readability": {{"score": <int>, "feedback": "<text>"}}, "voice": {{"score": <int>, "feedback": "<text>"}}}}

JOB DESCRIPTION ANALYSIS:
{jd_analysis}

RESUME:
{cv}"""

_SCORE_DIMS = ["keyword_coverage", "achievement_specificity", "jd_alignment", "readability", "voice"]


# ── State ─────────────────────────────────────────────────────────────────────
class EvaluateState(TypedDict):
    jd: str
    master_cv: str
    jd_id: str
    from_cache: bool
    baselines: Dict[str, str]    # baseline_id -> CV text
    jd_analysis: dict
    gap_analysis: str
    scores: Dict[str, dict]      # baseline_id -> {dim: {score, feedback}, total}
    best_baseline: str


# ── Cache helpers ─────────────────────────────────────────────────────────────
def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


# ── Nodes ─────────────────────────────────────────────────────────────────────
def intake(state: EvaluateState) -> dict:
    jd_id = state["jd_id"]
    jd    = loadfile("data/inputs/JobDescription.txt")
    cv    = loadfile("data/inputs/CV.md")
    jd_hash   = hashlib.sha256(jd.encode()).hexdigest()[:12]
    force     = "--force" in sys.argv
    cache_id  = jd_id if jd_id else f"master_{jd_hash}"
    cache_key = f"{cache_id}_{SCORER_VERSION}"

    if not force:
        cache = _load_cache()
        if cache_key in cache:
            entry = cache[cache_key]
            scores = entry["baselines"]
            best_baseline = max(
                (k for k in scores if k != "master_cv"),
                key=lambda k: scores[k]["total"],
                default="master_cv",
            )
            return {
                "jd": jd, "master_cv": cv, "jd_id": jd_id,
                "from_cache": True,
                "baselines": {},
                "jd_analysis": entry.get("jd_analysis", {}),
                "gap_analysis": entry.get("gap_analysis", ""),
                "scores": scores,
                "best_baseline": best_baseline,
            }

    return {
        "jd": jd, "master_cv": cv, "jd_id": jd_id,
        "from_cache": False,
        "baselines": {}, "jd_analysis": {}, "gap_analysis": "",
        "scores": {}, "best_baseline": "",
    }


def jd_analyzer(state: EvaluateState) -> dict:
    print("  Analyzing JD...")
    llm  = get_llm()
    raw  = extract_text(llm.invoke(JD_ANALYZER_PROMPT.format(jd=state["jd"]))).strip()
    raw  = strip_fences(raw)
    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError:
        print("    Warning: could not parse JD analysis.")
        analysis = {}
    return {"jd_analysis": analysis}


def gap_analyzer(state: EvaluateState) -> dict:
    print("  Analyzing gaps...")
    llm = get_llm()
    prompt = GAP_ANALYZER_PROMPT.format(
        jd_analysis=json.dumps(state["jd_analysis"], indent=2),
        master_cv=state["master_cv"],
    )
    gap_text = extract_text(llm.invoke(prompt)).strip()
    return {"gap_analysis": gap_text}


def load_baselines(state: EvaluateState) -> dict:
    jd_id = state["jd_id"]

    if not jd_id:
        print("  No --run specified; scoring master CV only.")
        return {"baselines": {"master_cv": state["master_cv"]}}

    baselines_dir = os.path.join(BASELINES_DIR, jd_id)

    if not os.path.isdir(baselines_dir):
        raise FileNotFoundError(
            f"No baselines folder for jd_id '{jd_id}'. Expected: {baselines_dir}"
        )

    baselines: Dict[str, str] = {"master_cv": state["master_cv"]}

    for skill_id in sorted(os.listdir(baselines_dir)):
        skill_dir = os.path.join(baselines_dir, skill_id)
        if not os.path.isdir(skill_dir):
            continue
        if skill_id not in SKILLS:
            print(f"  Warning: unknown skill_id '{skill_id}', skipping.")
            continue
        for filename in sorted(os.listdir(skill_dir)):
            filepath = os.path.join(skill_dir, filename)
            if not os.path.isfile(filepath):
                continue
            if not _FILENAME_RE.match(filename):
                raise ValueError(
                    f"Invalid baseline filename '{filename}' in {skill_dir}. "
                    f"Expected: {{model}}__{{version}}.txt"
                )
            baseline_id = f"{skill_id}/{filename[:-4]}"
            with open(filepath, "r", encoding="utf-8") as f:
                baselines[baseline_id] = f.read()
            print(f"  Loaded: {baseline_id}")

    print(f"  Baselines loaded: {len(baselines)} (includes master_cv)")
    return {"baselines": baselines}


def score_all(state: EvaluateState) -> dict:
    llm          = get_llm()
    jd_analysis  = json.dumps(state["jd_analysis"], indent=2)
    scores: Dict[str, dict] = {}

    for baseline_id, cv_text in state["baselines"].items():
        print(f"  Scoring: {baseline_id}...")
        raw = extract_text(llm.invoke(SCORER_PROMPT.format(
            jd_analysis=jd_analysis, cv=cv_text,
        ))).strip()
        raw = strip_fences(raw)

        try:
            parsed = json.loads(raw)
            dims = {}
            for d in _SCORE_DIMS:
                dim_data = parsed.get(d, {})
                dims[d] = {
                    "score":    max(0, min(10, int(dim_data.get("score", 0)))),
                    "feedback": str(dim_data.get("feedback", "")),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            print(f"    Warning: could not parse score for {baseline_id}, using zeros.")
            dims = {d: {"score": 0, "feedback": "parse error"} for d in _SCORE_DIMS}

        total = round(sum(dims[d]["score"] for d in _SCORE_DIMS) / len(_SCORE_DIMS) * 10, 1)
        scores[baseline_id] = {**dims, "total": total}

    best_baseline = max(
        (k for k in scores if k != "master_cv"),
        key=lambda k: scores[k]["total"],
        default="master_cv",
    )
    return {"scores": scores, "best_baseline": best_baseline}


def report(state: EvaluateState) -> dict:
    scores  = state["scores"]
    best_id = state["best_baseline"]

    if state["from_cache"]:
        print("[cached - use --force to re-evaluate]\n")

    if state.get("gap_analysis"):
        print("Gap Analysis (master CV vs JD):")
        print(state["gap_analysis"])
        print()

    # Comparison table
    dim_labels = {
        "keyword_coverage":        "Keywords",
        "achievement_specificity": "Achieve.",
        "jd_alignment":            "Alignment",
        "readability":             "Readability",
        "voice":                   "Voice",
    }
    col_w  = 45
    header = f"{'Baseline':<{col_w}} | " + " | ".join(f"{v:>11}" for v in dim_labels.values()) + " | Total"
    print(header)
    print("-" * len(header))

    order = ["master_cv"] + [k for k in scores if k != "master_cv"]
    for baseline_id in order:
        s      = scores[baseline_id]
        dims_s = " | ".join(f"{s[d]['score']:>10}/10" for d in _SCORE_DIMS)
        marker = " <--" if baseline_id == best_id else ""
        print(f"{baseline_id:<{col_w}} | {dims_s} | {s['total']:>5.1f}{marker}")

    print()
    print(f"Best baseline: {best_id} ({scores[best_id]['total']:.1f}/100)")

    # Per-baseline feedback
    print("\n--- Feedback Report ---\n")
    for baseline_id in order:
        s = scores[baseline_id]
        print(f"[{baseline_id}]  total: {s['total']:.1f}/100")
        for d in _SCORE_DIMS:
            print(f"  {d} ({s[d]['score']}/10): {s[d]['feedback']}")
        print()

    # Persist to cache
    cache     = _load_cache()
    cache_key = f"{state['jd_id']}_{SCORER_VERSION}"
    cache[cache_key] = {
        "date":         str(date.today()),
        "jd_analysis":  state.get("jd_analysis", {}),
        "gap_analysis": state.get("gap_analysis", ""),
        "baselines":    scores,
    }
    _save_cache(cache)
    print(f"Results cached to {CACHE_PATH}")

    return {}


# ── Graph ─────────────────────────────────────────────────────────────────────
builder = StateGraph(EvaluateState)
builder.add_node("intake",         intake)
builder.add_node("jd_analyzer",    jd_analyzer)
builder.add_node("gap_analyzer",   gap_analyzer)
builder.add_node("load_baselines", load_baselines)
builder.add_node("score_all",      score_all)
builder.add_node("report",         report)

builder.set_entry_point("intake")
builder.add_conditional_edges(
    "intake",
    lambda s: "report" if s["from_cache"] else "jd_analyzer",
)
builder.add_edge("jd_analyzer",    "gap_analyzer")
builder.add_edge("gap_analyzer",   "load_baselines")
builder.add_edge("load_baselines", "score_all")
builder.add_edge("score_all",      "report")
builder.add_edge("report",         END)

graph = builder.compile()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CV baselines against a job description.")
    parser.add_argument("--run",   default="", metavar="JD_ID",
                        help="Job application folder under data/inputs/baselines/ (omit to score master CV only)")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached results and re-evaluate")
    args = parser.parse_args()

    graph.invoke({
        "jd": "", "master_cv": "", "jd_id": args.run,
        "from_cache": False,
        "baselines": {}, "jd_analysis": {}, "gap_analysis": "",
        "scores": {}, "best_baseline": "",
    })
