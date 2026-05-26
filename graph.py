import os
import json
import hashlib
import sys
from datetime import date
from typing import Dict, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from skills_registry import SKILLS

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
SCORER_VERSION       = "v1.0"
SATURATION_THRESHOLD = 85
HEADROOM_THRESHOLD   = 10
PROVIDER             = "gemini"

_DIR       = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(_DIR, "data", "cache", "baselines.json")


def _load_skill_prompts() -> None:
    skills_dir = os.path.join(_DIR, "skills")
    for skill_id, skill in SKILLS.items():
        path = os.path.join(skills_dir, skill_id, "SKILL.md")
        with open(path, "r", encoding="utf-8") as f:
            skill["system"] = f.read()

_load_skill_prompts()


# ── LLM factory ──────────────────────────────────────────────────────────────
def get_llm(provider: str = PROVIDER, model: str = None):
    if provider == "ollama":
        return ChatOllama(model=model or "qwen3:4b")
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.5-flash",
            temperature=0.3,
        )
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


# ── Scorer prompt (version-locked) ───────────────────────────────────────────
SCORER_PROMPT = """\
Score this resume against the job description on four dimensions (0-10 each).

Keyword coverage        - what % of the JD's must-have and nice-to-have keywords appear naturally
Achievement specificity - ratio of quantified, impact-first bullets vs vague statements
JD alignment            - how well the CV summary and key bullets match JD requirements
Readability             - sentence clarity, active voice, ATS-safe formatting (no tables/special chars)

Return JSON only, no prose, no markdown fences:
{{"keyword_coverage": <int>, "achievement_specificity": <int>, "jd_alignment": <int>, "readability": <int>, "reasoning": "<one sentence>"}}

JOB DESCRIPTION:
{jd}

RESUME:
{cv}"""


# ── State ─────────────────────────────────────────────────────────────────────
class EvaluateState(TypedDict):
    jd: str
    master_cv: str
    jd_hash: str
    from_cache: bool
    skill_outputs: Dict[str, str]
    scores: Dict[str, dict]
    best_skill: str
    saturated: bool
    headroom: Dict[str, float]


# ── Helpers ───────────────────────────────────────────────────────────────────
_SCORE_DIMS = ["keyword_coverage", "achievement_specificity", "jd_alignment", "readability"]


def _compute_derived(scores: dict, best_skill: str) -> tuple[bool, Dict[str, float]]:
    best = scores[best_skill]
    headroom = {dim: round(100 - best[dim] * 10, 1) for dim in _SCORE_DIMS}
    saturated = (
        best["total"] >= SATURATION_THRESHOLD
        and all(v < HEADROOM_THRESHOLD for v in headroom.values())
    )
    return saturated, headroom


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
    jd = loadfile("data/inputs/JobDescription.txt")
    cv = loadfile("data/inputs/CV.md")
    jd_hash = hashlib.sha256(jd.encode()).hexdigest()[:12]
    cache_key = f"{jd_hash}_{SCORER_VERSION}"
    force = "--force" in sys.argv

    if not force:
        cache = _load_cache()
        if cache_key in cache:
            scores = cache[cache_key]["skills"]
            best_skill = max(
                (k for k in scores if k != "master_cv"),
                key=lambda k: scores[k]["total"],
            )
            saturated, headroom = _compute_derived(scores, best_skill)
            return {
                "jd": jd, "master_cv": cv, "jd_hash": jd_hash,
                "from_cache": True,
                "skill_outputs": {},
                "scores": scores,
                "best_skill": best_skill,
                "saturated": saturated,
                "headroom": headroom,
            }

    return {
        "jd": jd, "master_cv": cv, "jd_hash": jd_hash,
        "from_cache": False,
        "skill_outputs": {}, "scores": {}, "best_skill": "", "saturated": False, "headroom": {},
    }


def run_skills(state: EvaluateState) -> dict:
    llm = get_llm()
    outputs: Dict[str, str] = {"master_cv": state["master_cv"]}

    for skill_id, skill in SKILLS.items():
        print(f"  Running: {skill['name']}...")
        user_msg = skill["user_template"].format(jd=state["jd"], cv=state["master_cv"])
        response = llm.invoke([
            SystemMessage(content=skill["system"]),
            HumanMessage(content=user_msg),
        ])
        outputs[skill_id] = extract_text(response)

    return {"skill_outputs": outputs}


def score_all(state: EvaluateState) -> dict:
    llm = get_llm()
    scores: Dict[str, dict] = {}

    for skill_id, cv_text in state["skill_outputs"].items():
        print(f"  Scoring: {skill_id}...")
        prompt = SCORER_PROMPT.format(jd=state["jd"], cv=cv_text)
        raw = extract_text(llm.invoke(prompt)).strip()

        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

        try:
            parsed = json.loads(raw)
            dims = {d: max(0, min(10, int(parsed.get(d, 0)))) for d in _SCORE_DIMS}
            reasoning = parsed.get("reasoning", "")
        except (json.JSONDecodeError, ValueError):
            print(f"    Warning: could not parse score for {skill_id}, using zeros.")
            dims = {d: 0 for d in _SCORE_DIMS}
            reasoning = "parse error"

        total = round(sum(dims.values()) / len(_SCORE_DIMS) * 10, 1)
        scores[skill_id] = {**dims, "total": total, "reasoning": reasoning}

    best_skill = max(
        (k for k in scores if k != "master_cv"),
        key=lambda k: scores[k]["total"],
        default="master_cv",
    )
    saturated, headroom = _compute_derived(scores, best_skill)

    return {"scores": scores, "best_skill": best_skill, "saturated": saturated, "headroom": headroom}


def report(state: EvaluateState) -> dict:
    scores  = state["scores"]
    best_id = state["best_skill"]

    if state["from_cache"]:
        print("[cached - use --force to re-evaluate]\n")

    dim_headers = {"keyword_coverage": "Keyword", "achievement_specificity": "Achieve.",
                   "jd_alignment": "Alignment", "readability": "Readability"}

    col_w = 30
    header = f"{'Skill':<{col_w}} | " + " | ".join(f"{v:>10}" for v in dim_headers.values()) + " | Total"
    print(header)
    print("-" * len(header))

    order = ["master_cv"] + [k for k in scores if k != "master_cv"]
    for skill_id in order:
        s = scores[skill_id]
        label = SKILLS[skill_id]["name"] if skill_id in SKILLS else "master_cv (baseline)"
        dim_str = " | ".join(f"{s[d]:>9}/10" for d in _SCORE_DIMS)
        marker = " <--" if skill_id == best_id else ""
        print(f"{label:<{col_w}} | {dim_str} | {s['total']:>5.1f}{marker}")

    print()
    best = scores[best_id]
    best_label = SKILLS[best_id]["name"] if best_id in SKILLS else best_id
    print(f"Best skill: {best_label} ({best['total']:.1f}/100)")

    if state["saturated"]:
        headroom = state["headroom"]
        dims_open = {k: v for k, v in headroom.items() if v >= 1}
        hw_str = ", ".join(f"{k}: +{v:.0f}" for k, v in dims_open.items()) or "none"
        print(f"\nWARNING: Best public skill scored {best['total']:.1f}/100.")
        print(f"  Dimensions with remaining headroom: [{hw_str}]")
        print("  Running the optimisation pipeline is unlikely to add meaningful value.")

    # persist to cache (store without per-run reasoning to keep cache clean)
    cache = _load_cache()
    cache_key = f"{state['jd_hash']}_{SCORER_VERSION}"
    cache[cache_key] = {
        "date": str(date.today()),
        "skills": scores,
    }
    _save_cache(cache)
    print(f"\nResults cached to {CACHE_PATH}")

    return {}


# ── Graph ─────────────────────────────────────────────────────────────────────
builder = StateGraph(EvaluateState)
builder.add_node("intake",     intake)
builder.add_node("run_skills", run_skills)
builder.add_node("score_all",  score_all)
builder.add_node("report",     report)

builder.set_entry_point("intake")
builder.add_conditional_edges(
    "intake",
    lambda s: "report" if s["from_cache"] else "run_skills",
)
builder.add_edge("run_skills", "score_all")
builder.add_edge("score_all",  "report")
builder.add_edge("report",     END)

graph = builder.compile()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    graph.invoke({
        "jd": "", "master_cv": "", "jd_hash": "",
        "from_cache": False,
        "skill_outputs": {}, "scores": {}, "best_skill": "", "saturated": False, "headroom": {},
    })
