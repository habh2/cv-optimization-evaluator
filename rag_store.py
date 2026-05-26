import os
import json
from datetime import date
from typing import Optional

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

_DIR        = os.path.dirname(os.path.abspath(__file__))
_STORE_PATH = os.path.join(_DIR, "data", "rag_store")
_COLLECTION = "cv_runs"

SIMILARITY_THRESHOLD = 0.55
TOP_K                = 5

_OUTCOME_WEIGHT = {"contacted": 3, "no_response": 2, "rejected": 1, None: 1}


def _client() -> chromadb.PersistentClient:
    os.makedirs(_STORE_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=_STORE_PATH)


def _collection() -> chromadb.Collection:
    return _client().get_or_create_collection(_COLLECTION, metadata={"hnsw:space": "cosine"})


def _embedder() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


def embed_text(text: str) -> list[float]:
    return _embedder().embed_query(text)


def upsert_run(
    jd_id: str,
    jd_embedding: list[float],
    role_category: str,
    best_scores: dict,
    outcome: Optional[str],
    run_date: Optional[str] = None,
) -> None:
    col = _collection()
    doc_id = jd_id
    metadata = {
        "jd_id":         jd_id,
        "role_category": role_category,
        "outcome":       outcome or "null",
        "date":          run_date or str(date.today()),
        "best_scores":   json.dumps(best_scores),
    }
    col.upsert(
        ids=[doc_id],
        embeddings=[jd_embedding],
        documents=[jd_id],
        metadatas=[metadata],
    )


def retrieve_similar(
    jd_embedding: list[float],
    role_category: str,
    top_k: int = TOP_K,
) -> list[dict]:
    col = _collection()
    if col.count() == 0:
        return []

    results = col.query(
        query_embeddings=[jd_embedding],
        n_results=min(top_k, col.count()),
        where={"role_category": role_category} if role_category != "general" else None,
        include=["metadatas", "distances"],
    )

    runs = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        similarity = 1.0 - dist
        if similarity < SIMILARITY_THRESHOLD:
            continue
        outcome_val = meta["outcome"] if meta["outcome"] != "null" else None
        runs.append({
            "jd_id":         meta["jd_id"],
            "role_category": meta["role_category"],
            "outcome":       outcome_val,
            "date":          meta["date"],
            "best_scores":   json.loads(meta["best_scores"]),
            "similarity":    round(similarity, 3),
            "weight":        _OUTCOME_WEIGHT.get(outcome_val, 1),
        })

    runs.sort(key=lambda r: (r["weight"], r["similarity"]), reverse=True)
    return runs


def summarize_context(runs: list[dict]) -> str:
    if not runs:
        return ""

    contacted = [r for r in runs if r["outcome"] == "contacted"]
    rejected  = [r for r in runs if r["outcome"] == "rejected"]

    dims = ["keyword_coverage", "achievement_specificity", "jd_alignment", "readability", "voice"]

    def avg_scores(group: list[dict]) -> dict:
        if not group:
            return {}
        totals = {d: 0.0 for d in dims}
        for r in group:
            for d in dims:
                totals[d] += r["best_scores"].get(d, 0)
        return {d: round(totals[d] / len(group), 1) for d in dims}

    role = runs[0]["role_category"]
    lines = [
        f"RAG context — {len(runs)} similar past run(s) for role '{role}':",
    ]

    if contacted:
        avgs = avg_scores(contacted)
        score_str = ", ".join(f"{d}={v}" for d, v in avgs.items())
        lines.append(f"  Contacted ({len(contacted)} run(s)): {score_str}")

    if rejected:
        avgs = avg_scores(rejected)
        score_str = ", ".join(f"{d}={v}" for d, v in avgs.items())
        lines.append(f"  Rejected ({len(rejected)} run(s)):  {score_str}")

    no_outcome = [r for r in runs if r["outcome"] is None or r["outcome"] == "no_response"]
    if no_outcome and not contacted and not rejected:
        lines.append(f"  No outcome data yet ({len(no_outcome)} run(s)) — use as general context only.")

    lines.append(
        "Use these patterns to calibrate your scoring: dimensions where contacted candidates "
        "scored significantly higher than rejected ones should be weighted more critically."
    )
    return "\n".join(lines)
