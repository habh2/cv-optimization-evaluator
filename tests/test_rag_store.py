"""Tests for rag_store summarize_context and retrieve_similar filtering."""

from rag_store import summarize_context, SIMILARITY_THRESHOLD


# ── Helpers ───────────────────────────────────────────────────────────────────

_DIMS = [
    "keyword_coverage",
    "achievement_specificity",
    "jd_alignment",
    "readability",
    "voice",
]


def _run(outcome, scores=None):
    return {
        "jd_id": "test_run",
        "role_category": "ml-engineer",
        "outcome": outcome,
        "date": "2026-05-26",
        "best_scores": scores or {d: 6 for d in _DIMS},
        "similarity": 0.80,
        "weight": 3 if outcome == "contacted" else 1,
    }


# ── summarize_context ─────────────────────────────────────────────────────────


class TestSummarizeContext:
    def test_empty_runs_returns_empty_string(self):
        assert summarize_context([]) == ""

    def test_includes_role_category(self):
        result = summarize_context([_run("contacted")])
        assert "ml-engineer" in result

    def test_contacted_group_appears(self):
        result = summarize_context([_run("contacted")])
        assert "Contacted" in result

    def test_rejected_group_appears(self):
        result = summarize_context([_run("rejected")])
        assert "Rejected" in result

    def test_no_outcome_data_message_when_only_null(self):
        result = summarize_context([_run(None)])
        assert "No outcome data yet" in result

    def test_no_outcome_message_suppressed_when_contacted_present(self):
        result = summarize_context([_run("contacted"), _run(None)])
        assert "No outcome data yet" not in result

    def test_averages_contacted_scores_correctly(self):
        runs = [
            _run("contacted", {d: 8 for d in _DIMS}),
            _run("contacted", {d: 6 for d in _DIMS}),
        ]
        result = summarize_context(runs)
        # avg = 7.0 for all dims
        assert "7.0" in result

    def test_calibration_instruction_always_present(self):
        result = summarize_context([_run("contacted")])
        assert "calibrate" in result.lower()

    def test_run_count_in_header(self):
        runs = [_run("contacted"), _run("rejected")]
        result = summarize_context(runs)
        assert "2 similar past run" in result


# ── SIMILARITY_THRESHOLD ──────────────────────────────────────────────────────


def test_similarity_threshold_is_reasonable():
    # Threshold should be between 0.4 and 0.7 to be practically useful
    assert 0.4 <= SIMILARITY_THRESHOLD <= 0.7
