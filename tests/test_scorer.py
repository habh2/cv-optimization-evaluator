"""Tests for score_baseline node and supporting logic in graph.py."""
import json
from unittest.mock import MagicMock, patch

import pytest

from graph import (
    _SCORE_DIMS,
    _cache_key,
    _merge_scores,
    score_baseline,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_llm_response(scores: dict) -> MagicMock:
    """Return a mock LLM that produces a well-formed JSON scorer response."""
    payload = {d: {"score": scores.get(d, 5), "feedback": f"feedback for {d}"} for d in _SCORE_DIMS}
    mock = MagicMock()
    mock.content = json.dumps(payload)
    return mock


def _scorer_state(overrides: dict = None) -> dict:
    base = {
        "baseline_id":  "test/gemini-2.5-flash__v1",
        "cv_text":      "Software engineer with Python experience.",
        "jd_analysis":  {"must_haves": ["Python"], "keywords": ["Python"]},
        "gap_analysis": "Missing distributed systems experience.",
        "rag_context":  "",
    }
    if overrides:
        base.update(overrides)
    return base


# ── score_baseline ─────────────────────────────────────────────────────────────

class TestScoreBaseline:

    def test_happy_path_returns_all_dims(self):
        expected = {d: 6 for d in _SCORE_DIMS}
        with patch("graph.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _valid_llm_response(expected)
            result = score_baseline(_scorer_state())

        scores = result["scores"]["test/gemini-2.5-flash__v1"]
        for d in _SCORE_DIMS:
            assert scores[d]["score"] == 6
            assert "feedback for" in scores[d]["feedback"]

    def test_total_is_average_times_ten(self):
        # scores: [4, 6, 5, 7, 8] → avg=6.0 → total=60.0
        raw = {d: v for d, v in zip(_SCORE_DIMS, [4, 6, 5, 7, 8])}
        with patch("graph.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _valid_llm_response(raw)
            result = score_baseline(_scorer_state())

        total = result["scores"]["test/gemini-2.5-flash__v1"]["total"]
        assert total == 60.0

    def test_parse_failure_zeros_all_dims(self):
        mock = MagicMock()
        mock.content = "not valid json at all {{"
        with patch("graph.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = mock
            result = score_baseline(_scorer_state())

        scores = result["scores"]["test/gemini-2.5-flash__v1"]
        for d in _SCORE_DIMS:
            assert scores[d]["score"] == 0
            assert scores[d]["feedback"] == "parse error"
        assert scores["total"] == 0.0

    def test_score_clamped_above_ten(self):
        raw = {d: 15 for d in _SCORE_DIMS}
        with patch("graph.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _valid_llm_response(raw)
            result = score_baseline(_scorer_state())

        scores = result["scores"]["test/gemini-2.5-flash__v1"]
        for d in _SCORE_DIMS:
            assert scores[d]["score"] == 10

    def test_score_clamped_below_zero(self):
        raw = {d: -3 for d in _SCORE_DIMS}
        with patch("graph.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _valid_llm_response(raw)
            result = score_baseline(_scorer_state())

        scores = result["scores"]["test/gemini-2.5-flash__v1"]
        for d in _SCORE_DIMS:
            assert scores[d]["score"] == 0

    def test_result_keyed_by_baseline_id(self):
        with patch("graph.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = _valid_llm_response({})
            result = score_baseline(_scorer_state({"baseline_id": "composio/claude-sonnet-4-6__v2"}))

        assert "composio/claude-sonnet-4-6__v2" in result["scores"]

    def test_missing_dim_in_response_scores_zero(self):
        # LLM returns only 4 of 5 dims
        partial = {d: {"score": 7, "feedback": "ok"} for d in _SCORE_DIMS[:-1]}
        mock = MagicMock()
        mock.content = json.dumps(partial)
        with patch("graph.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = mock
            result = score_baseline(_scorer_state())

        missing_dim = _SCORE_DIMS[-1]
        assert result["scores"]["test/gemini-2.5-flash__v1"][missing_dim]["score"] == 0

    def test_markdown_fenced_json_is_parsed(self):
        payload = {d: {"score": 5, "feedback": "ok"} for d in _SCORE_DIMS}
        fenced = f"```json\n{json.dumps(payload)}\n```"
        mock = MagicMock()
        mock.content = fenced
        with patch("graph.get_llm") as mock_get_llm:
            mock_get_llm.return_value.invoke.return_value = mock
            result = score_baseline(_scorer_state())

        for d in _SCORE_DIMS:
            assert result["scores"]["test/gemini-2.5-flash__v1"][d]["score"] == 5


# ── _merge_scores ──────────────────────────────────────────────────────────────

class TestMergeScores:

    def test_merges_disjoint_keys(self):
        a = {"master_cv": {"total": 50.0}}
        b = {"composio/gemini__v1": {"total": 65.0}}
        merged = _merge_scores(a, b)
        assert "master_cv" in merged
        assert "composio/gemini__v1" in merged

    def test_new_key_overwrites_existing(self):
        a = {"master_cv": {"total": 50.0}}
        b = {"master_cv": {"total": 55.0}}
        merged = _merge_scores(a, b)
        assert merged["master_cv"]["total"] == 55.0

    def test_empty_existing_returns_new(self):
        assert _merge_scores({}, {"k": {"total": 1}}) == {"k": {"total": 1}}


# ── _cache_key ─────────────────────────────────────────────────────────────────

class TestCacheKey:

    def test_format(self):
        key = _cache_key("ml_eng_google", "abcdef1234567890")
        assert key == "ml_eng_google_abcdef12_v1.0"

    def test_uses_first_eight_chars_of_hash(self):
        key = _cache_key("job", "1234567890abcdef")
        assert key.startswith("job_12345678_")

    def test_different_hashes_produce_different_keys(self):
        k1 = _cache_key("job", "aaaa0000xxxxxxxx")
        k2 = _cache_key("job", "bbbb1111xxxxxxxx")
        assert k1 != k2
