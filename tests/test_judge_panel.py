"""
Tests for the multi-LLM judge panel (agent/judge.py).

All tests here are structural (no real API calls) unless the provider
keys are present.  Mocking is done via monkeypatch so no network hits.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Imports ───────────────────────────────────────────────────────────────────

class TestJudgeImports:

    def test_judge_importable(self):
        from agent.judge import score_with_panel, validate_with_panel, JudgeResult, JudgeScore
        assert score_with_panel is not None
        assert validate_with_panel is not None

    def test_judge_score_model(self):
        from agent.judge import JudgeScore
        s = JudgeScore(model="openai/gpt-4o", score=0.82, reason="Good answer", skipped=False)
        assert s.score == 0.82
        assert not s.skipped

    def test_judge_result_model(self):
        from agent.judge import JudgeResult, JudgeScore
        r = JudgeResult(
            scores=[
                JudgeScore(model="openai/gpt-4o", score=0.8, reason="good"),
                JudgeScore(model="groq/llama", score=0.6, reason="ok", skipped=True),
            ],
            ensemble_score=0.8,
            strategy="average",
            judges_used=1,
            judges_skipped=1,
        )
        assert r.judges_used == 1
        assert r.ensemble_score == 0.8


# ── Ensemble logic ────────────────────────────────────────────────────────────

class TestEnsembleStrategies:

    def test_average_strategy(self):
        from agent.judge import _ensemble
        assert _ensemble([0.6, 0.8, 1.0], "average") == pytest.approx(0.8)

    def test_conservative_strategy(self):
        from agent.judge import _ensemble
        assert _ensemble([0.6, 0.8, 1.0], "conservative") == 0.6

    def test_majority_strategy_odd(self):
        from agent.judge import _ensemble
        assert _ensemble([0.6, 0.8, 1.0], "majority") == 0.8

    def test_majority_strategy_even(self):
        from agent.judge import _ensemble
        # Even list: picks middle index
        result = _ensemble([0.5, 0.9], "majority")
        assert result in (0.5, 0.9)

    def test_empty_scores_returns_default(self):
        from agent.judge import _ensemble
        assert _ensemble([], "average") == 0.5


# ── Skipped-key behaviour ─────────────────────────────────────────────────────

class TestKeySkipping:

    def test_missing_key_causes_skip(self, monkeypatch):
        """Judge with no API key should return skipped=True without calling litellm."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from agent.judge import _call_judge_score
        result = run(_call_judge_score("openai/gpt-4o", "task", "answer"))
        assert result.skipped is True

    def test_missing_groq_key_skip(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        from agent.judge import _call_judge_score
        result = run(_call_judge_score("groq/llama-3.3-70b-versatile", "t", "a"))
        assert result.skipped is True

    def test_missing_google_key_skip(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        from agent.judge import _call_judge_score
        result = run(_call_judge_score("google/gemini-flash", "t", "a"))
        assert result.skipped is True


# ── Panel with all keys absent falls back gracefully ─────────────────────────

class TestPanelFallback:

    def test_score_panel_all_skipped_falls_back(self, monkeypatch):
        """When all external judges are skipped, panel uses Claude fallback mock."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        import agent.config as cfg
        monkeypatch.setattr(cfg, "JUDGE_MODELS", [
            "openai/gpt-4o", "google/gemini-flash", "groq/llama-3.3-70b-versatile"
        ])
        monkeypatch.setattr(cfg, "JUDGE_ENSEMBLE", "average")

        # Patch _call_judge_score so the Claude fallback also doesn't hit network
        import agent.judge as judge_mod
        from agent.judge import JudgeScore

        async def mock_call(model_id, task, answer):
            if "anthropic" in model_id:
                return JudgeScore(model=model_id, score=0.75, reason="mock fallback")
            return JudgeScore(model=model_id, score=0.0, reason="no key", skipped=True)

        monkeypatch.setattr(judge_mod, "_call_judge_score", mock_call)

        score, result = run(judge_mod.score_with_panel("task", "answer"))
        assert 0.0 <= score <= 1.0
        assert result.judges_skipped == 3

    def test_validate_panel_all_skipped_returns_default(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        import agent.config as cfg
        monkeypatch.setattr(cfg, "JUDGE_MODELS", [
            "openai/gpt-4o", "google/gemini-flash", "groq/llama-3.3-70b-versatile"
        ])
        monkeypatch.setattr(cfg, "JUDGE_ENSEMBLE", "average")

        import agent.judge as judge_mod

        async def mock_validate(model_id, task, answer, reasoning, confidence):
            if "anthropic" in model_id:
                return {
                    "passed": True, "score": 0.7,
                    "hallucination_risk": "low", "completeness": "complete",
                    "issues": [], "critique": "mock",
                }
            return None

        monkeypatch.setattr(judge_mod, "_call_judge_validate", mock_validate)

        result = run(judge_mod.validate_with_panel("task", "answer", "reasoning", 0.8))
        assert "passed" in result
        assert "score" in result


# ── Panel merging logic ───────────────────────────────────────────────────────

class TestPanelMerging:

    def test_conservative_verdict_any_fail(self, monkeypatch):
        """If any judge fails, merged result should have passed=False."""
        import agent.judge as judge_mod
        import agent.config as cfg

        monkeypatch.setattr(cfg, "JUDGE_MODELS", ["m1", "m2"])
        monkeypatch.setattr(cfg, "JUDGE_ENSEMBLE", "average")

        async def mock_validate(model_id, task, answer, reasoning, confidence):
            if model_id == "m1":
                return {"passed": True,  "score": 0.9, "hallucination_risk": "low",
                        "completeness": "complete", "issues": [], "critique": ""}
            return {"passed": False, "score": 0.4, "hallucination_risk": "high",
                    "completeness": "incomplete", "issues": ["bad claim"], "critique": "fix it"}

        monkeypatch.setattr(judge_mod, "_call_judge_validate", mock_validate)

        result = run(judge_mod.validate_with_panel("task", "answer", "r", 0.9))
        assert result["passed"] is False
        assert result["hallucination_risk"] == "high"
        assert result["completeness"] == "incomplete"
        assert "bad claim" in result["issues"]

    def test_average_score_correct(self, monkeypatch):
        import agent.judge as judge_mod
        import agent.config as cfg
        from agent.judge import JudgeScore

        monkeypatch.setattr(cfg, "JUDGE_MODELS", ["m1", "m2"])
        monkeypatch.setattr(cfg, "JUDGE_ENSEMBLE", "average")

        async def mock_score(model_id, task, answer):
            return JudgeScore(model=model_id, score=0.6 if model_id == "m1" else 0.8, reason="ok")

        monkeypatch.setattr(judge_mod, "_call_judge_score", mock_score)

        score, result = run(judge_mod.score_with_panel("task", "answer"))
        assert score == pytest.approx(0.7)
        assert result.judges_used == 2
        assert result.judges_skipped == 0


# ── Config flags ──────────────────────────────────────────────────────────────

class TestJudgeConfig:

    def test_judge_models_in_config(self):
        from agent.config import JUDGE_MODELS, JUDGE_ENSEMBLE
        assert isinstance(JUDGE_MODELS, list)
        assert len(JUDGE_MODELS) >= 1
        assert isinstance(JUDGE_ENSEMBLE, str)
        assert JUDGE_ENSEMBLE in ("average", "majority", "conservative")

    def test_judge_models_have_provider_prefix(self):
        from agent.config import JUDGE_MODELS
        for m in JUDGE_MODELS:
            assert "/" in m, f"Model '{m}' missing provider prefix (e.g. 'openai/...')"


# ── Live API tests (skipped if no keys) ──────────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") and not os.getenv("GROQ_API_KEY"),
    reason="needs at least one non-Anthropic judge API key",
)
class TestJudgePanelLive:

    def test_panel_returns_valid_score(self):
        from agent.judge import score_with_panel
        score, result = run(score_with_panel(
            task="What is 2 + 2?",
            answer="The answer is 4.",
        ))
        assert 0.0 <= score <= 1.0
        assert result.judges_used >= 1

    def test_panel_penalises_wrong_answer(self):
        from agent.judge import score_with_panel
        good_score, _ = run(score_with_panel("What is 2+2?", "4"))
        bad_score,  _ = run(score_with_panel("What is 2+2?", "The answer is 9999."))
        assert good_score > bad_score
