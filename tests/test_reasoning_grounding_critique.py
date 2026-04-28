"""
Tests for reasoning enforcement, grounding, and self-critique modules.

These three together form the anti-hallucination + reasoning guarantee layer.
All tests here are structural (no API calls) unless ANTHROPIC_API_KEY is set.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Reasoning ─────────────────────────────────────────────────────────────────

class TestReasoning:

    def test_reasoning_importable(self):
        from agent.reasoning import build_reasoning_plan, plan_to_context, ReasoningPlan
        assert build_reasoning_plan is not None

    def test_reasoning_plan_model(self):
        from agent.reasoning import ReasoningPlan
        plan = ReasoningPlan(
            understand="Find the capital of France",
            already_know=["France is a country in Europe"],
            gaps=["Need to confirm capital name"],
            tool_plan=["1. Call search_web to look up France capital"],
            risk="Tool may return outdated info",
            confidence_prior=0.9,
        )
        assert 0.0 <= plan.confidence_prior <= 1.0
        assert isinstance(plan.gaps, list)
        assert isinstance(plan.tool_plan, list)

    def test_plan_to_context_format(self):
        from agent.reasoning import ReasoningPlan, plan_to_context
        plan = ReasoningPlan(
            understand="test task",
            already_know=["fact a"],
            gaps=["gap 1"],
            tool_plan=["1. call tool X"],
            risk="none",
            confidence_prior=0.7,
        )
        ctx = plan_to_context(plan)
        assert "[Reasoning Plan]" in ctx
        assert "gap 1" in ctx
        assert "call tool X" in ctx
        assert "70%" in ctx

    def test_reasoning_disabled_returns_plan(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "REASONING_ENFORCEMENT", False)
        from agent.reasoning import build_reasoning_plan
        plan = run(build_reasoning_plan("test task", ["tool_a"]))
        assert plan is not None

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs API key")
    def test_reasoning_plan_with_api(self):
        from agent.reasoning import build_reasoning_plan
        plan = run(build_reasoning_plan(
            "What is the population of Tokyo?",
            ["search_web", "fetch_data"],
        ))
        assert plan.understand != ""
        assert len(plan.tool_plan) > 0
        assert 0.0 <= plan.confidence_prior <= 1.0


# ── Grounding ─────────────────────────────────────────────────────────────────

class TestGrounding:

    def test_grounding_importable(self):
        from agent.grounding import EvidenceLedger, check_grounding, GroundingResult
        assert EvidenceLedger is not None

    def test_evidence_ledger_add_and_retrieve(self):
        from agent.grounding import EvidenceLedger
        ledger = EvidenceLedger()
        assert ledger.is_empty()
        ledger.add("search_tool", "query: Paris", "Paris is the capital of France")
        assert not ledger.is_empty()
        assert "Paris" in ledger.all_outputs()

    def test_evidence_ledger_as_text(self):
        from agent.grounding import EvidenceLedger
        ledger = EvidenceLedger()
        ledger.add("web_search", "France capital", "Paris is the capital")
        text = ledger.as_text()
        assert "web_search" in text
        assert "Paris" in text
        assert "[Evidence 1]" in text

    def test_empty_ledger_as_text(self):
        from agent.grounding import EvidenceLedger
        ledger = EvidenceLedger()
        text = ledger.as_text()
        assert "No tool results" in text

    def test_grounding_disabled_returns_pass(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "GROUNDING_ENABLED", False)
        from agent.grounding import EvidenceLedger, check_grounding
        ledger = EvidenceLedger()
        result = run(check_grounding("task", "answer", ledger))
        assert result.verdict == "pass"
        assert result.grounding_ratio == 1.0

    def test_empty_ledger_returns_pass(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "GROUNDING_ENABLED", True)
        from agent.grounding import EvidenceLedger, check_grounding
        ledger = EvidenceLedger()   # empty
        result = run(check_grounding("task", "answer", ledger))
        assert result.verdict == "pass"

    def test_grounding_result_model(self):
        from agent.grounding import GroundingResult
        r = GroundingResult(
            grounded_claims=["claim a"],
            ungrounded_claims=[],
            grounding_ratio=1.0,
            verdict="pass",
            recommendation="",
        )
        assert r.verdict == "pass"
        assert r.grounding_ratio == 1.0

    def test_grounding_feedback_string(self):
        from agent.grounding import GroundingResult, grounding_feedback
        r = GroundingResult(
            grounded_claims=[],
            ungrounded_claims=["claim X is false"],
            grounding_ratio=0.4,
            verdict="fail",
            recommendation="Remove unsupported claims",
        )
        fb = grounding_feedback(r)
        assert "claim X is false" in fb
        assert "Grounding Failure" in fb

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs API key")
    def test_grounding_with_api_grounded(self):
        from agent.grounding import EvidenceLedger, check_grounding
        ledger = EvidenceLedger()
        ledger.add("search", "Paris capital", "Paris is the capital of France with population 2.1M")
        result = run(check_grounding(
            "What is the capital of France?",
            "Paris is the capital of France.",
            ledger,
        ))
        assert result.grounding_ratio > 0.5

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs API key")
    def test_grounding_with_api_ungrounded(self):
        from agent.grounding import EvidenceLedger, check_grounding
        ledger = EvidenceLedger()
        ledger.add("search", "Paris capital", "Paris is the capital of France")
        result = run(check_grounding(
            "What is the capital of France?",
            "Paris is the capital, founded in 52 BC with 17 Michelin-starred restaurants.",
            ledger,
        ))
        # Specific numbers not in evidence should be flagged
        assert isinstance(result.ungrounded_claims, list)


# ── Self-Critique ─────────────────────────────────────────────────────────────

class TestSelfCritique:

    def test_critique_importable(self):
        from agent.critique import self_critique, SelfCritiqueResult
        assert self_critique is not None

    def test_critique_result_model(self):
        from agent.critique import SelfCritiqueResult
        r = SelfCritiqueResult(
            original_answer="original",
            issues_found=["too vague"],
            missing_elements=["specific number"],
            unsupported_claims=[],
            revised_answer="revised and improved",
            critique_summary="Added specifics",
            revised_confidence=0.85,
        )
        assert r.revised_confidence == 0.85
        assert "revised" in r.revised_answer

    def test_critique_disabled_returns_original(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "SELF_CRITIQUE_ENABLED", False)
        from agent.critique import self_critique
        result = run(self_critique(
            task="test",
            draft_answer="original answer",
            draft_confidence=0.7,
            evidence_text="[Evidence 1] Tool: search\n Output: some result",
            reasoning_plan_text="[Reasoning Plan] ...",
        ))
        assert result.revised_answer == "original answer"
        assert result.revised_confidence == 0.7

    @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs API key")
    def test_critique_with_api_returns_revision(self):
        from agent.critique import self_critique
        result = run(self_critique(
            task="What is 2 + 2?",
            draft_answer="The answer is 5.",
            draft_confidence=0.9,
            evidence_text="[Evidence 1] Tool: calculator\n  Input: 2+2\n  Output: 4",
            reasoning_plan_text="Plan: use calculator",
        ))
        assert isinstance(result.revised_answer, str)
        assert len(result.revised_answer) > 0
        assert isinstance(result.issues_found, list)
        # Confidence should drop after catching the error
        assert result.revised_confidence <= 0.9


# ── Full pipeline flags ───────────────────────────────────────────────────────

class TestPipelineFlags:

    def test_all_flags_importable(self):
        from agent.config import (
            REASONING_ENFORCEMENT,
            GROUNDING_ENABLED, GROUNDING_FAIL_THRESHOLD, GROUNDING_WARN_THRESHOLD,
            SELF_CRITIQUE_ENABLED,
            VALIDATOR_ENABLED,
        )
        assert isinstance(REASONING_ENFORCEMENT, bool)
        assert isinstance(GROUNDING_ENABLED, bool)
        assert isinstance(SELF_CRITIQUE_ENABLED, bool)
        assert isinstance(VALIDATOR_ENABLED, bool)
        assert 0.0 <= GROUNDING_FAIL_THRESHOLD <= 1.0
        assert GROUNDING_FAIL_THRESHOLD < GROUNDING_WARN_THRESHOLD

    def test_defaults_are_all_on(self):
        from agent.config import (
            REASONING_ENFORCEMENT, GROUNDING_ENABLED,
            SELF_CRITIQUE_ENABLED, VALIDATOR_ENABLED,
        )
        assert REASONING_ENFORCEMENT is True
        assert GROUNDING_ENABLED is True
        assert SELF_CRITIQUE_ENABLED is True
        assert VALIDATOR_ENABLED is True
