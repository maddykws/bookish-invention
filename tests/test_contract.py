"""
Tests for the Final Output Contract (agent/contract.py).

All tests are structural — no API calls required.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_output(**overrides):
    """Build a valid FinalOutput dict, with optional overrides."""
    defaults = dict(
        answer="Paris is the capital of France.",
        confidence=0.9,
        reasoning="Searched web, found authoritative source.",
        tools_used=["web_search"],
        grounding_ratio=0.95,
        grounding_verdict="pass",
        sources=["web_search: Paris is the capital of France with population 2.1M"],
        reflections=0,
        latency_ms=1200,
        model="claude-sonnet-4-6",
    )
    defaults.update(overrides)
    return defaults


# ── Import checks ──────────────────────────────────────────────────────────────

class TestContractImports:

    def test_importable(self):
        from agent.contract import enforce_contract, FinalOutput, ContractResult, ContractViolation
        assert enforce_contract is not None

    def test_final_output_model(self):
        from agent.contract import FinalOutput
        o = FinalOutput(**_make_output())
        assert o.contract_version == "1.0"
        assert o.grounding_verdict == "pass"

    def test_confidence_clamped(self):
        from agent.contract import FinalOutput
        o = FinalOutput(**_make_output(confidence=1.5))
        assert o.confidence == 1.0

    def test_ratio_clamped_low(self):
        from agent.contract import FinalOutput
        o = FinalOutput(**_make_output(grounding_ratio=-0.1))
        assert o.grounding_ratio == 0.0

    def test_invalid_verdict_normalised(self):
        from agent.contract import FinalOutput
        o = FinalOutput(**_make_output(grounding_verdict="unknown"))
        assert o.grounding_verdict == "warn"


# ── Happy-path ────────────────────────────────────────────────────────────────

class TestContractPass:

    def test_valid_output_passes(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONTRACT_ENABLED", True)
        from agent.contract import enforce_contract
        result = enforce_contract(**_make_output())
        assert result.passed is True
        assert result.violations == []

    def test_contract_disabled_always_passes(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONTRACT_ENABLED", False)
        from agent.contract import enforce_contract
        result = enforce_contract(**_make_output(answer="", reasoning=""))
        assert result.passed is True


# ── Field violations ──────────────────────────────────────────────────────────

class TestFieldViolations:

    def test_empty_answer_fails(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONTRACT_ENABLED", True)
        from agent.contract import enforce_contract
        result = enforce_contract(**_make_output(answer=""))
        assert result.passed is False
        assert any(v.field == "answer" for v in result.violations)

    def test_short_answer_fails(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONTRACT_ENABLED", True)
        from agent.contract import enforce_contract
        result = enforce_contract(**_make_output(answer="ok"))
        assert result.passed is False

    def test_empty_reasoning_fails(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONTRACT_ENABLED", True)
        from agent.contract import enforce_contract
        result = enforce_contract(**_make_output(reasoning=""))
        assert result.passed is False
        assert any(v.field == "reasoning" for v in result.violations)

    def test_grounding_fail_verdict_triggers_violation(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONTRACT_ENABLED", True)
        from agent.contract import enforce_contract
        result = enforce_contract(**_make_output(grounding_verdict="fail", grounding_ratio=0.4))
        assert result.passed is False
        assert any(v.field == "grounding_verdict" for v in result.violations)

    def test_empty_model_uses_config_fallback(self, monkeypatch):
        """Empty model string falls back to config MODEL, so no violation."""
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONTRACT_ENABLED", True)
        from agent.contract import enforce_contract
        result = enforce_contract(**_make_output(model=""))
        # The contract fills in the MODEL from config — output should be non-empty
        assert result.output.model != ""


# ── Forbidden-pattern detection ───────────────────────────────────────────────

class TestForbiddenPatterns:

    def test_bare_i_dont_know(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONTRACT_ENABLED", True)
        from agent.contract import enforce_contract
        result = enforce_contract(**_make_output(answer="I don't know what the capital is."))
        violations = [v.rule for v in result.violations]
        assert any("don" in r for r in violations)

    def test_todo_placeholder(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONTRACT_ENABLED", True)
        from agent.contract import enforce_contract
        result = enforce_contract(**_make_output(answer="The answer is TODO — fill in later."))
        # Should be auto-repaired, not a hard fail
        assert result.output.answer != "The answer is TODO — fill in later."

    def test_tbd_placeholder_repaired(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONTRACT_ENABLED", True)
        from agent.contract import enforce_contract
        result = enforce_contract(**_make_output(answer="Result: [TBD] more research needed."))
        assert "[TBD]" not in result.output.answer

    def test_overconfident_with_low_confidence(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONTRACT_ENABLED", True)
        from agent.contract import enforce_contract
        result = enforce_contract(**_make_output(
            answer="I am absolutely certain the answer is 42.",
            confidence=0.3,
        ))
        # Should auto-repair confidence downward
        assert result.output.confidence < 0.5

    def test_confident_language_high_confidence_ok(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONTRACT_ENABLED", True)
        from agent.contract import enforce_contract
        result = enforce_contract(**_make_output(
            answer="I am absolutely certain the answer is 42.",
            confidence=0.95,
        ))
        # High confidence + certain language = no violation on this rule
        cert_violations = [v for v in result.violations if "claims certainty" in v.rule]
        assert len(cert_violations) == 0


# ── Source extraction ─────────────────────────────────────────────────────────

class TestSourceExtraction:

    def test_extract_sources_from_ledger(self):
        from agent.contract import extract_sources
        from agent.grounding import EvidenceLedger
        ledger = EvidenceLedger()
        ledger.add("web_search", "France capital", "Paris is the capital of France")
        ledger.add("calculator", "1+1", "2")
        sources = extract_sources(ledger)
        assert len(sources) == 2
        assert any("web_search" in s for s in sources)
        assert any("Paris" in s for s in sources)

    def test_empty_ledger_returns_empty_list(self):
        from agent.contract import extract_sources
        from agent.grounding import EvidenceLedger
        assert extract_sources(EvidenceLedger()) == []


# ── Contract version ──────────────────────────────────────────────────────────

class TestContractVersion:

    def test_version_in_output(self, monkeypatch):
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONTRACT_ENABLED", True)
        from agent.contract import enforce_contract, CONTRACT_VERSION
        result = enforce_contract(**_make_output())
        assert result.output.contract_version == CONTRACT_VERSION

    def test_version_is_semver_like(self):
        from agent.contract import CONTRACT_VERSION
        parts = CONTRACT_VERSION.split(".")
        assert len(parts) >= 2
        assert all(p.isdigit() for p in parts)


# ── Config flag ───────────────────────────────────────────────────────────────

class TestContractConfig:

    def test_contract_enabled_flag_exists(self):
        from agent.config import CONTRACT_ENABLED
        assert isinstance(CONTRACT_ENABLED, bool)

    def test_contract_enabled_by_default(self):
        from agent.config import CONTRACT_ENABLED
        assert CONTRACT_ENABLED is True
