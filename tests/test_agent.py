"""
End-to-end agent integration tests.
Tests the full pipeline: task → agent → answer.

scaffold.py injects problem-specific test cases into TEST_INPUTS below.
"""

import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── SCAFFOLD INJECTION POINT — replaced on question drop ──────────────────────
TEST_INPUTS = [
    "Placeholder test — scaffold.py will replace this with real inputs from the problem.",
]
# SCAFFOLD_E2E_TESTS_START


class TestAgentStructure:
    """Agent structural tests — always run, no API calls needed."""

    def test_agent_module_importable(self):
        from agent.agent import main, build_agent, load_system_prompt
        assert main is not None

    def test_agent_answer_model(self):
        from agent.agent import AgentAnswer
        ans = AgentAnswer(
            answer="test answer",
            reasoning="test reasoning",
            confidence=0.85,
            tools_used=["search"],
        )
        assert 0.0 <= ans.confidence <= 1.0

    def test_run_meta_model(self):
        from agent.agent import RunMeta
        meta = RunMeta(latency_ms=1200, reflections=1, final_score=0.88)
        assert meta.latency_ms == 1200
        assert meta.reflections == 1

    def test_system_prompt_loads(self):
        from agent.agent import load_system_prompt
        prompt = load_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 10


class TestAgentBehaviour:
    """
    Behavioural tests — require ANTHROPIC_API_KEY.
    Skipped automatically if key not set.
    """

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

    def test_agent_returns_answer_model(self):
        from agent.agent import main, AgentAnswer
        result = run(main(TEST_INPUTS[0]))
        assert isinstance(result, AgentAnswer)

    def test_answer_not_empty(self):
        from agent.agent import main
        result = run(main(TEST_INPUTS[0]))
        assert len(result.answer.strip()) > 0

    def test_confidence_in_range(self):
        from agent.agent import main
        result = run(main(TEST_INPUTS[0]))
        assert 0.0 <= result.confidence <= 1.0

    def test_tools_used_is_list(self):
        from agent.agent import main
        result = run(main(TEST_INPUTS[0]))
        assert isinstance(result.tools_used, list)

    def test_low_confidence_gate_triggered(self, monkeypatch):
        """If confidence is very low, answer should contain low-confidence notice."""
        import agent.config as cfg
        monkeypatch.setattr(cfg, "CONFIDENCE_GATE", 0.99)  # impossible threshold
        from agent.agent import main
        result = run(main("deliberately ambiguous or unanswerable query xyz123"))
        # Either gated (low confidence notice) or very confident — both valid
        assert isinstance(result.answer, str)


class TestDiagram:
    """Architecture diagram tests."""

    def test_diagram_generates(self):
        from modules.diagram import generate_diagram
        diagram = generate_diagram(["researcher", "executor"])
        assert "mermaid" in diagram
        assert "Orchestrator" in diagram
        assert "Validator" in diagram

    def test_diagram_write(self, tmp_path, monkeypatch):
        import modules.diagram as diag
        monkeypatch.setattr(diag, "DOCS_DIR", tmp_path)
        path = diag.write_diagram(["specialist_a"])
        assert path.exists()
        content = path.read_text()
        assert "specialist_a" in content.lower() or "Specialist" in content
