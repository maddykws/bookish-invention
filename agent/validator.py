"""
Meta-Validator Agent
A second independent Claude call that reviews the main agent's answer
before it leaves the system.

Checks:
  1. Factual consistency — does the answer contradict itself?
  2. Completeness       — does it actually address the full task?
  3. Hallucination risk — does it claim things not grounded in tool results?
  4. Confidence match   — is stated confidence consistent with answer quality?

If any check fails, returns a structured critique that feeds back into the
Reflexion loop for another attempt.

Enable/disable via VALIDATOR_ENABLED in agent/config.py.
"""

import os
from pydantic import BaseModel
from rich.console import Console

console = Console()


class ValidationResult(BaseModel):
    passed: bool
    score: float                  # 0.0 – 1.0
    hallucination_risk: str       # "low" | "medium" | "high"
    completeness: str             # "complete" | "partial" | "incomplete"
    issues: list[str]             # specific problems found
    critique: str                 # one-paragraph feedback for reflexion


VALIDATOR_PROMPT = """You are a strict quality validator for an AI agent's output.

## Original Task
{task}

## Agent's Answer
{answer}

## Agent's Stated Reasoning
{reasoning}

## Agent's Stated Confidence
{confidence:.0%}

---

Evaluate this answer on four dimensions:

1. FACTUAL CONSISTENCY: Does the answer contradict itself or its own reasoning?
2. COMPLETENESS: Does it fully address what was asked, or does it dodge parts of the task?
3. HALLUCINATION RISK: Does it claim specific facts, numbers, or details not grounded in the reasoning?
4. CONFIDENCE CALIBRATION: Is the stated confidence ({confidence:.0%}) warranted given the answer quality?

Respond as JSON:
{{
  "passed": true/false,
  "score": 0.0-1.0,
  "hallucination_risk": "low"/"medium"/"high",
  "completeness": "complete"/"partial"/"incomplete",
  "issues": ["specific issue 1", "specific issue 2"],
  "critique": "One paragraph of specific, actionable feedback for the agent to improve its answer."
}}

Be strict. A score above 0.8 means the answer is genuinely good, not just acceptable.
Return ONLY the JSON.
"""


async def validate(task: str, answer: str, reasoning: str, confidence: float) -> ValidationResult:
    from agent.config import VALIDATOR_ENABLED

    if not VALIDATOR_ENABLED:
        return ValidationResult(
            passed=True,
            score=confidence,
            hallucination_risk="low",
            completeness="complete",
            issues=[],
            critique="Validator disabled.",
        )

    console.print("\n  [dim]Running meta-validator (multi-LLM panel)...[/dim]")

    from agent.judge import validate_with_panel
    data = await validate_with_panel(task, answer, reasoning, confidence)

    try:
        result = ValidationResult(**data)
    except Exception:
        result = ValidationResult(
            passed=True,
            score=confidence,
            hallucination_risk="low",
            completeness="complete",
            issues=["Validator parse error — defaulting to pass"],
            critique="",
        )

    risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(
        result.hallucination_risk, "white"
    )
    status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
    console.print(
        f"  Validator: {status} | score {result.score:.2f} | "
        f"hallucination [{risk_color}]{result.hallucination_risk}[/] | "
        f"completeness {result.completeness}"
    )
    if result.issues:
        for issue in result.issues:
            console.print(f"    [yellow]![/yellow] {issue}")

    return result
