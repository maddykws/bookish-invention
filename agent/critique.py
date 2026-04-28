"""
Self-Critique Module
The agent critiques its own draft answer before it leaves the system.

This is NOT the external validator (agent/validator.py).
This is the SAME agent, looking at its own draft against:
  - the original task
  - the evidence it collected
  - its own reasoning plan

It asks: "What did I get wrong? What did I miss? What am I claiming
I cannot prove? Is there a simpler, more direct answer?"

Then it produces a revised answer incorporating the critique.

Chain position:
  reasoning → execute → [self-critique] → grounding check → external validator

Anti-pattern prevented: agent locks in on a first-pass answer that is
technically not wrong but misses the core of what was asked, or is
more verbose and less precise than it could be.
"""

import os
from pydantic import BaseModel
from rich.console import Console

console = Console()


class SelfCritiqueResult(BaseModel):
    original_answer:    str
    issues_found:       list[str]   # specific problems with the draft
    missing_elements:   list[str]   # things the task asked for that are absent
    unsupported_claims: list[str]   # claims the agent made without tool backing
    revised_answer:     str         # improved answer after critique
    critique_summary:   str         # one-sentence summary of what changed
    revised_confidence: float       # recalibrated confidence after critique


SELF_CRITIQUE_PROMPT = """You are reviewing your own draft answer to a task.
Be ruthless. Your goal is to produce the best possible answer, not to defend your first attempt.

## Original Task
{task}

## Evidence You Collected (tool results)
{evidence}

## Your Reasoning Plan
{reasoning_plan}

## Your Draft Answer
{draft_answer}

Critique your draft on these dimensions:
1. COMPLETENESS: Does it answer every part of the task? What is missing?
2. ACCURACY: Are there claims not supported by the evidence? List them specifically.
3. CLARITY: Is the answer direct and precise, or verbose and vague?
4. CONFIDENCE CALIBRATION: Is your stated confidence ({draft_confidence:.0%}) warranted?

Then write a REVISED answer that:
- Fixes every issue you identified
- Only asserts what the evidence supports
- Is direct — no padding, no hedging beyond genuine uncertainty
- Answers exactly what was asked

Return JSON:
{{
  "original_answer": "{draft_answer_escaped}",
  "issues_found": ["specific issue 1", "..."],
  "missing_elements": ["thing the task asked for that is absent", "..."],
  "unsupported_claims": ["claim not backed by evidence", "..."],
  "revised_answer": "your improved answer here",
  "critique_summary": "one sentence: what changed and why",
  "revised_confidence": 0.0
}}

Return ONLY the JSON.
"""


async def self_critique(
    task: str,
    draft_answer: str,
    draft_confidence: float,
    evidence_text: str,
    reasoning_plan_text: str,
) -> SelfCritiqueResult:
    import anthropic
    import json
    from agent.config import MODEL, SELF_CRITIQUE_ENABLED

    if not SELF_CRITIQUE_ENABLED:
        return SelfCritiqueResult(
            original_answer=draft_answer,
            issues_found=[],
            missing_elements=[],
            unsupported_claims=[],
            revised_answer=draft_answer,
            critique_summary="Self-critique disabled.",
            revised_confidence=draft_confidence,
        )

    console.print("\n  [dim]Running self-critique...[/dim]")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": SELF_CRITIQUE_PROMPT.format(
            task=task,
            evidence=evidence_text,
            reasoning_plan=reasoning_plan_text,
            draft_answer=draft_answer,
            draft_confidence=draft_confidence,
            draft_answer_escaped=draft_answer.replace('"', '\\"')[:300],
        )}],
    )

    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("```")

    try:
        data = json.loads(raw)
        result = SelfCritiqueResult(**data)
    except Exception:
        result = SelfCritiqueResult(
            original_answer=draft_answer,
            issues_found=["Parse error in self-critique"],
            missing_elements=[],
            unsupported_claims=[],
            revised_answer=draft_answer,
            critique_summary="Self-critique parse failed — using original.",
            revised_confidence=draft_confidence,
        )

    improved = result.revised_confidence > draft_confidence
    delta = result.revised_confidence - draft_confidence
    delta_str = f"+{delta:.0%}" if improved else f"{delta:.0%}"

    console.print(
        f"  Self-critique: {len(result.issues_found)} issue(s) | "
        f"{len(result.missing_elements)} missing element(s) | "
        f"confidence {draft_confidence:.0%} → {result.revised_confidence:.0%} "
        f"[{'green' if improved else 'yellow'}]({delta_str})[/]"
    )
    console.print(f"  [dim]{result.critique_summary}[/dim]")

    return result
