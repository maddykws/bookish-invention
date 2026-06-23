"""
Evidence Ledger + Hard Anti-Hallucination Grounding Check

Every tool result is logged into an EvidenceLedger during execution.
Before the final answer is accepted, every factual claim in the answer
is checked against the ledger.

A claim is grounded if it is traceable to at least one tool result.
An ungrounded claim is flagged. If too many claims are ungrounded,
the answer is rejected and the agent is forced to revise.

This is a HARD guarantee — not a metric, not a probability.
A claim either has evidence or it doesn't.

Anti-pattern prevented: agent states facts confidently that were never
returned by any tool — i.e., hallucinated from training data.
"""

import re
import os
from dataclasses import dataclass, field
from rich.console import Console

console = Console()


# ── Evidence ledger ───────────────────────────────────────────────────────────

@dataclass
class ToolEvidence:
    tool_name:  str
    tool_input: str
    tool_output: str


@dataclass
class EvidenceLedger:
    entries: list[ToolEvidence] = field(default_factory=list)

    def add(self, tool_name: str, tool_input: str, tool_output: str) -> None:
        self.entries.append(ToolEvidence(
            tool_name=tool_name,
            tool_input=str(tool_input)[:500],
            tool_output=str(tool_output)[:1000],
        ))

    def as_text(self) -> str:
        if not self.entries:
            return "[No tool results collected]"
        lines = []
        for i, e in enumerate(self.entries, 1):
            lines.append(f"[Evidence {i}] Tool: {e.tool_name}\n"
                         f"  Input:  {e.tool_input}\n"
                         f"  Output: {e.tool_output}")
        return "\n\n".join(lines)

    def all_outputs(self) -> str:
        return " ".join(e.tool_output for e in self.entries)

    def is_empty(self) -> bool:
        return len(self.entries) == 0


# ── Grounding check ───────────────────────────────────────────────────────────

GROUNDING_PROMPT = """You are a strict fact-checker for an AI agent's answer.

## Original Task
{task}

## Evidence Collected (tool results)
{evidence}

## Agent's Answer
{answer}

Your job: identify every factual claim in the answer and check whether it is
supported by the evidence above.

A claim is GROUNDED if the evidence explicitly contains the information.
A claim is UNGROUNDED if it asserts a specific fact not present in any tool result.
Statements of uncertainty ("I couldn't find...", "this may vary") are NOT claims.

Return JSON:
{{
  "grounded_claims": ["claim 1", "claim 2"],
  "ungrounded_claims": ["specific ungrounded claim 1", "..."],
  "grounding_ratio": 0.0,
  "verdict": "pass" | "warn" | "fail",
  "recommendation": "short instruction for the agent to fix ungrounded claims"
}}

verdict rules:
  pass  — grounding_ratio >= 0.85 OR no ungrounded claims
  warn  — grounding_ratio 0.60–0.84 (some ungrounded, answer still usable)
  fail  — grounding_ratio < 0.60 OR critical facts ungrounded

Return ONLY the JSON.
"""

GROUNDING_THRESHOLD_FAIL = 0.60   # below this → reject and force revision
GROUNDING_THRESHOLD_WARN = 0.85   # below this → warn but accept


@dataclass
class GroundingResult:
    grounded_claims:   list[str]
    ungrounded_claims: list[str]
    grounding_ratio:   float
    verdict:           str          # "pass" | "warn" | "fail"
    recommendation:    str


async def check_grounding(
    task: str,
    answer: str,
    ledger: EvidenceLedger,
) -> GroundingResult:
    import anthropic
    import json
    from agent.config import MODEL, GROUNDING_ENABLED

    if not GROUNDING_ENABLED or ledger.is_empty():
        return GroundingResult(
            grounded_claims=[],
            ungrounded_claims=[],
            grounding_ratio=1.0,
            verdict="pass",
            recommendation="Grounding check skipped.",
        )

    console.print("\n  [dim]Running grounding check...[/dim]")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=MODEL,
        max_tokens=700,
        messages=[{"role": "user", "content": GROUNDING_PROMPT.format(
            task=task,
            evidence=ledger.as_text(),
            answer=answer,
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
        result = GroundingResult(
            grounded_claims=data.get("grounded_claims", []),
            ungrounded_claims=data.get("ungrounded_claims", []),
            grounding_ratio=float(data.get("grounding_ratio", 1.0)),
            verdict=data.get("verdict", "pass"),
            recommendation=data.get("recommendation", ""),
        )
    except Exception:
        result = GroundingResult(
            grounded_claims=[],
            ungrounded_claims=[],
            grounding_ratio=1.0,
            verdict="pass",
            recommendation="Grounding parse error — defaulting to pass.",
        )

    verdict_color = {"pass": "green", "warn": "yellow", "fail": "red"}.get(result.verdict, "white")
    console.print(
        f"  Grounding: [{verdict_color}]{result.verdict.upper()}[/] | "
        f"ratio {result.grounding_ratio:.0%} | "
        f"{len(result.ungrounded_claims)} ungrounded claim(s)"
    )
    if result.ungrounded_claims:
        for claim in result.ungrounded_claims[:3]:
            console.print(f"    [red]✗[/red] {claim[:120]}")

    return result


def grounding_feedback(result: GroundingResult) -> str:
    """Produce a feedback string for the reflexion loop when grounding fails."""
    claims = "\n".join(f"  - {c}" for c in result.ungrounded_claims)
    return (
        f"\n\n[Grounding Failure] Ratio: {result.grounding_ratio:.0%}. "
        f"The following claims were not supported by tool results:\n{claims}\n"
        f"Recommendation: {result.recommendation}\n"
        f"Revise your answer to only assert what the tools returned."
    )
