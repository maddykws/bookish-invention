"""
Reasoning Enforcement Module
Forces the agent to think explicitly before acting.

The agent must produce a structured ReasoningPlan before any tool is called:
  1. UNDERSTAND  — restate what is actually being asked + constraints
  2. KNOW        — what the agent already knows that is relevant
  3. GAPS        — what it does NOT know and must look up
  4. PLAN        — which tools to call, in what order, and why
  5. RISK        — what could go wrong, what would make the answer unreliable

This plan is injected into the main agent's context as a prefix, forcing
tool selection and reasoning to be deliberate rather than reflexive.

Anti-pattern prevented: agent jumps straight to tool calls without understanding
the task, producing a confident-sounding answer built on the wrong foundation.
"""

import os
from pydantic import BaseModel
from rich.console import Console

console = Console()


class ReasoningPlan(BaseModel):
    understand:  str          # exact restatement of the task + constraints
    already_know: list[str]   # facts already known without tools
    gaps:        list[str]    # specific things that must be looked up
    tool_plan:   list[str]    # ordered list: "call X to find Y because Z"
    risk:        str          # what could make the answer wrong or incomplete
    confidence_prior: float   # prior confidence before seeing tool results (0–1)


REASONING_PROMPT = """You are about to answer a task using a set of tools.
Before calling any tool, produce a structured reasoning plan.

Task: {task}

Available tools: {tool_names}

Return a JSON object with exactly these fields:
{{
  "understand": "Restate the task precisely in your own words. Include all constraints and what a complete answer must cover.",
  "already_know": ["fact you know without tools", "..."],
  "gaps": ["specific thing you must look up", "..."],
  "tool_plan": ["1. Call X to find Y because Z", "2. Call A to verify B", "..."],
  "risk": "One sentence: what would make this answer wrong or unreliable?",
  "confidence_prior": 0.0
}}

Be honest about gaps. If you're unsure whether a tool will answer a gap, say so in the plan.
Return ONLY the JSON.
"""


async def build_reasoning_plan(task: str, tool_names: list[str]) -> ReasoningPlan:
    import anthropic
    import json
    from agent.config import MODEL, REASONING_ENFORCEMENT

    if not REASONING_ENFORCEMENT:
        return ReasoningPlan(
            understand=task,
            already_know=[],
            gaps=["reasoning enforcement disabled"],
            tool_plan=["proceed directly"],
            risk="none assessed",
            confidence_prior=0.5,
        )

    console.print("\n  [dim]Building reasoning plan...[/dim]")

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": REASONING_PROMPT.format(
            task=task,
            tool_names=", ".join(tool_names) if tool_names else "none",
        )}],
    )

    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip().rstrip("```")

    try:
        plan = ReasoningPlan(**json.loads(raw))
    except Exception:
        plan = ReasoningPlan(
            understand=task,
            already_know=[],
            gaps=["plan parse failed"],
            tool_plan=["proceed directly"],
            risk="reasoning plan could not be parsed",
            confidence_prior=0.4,
        )

    console.print(f"  [dim]Gaps to fill: {len(plan.gaps)} | "
                  f"Tool calls planned: {len(plan.tool_plan)} | "
                  f"Prior confidence: {plan.confidence_prior:.0%}[/dim]")

    return plan


def plan_to_context(plan: ReasoningPlan) -> str:
    """Serialize reasoning plan as a context prefix injected into the agent's task."""
    gaps = "\n".join(f"  - {g}" for g in plan.gaps)
    tool_steps = "\n".join(f"  {s}" for s in plan.tool_plan)
    known = "\n".join(f"  - {k}" for k in plan.already_know) or "  - nothing yet"

    return (
        f"[Reasoning Plan]\n"
        f"Task understood as: {plan.understand}\n"
        f"Already known:\n{known}\n"
        f"Gaps to fill:\n{gaps}\n"
        f"Tool execution plan:\n{tool_steps}\n"
        f"Risk: {plan.risk}\n"
        f"Prior confidence: {plan.confidence_prior:.0%}\n"
        f"[End Reasoning Plan]\n\n"
        f"Now execute the plan above. Follow the tool order. "
        f"Do not claim anything not returned by a tool.\n\n"
    )
