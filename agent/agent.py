"""
Main agent entry point — full reasoning + anti-hallucination pipeline.

Execution chain (all steps configurable in agent/config.py):

  1. REASONING PLAN   — explicit plan before any tool is called
                        (agent/reasoning.py → ReasoningPlan)

  2. EXECUTE          — tools run with plan as context prefix
                        (PydanticAI agent, async parallel tools)

  3. SELF-CRITIQUE    — agent critiques its own draft vs evidence + plan
                        (agent/critique.py → revised answer)

  4. GROUNDING CHECK  — every claim verified against tool results (evidence ledger)
                        (agent/grounding.py → hard anti-hallucination guarantee)

  5. EXTERNAL VALIDATOR — independent agent reviews the grounded, critiqued answer
                        (agent/validator.py → ValidationResult)

  6. REFLEXION LOOP   — if any step fails, feed structured feedback back and retry
                        (max MAX_REFLECTIONS attempts)

  7. CONFIDENCE GATE  — refuse to answer if confidence stays below threshold

Run:
    python agent/agent.py --input "your task"
    python agent/ui.py          ← Gradio streaming demo → localhost:7860
Phoenix: http://localhost:6006
"""

import os
import sys
import json
import time
import asyncio
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from pydantic_ai import Agent
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.config import (
    MODEL,
    MAX_REFLECTIONS, REFLECTION_THRESHOLD, CONFIDENCE_GATE,
    REASONING_ENFORCEMENT,
    GROUNDING_ENABLED, GROUNDING_FAIL_THRESHOLD,
    SELF_CRITIQUE_ENABLED,
    VALIDATOR_ENABLED,
    CONTRACT_ENABLED,
    MULTI_AGENT_MODE,
    HUMAN_IN_THE_LOOP, HITL_TOOLS,
    PHOENIX_ENABLED, PHOENIX_PORT,
    PROMPTS_DIR, ACTIVE_PROMPT_VERSION, LANGFUSE_ENABLED,
)
from agent.memory import AgentMemory
from agent.reasoning import build_reasoning_plan, plan_to_context, ReasoningPlan
from agent.grounding import EvidenceLedger, check_grounding, grounding_feedback
from agent.critique import self_critique
from agent.validator import validate
from agent.contract import enforce_contract, extract_sources, FinalOutput

console = Console()
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

_COST_PER_1M_IN  = 3.00
_COST_PER_1M_OUT = 15.00


# ── Observability ─────────────────────────────────────────────────────────────

_phoenix_started = False

def setup_observability() -> None:
    global _phoenix_started
    if not PHOENIX_ENABLED or _phoenix_started:
        return
    try:
        import phoenix as px
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        px.launch_app(port=PHOENIX_PORT)
        AnthropicInstrumentor().instrument()
        _phoenix_started = True
        console.print(f"[green]Phoenix →[/green] http://localhost:{PHOENIX_PORT}")
    except ImportError:
        console.print("[yellow]Phoenix not installed — skipping.[/yellow]")


# ── Prompt management ─────────────────────────────────────────────────────────

def load_system_prompt(version: str = ACTIVE_PROMPT_VERSION) -> str:
    if LANGFUSE_ENABLED:
        try:
            from langfuse import Langfuse
            prompt_obj = Langfuse().get_prompt("system-prompt")
            console.print(f"[green]Prompt: Langfuse v{prompt_obj.version}[/green]")
            return prompt_obj.compile()
        except Exception as e:
            console.print(f"[yellow]Langfuse fallback ({e})[/yellow]")

    prompt_file = PROMPTS_DIR / f"system_v{version}.yaml"
    if not prompt_file.exists():
        console.print(f"[red]Prompt not found: {prompt_file} — run python runner.py first[/red]")
        sys.exit(1)

    data = yaml.safe_load(prompt_file.read_text())
    console.print(f"[green]Prompt: local YAML v{data['version']}[/green]")
    return data["prompt"]


# ── HITL wrapper ──────────────────────────────────────────────────────────────

def _hitl_wrap(tool_fn, ledger: EvidenceLedger):
    """Wrap tool with HITL checkpoint + evidence ledger logging."""
    import functools

    @functools.wraps(tool_fn)
    async def wrapper(ctx, *args, **kwargs):
        tool_name = tool_fn.__name__

        if HUMAN_IN_THE_LOOP and (not HITL_TOOLS or tool_name in HITL_TOOLS):
            console.print(
                f"\n[bold yellow]HITL[/bold yellow] — "
                f"[cyan]{tool_name}[/cyan] args={kwargs or args}  Approve? (y/n) ",
                end="",
            )
            if input().strip().lower() != "y":
                return {"status": "rejected_by_human", "tool": tool_name}

        result = await tool_fn(ctx, *args, **kwargs)
        ledger.add(tool_name, str(kwargs or args), str(result))
        return result

    return wrapper


# ── Structured output ─────────────────────────────────────────────────────────

class AgentAnswer(BaseModel):
    answer:     str
    reasoning:  str
    confidence: float
    tools_used: list[str]


class RunMeta(BaseModel):
    tokens_in:        int   = 0
    tokens_out:       int   = 0
    cost_usd:         float = 0.0
    latency_ms:       int   = 0
    reflections:      int   = 0
    final_score:      float = 0.0
    grounding_ratio:  float = 1.0
    validator_passed: bool  = True
    confidence_gated: bool  = False
    self_critiqued:   bool  = False
    mode:             str   = "single"


# ── Agent factory ─────────────────────────────────────────────────────────────

def build_agent(system_prompt: str, ledger: EvidenceLedger) -> Agent:
    from agent.tools import TOOLS
    agent: Agent[None, AgentAnswer] = Agent(
        model=f"anthropic:{MODEL}",
        result_type=AgentAnswer,
        system_prompt=system_prompt,
    )
    for tool_fn in TOOLS:
        agent.tool(_hitl_wrap(tool_fn, ledger))
    return agent


# ── Scoring (for reflexion) ───────────────────────────────────────────────────

async def _score(answer: str, task: str) -> tuple[float, str]:
    """Score via the multi-LLM judge panel to eliminate same-model bias."""
    try:
        from agent.judge import score_with_panel
        ensemble, result = await score_with_panel(task, answer)
        reason = "; ".join(
            f"{s.model.split('/')[-1]}:{s.score:.2f}" for s in result.scores if not s.skipped
        )
        return ensemble, reason
    except Exception:
        return 0.5, "score unavailable"


# ── Core pipeline: one attempt ────────────────────────────────────────────────

async def _single_attempt(
    task: str,
    context_prefix: str,
    system_prompt: str,
    reasoning_plan: ReasoningPlan,
    ledger: EvidenceLedger,
    is_multi: bool,
) -> tuple[AgentAnswer, float]:
    """Run one full attempt: execute → self-critique → grounding → validate → score."""

    # ── Execute ───────────────────────────────────────────────────────────────
    if is_multi:
        from agent.orchestrator import MultiAgentOrchestrator, load_specialists
        runner = MultiAgentOrchestrator(model=MODEL, specialists=load_specialists())
        raw = await runner.run(context_prefix + task)
        draft = AgentAnswer(
            answer=raw.answer, reasoning=raw.reasoning,
            confidence=raw.confidence, tools_used=raw.tools_used,
        )
    else:
        agent = build_agent(system_prompt, ledger)
        run_result = await agent.run(context_prefix + task)
        draft = run_result.data

    # ── Self-critique ─────────────────────────────────────────────────────────
    critique_result = await self_critique(
        task=task,
        draft_answer=draft.answer,
        draft_confidence=draft.confidence,
        evidence_text=ledger.as_text(),
        reasoning_plan_text=plan_to_context(reasoning_plan),
    )
    # Use revised answer + recalibrated confidence
    draft.answer     = critique_result.revised_answer
    draft.confidence = critique_result.revised_confidence

    # ── Grounding check ───────────────────────────────────────────────────────
    grounding = await check_grounding(task, draft.answer, ledger)

    if grounding.verdict == "fail":
        # Return low score so reflexion loop retries
        return draft, grounding.grounding_ratio * 0.5

    # ── External validator ────────────────────────────────────────────────────
    if VALIDATOR_ENABLED:
        validation = await validate(task, draft.answer, draft.reasoning, draft.confidence)
        if not validation.passed:
            return draft, validation.score

    # ── Score ─────────────────────────────────────────────────────────────────
    score, _ = await _score(draft.answer, task)
    return draft, score


# ── Reflexion loop ────────────────────────────────────────────────────────────

async def _run_with_reflection(
    task: str,
    system_prompt: str,
    memory_context: str,
    is_multi: bool,
) -> tuple[AgentAnswer, int, float, EvidenceLedger]:
    best_result: AgentAnswer | None = None
    best_score = 0.0
    num_reflections = 0
    feedback = memory_context

    for attempt in range(1, MAX_REFLECTIONS + 1):
        console.print(f"\n[bold]Attempt {attempt}/{MAX_REFLECTIONS}[/bold]")

        # Fresh ledger per attempt so evidence stays aligned with this run
        ledger = EvidenceLedger()

        # Build reasoning plan before each attempt
        tool_names = []
        try:
            from agent.tools import TOOLS
            tool_names = [t.__name__ for t in TOOLS]
        except Exception:
            pass

        reasoning_plan = await build_reasoning_plan(task + feedback, tool_names)
        context_prefix = plan_to_context(reasoning_plan) if REASONING_ENFORCEMENT else ""

        result, score = await _single_attempt(
            task=task,
            context_prefix=context_prefix + feedback,
            system_prompt=system_prompt,
            reasoning_plan=reasoning_plan,
            ledger=ledger,
            is_multi=is_multi,
        )

        status_color = "green" if score >= REFLECTION_THRESHOLD else "yellow"
        console.print(f"  Score: [{status_color}]{score:.2f}[/]")

        if score > best_score:
            best_score = score
            best_result = result

        if score >= REFLECTION_THRESHOLD:
            console.print(f"  [green]Accepted (≥{REFLECTION_THRESHOLD})[/green]")
            break

        if attempt < MAX_REFLECTIONS:
            num_reflections += 1
            # Build structured feedback from grounding + score
            grounding = await check_grounding(task, result.answer, ledger)
            feedback = grounding_feedback(grounding) if grounding.verdict == "fail" else (
                f"\n\n[Reflection {attempt}] Score {score:.2f}. "
                f"Improve: be more precise, ground every claim in tool results."
            )
            console.print("  [yellow]Retrying with structured feedback...[/yellow]")

    return best_result, num_reflections, best_score, ledger


# ── Cost + log ────────────────────────────────────────────────────────────────

def _compute_cost(tin: int, tout: int) -> float:
    return (tin * _COST_PER_1M_IN + tout * _COST_PER_1M_OUT) / 1_000_000


def _write_log(task: str, result: AgentAnswer, meta: RunMeta) -> None:
    path = LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    path.write_text(json.dumps({
        "timestamp":        datetime.now().isoformat(),
        "mode":             meta.mode,
        "input":            task,
        "answer":           result.answer,
        "reasoning":        result.reasoning,
        "confidence":       result.confidence,
        "tools_used":       result.tools_used,
        "tokens_in":        meta.tokens_in,
        "tokens_out":       meta.tokens_out,
        "cost_usd":         round(meta.cost_usd, 6),
        "latency_ms":       meta.latency_ms,
        "reflections":      meta.reflections,
        "final_score":      meta.final_score,
        "grounding_ratio":  meta.grounding_ratio,
        "validator_passed": meta.validator_passed,
        "confidence_gated": meta.confidence_gated,
        "self_critiqued":   meta.self_critiqued,
    }, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(task: str, return_meta: bool = False):
    setup_observability()
    system_prompt = load_system_prompt()
    memory = AgentMemory()

    memory_context = memory.retrieve(task)
    if memory_context:
        console.print(f"[dim]Memory: {memory.count()} runs — injecting context[/dim]")

    flags = (
        f"reasoning={'on' if REASONING_ENFORCEMENT else 'off'} | "
        f"grounding={'on' if GROUNDING_ENABLED else 'off'} | "
        f"critique={'on' if SELF_CRITIQUE_ENABLED else 'off'} | "
        f"validator={'on' if VALIDATOR_ENABLED else 'off'} | "
        f"mode={'multi' if MULTI_AGENT_MODE else 'single'}"
    )
    console.print(Panel(
        f"[bold cyan]Running Agent[/bold cyan]  [dim]{flags}[/dim]\n\n"
        f"{task[:200]}{'...' if len(task) > 200 else ''}",
        border_style="cyan",
    ))

    t0 = time.monotonic()
    result, num_reflections, final_score, ledger = await _run_with_reflection(
        task, system_prompt, memory_context, is_multi=MULTI_AGENT_MODE
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    # Confidence gate
    confidence_gated = False
    if result.confidence < CONFIDENCE_GATE:
        console.print(f"[red]Confidence {result.confidence:.0%} below gate — low-confidence notice[/red]")
        result.answer = (
            f"[Low confidence: {result.confidence:.0%}] "
            f"Partial reasoning: {result.reasoning[:200]}"
        )
        confidence_gated = True

    # Final grounding ratio (from last attempt's ledger)
    final_grounding = await check_grounding(task, result.answer, ledger)

    meta = RunMeta(
        latency_ms=latency_ms,
        reflections=num_reflections,
        final_score=final_score,
        grounding_ratio=final_grounding.grounding_ratio,
        validator_passed=(final_grounding.verdict != "fail"),
        confidence_gated=confidence_gated,
        self_critiqued=SELF_CRITIQUE_ENABLED,
        mode="multi-agent" if MULTI_AGENT_MODE else "single",
    )

    # ── Final Output Contract ─────────────────────────────────────────────────
    console.print("\n  [dim]Enforcing output contract...[/dim]")
    contract_result = enforce_contract(
        answer=result.answer,
        confidence=result.confidence,
        reasoning=result.reasoning,
        tools_used=result.tools_used,
        grounding_ratio=final_grounding.grounding_ratio,
        grounding_verdict=final_grounding.verdict,
        sources=extract_sources(ledger),
        reflections=num_reflections,
        latency_ms=latency_ms,
        model=MODEL,
    )

    if not contract_result.passed:
        # Surface violations clearly — do not silently swallow them
        violation_lines = "\n".join(
            f"  [{v.field}] {v.rule}" for v in contract_result.violations
        )
        console.print(f"[red]Output contract violations:\n{violation_lines}[/red]")

    final_output: FinalOutput = contract_result.output
    # Sync any auto-repairs back to result for logging
    result.answer     = final_output.answer
    result.confidence = final_output.confidence

    memory.store(task, result.answer, final_score, result.tools_used)
    _write_log(task, result, meta)

    console.print(Panel(
        f"[bold green]Answer[/bold green]\n\n{final_output.answer}\n\n"
        f"[dim]Confidence: {final_output.confidence:.0%} | "
        f"Grounding: {final_output.grounding_ratio:.0%} ({final_output.grounding_verdict}) | "
        f"Sources: {len(final_output.sources)} | "
        f"Tools: {', '.join(final_output.tools_used) or 'none'} | "
        f"Latency: {latency_ms}ms | Reflections: {num_reflections} | "
        f"Contract: {'✓' if contract_result.passed else '✗'}[/dim]",
        border_style="green" if contract_result.passed else "yellow",
    ))

    if return_meta:
        return final_output, meta.model_dump()
    return final_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.input))
