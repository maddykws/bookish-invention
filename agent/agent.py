"""
Main agent entry point.
Full stack: PydanticAI + Arize Phoenix + Langfuse + Reflexion +
            ChromaDB memory + Meta-validator + Human-in-the-loop +
            Multi-agent orchestrator + JSON logs + Cost tracking

Modes (set in agent/config.py):
  MULTI_AGENT_MODE = True   → orchestrator routes to specialist agents
  MULTI_AGENT_MODE = False  → single flat agent (faster for simple tasks)
  VALIDATOR_ENABLED = True  → meta-validator checks every answer
  HUMAN_IN_THE_LOOP = True  → pauses before high-stakes tool calls

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
    MODEL, MAX_REFLECTIONS, REFLECTION_THRESHOLD, CONFIDENCE_GATE,
    VALIDATOR_ENABLED, MULTI_AGENT_MODE,
    HUMAN_IN_THE_LOOP, HITL_TOOLS,
    PHOENIX_ENABLED, PHOENIX_PORT,
    PROMPTS_DIR, ACTIVE_PROMPT_VERSION, LANGFUSE_ENABLED,
)
from agent.memory import AgentMemory
from agent.validator import validate

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
        console.print("[yellow]Phoenix not installed — skipping. pip install arize-phoenix[/yellow]")


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


# ── Human-in-the-loop wrapper ─────────────────────────────────────────────────

def _hitl_wrap(tool_fn):
    """Wrap a tool function with a human approval checkpoint."""
    import functools

    @functools.wraps(tool_fn)
    async def wrapper(ctx, *args, **kwargs):
        tool_name = tool_fn.__name__
        needs_approval = HUMAN_IN_THE_LOOP and (
            not HITL_TOOLS or tool_name in HITL_TOOLS
        )

        if needs_approval:
            console.print(
                f"\n[bold yellow]HITL Checkpoint[/bold yellow] — "
                f"Agent wants to call [cyan]{tool_name}[/cyan]\n"
                f"Args: {kwargs or args}\n"
                "Approve? [bold](y/n)[/bold] ",
                end="",
            )
            answer = input().strip().lower()
            if answer != "y":
                return {"status": "rejected_by_human", "tool": tool_name}

        return await tool_fn(ctx, *args, **kwargs)

    return wrapper


# ── Structured output ─────────────────────────────────────────────────────────

class AgentAnswer(BaseModel):
    answer: str
    reasoning: str
    confidence: float
    tools_used: list[str]


class RunMeta(BaseModel):
    tokens_in:        int   = 0
    tokens_out:       int   = 0
    cost_usd:         float = 0.0
    latency_ms:       int   = 0
    reflections:      int   = 0
    final_score:      float = 0.0
    validator_passed: bool  = True
    confidence_gated: bool  = False
    mode:             str   = "single"   # "single" | "multi-agent"


# ── Single-agent builder ──────────────────────────────────────────────────────

def build_agent(system_prompt: str) -> Agent:
    from agent.tools import TOOLS
    agent: Agent[None, AgentAnswer] = Agent(
        model=f"anthropic:{MODEL}",
        result_type=AgentAnswer,
        system_prompt=system_prompt,
    )
    for tool_fn in TOOLS:
        agent.tool(_hitl_wrap(tool_fn) if HUMAN_IN_THE_LOOP else tool_fn)
    return agent


# ── Scoring ───────────────────────────────────────────────────────────────────

async def _score(result: AgentAnswer, task: str) -> tuple[float, str]:
    try:
        import anthropic as anth
        client = anth.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": (
                f"Task: {task}\nAnswer: {result.answer}\n\n"
                "Rate 0.0–1.0. Reply ONLY: <score>|<reason>. E.g. 0.85|Correct and complete."
            )}],
        )
        parts = resp.content[0].text.strip().split("|", 1)
        return float(parts[0].strip()), parts[1].strip() if len(parts) > 1 else ""
    except Exception:
        return result.confidence, "Derived from confidence"


# ── Reflexion loop ────────────────────────────────────────────────────────────

async def _run_with_reflection(
    runner,           # Agent or MultiAgentOrchestrator
    task: str,
    memory_context: str,
    is_multi: bool,
) -> tuple[AgentAnswer, int, float]:
    feedback = memory_context
    best_result: AgentAnswer | None = None
    best_score = 0.0
    num_reflections = 0

    for attempt in range(1, MAX_REFLECTIONS + 1):
        console.print(f"\n[bold]Attempt {attempt}/{MAX_REFLECTIONS}[/bold]")

        if is_multi:
            raw = await runner.run(task + feedback)
            # Convert AggregatedAnswer to AgentAnswer
            result = AgentAnswer(
                answer=raw.answer,
                reasoning=raw.reasoning,
                confidence=raw.confidence,
                tools_used=raw.tools_used,
            )
        else:
            run_result = await runner.run(task + feedback)
            result = run_result.data

        # Meta-validator check
        if VALIDATOR_ENABLED:
            validation = await validate(task, result.answer, result.reasoning, result.confidence)
            if not validation.passed and attempt < MAX_REFLECTIONS:
                score = validation.score
                reason = validation.critique
                console.print(f"  Score: [yellow]{score:.2f}[/] (validator failed)")
                if score > best_score:
                    best_score = score
                    best_result = result
                num_reflections += 1
                feedback = (
                    f"\n\n[Reflection {attempt}] Validator score {score:.2f}/1.0. "
                    f"Issues: {'; '.join(validation.issues)}. "
                    f"Feedback: {validation.critique}"
                )
                console.print("  [yellow]Validator failed — reflecting...[/yellow]")
                continue
            score = validation.score
            reason = "Validator passed"
        else:
            score, reason = await _score(result, task)

        status_color = "green" if score >= REFLECTION_THRESHOLD else "yellow"
        console.print(f"  Score: [{status_color}]{score:.2f}[/] — {reason}")

        if score > best_score:
            best_score = score
            best_result = result

        if score >= REFLECTION_THRESHOLD:
            console.print(f"  [green]Accepted (≥{REFLECTION_THRESHOLD})[/green]")
            break

        if attempt < MAX_REFLECTIONS:
            num_reflections += 1
            feedback = (
                f"\n\n[Reflection {attempt}] Score {score:.2f}. "
                f"Issue: {reason}. Be more precise and complete."
            )
            console.print("  [yellow]Reflecting and retrying...[/yellow]")

    return best_result, num_reflections, best_score


# ── Cost ──────────────────────────────────────────────────────────────────────

def _compute_cost(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * _COST_PER_1M_IN + tokens_out * _COST_PER_1M_OUT) / 1_000_000


# ── JSON log ──────────────────────────────────────────────────────────────────

def _write_log(task: str, result: AgentAnswer, meta: RunMeta) -> None:
    log_path = LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    log_path.write_text(json.dumps({
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
        "validator_passed": meta.validator_passed,
        "confidence_gated": meta.confidence_gated,
    }, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(task: str, return_meta: bool = False):
    setup_observability()
    system_prompt = load_system_prompt()
    memory = AgentMemory()

    memory_context = memory.retrieve(task)
    if memory_context:
        console.print(f"[dim]Memory: {memory.count()} stored runs — injecting context[/dim]")

    console.print(Panel(
        f"[bold cyan]Running Agent[/bold cyan]  "
        f"[dim]mode={'multi-agent' if MULTI_AGENT_MODE else 'single'} | "
        f"validator={'on' if VALIDATOR_ENABLED else 'off'} | "
        f"HITL={'on' if HUMAN_IN_THE_LOOP else 'off'}[/dim]\n\n"
        f"{task[:200]}{'...' if len(task) > 200 else ''}",
        border_style="cyan",
    ))

    t0 = time.monotonic()

    if MULTI_AGENT_MODE:
        from agent.orchestrator import MultiAgentOrchestrator, load_specialists
        runner = MultiAgentOrchestrator(model=MODEL, specialists=load_specialists())
        mode = "multi-agent"
    else:
        runner = build_agent(system_prompt)
        mode = "single"

    result, num_reflections, final_score = await _run_with_reflection(
        runner, task, memory_context, is_multi=MULTI_AGENT_MODE
    )
    latency_ms = int((time.monotonic() - t0) * 1000)

    # Confidence gate
    confidence_gated = False
    if result.confidence < CONFIDENCE_GATE:
        console.print(f"[red]Confidence {result.confidence:.0%} below gate — returning low-confidence notice[/red]")
        result.answer = (
            f"[Low confidence: {result.confidence:.0%}] "
            f"The agent could not produce a reliable answer. "
            f"Partial reasoning: {result.reasoning[:200]}"
        )
        confidence_gated = True

    meta = RunMeta(
        latency_ms=latency_ms,
        reflections=num_reflections,
        final_score=final_score,
        confidence_gated=confidence_gated,
        mode=mode,
    )

    memory.store(task, result.answer, final_score, result.tools_used)
    _write_log(task, result, meta)

    console.print(Panel(
        f"[bold green]Answer[/bold green]\n\n{result.answer}\n\n"
        f"[dim]Confidence: {result.confidence:.0%} | "
        f"Tools: {', '.join(result.tools_used) or 'none'} | "
        f"Latency: {latency_ms}ms | Reflections: {num_reflections} | "
        f"Mode: {mode}[/dim]",
        border_style="green",
    ))

    if return_meta:
        return result, meta.model_dump()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.input))
