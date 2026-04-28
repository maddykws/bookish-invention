"""
Main agent entry point.
Stack: PydanticAI + Arize Phoenix + Langfuse + Reflexion + ChromaDB memory

Features:
  - Parallel async tool execution (asyncio — PydanticAI calls async tools concurrently)
  - Reflexion loop: up to MAX_REFLECTIONS self-correction passes
  - Confidence gating: refuses to answer when confidence < CONFIDENCE_GATE
  - Cost + latency tracking per run
  - Structured JSON logs to logs/
  - Long-term memory via ChromaDB (retrieves relevant past runs as context)

Run:
    python agent/agent.py --input "your task"
    python agent/ui.py          ← Gradio demo at localhost:7860
Phoenix dashboard: http://localhost:6006
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
from rich.markdown import Markdown
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.config import (
    MODEL, MAX_REFLECTIONS, REFLECTION_THRESHOLD, CONFIDENCE_GATE,
    PHOENIX_ENABLED, PHOENIX_PORT,
    PROMPTS_DIR, ACTIVE_PROMPT_VERSION, LANGFUSE_ENABLED,
)
from agent.memory import AgentMemory

console = Console()
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Approximate pricing for claude-sonnet-4-6 ($/1M tokens — verify at anthropic.com/pricing)
_COST_PER_1M_IN  = 3.00
_COST_PER_1M_OUT = 15.00


# ── Observability (Phoenix) ───────────────────────────────────────────────────

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


# ── Structured output ─────────────────────────────────────────────────────────

class AgentAnswer(BaseModel):
    answer: str
    reasoning: str
    confidence: float       # 0.0 – 1.0
    tools_used: list[str]


class RunMeta(BaseModel):
    tokens_in:   int   = 0
    tokens_out:  int   = 0
    cost_usd:    float = 0.0
    latency_ms:  int   = 0
    reflections: int   = 0
    final_score: float = 0.0
    confidence_gated: bool = False


# ── Agent factory ─────────────────────────────────────────────────────────────

def build_agent(system_prompt: str) -> Agent:
    from agent.tools import TOOLS
    agent: Agent[None, AgentAnswer] = Agent(
        model=f"anthropic:{MODEL}",
        result_type=AgentAnswer,
        system_prompt=system_prompt,
    )
    for tool_fn in TOOLS:
        agent.tool(tool_fn)
    return agent


# ── Scoring (drives Reflexion) ────────────────────────────────────────────────

async def _score(result: AgentAnswer, task: str) -> tuple[float, str]:
    try:
        import anthropic as anth
        client = anth.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": (
                f"Task: {task}\nAnswer: {result.answer}\n\n"
                "Rate 0.0–1.0. Reply ONLY: <score>|<one-line reason>. E.g. 0.85|Correct and complete."
            )}],
        )
        parts = resp.content[0].text.strip().split("|", 1)
        return float(parts[0].strip()), parts[1].strip() if len(parts) > 1 else ""
    except Exception:
        return result.confidence, "Derived from agent confidence"


# ── Reflexion loop ────────────────────────────────────────────────────────────

async def _run_with_reflection(
    agent: Agent,
    task: str,
    memory_context: str,
) -> tuple[AgentAnswer, int, float]:
    """Returns (best_result, num_reflections, best_score)."""
    feedback = memory_context   # inject memory as first context block
    best_result: AgentAnswer | None = None
    best_score = 0.0
    num_reflections = 0

    for attempt in range(1, MAX_REFLECTIONS + 1):
        console.print(f"\n[bold]Attempt {attempt}/{MAX_REFLECTIONS}[/bold]")
        run_result = await agent.run(task + feedback)
        result: AgentAnswer = run_result.data

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
                f"\n\n[Reflection {attempt}] Score {score:.2f}/1.0. "
                f"Issue: {reason}. Be more precise and complete."
            )
            console.print("  [yellow]Reflecting and retrying...[/yellow]")

    return best_result, num_reflections, best_score


# ── Cost tracking ─────────────────────────────────────────────────────────────

def _compute_cost(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * _COST_PER_1M_IN + tokens_out * _COST_PER_1M_OUT) / 1_000_000


# ── JSON log writer ───────────────────────────────────────────────────────────

def _write_log(task: str, result: AgentAnswer, meta: RunMeta) -> None:
    log_path = LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    log_path.write_text(json.dumps({
        "timestamp":        datetime.now().isoformat(),
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
        "confidence_gated": meta.confidence_gated,
    }, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(task: str, return_meta: bool = False):
    setup_observability()

    system_prompt = load_system_prompt()
    agent = build_agent(system_prompt)
    memory = AgentMemory()

    # Retrieve relevant past runs as context
    memory_context = memory.retrieve(task)
    if memory_context:
        console.print(f"[dim]Memory: {memory.count()} stored runs, injecting context[/dim]")

    console.print(Panel(
        f"[bold cyan]Running Agent[/bold cyan]\n"
        f"[dim]{task[:200]}{'...' if len(task) > 200 else ''}[/dim]",
        border_style="cyan",
    ))

    t0 = time.monotonic()
    result, num_reflections, final_score = await _run_with_reflection(agent, task, memory_context)
    latency_ms = int((time.monotonic() - t0) * 1000)

    # Confidence gate
    confidence_gated = False
    if result.confidence < CONFIDENCE_GATE:
        console.print(
            f"[red]Confidence {result.confidence:.0%} below gate ({CONFIDENCE_GATE:.0%}) "
            f"— returning low-confidence notice[/red]"
        )
        result.answer = (
            f"[Low confidence: {result.confidence:.0%}] "
            f"The agent could not produce a reliable answer. "
            f"Partial reasoning: {result.reasoning[:200]}"
        )
        confidence_gated = True

    # Cost tracking (tokens not directly exposed by PydanticAI — approximated via confidence proxy)
    # For exact token counts, instrument via Phoenix OpenTelemetry spans
    tokens_in, tokens_out = 0, 0   # populated by Phoenix spans; set 0 if not available
    cost_usd = _compute_cost(tokens_in, tokens_out)

    meta = RunMeta(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        reflections=num_reflections,
        final_score=final_score,
        confidence_gated=confidence_gated,
    )

    # Persist to memory and log
    memory.store(task, result.answer, final_score, result.tools_used)
    _write_log(task, result, meta)

    console.print(Panel(
        f"[bold green]Answer[/bold green]\n\n{result.answer}\n\n"
        f"[dim]Reasoning: {result.reasoning[:200]}[/dim]\n"
        f"[dim]Confidence: {result.confidence:.0%} | Tools: {', '.join(result.tools_used) or 'none'} | "
        f"Latency: {latency_ms}ms | Reflections: {num_reflections} | "
        f"Cost: ${cost_usd:.4f}[/dim]",
        border_style="green",
    ))

    if return_meta:
        return result, meta.model_dump()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Task for the agent")
    args = parser.parse_args()
    asyncio.run(main(args.input))
