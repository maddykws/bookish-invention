"""
HackerRank Orchestrate — End-to-End Pipeline
============================================
Question drops → this pipeline runs → you build → agent solves problem → you win.

PIPELINE (python runner.py):
  Phase 1   Decompose   → domain, agent type, success criteria, search queries
  Phase 2a  ArXiv       → cutting-edge papers (2022–2026, recency signal)
  Phase 2b  S2          → high-impact papers  (2020–2026, citation signal ≥50)
  Phase 2c  GitHub      → existing implementations (50+ stars/forks)
  Phase 3   Judge tools → shortlist + live search
  Phase 4   Synthesize  → architecture plan + 24h time budget
  Phase 5   Scaffold    → tools, specialists, prompt, eval cases, pytest tests, diagram

THEN YOU:
  Fill in   agent/tools.py  TODO blocks
  Test      make test                             (pytest — run after filling tools)
  Run       python agent/agent.py --input "..."   (Phoenix traces → localhost:6006)
  Demo      python agent/ui.py                    (Gradio streaming UI → localhost:7860)
  Eval      python agent/eval.py                  (DeepEval scores)
  A/B       python agent/eval.py --ab 1.0 2.0     (compare prompt versions)
  Submit    python runner.py --self-eval           (AI judge simulation)

MODES:
  python runner.py                          # Full pipeline
  python runner.py --judge-tools-only       # Show pre-researched judge tools
  python runner.py --self-eval              # AI judge simulation
  python runner.py --checklist              # Pre-submission checklist
"""

import os
import sys
import time
import argparse
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

from modules.decompose import decompose_question
from modules.arxiv_search import search_arxiv
from modules.semantic_scholar import search_semantic_scholar
from modules.github_search import search_github_implementations, search_github_judge_tools
from modules.judge_tools import display_judge_tools_shortlist, get_judge_tool_github_queries, PRERESEARCHED_JUDGE_TOOLS
from modules.synthesize import synthesize
from modules.scaffold import generate_scaffold
from modules.self_eval import run_self_eval, display_checklist

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# PASTE THE HACKATHON QUESTION HERE WHEN IT DROPS AT 11 AM IST ON MAY 1
# ─────────────────────────────────────────────────────────────────────────────
QUESTION = """
[PASTE THE HACKATHON QUESTION HERE WHEN IT DROPS AT 11 AM IST ON MAY 1]
"""
# ─────────────────────────────────────────────────────────────────────────────

RESEARCH_TIME_BUDGET_MINUTES = 90


def _check_env():
    missing = []
    if not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.getenv("GITHUB_TOKEN"):
        console.print("[yellow]Warning:[/yellow] GITHUB_TOKEN not set — GitHub API rate limit: 60 req/hr")
    if missing:
        console.print(f"[red]Missing required env vars:[/red] {', '.join(missing)}")
        console.print("Copy .env.example → .env and fill in your keys.")
        sys.exit(1)


def run_full_pipeline(question: str) -> dict:
    start = time.time()

    console.print(Panel(
        "[bold green]HackerRank Orchestrate — Full Pipeline[/bold green]\n"
        f"[dim]Research cap: {RESEARCH_TIME_BUDGET_MINUTES} min → scaffold → build → eval → submit[/dim]",
        border_style="green",
    ))

    # ── Phase 1: Decompose ────────────────────────────────────────────────────
    console.print(Rule("[bold]PHASE 1 / 5 — Decompose Question[/bold]"))
    decomposition = decompose_question(question)

    # ── Phase 2a: ArXiv (2022–2026, recency signal) ───────────────────────────
    console.print(Rule("[bold]PHASE 2a / 6 — ArXiv (2022–2026)[/bold]"))
    arxiv_papers = search_arxiv(decomposition["arxiv_queries"])

    # ── Phase 2b: Semantic Scholar (2020–2026, citation signal) ───────────────
    console.print(Rule("[bold]PHASE 2b / 6 — Semantic Scholar (2020–2026, ≥50 citations)[/bold]"))
    s2_papers = search_semantic_scholar(decomposition["arxiv_queries"])

    all_papers = arxiv_papers + s2_papers

    # ── Phase 2c: GitHub implementations ─────────────────────────────────────
    console.print(Rule("[bold]PHASE 2c / 6 — GitHub Implementations (50+ stars/forks)[/bold]"))
    github_repos = search_github_implementations(decomposition["github_queries"])

    # ── Phase 3: AI judge tools ───────────────────────────────────────────────
    console.print(Rule("[bold]PHASE 3 / 6 — AI Judge & Eval Tools[/bold]"))
    display_judge_tools_shortlist()
    live_judge_repos = search_github_judge_tools(get_judge_tool_github_queries())

    # ── Phase 4: Synthesize ───────────────────────────────────────────────────
    console.print(Rule("[bold]PHASE 4 / 6 — Synthesize → Architecture Plan[/bold]"))
    architecture_plan = synthesize(decomposition, all_papers, github_repos, PRERESEARCHED_JUDGE_TOOLS)

    # ── Phase 5: Scaffold ─────────────────────────────────────────────────────
    console.print(Rule("[bold]PHASE 5 / 6 — Generate Agent Scaffold[/bold]"))
    generate_scaffold(decomposition, architecture_plan)

    elapsed = (time.time() - start) / 60
    remaining = RESEARCH_TIME_BUDGET_MINUTES - elapsed

    console.print(Rule())
    console.print(Panel(
        f"[bold green]Pipeline complete in {elapsed:.1f} min[/bold green]  "
        f"[dim](budget: {RESEARCH_TIME_BUDGET_MINUTES} min)[/dim]\n\n"

        "[bold cyan]What was generated:[/bold cyan]\n"
        "  [cyan]agent/tools.py[/cyan]         ← fill in # TODO blocks\n"
        "  [cyan]agent/specialists.py[/cyan]   ← multi-agent specialist definitions\n"
        "  [cyan]prompts/system_v1.0.yaml[/cyan]\n"
        "  [cyan]agent/eval.py[/cyan]          ← 5 generated DeepEval test cases\n"
        "  [cyan]tests/test_tools.py[/cyan]    ← pytest per-tool tests\n"
        "  [cyan]tests/test_agent.py[/cyan]    ← e2e integration tests\n"
        "  [cyan]docs/architecture.md[/cyan]   ← Mermaid diagram\n\n"

        "[bold yellow]Build loop:[/bold yellow]\n"
        "  1. Fill in [cyan]agent/tools.py[/cyan] TODO blocks\n"
        "  2. [cyan]make test[/cyan]                              — pytest (target: all green)\n"
        "  3. [cyan]make agent TASK='test input'[/cyan]           — run agent\n"
        "  4. [cyan]http://localhost:6006[/cyan]                  — Phoenix traces\n"
        "  5. [cyan]make eval[/cyan]                              — DeepEval scores\n"
        "  6. Bump prompt → [cyan]prompts/changelog.yaml[/cyan]\n"
        "  7. [cyan]make ab VA=1.0 VB=2.0[/cyan]                 — A/B compare\n"
        "  8. [cyan]make ui[/cyan]                                — Gradio streaming demo\n"
        "  9. [cyan]make diagram[/cyan]                          — regenerate architecture diagram\n"
        " 10. [cyan]make self-eval[/cyan]                         — AI judge simulation\n"
        " 11. Submit\n\n"

        f"[dim]Remaining time budget: {remaining:.1f} min[/dim]",
        border_style="green",
    ))

    return {
        "question":          question,
        "decomposition":     decomposition,
        "arxiv_papers":      arxiv_papers,
        "s2_papers":         s2_papers,
        "all_papers":        all_papers,
        "github_repos":      github_repos,
        "live_judge_repos":  live_judge_repos,
        "architecture_plan": architecture_plan,
    }


def main():
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate — End-to-End Pipeline")
    parser.add_argument("--question", type=str, help="Hackathon question (overrides QUESTION constant)")
    parser.add_argument("--judge-tools-only", action="store_true", help="Show pre-researched judge tools")
    parser.add_argument("--self-eval", action="store_true", help="AI judge simulation (run before submitting)")
    parser.add_argument("--agent-description", type=str, default="", help="Describe your built agent for self-eval")
    parser.add_argument("--checklist", action="store_true", help="Pre-submission checklist")
    args = parser.parse_args()

    _check_env()

    if args.checklist:
        display_checklist()
        return

    if args.judge_tools_only:
        display_judge_tools_shortlist()
        return

    question = args.question or QUESTION.strip()

    if "[PASTE THE HACKATHON QUESTION HERE" in question:
        console.print("[red]Error:[/red] Paste the question into QUESTION in runner.py (or use --question flag)")
        sys.exit(1)

    if args.self_eval:
        console.print(Panel(
            "[bold yellow]Self-Evaluation Mode[/bold yellow]\n"
            "[dim]Simulating AI judge round — run this before submitting[/dim]"
        ))
        decomposition = decompose_question(question)
        arch_plan = (
            f"Domain: {decomposition['domain']}\n"
            f"Agent Type: {decomposition['agent_type']}\n"
            f"Architecture Hints: {decomposition['architecture_hints']}"
        )
        run_self_eval(question, arch_plan, args.agent_description)
        return

    run_full_pipeline(question)


if __name__ == "__main__":
    main()
