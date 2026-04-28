"""
HackerRank Orchestrate Hackathon — Research & Planning Runner
=============================================================
USAGE:
  On question drop (11 AM IST, May 1):
    1. Paste the question into QUESTION below (or pass via --question flag)
    2. Run: python runner.py
    3. Review the full research report (target: done in 90 minutes)
    4. Start building

  Before submission:
    python runner.py --self-eval --agent-description "describe what you built"

MODES:
  python runner.py                          # Full research pipeline
  python runner.py --judge-tools-only       # Show pre-researched judge tools
  python runner.py --self-eval              # Run AI judge simulation
  python runner.py --checklist              # Show pre-submission checklist
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
from modules.github_search import search_github_implementations, search_github_judge_tools
from modules.judge_tools import display_judge_tools_shortlist, get_judge_tool_github_queries, PRERESEARCHED_JUDGE_TOOLS
from modules.synthesize import synthesize
from modules.self_eval import run_self_eval, display_checklist

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# PASTE THE HACKATHON QUESTION HERE WHEN IT DROPS
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
        console.print("[yellow]Warning:[/yellow] GITHUB_TOKEN not set — GitHub API rate limit will be 60 req/hr (unauthenticated)")
    if missing:
        console.print(f"[red]Missing required env vars:[/red] {', '.join(missing)}")
        console.print("Copy .env.example → .env and fill in your keys.")
        sys.exit(1)


def run_full_pipeline(question: str) -> dict:
    start = time.time()

    console.print(Panel(
        f"[bold green]HackerRank Orchestrate — Research Pipeline[/bold green]\n"
        f"[dim]Time budget: {RESEARCH_TIME_BUDGET_MINUTES} minutes max[/dim]",
        border_style="green",
    ))

    # ── Point 2: Question Decomposition Framework ──────────────────────────
    console.print(Rule("[bold]PHASE 1: Decompose[/bold]"))
    decomposition = decompose_question(question)

    # ── Point 1: ArXiv Search ──────────────────────────────────────────────
    console.print(Rule("[bold]PHASE 2a: ArXiv Research[/bold]"))
    arxiv_papers = search_arxiv(decomposition["arxiv_queries"])

    # ── Point 1: GitHub Search (implementations) ───────────────────────────
    console.print(Rule("[bold]PHASE 2b: GitHub Implementations[/bold]"))
    github_repos = search_github_implementations(decomposition["github_queries"])

    # ── Point 3 (pre-done) + live search: AI Judge Tools ──────────────────
    console.print(Rule("[bold]PHASE 3: AI Judge Tools[/bold]"))
    display_judge_tools_shortlist()
    live_judge_repos = search_github_judge_tools(get_judge_tool_github_queries())

    # ── Point 4: Synthesis → Architecture Decision ─────────────────────────
    console.print(Rule("[bold]PHASE 4: Synthesis → Architecture Plan[/bold]"))
    architecture_plan = synthesize(decomposition, arxiv_papers, github_repos, PRERESEARCHED_JUDGE_TOOLS)

    elapsed = (time.time() - start) / 60
    remaining = RESEARCH_TIME_BUDGET_MINUTES - elapsed

    console.print(Rule())
    console.print(Panel(
        f"[bold green]Research complete in {elapsed:.1f} minutes.[/bold green]\n"
        f"[cyan]Remaining research budget: {remaining:.1f} minutes[/cyan]\n\n"
        f"[bold yellow]→ START BUILDING NOW[/bold yellow]\n"
        f"[dim]Run  python runner.py --self-eval  before submitting[/dim]",
        border_style="green",
    ))

    return {
        "question": question,
        "decomposition": decomposition,
        "arxiv_papers": arxiv_papers,
        "github_repos": github_repos,
        "live_judge_repos": live_judge_repos,
        "architecture_plan": architecture_plan,
    }


def main():
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate Research Pipeline")
    parser.add_argument("--question", type=str, help="Hackathon question (overrides QUESTION constant)")
    parser.add_argument("--judge-tools-only", action="store_true", help="Show pre-researched judge tools and exit")
    parser.add_argument("--self-eval", action="store_true", help="Run AI judge simulation (use after building)")
    parser.add_argument("--agent-description", type=str, default="", help="Describe your built agent for self-eval")
    parser.add_argument("--checklist", action="store_true", help="Show pre-submission checklist")
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
        console.print("[red]Error:[/red] Paste the actual hackathon question into QUESTION in runner.py (or use --question flag)")
        sys.exit(1)

    if args.self_eval:
        # ── Point 5: Self-eval loop before AI judge round ─────────────────
        console.print(Panel(
            "[bold yellow]Self-Evaluation Mode[/bold yellow]\n"
            "[dim]Simulating AI judge round — run this before submitting[/dim]"
        ))
        # Re-decompose to get architecture plan context
        decomposition = decompose_question(question)
        # Quick synthesis with no papers (just problem context)
        arch_plan = f"Domain: {decomposition['domain']}\nAgent Type: {decomposition['agent_type']}\nHints: {decomposition['architecture_hints']}"
        run_self_eval(question, arch_plan, args.agent_description)
        return

    # Full pipeline
    run_full_pipeline(question)


if __name__ == "__main__":
    main()
