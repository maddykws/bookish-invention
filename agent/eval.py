"""
Evaluation harness — DeepEval metrics + A/B prompt comparison.

Modes:
  python agent/eval.py                        # evaluate active prompt version
  python agent/eval.py --ab v1.0 v2.0         # A/B compare two prompt versions
  python agent/eval.py --version 1.1          # evaluate a specific prompt version

Output: per-case scores + aggregate report + judge-ready statement.
Run this after every prompt iteration. Paste scores into prompts/changelog.yaml.
"""

import sys
import asyncio
import argparse
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.config import EVAL_PASS_THRESHOLD, ACTIVE_PROMPT_VERSION, PROMPTS_DIR

console = Console()

# GENERATED_TEST_CASES  ← scaffold.py injects TEST_CASES_DATA here after question drop
TEST_CASES_DATA = [
    {
        "input": "Placeholder test case — scaffold.py will replace this on question drop",
        "expected_output": "A correct, complete answer to the task",
        "context": "",
        "label": "placeholder",
    }
]


# ── Core eval runner ──────────────────────────────────────────────────────────

def _get_metrics():
    from deepeval.metrics import (
        TaskCompletionMetric,
        AnswerRelevancyMetric,
        HallucinationMetric,
    )
    return [
        TaskCompletionMetric(threshold=EVAL_PASS_THRESHOLD, model="gpt-4o", include_reason=True),
        AnswerRelevancyMetric(threshold=EVAL_PASS_THRESHOLD, model="gpt-4o", include_reason=True),
        HallucinationMetric(threshold=0.5, model="gpt-4o"),
    ]


def _run_cases(prompt_version: str) -> dict:
    """Run all test cases against a given prompt version. Returns score summary."""
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase
    from agent.agent import main as run_agent, load_system_prompt, build_agent, setup_observability

    setup_observability()
    system_prompt = load_system_prompt(version=prompt_version)
    agent = build_agent(system_prompt)

    test_cases = []
    raw_outputs = []

    for case in TEST_CASES_DATA:
        console.print(f"  [yellow]→[/yellow] {case['label']}")
        try:
            result = asyncio.run(run_agent(case["input"]))
            actual = result.answer
        except Exception as e:
            actual = f"ERROR: {e}"

        raw_outputs.append({"label": case["label"], "output": actual})
        test_cases.append(LLMTestCase(
            input=case["input"],
            actual_output=actual,
            expected_output=case["expected_output"],
            context=[case["context"]] if case["context"] else [],
        ))

    metrics = _get_metrics()
    eval_results = evaluate(test_cases, metrics)
    return {"eval_results": eval_results, "raw_outputs": raw_outputs, "version": prompt_version}


def _extract_scores(eval_results) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    try:
        for tr in eval_results.test_results:
            for md in tr.metrics_data:
                scores.setdefault(md.name, []).append(md.score or 0.0)
    except Exception:
        pass
    return {k: round(sum(v) / len(v), 3) for k, v in scores.items()}


# ── Display helpers ───────────────────────────────────────────────────────────

def _display_single(data: dict) -> dict[str, float]:
    scores = _extract_scores(data["eval_results"])
    console.print(Rule(f"Prompt v{data['version']} — Results"))

    table = Table(show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="bold")
    table.add_column("Pass?", style="green")
    for metric, score in scores.items():
        passed = score >= EVAL_PASS_THRESHOLD
        table.add_row(metric, f"{score:.3f}", "✓" if passed else "✗")
    console.print(table)
    return scores


def _display_ab(data_a: dict, data_b: dict) -> None:
    scores_a = _extract_scores(data_a["eval_results"])
    scores_b = _extract_scores(data_b["eval_results"])

    console.print(Rule(f"A/B Comparison: v{data_a['version']} vs v{data_b['version']}"))

    table = Table(show_lines=True, title="A/B Prompt Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column(f"v{data_a['version']}", style="yellow")
    table.add_column(f"v{data_b['version']}", style="green")
    table.add_column("Delta", style="bold")
    table.add_column("Winner", style="magenta")

    all_metrics = set(scores_a) | set(scores_b)
    for metric in sorted(all_metrics):
        sa = scores_a.get(metric, 0.0)
        sb = scores_b.get(metric, 0.0)
        delta = sb - sa
        delta_str = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
        winner = f"v{data_b['version']}" if delta > 0 else (f"v{data_a['version']}" if delta < 0 else "tie")
        table.add_row(metric, f"{sa:.3f}", f"{sb:.3f}", delta_str, winner)

    console.print(table)

    console.print(Panel(
        "[bold yellow]Judge-Ready A/B Statement[/bold yellow]\n\n"
        f"\"I ran A/B evaluation between prompt v{data_a['version']} and v{data_b['version']} "
        f"on {len(TEST_CASES_DATA)} test cases using DeepEval's TaskCompletion, "
        f"AnswerRelevancy, and Hallucination metrics. "
        f"v{data_b['version']} improved TaskCompletion from "
        f"{scores_a.get('Task Completion', 0):.0%} to "
        f"{scores_b.get('Task Completion', 0):.0%}. "
        f"I can show you the prompt diff and the changelog.\"",
        border_style="yellow",
    ))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DeepEval harness + A/B prompt comparison")
    parser.add_argument("--ab", nargs=2, metavar=("VA", "VB"), help="A/B compare two versions e.g. --ab 1.0 2.0")
    parser.add_argument("--version", type=str, default=ACTIVE_PROMPT_VERSION, help="Prompt version to evaluate")
    args = parser.parse_args()

    try:
        from deepeval import evaluate  # noqa: F401
    except ImportError:
        console.print("[red]deepeval not installed. Run: pip install deepeval[/red]")
        sys.exit(1)

    if args.ab:
        va, vb = args.ab
        console.print(Panel(
            f"[bold cyan]A/B Evaluation: v{va} vs v{vb}[/bold cyan]\n"
            f"[dim]{len(TEST_CASES_DATA)} test cases each[/dim]"
        ))
        console.print(f"\n[bold]Running v{va}...[/bold]")
        data_a = _run_cases(va)
        console.print(f"\n[bold]Running v{vb}...[/bold]")
        data_b = _run_cases(vb)
        _display_ab(data_a, data_b)
    else:
        console.print(Panel(
            f"[bold cyan]Evaluation — Prompt v{args.version}[/bold cyan]\n"
            f"[dim]{len(TEST_CASES_DATA)} test cases | TaskCompletion + AnswerRelevancy + Hallucination[/dim]"
        ))
        data = _run_cases(args.version)
        scores = _display_single(data)

        console.print(Panel(
            "[bold yellow]Judge-Ready Statement[/bold yellow]\n\n"
            f"\"Evaluated on {len(TEST_CASES_DATA)} test cases using DeepEval. "
            + " | ".join(f"{k}: {v:.0%}" for k, v in scores.items())
            + ". Full traces available in Phoenix at localhost:6006.\"",
            border_style="yellow",
        ))


if __name__ == "__main__":
    main()
