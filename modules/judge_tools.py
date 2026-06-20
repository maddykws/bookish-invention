"""
AI Judge Tools Module
Pre-researched shortlist of AI evaluation/judge tools (as of April 2026).
Also runs a live GitHub search for latest tools.
These will be used to evaluate and stress-test the agent before submission.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Pre-researched shortlist — validated April 2026
PRERESEARCHED_JUDGE_TOOLS = [
    {
        "name": "confident-ai/deepeval",
        "url": "https://github.com/confident-ai/deepeval",
        "description": "Pytest-style LLM evaluation. Metrics: G-Eval, task completion, answer relevancy, hallucination. Runs locally.",
        "use_case": "Unit-test your agent's outputs against custom criteria before submission.",
        "install": "pip install deepeval",
        "key_metrics": ["GEval", "TaskCompletion", "AnswerRelevancy", "Hallucination", "ContextualRelevancy"],
    },
    {
        "name": "explodinggradients/ragas",
        "url": "https://github.com/explodinggradients/ragas",
        "description": "Evaluation toolkit for RAG pipelines. Faithfulness, answer relevance, context precision.",
        "use_case": "If your agent is RAG-based, use Ragas to measure retrieval + generation quality.",
        "install": "pip install ragas",
        "key_metrics": ["Faithfulness", "AnswerRelevance", "ContextPrecision", "ContextRecall"],
    },
    {
        "name": "langchain-ai/agentevals",
        "url": "https://github.com/langchain-ai/agentevals",
        "description": "Evaluators for agent trajectories. Checks tool selection, step correctness, goal completion.",
        "use_case": "Evaluate the full agent trajectory — not just final answer but each reasoning/tool step.",
        "install": "pip install agentevals",
        "key_metrics": ["TrajectoryExactMatch", "TrajectoryUnordered", "LLMAsJudge"],
    },
    {
        "name": "THUDM/AgentBench",
        "url": "https://github.com/THUDM/AgentBench",
        "description": "Comprehensive benchmark across 8 environments (OS, DB, web, code). ICLR'24.",
        "use_case": "Test your agent in real environments similar to hackathon problem domains.",
        "install": "git clone + docker",
        "key_metrics": ["TaskSuccessRate", "EnvironmentInteraction", "MultiTurnReasoning"],
    },
    {
        "name": "wandb/odsc-2025-agent-eval",
        "url": "https://github.com/wandb/odsc-2025-agent-eval",
        "description": "Ground truth eval + tool selection validation + GPT-4 LLM-as-judge + adversarial testing.",
        "use_case": "Full eval pipeline: ground truth comparison, tool validation, adversarial robustness.",
        "install": "pip install wandb",
        "key_metrics": ["GroundTruthMatch", "ToolSelectionAccuracy", "LLMJudgeScore", "AdversarialRobustness"],
    },
    {
        "name": "BerriAI/litellm",
        "url": "https://github.com/BerriAI/litellm",
        "description": "Universal LLM API wrapper. Swap models easily during evaluation.",
        "use_case": "Evaluate your agent across different LLM backends quickly.",
        "install": "pip install litellm",
        "key_metrics": ["ModelPortability", "CostTracking", "LatencyMetrics"],
    },
]

# GitHub queries for live search (used by github_search module)
JUDGE_TOOL_QUERIES = [
    "LLM evaluation framework agent judge open source",
    "AI agent evaluation benchmark task completion",
    "LLM as judge evaluation metrics hallucination",
    "agent trajectory evaluation tool use accuracy",
]


def display_judge_tools_shortlist():
    console.print(Panel(
        "[bold cyan]Pre-Researched AI Judge Tools Shortlist[/bold cyan]\n"
        "[dim]Validated April 2026 — ready to use on question drop[/dim]"
    ))

    table = Table(show_lines=True)
    table.add_column("Tool", style="cyan", max_width=30)
    table.add_column("Use Case", style="green", max_width=45)
    table.add_column("Key Metrics", style="yellow", max_width=40)
    table.add_column("Install", style="magenta", max_width=25)

    for t in PRERESEARCHED_JUDGE_TOOLS:
        table.add_row(
            t["name"],
            t["use_case"],
            ", ".join(t["key_metrics"][:3]),
            t["install"],
        )

    console.print(table)

    console.print("\n[bold]Recommended Quick Stack for Hackathon:[/bold]")
    console.print("  1. [cyan]deepeval[/cyan] — fast unit tests on agent outputs (any agent type)")
    console.print("  2. [cyan]agentevals[/cyan] — trajectory eval if multi-step reasoning matters")
    console.print("  3. [cyan]ragas[/cyan] — add only if RAG is core to your agent")
    console.print()

    return PRERESEARCHED_JUDGE_TOOLS


def get_judge_tool_github_queries() -> list[str]:
    return JUDGE_TOOL_QUERIES
