"""
Synthesis Module
Takes all research results and produces:
  - Architecture recommendation
  - Build vs adapt vs compose decision
  - Tech stack suggestion
  - Implementation priority order
  - Time budget breakdown for 24 hours
"""

import os
import json
import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

SYNTHESIS_PROMPT = """You are a senior AI engineer with 24 hours to build and submit an AI agent at a hackathon.

## Problem Decomposition
{decomposition}

## ArXiv Research Found
{arxiv_papers}

## GitHub Repos Found (existing implementations, 50+ stars)
{github_repos}

## Available AI Judge/Eval Tools
{judge_tools}

---

Based on all the above, produce a concise technical action plan. Return as markdown with these sections:

### Architecture Decision
Specify the exact architecture pattern (e.g. ReAct, RAG+Tool-use, Plan-and-Execute, etc.).
Explain WHY this pattern fits the problem.

### Build vs Adapt vs Compose
- Which GitHub repos (if any) to directly adapt or use as base?
- Which to use as a library/component?
- What must be built from scratch?

### Tech Stack
List the exact libraries/frameworks to use (no alternatives — commit to one stack).

### Implementation Order
Number the implementation steps in order of priority.
Earlier steps = must-have. Later = nice-to-have if time allows.

### Time Budget (24 hours)
Break down the 24 hours:
- Research synthesis: already done (you have this plan)
- Implementation phases with hour estimates
- Self-eval + judge prep
- Buffer

### Judge Prep
What questions will the AI judge likely ask? List 5 specific questions based on this problem and architecture.

### Risks & Mitigations
Top 3 risks and how to handle each.
"""


def synthesize(
    decomposition: dict,
    arxiv_papers: list[dict],
    github_repos: list[dict],
    judge_tools: list[dict],
) -> str:
    console.print(Panel("[bold cyan]Step 4: Synthesizing Research → Architecture Plan[/bold cyan]"))

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    arxiv_summary = json.dumps(
        [{"title": p["title"], "published": p["published"], "abstract": p["abstract"]} for p in arxiv_papers[:8]],
        indent=2,
    )

    github_summary = json.dumps(
        [{"repo": r["name"], "stars": r["stars"], "description": r["description"]} for r in github_repos[:8]],
        indent=2,
    )

    judge_summary = json.dumps(
        [{"tool": t["name"], "use_case": t["use_case"], "metrics": t["key_metrics"]} for t in judge_tools],
        indent=2,
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        messages=[
            {
                "role": "user",
                "content": SYNTHESIS_PROMPT.format(
                    decomposition=json.dumps(decomposition, indent=2),
                    arxiv_papers=arxiv_summary,
                    github_repos=github_summary,
                    judge_tools=judge_summary,
                ),
            }
        ],
    )

    plan = response.content[0].text.strip()
    console.print(Markdown(plan))
    return plan
