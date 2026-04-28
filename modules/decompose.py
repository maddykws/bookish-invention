"""
Question Decomposition Module
Breaks down the hackathon problem into structured search targets.
"""

import os
import json
import anthropic
from rich.console import Console
from rich.panel import Panel

console = Console()

DECOMPOSE_PROMPT = """You are a senior AI engineer preparing for a 24-hour hackathon.
You have been given the following problem statement. Break it down into structured components
that will drive targeted research on ArXiv and GitHub.

Problem Statement:
{question}

Return a JSON object with exactly these fields:
{{
  "domain": "The broad field this problem belongs to (e.g. NLP, Computer Vision, Robotics, Finance, Healthcare)",
  "agent_type": "Type of agent needed (e.g. RAG agent, planning agent, tool-use agent, multi-agent, code agent)",
  "input_shape": "What the agent receives as input",
  "output_shape": "What the agent must produce as output",
  "success_criteria": "How success is measured for this problem",
  "core_challenges": ["list", "of", "key", "technical", "challenges"],
  "arxiv_queries": ["3-4 specific search queries for ArXiv", "each targeting a different aspect"],
  "github_queries": ["3-4 specific search queries for GitHub", "focused on existing implementations"],
  "architecture_hints": ["likely patterns or frameworks that apply", "e.g. ReAct, RAG, Chain-of-Thought"],
  "build_vs_adapt": "Initial gut feeling: build from scratch or adapt existing? Why?"
}}

Return ONLY the JSON, no markdown, no explanation.
"""


def decompose_question(question: str) -> dict:
    console.print(Panel("[bold cyan]Step 1: Decomposing the Problem[/bold cyan]"))

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(question=question)}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    decomposition = json.loads(raw.strip())

    console.print("[green]Domain:[/green]", decomposition["domain"])
    console.print("[green]Agent Type:[/green]", decomposition["agent_type"])
    console.print("[green]Core Challenges:[/green]", ", ".join(decomposition["core_challenges"]))
    console.print("[green]Build vs Adapt:[/green]", decomposition["build_vs_adapt"])
    console.print()

    return decomposition
