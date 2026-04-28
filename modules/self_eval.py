"""
Self-Evaluation Module
Simulates the AI judge round before actual submission.
Run this after implementation is complete to stress-test the agent
and prepare answers for the mandatory AI judge session.
"""

import os
import anthropic
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

JUDGE_SIMULATION_PROMPT = """You are a strict AI judge evaluating a hackathon submission for HackerRank Orchestrate.
The participant built an AI agent to solve the following problem.

## Original Problem
{question}

## Participant's Architecture Plan
{architecture_plan}

## Agent Description (provided by participant)
{agent_description}

---

Conduct a rigorous judge session. Ask 8 probing questions that test:
1. Depth of understanding of the problem
2. Technical soundness of the architecture
3. Design decisions and trade-offs made
4. How the agent handles edge cases and failures
5. Scalability and extensibility
6. Evaluation methodology — how did they verify it works?
7. What they would do differently with more time
8. The most critical flaw and how they addressed it

Format each question as:
**Q[N]: [Category]**
[Question text]

After listing questions, add a section:
## Scoring Rubric
Rate the submitted approach on:
- Problem Understanding (1-10)
- Technical Implementation (1-10)
- Code Quality & Structure (1-10)
- Agent Robustness (1-10)
- Presentation Readiness (1-10)

For each dimension, explain what a 10/10 answer looks like.
"""

QUICK_CHECKLIST = """
## Pre-Submission Checklist

Before you submit, verify each item:

### Agent Functionality
- [ ] Agent solves the core problem end-to-end
- [ ] Handles at least 3 edge cases
- [ ] Returns structured, parseable output
- [ ] Does not crash on malformed input

### Code Quality
- [ ] Clear entry point (e.g. main.py or agent.py)
- [ ] Dependencies listed in requirements.txt
- [ ] No hardcoded secrets or API keys in code
- [ ] Code is readable — judge may look at it

### Architecture Clarity
- [ ] Can explain architecture in 2 minutes verbally
- [ ] Can explain WHY each tool/library was chosen
- [ ] Can describe the agent's reasoning loop
- [ ] Know what happens at each step of tool-use chain

### Evaluation
- [ ] Tested on at least 5 different inputs
- [ ] Measured at least one metric (accuracy, latency, etc.)
- [ ] Can quantify improvement over naive baseline
- [ ] Have a concrete result to present (number, demo, output sample)

### Judge Round Prep
- [ ] Rehearsed 2-minute overview of what you built
- [ ] Prepared answers to "what would you do differently?"
- [ ] Know your top 3 risks and mitigations
- [ ] Have a live demo or recorded output ready
"""


def run_self_eval(question: str, architecture_plan: str, agent_description: str = "") -> str:
    console.print(Panel(
        "[bold cyan]Step 5: Self-Evaluation — AI Judge Simulation[/bold cyan]\n"
        "[dim]Simulating the mandatory AI judge round[/dim]"
    ))

    if not agent_description:
        agent_description = "Agent description not provided yet — run this after implementation."

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2500,
        messages=[
            {
                "role": "user",
                "content": JUDGE_SIMULATION_PROMPT.format(
                    question=question,
                    architecture_plan=architecture_plan,
                    agent_description=agent_description,
                ),
            }
        ],
    )

    judge_session = response.content[0].text.strip()
    console.print(Markdown(judge_session))

    console.print(Panel("[bold yellow]Pre-Submission Checklist[/bold yellow]"))
    console.print(Markdown(QUICK_CHECKLIST))

    return judge_session


def display_checklist():
    console.print(Panel("[bold yellow]Pre-Submission Checklist[/bold yellow]"))
    console.print(Markdown(QUICK_CHECKLIST))
