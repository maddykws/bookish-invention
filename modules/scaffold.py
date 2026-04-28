"""
Scaffold Module
Generates all problem-specific code from the architecture plan.

Writes:
  agent/tools.py         — typed tool stubs (Pydantic in/out, # TODO bodies)
  agent/specialists.py   — multi-agent specialist definitions (orchestrator picks these up)
  prompts/system_v1.0.yaml — versioned system prompt
  agent/eval.py          — DeepEval test cases (3 normal, 1 edge, 1 adversarial)
  tests/test_tools.py    — pytest unit tests per tool (injected at marker)
  tests/test_agent.py    — e2e integration tests (injected at marker)
  docs/architecture.md   — Mermaid architecture diagram
"""

import os
import json
import yaml
import anthropic
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

console = Console()

REPO_ROOT   = Path(__file__).parent.parent
AGENT_DIR   = REPO_ROOT / "agent"
PROMPTS_DIR = REPO_ROOT / "prompts"
TESTS_DIR   = REPO_ROOT / "tests"
DOCS_DIR    = REPO_ROOT / "docs"

# ── Claude prompts ─────────────────────────────────────────────────────────────

TOOLS_PROMPT = """You are a senior Python AI engineer. Based on the problem below,
generate tool stubs for a PydanticAI agent.

## Problem Decomposition
{decomposition}

## Architecture Plan
{architecture_plan}

Requirements:
1. Each tool must have a Pydantic input model and return a Pydantic model (not raw dicts)
2. Generate 3-5 tools that together solve the problem end-to-end
3. Add # TODO: implement inside each tool body — the human fills this in
4. Import only: pydantic, pydantic_ai, typing, standard library
5. TOOLS list at the bottom registers all tool functions
6. Tool names: snake_case, descriptive, problem-specific

Return ONLY valid Python code, no markdown fences, no explanation.
"""

SPECIALISTS_PROMPT = """You are a senior AI architect designing a multi-agent system.

## Problem Decomposition
{decomposition}

## Architecture Plan
{architecture_plan}

## Available Tools
{tool_names}

Design 3-4 specialist agents. Each specialist gets a subset of tools and a focused role.
Together they should cover the full problem.

Return Python code for agent/specialists.py defining:

from agent.orchestrator import Specialist
from agent.tools import <relevant tools per specialist>

SPECIALISTS = [
    Specialist(
        name="specialist_name",          # snake_case
        description="what this handles", # one sentence, used by router
        system_prompt="You are a ...",   # focused system prompt for this specialist
        tools=[tool_a, tool_b],          # subset of imported tools
    ),
    ...
]

Return ONLY valid Python code, no markdown fences, no explanation.
"""

SYSTEM_PROMPT_PROMPT = """You are a prompt engineer. Write a system prompt (v1.0)
for an AI orchestrator agent that routes tasks to specialist sub-agents.

## Problem Decomposition
{decomposition}

## Architecture Plan
{architecture_plan}

The system prompt must:
1. State the orchestrator's goal clearly
2. Describe the ReAct reasoning loop (think → route → observe → synthesize)
3. Explain available specialists and when to use each
4. Define what a complete, correct final answer looks like
5. Handle uncertainty: when to ask for clarification vs attempt answer
6. Be under 400 words

Return ONLY the prompt text, no markdown, no explanation.
"""

EVAL_CASES_PROMPT = """You are a QA engineer. Generate DeepEval test cases for this problem.

## Problem Decomposition
{decomposition}

## Architecture Plan
{architecture_plan}

Generate exactly 5 test cases as a Python list of dicts:
- "input": task/query given to agent
- "expected_output": what a correct answer looks like
- "context": relevant context (empty string if none)
- "label": short snake_case name

Cover: 3 normal cases, 1 edge case, 1 adversarial/tricky case.

Return ONLY a valid Python list literal, no markdown, no variable assignment.
"""

PYTEST_TOOLS_PROMPT = """You are a QA engineer writing pytest tests for AI agent tools.

## Problem Decomposition
{decomposition}

## Tool Stubs
{tools_code}

Write pytest test classes — one class per tool — each with 3 test methods:
1. test_{{tool_name}}_happy_path     — valid input, check output type + key fields
2. test_{{tool_name}}_edge_case      — boundary/empty input
3. test_{{tool_name}}_invalid_input  — malformed input, should raise or return error gracefully

Use:
  import pytest, asyncio, sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).parent.parent))
  from agent.tools import <tool_input_models>, <tool_functions>
  def run(coro): return asyncio.get_event_loop().run_until_complete(coro)

Mark tests that need ANTHROPIC_API_KEY with:
  @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs API key")

Return ONLY valid Python code, no markdown fences.
"""

PYTEST_E2E_PROMPT = """You are a QA engineer writing end-to-end pytest tests for an AI agent.

## Problem Decomposition
{decomposition}

## Test Cases
{test_cases}

Write 3-5 pytest test functions using these inputs.
Each test:
  1. Calls asyncio.run(main(input))
  2. Asserts result is AgentAnswer
  3. Asserts result.answer is non-empty
  4. Asserts result.confidence >= 0.0

Mark all tests with:
  @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="needs API key")

Imports needed:
  import pytest, asyncio, os
  from agent.agent import main, AgentAnswer

Return ONLY valid Python test functions (no class, no imports — just the functions), no markdown.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _call_claude(prompt: str, max_tokens: int = 3000) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def _strip_fences(code: str) -> str:
    if code.startswith("```"):
        lines = code.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return code


def _tool_names_from_code(tools_code: str) -> list[str]:
    names = []
    for line in tools_code.split("\n"):
        if line.strip().startswith("async def "):
            name = line.strip().split("async def ")[1].split("(")[0]
            names.append(name)
    return names


# ── Writers ───────────────────────────────────────────────────────────────────

def _write_tools(tools_code: str) -> None:
    path = AGENT_DIR / "tools.py"
    header = (
        '"""\nTool stubs — generated by scaffold.py on question drop.\n'
        'Fill in each # TODO block with the actual implementation.\n'
        'All tools are async. All inputs/outputs are Pydantic models.\n"""\n\n'
    )
    path.write_text(header + tools_code)
    console.print("  [green]✓[/green] agent/tools.py")


def _write_specialists(specialists_code: str) -> None:
    path = AGENT_DIR / "specialists.py"
    header = (
        '"""\nSpecialist agents — generated by scaffold.py on question drop.\n'
        'Each specialist handles a focused subtask with its own tools + prompt.\n'
        'Loaded by agent/orchestrator.py at runtime.\n"""\n\n'
    )
    path.write_text(header + specialists_code)
    console.print("  [green]✓[/green] agent/specialists.py")


def _write_system_prompt(prompt_text: str) -> None:
    data = {
        "version": "1.0",
        "model": "claude-sonnet-4-6",
        "created": datetime.now().strftime("%Y-%m-%d"),
        "changed_from": None,
        "reason": "Initial generated prompt from question decomposition",
        "metric_before": None,
        "metric_after": None,
        "prompt": prompt_text,
    }
    path = PROMPTS_DIR / "system_v1.0.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    changelog_path = PROMPTS_DIR / "changelog.yaml"
    if not changelog_path.exists():
        with open(changelog_path, "w") as f:
            yaml.dump({
                "versions": [{
                    "version": "1.0",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "reason": "Initial generated prompt",
                    "task_completion": None,
                    "answer_relevancy": None,
                }]
            }, f, allow_unicode=True, default_flow_style=False)

    console.print("  [green]✓[/green] prompts/system_v1.0.yaml")


def _write_eval_cases(eval_cases_raw: str) -> None:
    eval_path = AGENT_DIR / "eval.py"
    if eval_path.exists():
        content = eval_path.read_text()
        if "# GENERATED_TEST_CASES" in content:
            # Replace placeholder TEST_CASES_DATA
            import re
            content = re.sub(
                r"TEST_CASES_DATA = \[.*?\]",
                f"TEST_CASES_DATA = {eval_cases_raw}",
                content,
                flags=re.DOTALL,
            )
            eval_path.write_text(content)
    console.print("  [green]✓[/green] agent/eval.py (test cases injected)")


def _inject_pytest_tools(pytest_tools_code: str) -> None:
    path = TESTS_DIR / "test_tools.py"
    if path.exists():
        content = path.read_text()
        marker = "# SCAFFOLD_TOOL_TESTS_START"
        if marker in content:
            content = content[:content.index(marker) + len(marker)]
            content += "\n\n" + pytest_tools_code
            path.write_text(content)
    console.print("  [green]✓[/green] tests/test_tools.py (tool tests injected)")


def _inject_pytest_e2e(pytest_e2e_code: str, test_cases_raw: str) -> None:
    path = TESTS_DIR / "test_agent.py"
    if path.exists():
        content = path.read_text()
        # Replace TEST_INPUTS placeholder
        import re
        content = re.sub(
            r"TEST_INPUTS = \[.*?\]",
            f"TEST_INPUTS = [{', '.join(repr(c['input']) for c in eval(test_cases_raw))}]",
            content,
            flags=re.DOTALL,
        )
        # Inject generated e2e functions
        marker = "# SCAFFOLD_E2E_TESTS_START"
        if marker in content:
            content = content[:content.index(marker) + len(marker)]
            content += "\n\n" + pytest_e2e_code
        path.write_text(content)
    console.print("  [green]✓[/green] tests/test_agent.py (e2e tests injected)")


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_scaffold(decomposition: dict, architecture_plan: str) -> None:
    console.print(Panel(
        "[bold cyan]Phase 5: Generating Agent Scaffold[/bold cyan]\n"
        "[dim]tools → specialists → prompt → eval cases → tests → diagram[/dim]"
    ))

    for d in [AGENT_DIR, PROMPTS_DIR, TESTS_DIR, DOCS_DIR]:
        d.mkdir(exist_ok=True)

    decomp_str = json.dumps(decomposition, indent=2)

    # 1. Tools
    console.print("[yellow]1/6 Generating tool stubs...[/yellow]")
    tools_code = _strip_fences(_call_claude(
        TOOLS_PROMPT.format(decomposition=decomp_str, architecture_plan=architecture_plan)
    ))
    _write_tools(tools_code)
    tool_names = _tool_names_from_code(tools_code)

    # 2. Specialists (multi-agent)
    console.print("[yellow]2/6 Generating specialist agents...[/yellow]")
    specialists_code = _strip_fences(_call_claude(
        SPECIALISTS_PROMPT.format(
            decomposition=decomp_str,
            architecture_plan=architecture_plan,
            tool_names="\n".join(f"- {n}" for n in tool_names),
        )
    ))
    _write_specialists(specialists_code)

    # 3. System prompt
    console.print("[yellow]3/6 Generating system prompt v1.0...[/yellow]")
    system_prompt_text = _call_claude(
        SYSTEM_PROMPT_PROMPT.format(decomposition=decomp_str, architecture_plan=architecture_plan)
    )
    _write_system_prompt(system_prompt_text)

    # 4. Eval test cases
    console.print("[yellow]4/6 Generating eval test cases...[/yellow]")
    eval_cases_raw = _strip_fences(_call_claude(
        EVAL_CASES_PROMPT.format(decomposition=decomp_str, architecture_plan=architecture_plan)
    ))
    _write_eval_cases(eval_cases_raw)

    # 5. Pytest tool tests
    console.print("[yellow]5/6 Generating pytest tool tests...[/yellow]")
    pytest_tools_code = _strip_fences(_call_claude(
        PYTEST_TOOLS_PROMPT.format(decomposition=decomp_str, tools_code=tools_code),
        max_tokens=4000,
    ))
    _inject_pytest_tools(pytest_tools_code)

    # 6. Pytest e2e tests
    console.print("[yellow]6/6 Generating e2e tests + architecture diagram...[/yellow]")
    try:
        pytest_e2e_code = _strip_fences(_call_claude(
            PYTEST_E2E_PROMPT.format(decomposition=decomp_str, test_cases=eval_cases_raw)
        ))
        _inject_pytest_e2e(pytest_e2e_code, eval_cases_raw)
    except Exception as e:
        console.print(f"  [yellow]E2E test injection skipped: {e}[/yellow]")

    # 7. Architecture diagram
    try:
        from modules.diagram import write_diagram
        write_diagram(tool_names[:5])
    except Exception as e:
        console.print(f"  [yellow]Diagram skipped: {e}[/yellow]")

    console.print("\n[bold green]Scaffold complete:[/bold green]")
    console.print("  [cyan]agent/tools.py[/cyan]        ← fill in # TODO blocks")
    console.print("  [cyan]agent/specialists.py[/cyan]  ← specialist agents (review routing)")
    console.print("  [cyan]prompts/system_v1.0.yaml[/cyan]")
    console.print("  [cyan]agent/eval.py[/cyan]         ← 5 generated test cases")
    console.print("  [cyan]tests/test_tools.py[/cyan]   ← pytest per-tool tests")
    console.print("  [cyan]tests/test_agent.py[/cyan]   ← e2e integration tests")
    console.print("  [cyan]docs/architecture.md[/cyan]  ← Mermaid diagram\n")
