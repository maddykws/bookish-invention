"""
Tool stubs — PLACEHOLDER file.
This file is OVERWRITTEN by scaffold.py when the question drops.
scaffold.py generates problem-specific tools based on the question decomposition.

After scaffold runs:
  1. Each tool will have a Pydantic input + output model
  2. Each tool body will have a # TODO: implement comment
  3. TOOLS list at the bottom registers all tools with the agent

This placeholder is here so the repo runs before question drop.
"""

from pydantic import BaseModel
from pydantic_ai import RunContext


class ExampleInput(BaseModel):
    query: str


class ExampleOutput(BaseModel):
    result: str
    source: str


async def example_tool(ctx: RunContext[None], input: ExampleInput) -> ExampleOutput:
    """Placeholder tool — returns a stub result until scaffold.py overwrites this
    file with problem-specific tools on question drop."""
    # TODO: implement — replace this entire file via scaffold.py on question drop
    return ExampleOutput(result="placeholder", source="placeholder")


# All tools registered with the agent — scaffold will populate this list
TOOLS = [example_tool]
