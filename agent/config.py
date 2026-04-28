"""
Central config for the agent. All knobs in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"

# ── Multi-agent mode ──────────────────────────────────────────────────────────
MULTI_AGENT_MODE = True      # True  → orchestrator routes to specialist agents
                              # False → single flat agent (simpler, faster)

# ── Reflexion ─────────────────────────────────────────────────────────────────
MAX_REFLECTIONS = 3          # max self-correction loops
REFLECTION_THRESHOLD = 0.75  # score below this triggers a reflection loop

# ── Confidence gate ───────────────────────────────────────────────────────────
CONFIDENCE_GATE = 0.35       # refuse to answer if confidence stays below this

# ── Meta-validator ────────────────────────────────────────────────────────────
VALIDATOR_ENABLED = True     # second agent independently reviews every answer

# ── Human-in-the-loop ────────────────────────────────────────────────────────
HUMAN_IN_THE_LOOP = False    # True → pause before high-stakes tool calls for confirmation
HITL_TOOLS = []              # tool names that require HITL approval (empty = all tools)
                              # e.g. ["write_database", "send_email", "make_api_call"]

# ── Observability (Arize Phoenix) ─────────────────────────────────────────────
PHOENIX_ENABLED = True
PHOENIX_PORT = 6006

# ── Prompt management ─────────────────────────────────────────────────────────
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
ACTIVE_PROMPT_VERSION = "1.0"
LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_SECRET_KEY"))

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_PASS_THRESHOLD = 0.7
EVAL_MODEL = "claude-sonnet-4-6"
