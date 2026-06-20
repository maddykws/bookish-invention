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

# ── Reasoning enforcement ────────────────────────────────────────────────────
REASONING_ENFORCEMENT = True  # True → agent must produce explicit ReasoningPlan before tools

# ── Grounding (anti-hallucination) ────────────────────────────────────────────
GROUNDING_ENABLED = True      # True → every claim in the answer checked against tool results
GROUNDING_FAIL_THRESHOLD = 0.60   # below this ratio → reject and force revision
GROUNDING_WARN_THRESHOLD = 0.85   # below this → warn but accept

# ── Self-critique ─────────────────────────────────────────────────────────────
SELF_CRITIQUE_ENABLED = True  # True → agent critiques its own draft before validator

# ── Meta-validator ────────────────────────────────────────────────────────────
VALIDATOR_ENABLED = True      # True → second independent agent reviews every answer

# ── Human-in-the-loop ────────────────────────────────────────────────────────
HUMAN_IN_THE_LOOP = False     # True → pause before high-stakes tool calls for confirmation
HITL_TOOLS = []              # tool names that require HITL approval (empty = all tools)
                              # e.g. ["write_database", "send_email", "make_api_call"]

# ── Observability (Arize Phoenix) ─────────────────────────────────────────────
PHOENIX_ENABLED = True
PHOENIX_PORT = 6006

# ── Prompt management ─────────────────────────────────────────────────────────
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
ACTIVE_PROMPT_VERSION = "1.0"
LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_SECRET_KEY"))

# ── Final output contract ─────────────────────────────────────────────────────
CONTRACT_ENABLED = True   # True → every answer must pass the output contract before delivery

# ── Multi-LLM judge panel (eliminates same-model bias) ────────────────────────
# Models are tried in order; any without an API key are silently skipped.
# Claude is the automatic fallback if all external judges are unavailable.
JUDGE_MODELS = [
    "openai/gpt-4o",               # needs OPENAI_API_KEY
    "google/gemini-flash",         # needs GOOGLE_API_KEY
    "groq/llama-3.3-70b-versatile",# needs GROQ_API_KEY  (free tier available)
]
JUDGE_ENSEMBLE = "average"         # "average" | "majority" | "conservative"

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_PASS_THRESHOLD = 0.7
EVAL_MODEL = "claude-sonnet-4-6"
