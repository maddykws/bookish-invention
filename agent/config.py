"""
Central config for the agent. Change these to tune behaviour.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"

# ── Reflexion ─────────────────────────────────────────────────────────────────
MAX_REFLECTIONS = 3          # max self-correction loops before accepting best result
REFLECTION_THRESHOLD = 0.75  # DeepEval score below this triggers a reflection loop

# ── Observability (Arize Phoenix) ─────────────────────────────────────────────
PHOENIX_ENABLED = True
PHOENIX_PORT = 6006

# ── Prompt management ─────────────────────────────────────────────────────────
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
ACTIVE_PROMPT_VERSION = "1.0"       # bump this when you create a new prompt version
LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_SECRET_KEY"))  # auto-detected

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_PASS_THRESHOLD = 0.7    # minimum score for task completion metric
EVAL_MODEL = "claude-sonnet-4-6"   # model used by DeepEval LLM-as-judge
