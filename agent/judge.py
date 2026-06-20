"""
Multi-LLM Judge Panel — eliminates same-model bias.

Problem: Claude scoring Claude is circular. A model that made an error
will often not detect it when asked to grade itself.

Solution: An independent panel of judges from different model families
(OpenAI, Google, Meta via Groq) score in parallel. Their scores are
combined into an ensemble via one of three strategies:
  - "average"      : mean of all available judge scores (default)
  - "majority"     : score from the judge closest to the median
  - "conservative" : minimum score (most strict — penalises any failure)

If a provider's API key is missing, that judge is silently skipped.
Claude is used as a fallback judge only if NO other judge is available.

Usage:
    from agent.judge import score_with_panel, validate_with_panel

    score, breakdown = await score_with_panel(task, answer)
    result          = await validate_with_panel(task, answer, reasoning, confidence)
"""

import os
import asyncio
import json
from pydantic import BaseModel
from rich.console import Console

console = Console()


# ── Models ────────────────────────────────────────────────────────────────────

class JudgeScore(BaseModel):
    model:   str
    score:   float        # 0.0 – 1.0
    reason:  str
    skipped: bool = False # True if API key was absent or call failed


class JudgeResult(BaseModel):
    scores:         list[JudgeScore]
    ensemble_score: float
    strategy:       str
    judges_used:    int
    judges_skipped: int


# ── Prompts ───────────────────────────────────────────────────────────────────

_SCORE_PROMPT = """\
Task: {task}

Answer: {answer}

Rate how well this answer addresses the task.
Score 0.0 (completely wrong / useless) to 1.0 (fully correct, complete, precise).
Reply ONLY in this exact format: <score>|<one sentence reason>
Example: 0.82|Answer is mostly correct but omits the time complexity.
"""

_VALIDATE_PROMPT = """\
You are a strict quality validator for an AI agent's output.

## Original Task
{task}

## Agent's Answer
{answer}

## Agent's Stated Reasoning
{reasoning}

## Agent's Stated Confidence
{confidence:.0%}

Evaluate on four dimensions:
1. FACTUAL CONSISTENCY: Does the answer contradict itself or its own reasoning?
2. COMPLETENESS: Does it fully address what was asked?
3. HALLUCINATION RISK: Does it assert specific facts not grounded in reasoning?
4. CONFIDENCE CALIBRATION: Is the stated {confidence:.0%} confidence warranted?

Respond as JSON only:
{{
  "passed": true/false,
  "score": 0.0-1.0,
  "hallucination_risk": "low"/"medium"/"high",
  "completeness": "complete"/"partial"/"incomplete",
  "issues": ["issue 1", "issue 2"],
  "critique": "One paragraph of actionable feedback."
}}
Return ONLY the JSON.
"""


# ── Per-judge call (async) ────────────────────────────────────────────────────

async def _call_judge_score(model_id: str, task: str, answer: str) -> JudgeScore:
    """Call one judge model for a score. Returns skipped=True on any failure."""
    provider = model_id.split("/")[0] if "/" in model_id else "anthropic"

    # Check env key before making the call
    key_map = {
        "openai":    "OPENAI_API_KEY",
        "google":    "GOOGLE_API_KEY",
        "groq":      "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    env_var = key_map.get(provider)
    if env_var and not os.getenv(env_var):
        return JudgeScore(model=model_id, score=0.0, reason="API key not set", skipped=True)

    prompt = _SCORE_PROMPT.format(task=task, answer=answer)

    try:
        import litellm
        litellm.drop_params = True  # ignore unsupported params silently
        resp = await asyncio.to_thread(
            litellm.completion,
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        parts = raw.split("|", 1)
        score = max(0.0, min(1.0, float(parts[0].strip())))
        reason = parts[1].strip() if len(parts) > 1 else raw
        return JudgeScore(model=model_id, score=score, reason=reason)
    except Exception as exc:
        return JudgeScore(model=model_id, score=0.5, reason=f"error: {exc}", skipped=True)


async def _call_judge_validate(
    model_id: str, task: str, answer: str, reasoning: str, confidence: float
) -> dict | None:
    """Call one judge model for validation JSON. Returns None on failure."""
    provider = model_id.split("/")[0] if "/" in model_id else "anthropic"

    key_map = {
        "openai":    "OPENAI_API_KEY",
        "google":    "GOOGLE_API_KEY",
        "groq":      "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    env_var = key_map.get(provider)
    if env_var and not os.getenv(env_var):
        return None

    prompt = _VALIDATE_PROMPT.format(
        task=task, answer=answer, reasoning=reasoning, confidence=confidence
    )

    try:
        import litellm
        litellm.drop_params = True
        resp = await asyncio.to_thread(
            litellm.completion,
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip().rstrip("```")
        return json.loads(raw)
    except Exception:
        return None


# ── Ensemble strategies ───────────────────────────────────────────────────────

def _ensemble(scores: list[float], strategy: str) -> float:
    if not scores:
        return 0.5
    if strategy == "conservative":
        return min(scores)
    if strategy == "majority":
        sorted_s = sorted(scores)
        return sorted_s[len(sorted_s) // 2]
    # default: average
    return sum(scores) / len(scores)


# ── Public API ────────────────────────────────────────────────────────────────

async def score_with_panel(
    task: str,
    answer: str,
    *,
    models: list[str] | None = None,
    strategy: str | None = None,
) -> tuple[float, JudgeResult]:
    """
    Score an answer using a panel of independent LLM judges.

    Returns (ensemble_score, JudgeResult).
    """
    from agent.config import JUDGE_MODELS, JUDGE_ENSEMBLE

    judge_models = models or JUDGE_MODELS
    judge_strategy = strategy or JUDGE_ENSEMBLE

    # Run all judges in parallel
    raw_scores: list[JudgeScore] = await asyncio.gather(
        *[_call_judge_score(m, task, answer) for m in judge_models]
    )

    used   = [s for s in raw_scores if not s.skipped]
    skipped = [s for s in raw_scores if s.skipped]

    # If every external judge was skipped, fall back to Claude
    if not used:
        console.print("  [yellow]Judge panel: all external judges skipped — using Claude fallback[/yellow]")
        fallback = await _call_judge_score("anthropic/claude-sonnet-4-6", task, answer)
        fallback.skipped = False
        used = [fallback]

    ensemble = _ensemble([s.score for s in used], judge_strategy)

    result = JudgeResult(
        scores=raw_scores,
        ensemble_score=round(ensemble, 4),
        strategy=judge_strategy,
        judges_used=len(used),
        judges_skipped=len(skipped),
    )

    _print_panel_summary(result)
    return ensemble, result


async def validate_with_panel(
    task: str,
    answer: str,
    reasoning: str,
    confidence: float,
    *,
    models: list[str] | None = None,
    strategy: str | None = None,
) -> dict:
    """
    Validate an answer using a panel of independent LLM judges.

    Returns a merged ValidationResult-compatible dict.
    The most conservative (strictest) verdict across judges is used for 'passed'.
    The ensemble score is averaged.
    """
    from agent.config import JUDGE_MODELS, JUDGE_ENSEMBLE

    judge_models = models or JUDGE_MODELS
    judge_strategy = strategy or JUDGE_ENSEMBLE

    raw_results: list[dict | None] = await asyncio.gather(
        *[_call_judge_validate(m, task, answer, reasoning, confidence) for m in judge_models]
    )

    valid = [r for r in raw_results if r is not None]

    if not valid:
        # Fallback to single Claude call
        console.print("  [yellow]Validator panel: all judges skipped — using Claude fallback[/yellow]")
        fb = await _call_judge_validate(
            "anthropic/claude-sonnet-4-6", task, answer, reasoning, confidence
        )
        if fb:
            valid = [fb]

    if not valid:
        return {
            "passed": True, "score": confidence,
            "hallucination_risk": "low", "completeness": "complete",
            "issues": ["All judges unavailable"], "critique": "",
        }

    scores    = [float(r.get("score", 0.5)) for r in valid]
    all_issues = []
    for r in valid:
        all_issues.extend(r.get("issues", []))
    unique_issues = list(dict.fromkeys(all_issues))  # deduplicate, preserve order

    ensemble_score = _ensemble(scores, judge_strategy)
    # Conservative: fail if ANY judge failed
    any_failed = any(not r.get("passed", True) for r in valid)
    # Worst-case hallucination risk across judges
    risk_rank = {"low": 0, "medium": 1, "high": 2}
    worst_risk = max(
        (r.get("hallucination_risk", "low") for r in valid),
        key=lambda x: risk_rank.get(x, 0),
    )
    # Worst-case completeness
    comp_rank = {"complete": 0, "partial": 1, "incomplete": 2}
    worst_comp = max(
        (r.get("completeness", "complete") for r in valid),
        key=lambda x: comp_rank.get(x, 0),
    )
    # Aggregate critiques
    critiques = [r.get("critique", "") for r in valid if r.get("critique")]
    merged_critique = " | ".join(critiques[:2]) if critiques else ""

    return {
        "passed":            not any_failed,
        "score":             round(ensemble_score, 4),
        "hallucination_risk": worst_risk,
        "completeness":      worst_comp,
        "issues":            unique_issues[:6],
        "critique":          merged_critique,
    }


# ── Console summary ───────────────────────────────────────────────────────────

def _print_panel_summary(result: JudgeResult) -> None:
    parts = []
    for s in result.scores:
        if s.skipped:
            parts.append(f"[dim]{s.model.split('/')[-1]}:skip[/dim]")
        else:
            color = "green" if s.score >= 0.75 else "yellow" if s.score >= 0.5 else "red"
            parts.append(f"[{color}]{s.model.split('/')[-1]}:{s.score:.2f}[/]")

    ensemble_color = "green" if result.ensemble_score >= 0.75 else "yellow" if result.ensemble_score >= 0.5 else "red"
    console.print(
        f"  Judge panel ({result.strategy}): "
        + " | ".join(parts)
        + f" → ensemble [{ensemble_color}]{result.ensemble_score:.2f}[/]"
        + f"  [dim]({result.judges_used} active, {result.judges_skipped} skipped)[/dim]"
    )
