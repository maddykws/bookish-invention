"""
Final Output Contract — MANDATORY last gate before any answer leaves the system.

Every response the agent produces must satisfy this contract.
A response that violates it is either repaired automatically or rejected
with a structured ContractViolation so the caller knows exactly what broke.

Contract fields (all required in the final envelope):
  answer            str       non-empty, ≥ MIN_ANSWER_CHARS
  confidence        float     0.0 – 1.0
  reasoning         str       non-empty
  tools_used        list[str] may be empty (valid for no-tool queries)
  grounding_ratio   float     0.0 – 1.0
  grounding_verdict str       "pass" | "warn" | "fail"
  sources           list[str] evidence citations (tool_name: snippet)
  reflections       int       ≥ 0
  latency_ms        int       ≥ 0
  model             str       non-empty
  contract_version  str       semver string of this contract

Forbidden patterns (auto-detected in answer text):
  - Bare "I don't know" with no supporting reasoning
  - Placeholder text: "TODO", "FIXME", "<insert", "[TBD]"
  - Contradictory confidence: states "I'm certain" but confidence < 0.5

Enable/disable via CONTRACT_ENABLED in agent/config.py.
"""

import re
from pydantic import BaseModel, field_validator
from rich.console import Console

console = Console()

CONTRACT_VERSION = "1.0"
MIN_ANSWER_CHARS = 10   # sanity floor — answers shorter than this are empty


# ── Contract envelope ─────────────────────────────────────────────────────────

class FinalOutput(BaseModel):
    """The standardised envelope every agent response must fit into."""

    answer:            str
    confidence:        float
    reasoning:         str
    tools_used:        list[str]
    grounding_ratio:   float
    grounding_verdict: str
    sources:           list[str]        # ["tool_name: output snippet", ...]
    reflections:       int
    latency_ms:        int
    model:             str
    contract_version:  str = CONTRACT_VERSION

    @field_validator("confidence", "grounding_ratio")
    @classmethod
    def _clamp_ratio(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("grounding_verdict")
    @classmethod
    def _valid_verdict(cls, v: str) -> str:
        if v not in ("pass", "warn", "fail"):
            return "warn"
        return v


# ── Violation record ──────────────────────────────────────────────────────────

class ContractViolation(BaseModel):
    field:   str
    rule:    str
    value:   str        # string representation of the offending value


class ContractResult(BaseModel):
    passed:     bool
    output:     FinalOutput
    violations: list[ContractViolation]


# ── Forbidden-pattern checks ──────────────────────────────────────────────────

_FORBIDDEN = [
    (re.compile(r"\bI don['’]t know\b", re.IGNORECASE),
     "bare 'I don't know' without supporting reasoning"),
    (re.compile(r"\b(TODO|FIXME|TBD)\b"),
     "placeholder text left in answer"),
    (re.compile(r"<insert\b", re.IGNORECASE),
     "unfilled template placeholder"),
    (re.compile(r"\[TBD\]", re.IGNORECASE),
     "[TBD] placeholder in answer"),
]

_OVERCONFIDENT = re.compile(
    r"\b(I(?:'m| am) (absolutely |completely |100%? )?certain|guaranteed|without (any )?doubt)\b",
    re.IGNORECASE,
)


def _check_forbidden(answer: str, confidence: float) -> list[ContractViolation]:
    violations: list[ContractViolation] = []

    for pattern, rule in _FORBIDDEN:
        if pattern.search(answer):
            violations.append(ContractViolation(
                field="answer", rule=rule, value=answer[:120]
            ))

    if _OVERCONFIDENT.search(answer) and confidence < 0.5:
        violations.append(ContractViolation(
            field="answer",
            rule="answer claims certainty but confidence < 50%",
            value=f"confidence={confidence:.0%}",
        ))

    return violations


# ── Auto-repair ───────────────────────────────────────────────────────────────

def _repair(output: FinalOutput, violations: list[ContractViolation]) -> FinalOutput:
    """
    Attempt lightweight automatic repair for recoverable violations.
    Returns a (possibly unchanged) FinalOutput.
    """
    answer = output.answer

    for v in violations:
        if "placeholder" in v.rule or "TBD" in v.rule or "template" in v.rule:
            # Strip placeholder tokens
            for pattern, _ in _FORBIDDEN[1:]:
                answer = pattern.sub("[unavailable]", answer)

    # Clamp confidence to match overconfident-language violation
    confidence = output.confidence
    if any("claims certainty" in v.rule for v in violations):
        confidence = min(confidence, 0.49)

    return output.model_copy(update={"answer": answer, "confidence": confidence})


# ── Main enforcement function ─────────────────────────────────────────────────

def enforce_contract(
    answer:            str,
    confidence:        float,
    reasoning:         str,
    tools_used:        list[str],
    grounding_ratio:   float,
    grounding_verdict: str,
    sources:           list[str],
    reflections:       int,
    latency_ms:        int,
    model:             str,
) -> ContractResult:
    """
    Validate and enforce the output contract.

    Returns ContractResult with:
      - passed=True  → output is contract-compliant (possibly auto-repaired)
      - passed=False → violations that could not be auto-repaired
    """
    from agent.config import CONTRACT_ENABLED, MODEL

    # Build the envelope first (normalises types via Pydantic validators)
    output = FinalOutput(
        answer=answer,
        confidence=confidence,
        reasoning=reasoning,
        tools_used=tools_used,
        grounding_ratio=grounding_ratio,
        grounding_verdict=grounding_verdict,
        sources=sources,
        reflections=reflections,
        latency_ms=latency_ms,
        model=model or MODEL,
        contract_version=CONTRACT_VERSION,
    )

    if not CONTRACT_ENABLED:
        return ContractResult(passed=True, output=output, violations=[])

    violations: list[ContractViolation] = []

    # ── Field presence / length checks ───────────────────────────────────────
    if not output.answer.strip():
        violations.append(ContractViolation(
            field="answer", rule="answer must not be empty", value="<empty>"
        ))
    elif len(output.answer.strip()) < MIN_ANSWER_CHARS:
        violations.append(ContractViolation(
            field="answer",
            rule=f"answer must be ≥ {MIN_ANSWER_CHARS} characters",
            value=output.answer,
        ))

    if not output.reasoning.strip():
        violations.append(ContractViolation(
            field="reasoning", rule="reasoning must not be empty", value="<empty>"
        ))

    if not output.model.strip():
        violations.append(ContractViolation(
            field="model", rule="model identifier must be set", value="<empty>"
        ))

    # ── Forbidden-pattern checks ──────────────────────────────────────────────
    violations.extend(_check_forbidden(output.answer, output.confidence))

    # ── Grounding gate ────────────────────────────────────────────────────────
    if output.grounding_verdict == "fail":
        violations.append(ContractViolation(
            field="grounding_verdict",
            rule="grounding verdict is 'fail' — answer contains ungrounded claims",
            value=f"ratio={output.grounding_ratio:.0%}",
        ))

    # ── Try auto-repair on recoverable violations ─────────────────────────────
    recoverable = {"placeholder", "TBD", "template", "claims certainty"}
    repairable = [v for v in violations if any(kw in v.rule for kw in recoverable)]
    hard       = [v for v in violations if v not in repairable]

    if repairable:
        output = _repair(output, repairable)
        console.print(
            f"  [yellow]Contract: auto-repaired {len(repairable)} violation(s)[/yellow]"
        )

    passed = len(hard) == 0

    # ── Console summary ───────────────────────────────────────────────────────
    if passed and not repairable:
        console.print(
            f"  [green]Contract v{CONTRACT_VERSION}: PASS[/green] — "
            f"answer={len(output.answer)}ch | "
            f"confidence={output.confidence:.0%} | "
            f"grounding={output.grounding_ratio:.0%} ({output.grounding_verdict}) | "
            f"sources={len(output.sources)} | "
            f"reflections={output.reflections}"
        )
    elif not passed:
        console.print(f"  [red]Contract v{CONTRACT_VERSION}: FAIL[/red] — {len(hard)} hard violation(s):")
        for v in hard:
            console.print(f"    [red]✗[/red] [{v.field}] {v.rule}")

    return ContractResult(passed=passed, output=output, violations=hard + repairable)


# ── Source extraction helper ──────────────────────────────────────────────────

def extract_sources(ledger) -> list[str]:
    """Pull citation strings from an EvidenceLedger for the sources field."""
    sources = []
    for e in getattr(ledger, "entries", []):
        snippet = e.tool_output[:120].replace("\n", " ")
        sources.append(f"{e.tool_name}: {snippet}")
    return sources
