"""Stage 4 — assemble + validate the output, repair if needed, else safe defaults.

Local coercion first (clean the model dict into a valid ClaimOutput shape),
then the 5 consistency checks, then a targeted repair call (Haiku) only if still
broken, then safe_defaults as the floor. Never raises; always yields a valid
14-field ClaimOutput (§24.2).
"""

from __future__ import annotations

from code.config import Config
from code.pipeline.models import (
    CLAIM_STATUS_VALUES, ISSUE_TYPE_VALUES, OBJECT_PART_VOCAB, SEVERITY_VALUES,
    VALID_RISK_FLAGS, ClaimOutput, ClaimRow, safe_defaults,
)
from code.pipeline.prompts import STAGE3_STATIC_PREFIX, build_repair_user
from code.utils.llm_client import LLMClient
from code.utils.logger import get_logger

log = get_logger("pipeline.stage4")

_INTERNAL_OK = VALID_RISK_FLAGS | {"model_consensus_conflict"}


def build_output(
    *, claim: ClaimRow, verdict: dict, valid_image_ids: list[str], cfg: Config,
) -> tuple[ClaimOutput | None, list[str]]:
    """Coerce a raw verdict dict into a ClaimOutput. Returns (output|None, errors)."""
    errors: list[str] = []

    status = _pick(verdict.get("claim_status"), CLAIM_STATUS_VALUES, "not_enough_information")
    issue = _pick(verdict.get("issue_type"), ISSUE_TYPE_VALUES, "unknown")
    severity = _pick(verdict.get("severity"), SEVERITY_VALUES, "unknown")

    part_vocab = OBJECT_PART_VOCAB.get(claim.claim_object, frozenset({"unknown"}))
    part = str(verdict.get("object_part", "unknown")).strip()
    if part not in part_vocab:
        part = "unknown"

    risk = _clean_flags(verdict.get("risk_flags", "none"))
    support = _clean_support_ids(verdict.get("supporting_image_ids", "none"), valid_image_ids)

    # ── 5 consistency repairs (local, deterministic) ─────────────────────────
    ev_met = _as_bool(verdict.get("evidence_standard_met"), default=(status == "supported"))
    valid_img = _as_bool(verdict.get("valid_image"), default=True)

    if not ev_met and status == "supported":
        errors.append("evidence_standard_met=false with claim_status=supported")
    if severity == "high" and issue == "none":
        errors.append("severity=high with issue_type=none")
    if severity == "none" and status == "supported":
        errors.append("severity=none with claim_status=supported")
    if status == "supported" and support == "none":
        errors.append("claim_status=supported with supporting_image_ids=none")
    if status == "not_enough_information" and severity == "high":
        errors.append("not_enough_information with severity=high")

    # local normalization for the ones we can fix without the model
    if status == "not_enough_information":
        severity = "unknown"
    if issue == "none" and severity not in ("none", "unknown"):
        severity = "none"

    try:
        out = ClaimOutput(
            user_id=claim.user_id, image_paths=claim.image_paths,
            user_claim=claim.user_claim, claim_object=claim.claim_object,
            evidence_standard_met=ev_met,
            evidence_standard_met_reason=str(verdict.get("evidence_standard_met_reason", ""))[:500] or "n/a",
            risk_flags=risk, issue_type=issue,  # type: ignore[arg-type]
            object_part=part, claim_status=status,  # type: ignore[arg-type]
            claim_status_justification=str(verdict.get("claim_status_justification", ""))[:600] or "n/a",
            supporting_image_ids=support, valid_image=valid_img,
            severity=severity,  # type: ignore[arg-type]
        )
    except (ValueError, TypeError) as exc:
        errors.append(f"ClaimOutput construction failed: {exc}")
        return None, errors

    # hard contradictions that local fixes cannot resolve -> ask for repair
    hard = [e for e in errors if "supported" in e]
    return out, hard


def repair(
    *, claim: ClaimRow, verdict: dict, valid_image_ids: list[str],
    errors: list[str], client: LLMClient, cfg: Config,
) -> ClaimOutput:
    """Targeted repair loop, then safe_defaults floor."""
    import json
    current = dict(verdict)
    for attempt in range(cfg.max_repair_attempts):
        log.warning("Repair attempt %d for %s: %s", attempt + 1, claim.user_id, errors)
        res = client.complete_json(
            openrouter_model=cfg.stage3_repair_model,
            system_prefix=STAGE3_STATIC_PREFIX,
            user_text=build_repair_user(errors, json.dumps(current)),
            max_tokens=cfg.max_tokens_repair, x_title="stage4c-repair",
        )
        if res.ok:
            current = {**current, **res.content}
        out, hard = build_output(claim=claim, verdict=current,
                                 valid_image_ids=valid_image_ids, cfg=cfg)
        if out is not None and not hard:
            return out
        errors = hard or errors
    log.error("Repair exhausted for %s — safe defaults", claim.user_id)
    return safe_defaults(claim, "validation failed after repair")


# ── coercion helpers ──────────────────────────────────────────────────────────

def _pick(v: object, allowed: frozenset[str], default: str) -> str:
    s = str(v).strip() if v is not None else ""
    return s if s in allowed else default


def _clean_flags(v: object) -> str:
    raw = [f.strip() for f in str(v).split(";") if f.strip()]
    kept: list[str] = []
    for f in raw:
        if f == "model_consensus_conflict":
            f = "manual_review_required"
        if f in VALID_RISK_FLAGS and f != "none":
            kept.append(f)
    # dedupe, preserve order
    seen: set[str] = set()
    uniq = [x for x in kept if not (x in seen or seen.add(x))]
    return ";".join(uniq) if uniq else "none"


def _clean_support_ids(v: object, valid_ids: list[str]) -> str:
    raw = [s.strip() for s in str(v).split(";") if s.strip()]
    kept = [s for s in raw if s in valid_ids]
    return ";".join(kept) if kept else "none"


def _as_bool(v: object, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return default
