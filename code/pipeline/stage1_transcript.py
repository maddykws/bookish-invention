"""Stage 1 — parse the chat transcript into a structured ExtractedClaim (§4.1)."""

from __future__ import annotations

from code.config import Config
from code.pipeline.models import ClaimRow, ExtractedClaim
from code.pipeline.prompts import STAGE1_SYSTEM, build_stage1_user
from code.utils.injection import screen_text
from code.utils.llm_client import LLMClient
from code.utils.logger import get_logger

log = get_logger("pipeline.stage1")

_VALID_OBJECTS = {"car", "laptop", "package", "unknown"}


def truncate_transcript(transcript: str, max_turns: int = 8) -> str:
    turns = [t.strip() for t in transcript.split("|") if t.strip()]
    if len(turns) <= max_turns:
        return transcript
    return " | ".join(turns[-max_turns:])


def parse_transcript(
    claim: ClaimRow, client: LLMClient, cfg: Config,
) -> tuple[ExtractedClaim, bool]:
    """Return (ExtractedClaim, injection_detected). Always returns a usable claim
    even when the model call fails (falls back to CSV claim_object)."""
    sanitized, injected = screen_text(claim.user_claim)
    transcript = truncate_transcript(sanitized, cfg.transcript_max_turns)

    res = client.complete_json(
        openrouter_model=cfg.stage1_model,
        system_prefix=STAGE1_SYSTEM,
        user_text=build_stage1_user(transcript, claim.claim_object),
        max_tokens=cfg.max_tokens_stage1,
        x_title="stage1-transcript",
    )

    c = res.content if res.ok else {}
    obj = str(c.get("claim_object", claim.claim_object)).strip().lower()
    if obj not in _VALID_OBJECTS:
        obj = claim.claim_object
    # the CSV claim_object is authoritative for the 3 real types
    if claim.claim_object in ("car", "laptop", "package"):
        obj = claim.claim_object

    extracted = ExtractedClaim(
        claim_text=str(c.get("claim_text", sanitized[:200]) or sanitized[:200]),
        claim_object=obj,  # type: ignore[arg-type]
        claimed_part=str(c.get("claimed_part", "unknown") or "unknown"),
        issue_family=str(c.get("issue_family", "unknown") or "unknown"),
        claim_language=str(c.get("claim_language", "en") or "en"),
        confidence=_clamp(c.get("confidence", 0.5)),
    )
    return extracted, injected


def _clamp(v: object) -> float:
    try:
        return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return 0.5
