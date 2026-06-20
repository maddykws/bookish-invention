"""Stage 3 — primary verdict (Opus 4.8) with prompt caching on the static prefix.

Assembles the four inputs (claim, images, history, evidence requirements) into
the volatile per-claim suffix and calls the verdict model. Also: evidence-
requirement selection (data-driven, §4.4) and the history snippet (§4.3).
"""

from __future__ import annotations

import re

from code.config import Config
from code.pipeline.models import EvidenceRequirement, ExtractedClaim, UserHistory
from code.pipeline.prompts import STAGE3_STATIC_PREFIX, build_stage3_user
from code.utils.injection import screen_text
from code.utils.llm_client import LLMClient, LLMResult
from code.utils.logger import get_logger

log = get_logger("pipeline.stage3")

_CATCH_ALL = {"all", "any", "", "*", "general"}


def select_requirements(
    claim_object: str, issue_family: str, reqs: list[EvidenceRequirement],
) -> list[EvidenceRequirement]:
    """Data-driven selection keyed on claim_object + applies_to (ID-agnostic, §4.4)."""
    obj = claim_object.strip().lower()
    fam = issue_family.strip().lower()
    out: list[EvidenceRequirement] = []
    for req in reqs:
        rob = req.claim_object.strip().lower()
        if rob not in _CATCH_ALL and rob != obj:
            continue
        if _applies(fam, req.applies_to):
            out.append(req)
    return out


def _applies(issue_family: str, applies_to: str) -> bool:
    a = applies_to.strip().lower()
    if a in _CATCH_ALL:
        return True
    if issue_family in ("", "unknown"):
        return False
    fam_tokens = set(issue_family.replace("-", "_").split("_"))
    a_tokens = set(re.findall(r"[a-z]+", a))
    return bool(fam_tokens & a_tokens)


def history_snippet(history: UserHistory | None) -> tuple[str, list[str]]:
    """Return (text for the prompt, preflags). history_summary is injection-screened."""
    if history is None:
        return "", []
    summary, _ = screen_text(history.history_summary)
    flags: list[str] = []
    if history.is_high_risk:
        flags.append("user_history_risk")
    if "manual_review_required" in history.history_flags:
        flags.append("manual_review_required")
    text = (
        f"  summary: {summary}\n"
        f"  history_flags: {history.history_flags}\n"
        f"  rejection_rate: {history.rejection_rate:.0%} "
        f"({history.rejected_claim}/{history.past_claim_count})\n"
        f"  last_90_days_claims: {history.last_90_days_claim_count}"
    )
    return text, flags


def run_verdict(
    *, extracted: ExtractedClaim, image_ids: list[str], image_data_urls: list[str],
    history: UserHistory | None, reqs: list[EvidenceRequirement],
    preflags: list[str], client: LLMClient, cfg: Config,
) -> tuple[LLMResult, list[str]]:
    """Call the primary verdict model. Returns (result, combined_preflags)."""
    hist_text, hist_flags = history_snippet(history)
    selected = select_requirements(extracted.claim_object, extracted.issue_family, reqs)
    req_text = "\n".join(f"  [{r.requirement_id}] {r.minimum_image_evidence}" for r in selected)
    combined_flags = sorted(set(preflags) | set(hist_flags))
    preflag_note = "; ".join(combined_flags) if combined_flags else ""

    user_text = build_stage3_user(
        claim_text=extracted.claim_text, claim_object=extracted.claim_object,
        claimed_part=extracted.claimed_part, issue_family=extracted.issue_family,
        history_snippet=hist_text, requirements_text=req_text,
        image_ids=image_ids, preflag_note=preflag_note,
    )

    res = client.complete_json(
        openrouter_model=cfg.stage3_primary_model,
        system_prefix=STAGE3_STATIC_PREFIX,
        user_text=user_text,
        image_data_urls=image_data_urls,
        max_tokens=cfg.max_tokens_stage3,
        cache_prefix=cfg.use_prompt_cache,
        x_title="stage3-verdict",
    )
    if res.cache_read_tokens:
        log.info("Prompt cache HIT: %d cached tokens", res.cache_read_tokens)
    return res, combined_flags
