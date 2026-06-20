"""Per-claim orchestration — ties the stages together with batch isolation.

Every claim is processed inside a try/except so one failure can never kill the
batch; on any error the claim gets safe_defaults. Returns (ClaimOutput, metrics).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from code.config import Config
from code.pipeline.chroma_memory import ClaimMemory
from code.pipeline.models import (
    ClaimRow, ClaimOutput, EvidenceRequirement, PipelineMetrics, UserHistory,
    safe_defaults,
)
from code.pipeline import stage1_transcript as s1
from code.pipeline import stage2_images as s2
from code.pipeline import stage3_reason as s3
from code.pipeline import stage3_6_consensus as s36
from code.pipeline import stage4_validate as s4
from code.providers import ResolvedProvider
from code.utils.cache import ClaimCache
from code.utils.llm_client import LLMClient
from code.utils.logger import get_logger

log = get_logger("pipeline.orchestrator")


@dataclass
class BatchContext:
    cfg: Config
    provider: ResolvedProvider
    client: LLMClient
    history: dict[str, UserHistory]
    requirements: list[EvidenceRequirement]
    cache: ClaimCache
    memory: ClaimMemory
    seen_hashes: dict[str, str] = field(default_factory=dict)


def process_claim(claim: ClaimRow, ctx: BatchContext) -> tuple[ClaimOutput, PipelineMetrics]:
    metrics = PipelineMetrics(claim_id=claim.user_id)
    try:
        return _process(claim, ctx, metrics)
    except Exception as exc:  # noqa: BLE001 — claim-level isolation is the point
        log.error("Unhandled error on %s (%s) — safe defaults", claim.user_id, exc)
        metrics.safe_defaults_applied = True
        return safe_defaults(claim, f"unhandled: {type(exc).__name__}"), metrics


def _process(claim: ClaimRow, ctx: BatchContext, m: PipelineMetrics) -> tuple[ClaimOutput, PipelineMetrics]:
    cfg, client = ctx.cfg, ctx.client

    # Stage 2 first (local, cheap) so we can compute the cache key from image hashes
    processed, data_urls, preflags, exif_inj = s2.process_images(claim, cfg, ctx.seen_hashes)
    image_ids = [p.image_id for p in processed]
    valid_ids = [p.image_id for p in processed if p.valid]
    img_hashes = [p.sha256 for p in processed if p.sha256]

    # L1/L2 verdict cache (exact claim+images)
    cached = ctx.cache.get(claim.user_claim, img_hashes)
    if cached is not None:
        m.cache_hit = True
        log.info("Cache hit for %s", claim.user_id)
        return cached, m

    # Stage 1 — transcript parse
    extracted, injected = s1.parse_transcript(claim, client, cfg)
    m.injection_detected = injected or exif_inj
    if m.injection_detected:
        preflags.append("text_instruction_present")

    # Cross-claim fraud memory
    preflags.extend(ctx.memory.check_and_add(claim.user_id, extracted.claim_text))

    # Stage 3 — primary verdict (Opus) with caching
    res, combined_flags = s3.run_verdict(
        extracted=extracted, image_ids=image_ids, image_data_urls=data_urls,
        history=ctx.history.get(claim.user_id), reqs=ctx.requirements,
        preflags=preflags, client=client, cfg=cfg,
    )
    m.stage3_tokens = res.prompt_tokens + res.completion_tokens
    m.stage3_generation_id = res.generation_id
    m.model_used = res.model

    if not res.ok:
        out = safe_defaults(claim, "primary verdict unavailable")
        ctx.cache.put(claim.user_claim, img_hashes, out)
        return out, m

    verdict = dict(res.content)
    primary_status = str(verdict.get("claim_status", "not_enough_information"))
    primary_conf = _clamp(verdict.get("confidence", 0.5))
    m.primary_confidence = primary_conf

    # Stage 3.5 — escalation tier
    base_flags = s4._clean_flags(verdict.get("risk_flags", "none"))
    flag_count = 0 if base_flags == "none" else len(base_flags.split(";"))
    tier = s36.escalation_tier(primary_conf, flag_count,
                               str(verdict.get("claim_status_justification", "")),
                               primary_status, cfg)
    m.tier_reached = tier

    # Stage 3.6/3.7 — jury + aggregation
    jury = s36.run_jury(
        tier=tier, claim_text=extracted.claim_text, claim_object=extracted.claim_object,
        image_ids=valid_ids, image_data_urls=data_urls, cfg=cfg, primary=ctx.provider,
    )
    decision = s36.aggregate(primary_status=primary_status, primary_conf=primary_conf,
                             jury=jury, cfg=cfg)
    m.consensus_confidence = decision["confidence"] if jury else None

    # apply consensus override + merge flags into the verdict before validation
    verdict["claim_status"] = decision["status"]
    merged = _merge_flags(base_flags, combined_flags, decision["extra_flags"])
    verdict["risk_flags"] = merged
    if decision["note"]:
        verdict["claim_status_justification"] = (
            str(verdict.get("claim_status_justification", "")).rstrip(". ")
            + ". " + decision["note"]
        )[:600]
    if decision["status"] == "not_enough_information":
        verdict["severity"] = "unknown"

    # Stage 4 — validate / repair / safe defaults
    out, hard = s4.build_output(claim=claim, verdict=verdict,
                                valid_image_ids=valid_ids, cfg=cfg)
    if out is None or hard:
        m.repair_invocations = 1
        out = s4.repair(claim=claim, verdict=verdict, valid_image_ids=valid_ids,
                        errors=hard or ["construction failed"], client=client, cfg=cfg)
        m.repair_success = not out.evidence_standard_met_reason.startswith("System error")

    ctx.cache.put(claim.user_claim, img_hashes, out)
    return out, m


def _merge_flags(*groups) -> str:
    flags: list[str] = []
    for g in groups:
        if isinstance(g, str):
            flags += [f.strip() for f in g.split(";") if f.strip() and f != "none"]
        else:
            flags += [f for f in g if f]
    # remap internal + dedupe
    out: list[str] = []
    seen: set[str] = set()
    for f in flags:
        if f == "model_consensus_conflict":
            f = "manual_review_required"
        if f not in seen:
            seen.add(f)
            out.append(f)
    return ";".join(out) if out else "none"


def _clamp(v: object) -> float:
    try:
        return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return 0.5
