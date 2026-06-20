"""Stage 3.5/3.6/3.7 — escalation gate, multi-vendor jury, weighted aggregation.

The verdict is already set by Opus (Stage 3). The jury is independent cross-
vendor second opinions, run in parallel via a thread pool. Tiered: free models on
soft escalation, paid models added on hard cases. Aggregation is a weighted vote
normalized over whichever models actually ran (§27.3).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from code.config import Config
from code.pipeline.models import CrossCheckResult
from code.pipeline.prompts import JURY_SYSTEM, build_jury_user
from code.providers import ResolvedProvider, resolve_jury_provider
from code.utils.llm_client import LLMClient
from code.utils.logger import get_logger

log = get_logger("pipeline.consensus")

_HEDGE = ("might", "maybe", "unclear", "possibly", "hard to tell", "appears", "seems")


def escalation_tier(primary_conf: float, risk_flag_count: int, justification: str,
                    primary_status: str, cfg: Config) -> int:
    """Map the primary verdict to an escalation tier (0=fast path .. 3=ceiling)."""
    hedged = any(h in justification.lower() for h in _HEDGE)
    if primary_status == "not_enough_information" or primary_conf < cfg.confidence_tier2_threshold:
        return 2
    if (cfg.confidence_tier1_lower <= primary_conf < cfg.confidence_tier0_threshold
            or hedged or risk_flag_count >= cfg.consensus_trigger_threshold):
        return 1
    return 0


def run_jury(
    *, tier: int, claim_text: str, claim_object: str, image_ids: list[str],
    image_data_urls: list[str], cfg: Config, primary: ResolvedProvider,
) -> list[CrossCheckResult]:
    """Run the jury for this tier. Tier 0 -> no jury. Tier 1 -> free models.
    Tier 2/3 -> free + paid. Missing-provider jurors are skipped (degrade safe)."""
    if tier == 0 or not cfg.use_consensus:
        return []
    models = list(cfg.consensus_free_models)
    if tier >= 2:
        models += list(cfg.consensus_paid_models)

    results: list[CrossCheckResult] = []
    tasks: list[tuple[str, LLMClient]] = []
    for m in models:
        prov = resolve_jury_provider(m, cfg, primary)
        if prov is None:
            continue
        tasks.append((m, LLMClient(cfg, prov)))

    if not tasks:
        return []

    with ThreadPoolExecutor(max_workers=cfg.jury_max_workers) as pool:
        futs = {
            pool.submit(_one_juror, m, cl, claim_text, claim_object,
                        image_ids, image_data_urls, cfg): m
            for m, cl in tasks
        }
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                results.append(r)
    return results


def _one_juror(model, client, claim_text, claim_object, image_ids, image_urls, cfg):
    res = client.complete_json(
        openrouter_model=model, system_prefix=JURY_SYSTEM,
        user_text=build_jury_user(claim_text, claim_object, image_ids),
        image_data_urls=image_urls, max_tokens=cfg.max_tokens_jury,
        x_title="stage36-jury",
    )
    if not res.ok:
        return None
    c = res.content
    status = str(c.get("claim_status", "")).strip()
    if status not in ("supported", "contradicted", "not_enough_information"):
        return None
    return CrossCheckResult(
        model=model, claim_status=status,  # type: ignore[arg-type]
        confidence=_clamp(c.get("confidence", 0.5)),
        severity=str(c.get("severity", "unknown")) if c.get("severity") in
        ("none", "low", "medium", "high", "unknown") else "unknown",  # type: ignore[arg-type]
        issue_type=str(c.get("issue_type", "unknown")),
        reasoning_summary=str(c.get("reasoning_summary", "")),
    )


def aggregate(
    *, primary_status: str, primary_conf: float, jury: list[CrossCheckResult],
    cfg: Config,
) -> dict:
    """Weighted vote over primary + jury. Returns the consensus decision dict:
    {status, confidence, agreement, extra_flags, note}."""
    weights = cfg.consensus_weights
    votes: dict[str, float] = {}
    confs: dict[str, list[float]] = {}

    def add(vendor_key: str, status: str, conf: float) -> None:
        w = weights.get(vendor_key, 0.0)
        votes[status] = votes.get(status, 0.0) + w
        confs.setdefault(status, []).append(conf)

    add("primary", primary_status, primary_conf)
    for j in jury:
        add(_vendor_key(j.model), j.claim_status, j.confidence)

    if not jury:  # no consensus run — primary stands
        return {"status": primary_status, "confidence": primary_conf,
                "agreement": "primary_only", "extra_flags": [], "note": ""}

    winner = max(votes, key=votes.get)  # type: ignore[arg-type]
    total = sum(votes.values()) or 1.0
    win_share = votes[winner] / total

    all_statuses = {primary_status} | {j.claim_status for j in jury}
    summary = "Opus=%s; " % primary_status + "; ".join(
        f"{_short(j.model)}={j.claim_status}" for j in jury
    )

    if len(all_statuses) == 1:
        return {"status": winner, "confidence": _avg(confs[winner]),
                "agreement": "unanimous", "extra_flags": [], "note": ""}

    # disagreement exists
    if win_share < 0.5 or _no_majority(votes):
        return {"status": "not_enough_information", "confidence": 0.0,
                "agreement": "no_consensus",
                "extra_flags": ["manual_review_required", "model_consensus_conflict"],
                "note": f"Jury reached no consensus: {summary}."}

    extra = []
    if winner != primary_status:
        extra = ["manual_review_required", "model_consensus_conflict"]
    return {"status": winner, "confidence": _avg(confs[winner]),
            "agreement": "majority", "extra_flags": extra,
            "note": f"Majority verdict ({win_share:.0%}). {summary}."
            if extra else ""}


def _no_majority(votes: dict[str, float]) -> bool:
    if not votes:
        return True
    ordered = sorted(votes.values(), reverse=True)
    return len(ordered) > 1 and abs(ordered[0] - ordered[1]) < 1e-9


def _vendor_key(model: str) -> str:
    prefix = model.split("/", 1)[0]
    return {"openai": "openai", "x-ai": "xai", "google": "google",
            "meta-llama": "meta"}.get(prefix, "meta")


def _short(model: str) -> str:
    return model.split("/")[-1].split("-")[0]


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _clamp(v: object) -> float:
    try:
        return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return 0.5
