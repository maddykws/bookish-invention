"""Load the four input CSVs into Pydantic models / dict indexes (§4).

Tolerant of column-name drift and missing optional files — the pipeline must
never crash on ingestion; bad rows get flagged downstream, not dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from code.pipeline.models import (
    ClaimRow, ClaimRowValidator, EvidenceRequirement, UserHistory,
)
from code.utils.logger import get_logger

log = get_logger("pipeline.loaders")


def _read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        log.warning("CSV not found: %s — returning empty frame", path)
        return pd.DataFrame()
    return pd.read_csv(p, dtype=str, keep_default_na=False).fillna("")


def load_claims(path: str) -> tuple[list[ClaimRow], list[dict]]:
    """Return (valid ClaimRows, raw rows that failed validation)."""
    df = _read_csv(path)
    valid: list[ClaimRow] = []
    invalid: list[dict] = []
    for raw in df.to_dict(orient="records"):
        claim, errors = ClaimRowValidator.validate(raw)
        if claim is None:
            log.warning("Invalid claim row (%s): %s", raw.get("user_id", "?"), errors)
            invalid.append(raw)
        else:
            valid.append(claim)
    log.info("Loaded %d claims (%d invalid) from %s", len(valid), len(invalid), path)
    return valid, invalid


def load_user_history(path: str) -> dict[str, UserHistory]:
    df = _read_csv(path)
    index: dict[str, UserHistory] = {}
    for raw in df.to_dict(orient="records"):
        uid = str(raw.get("user_id", "")).strip()
        if not uid:
            continue
        try:
            index[uid] = UserHistory(
                user_id=uid,
                past_claim_count=_int(raw.get("past_claim_count")),
                accept_claim=_int(raw.get("accept_claim")),
                manual_review_claim=_int(raw.get("manual_review_claim")),
                rejected_claim=_int(raw.get("rejected_claim")),
                last_90_days_claim_count=_int(raw.get("last_90_days_claim_count")),
                history_flags=str(raw.get("history_flags", "none")).strip() or "none",
                history_summary=str(raw.get("history_summary", "")).strip(),
            )
        except (ValueError, TypeError) as exc:
            log.warning("Skipping malformed history row %s: %s", uid, exc)
    log.info("Loaded %d user-history rows from %s", len(index), path)
    return index


def load_evidence_requirements(path: str) -> list[EvidenceRequirement]:
    df = _read_csv(path)
    reqs: list[EvidenceRequirement] = []
    for raw in df.to_dict(orient="records"):
        try:
            reqs.append(EvidenceRequirement(
                requirement_id=str(raw.get("requirement_id", "")).strip(),
                claim_object=str(raw.get("claim_object", "all")).strip().lower() or "all",
                applies_to=str(raw.get("applies_to", "")).strip(),
                minimum_image_evidence=str(raw.get("minimum_image_evidence", "")).strip(),
            ))
        except (ValueError, TypeError) as exc:
            log.warning("Skipping malformed requirement row: %s", exc)
    log.info("Loaded %d evidence requirements from %s", len(reqs), path)
    return reqs


def _int(v: object) -> int:
    try:
        return int(float(str(v).strip())) if str(v).strip() else 0
    except (ValueError, TypeError):
        return 0
