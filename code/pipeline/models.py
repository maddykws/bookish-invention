"""All Pydantic models for the damage-claim pipeline.

Nothing flows between stages as a raw dict.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


# ── Input models ──────────────────────────────────────────────────────────────

class ClaimRow(BaseModel):
    user_id: str
    image_paths: str           # semicolon-separated paths as read from CSV
    user_claim: str            # actual column name in dataset CSV
    claim_object: Literal["car", "laptop", "package"]

    @property
    def image_path_list(self) -> list[str]:
        return [p.strip() for p in self.image_paths.split(";") if p.strip()]

    @property
    def image_count(self) -> int:
        return len(self.image_path_list)


class UserHistory(BaseModel):
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str         # "none" or "user_history_risk;manual_review_required"
    history_summary: str

    @property
    def rejection_rate(self) -> float:
        if self.past_claim_count == 0:
            return 0.0
        return self.rejected_claim / self.past_claim_count

    @property
    def is_high_risk(self) -> bool:
        return "user_history_risk" in self.history_flags


class EvidenceRequirement(BaseModel):
    requirement_id: str
    claim_object: str      # "all", "car", "laptop", "package"
    applies_to: str        # "dent or scratch", "multi-image rows", etc.
    minimum_image_evidence: str


# ── Pipeline-internal models ──────────────────────────────────────────────────

class ExtractedClaim(BaseModel):
    claim_text: str
    claim_object: Literal["car", "laptop", "package", "unknown"]
    claimed_part: str      # rear_bumper, screen, package_corner, etc.
    issue_family: str      # dent, scratch, crack, water_damage, etc.
    claim_language: str    # en, hi, mixed
    confidence: float      # how clearly claim was stated (0.0–1.0)


class ProcessedImage(BaseModel):
    path: Path
    image_id: str                      # "img_1", "img_2", etc.
    exists: bool
    valid: bool                        # passes all local checks
    is_blank: bool
    is_blurry: bool
    blur_score: float
    sha256: str
    is_duplicate: bool
    duplicate_of_claim: str | None
    yolo_detected_object: str | None   # "car", "laptop", "package", None
    clip_similarity: float
    exif_date: datetime | None
    resized_path: Path | None
    local_vlm_damage: str | None       # "yes" / "no" / "unclear" / None
    flags: list[str]

    class Config:
        arbitrary_types_allowed = True


class CrossCheckResult(BaseModel):
    model: str
    claim_status: Literal["supported", "contradicted", "not_enough_information"]
    confidence: float
    severity: Literal["none", "low", "medium", "high", "unknown"]
    issue_type: str
    reasoning_summary: str
    raw_response: str = ""


class PipelineMetrics(BaseModel):
    claim_id: str
    stage1_tokens: int = 0
    stage3_tokens: int = 0
    stage3_6_tokens: int = 0
    stage4c_tokens: int = 0
    stage1_cost_usd: float = 0.0
    stage3_cost_usd: float = 0.0
    stage3_6_cost_usd: float = 0.0
    stage4c_cost_usd: float = 0.0
    stage1_latency_ms: float = 0.0
    stage3_latency_ms: float = 0.0
    stage3_6_latency_ms: float = 0.0
    stage4c_latency_ms: float = 0.0
    tier_reached: int = 0
    repair_invocations: int = 0
    repair_success: bool = False
    safe_defaults_applied: bool = False
    cache_hit: bool = False
    injection_detected: bool = False
    stage1_generation_id: str = ""
    stage3_generation_id: str = ""
    primary_confidence: float = 0.0
    consensus_confidence: float | None = None
    model_used: str = ""


# ── Authoritative allowed-value vocabularies (from official spec) ─────────────

# The 14 sanctioned risk flags. This is the AUTHORITY — not the 20-sample subset.
# model_consensus_conflict is INTERNAL ONLY (remapped before output), never here.
VALID_RISK_FLAGS: frozenset[str] = frozenset({
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
})

# Per-object allowed object_part values. "unknown" is valid for every object.
OBJECT_PART_VOCAB: dict[str, frozenset[str]] = {
    "car": frozenset({
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
        "body", "unknown",
    }),
    "laptop": frozenset({
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
        "port", "base", "body", "unknown",
    }),
    "package": frozenset({
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item", "unknown",
    }),
}

ISSUE_TYPE_VALUES: frozenset[str] = frozenset({
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
})

CLAIM_STATUS_VALUES: frozenset[str] = frozenset({
    "supported", "contradicted", "not_enough_information",
})

SEVERITY_VALUES: frozenset[str] = frozenset({
    "none", "low", "medium", "high", "unknown",
})


# ── Output model ──────────────────────────────────────────────────────────────

class ClaimOutput(BaseModel):
    # 4 echoed input columns (verbatim)
    user_id: str
    image_paths: str
    user_claim: str        # echo input EXACTLY
    claim_object: Literal["car", "laptop", "package"]
    # 10 produced columns
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: str        # semicolon-separated
    issue_type: Literal[
        "dent", "scratch", "crack", "glass_shatter", "broken_part",
        "missing_part", "torn_packaging", "crushed_packaging",
        "water_damage", "stain", "none", "unknown"
    ]
    object_part: str             # validated against OBJECT_PART_VOCAB[claim_object]
    claim_status: Literal["supported", "contradicted", "not_enough_information"]
    claim_status_justification: str
    supporting_image_ids: str    # semicolon-separated or "none"
    valid_image: bool            # single bool: overall image set usable/authentic
    severity: Literal["none", "low", "medium", "high", "unknown"]

    @field_validator("risk_flags")
    @classmethod
    def validate_risk_flags(cls, v: str) -> str:
        flags = [f.strip() for f in v.split(";") if f.strip()]
        for flag in flags:
            if flag not in VALID_RISK_FLAGS:
                raise ValueError(
                    f"Invalid risk flag: {flag!r} (not in the 14 official flags)"
                )
        return v

    @model_validator(mode="after")
    def validate_object_part(self) -> "ClaimOutput":
        allowed = OBJECT_PART_VOCAB.get(self.claim_object)
        if allowed is not None and self.object_part not in allowed:
            raise ValueError(
                f"object_part {self.object_part!r} is not valid for "
                f"claim_object {self.claim_object!r}; allowed: {sorted(allowed)}"
            )
        return self


# ── Validator helper ──────────────────────────────────────────────────────────

class ClaimRowValidator:
    VALID_OBJECTS: frozenset[str] = frozenset({"car", "laptop", "package"})
    # Single source of truth — the 14 official flags (defined above).
    VALID_RISK_FLAGS: frozenset[str] = VALID_RISK_FLAGS
    # Internal-only flags → remap before writing output.csv
    INTERNAL_FLAG_MAP: dict[str, str] = {
        "model_consensus_conflict": "manual_review_required",
    }

    @staticmethod
    def validate(row: dict) -> tuple[ClaimRow | None, list[str]]:
        errors: list[str] = []

        if not row.get("user_id") or str(row["user_id"]).strip() == "":
            errors.append("user_id is null or empty")

        if not row.get("user_claim") or len(str(row["user_claim"]).strip()) < 2:
            errors.append("user_claim is empty or too short")

        obj = str(row.get("claim_object", "")).strip().lower()
        if obj not in ClaimRowValidator.VALID_OBJECTS:
            errors.append(f"claim_object '{obj}' is not one of: car, laptop, package")

        if not row.get("image_paths") or str(row["image_paths"]).strip() == "":
            errors.append("image_paths is empty — treating as no-image claim")
            # Non-fatal: proceed with image_count = 0

        if errors and any("user_id" in e or "claim_object" in e for e in errors):
            return None, errors

        try:
            claim = ClaimRow(
                user_id=str(row.get("user_id", "")).strip(),
                image_paths=str(row.get("image_paths", "")).strip(),
                user_claim=str(row.get("user_claim", "")).strip(),
                claim_object=obj,  # type: ignore[arg-type]
            )
            return claim, errors
        except Exception as exc:
            errors.append(f"ClaimRow construction failed: {exc}")
            return None, errors

    @staticmethod
    def remap_internal_flags(flags_str: str) -> str:
        """Map internal-only flags to output-safe equivalents."""
        flag_map = ClaimRowValidator.INTERNAL_FLAG_MAP
        flags = [f.strip() for f in flags_str.split(";") if f.strip()]
        remapped = []
        for flag in flags:
            remapped.append(flag_map.get(flag, flag))
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique = []
        for f in remapped:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        return ";".join(unique) if unique else "none"


# ── Output helpers ────────────────────────────────────────────────────────────

OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity",
]


def safe_defaults(claim: ClaimRow, reason: str) -> ClaimOutput:
    """Return a conservative, safe output when the pipeline cannot produce a valid verdict."""
    return ClaimOutput(
        user_id=claim.user_id,
        image_paths=claim.image_paths,
        user_claim=claim.user_claim,
        claim_object=claim.claim_object,
        evidence_standard_met=False,
        evidence_standard_met_reason=f"System error: {reason}",
        risk_flags="manual_review_required",
        issue_type="unknown",
        object_part="unknown",
        claim_status="not_enough_information",
        claim_status_justification=f"System could not produce valid output: {reason}",
        supporting_image_ids="none",
        valid_image=False,
        severity="unknown",
    )


def output_to_row(output: ClaimOutput) -> dict[str, str]:
    """Serialize ClaimOutput to a dict with lowercase booleans for CSV."""
    d = output.model_dump()
    # Remap any internal flags that slipped through
    d["risk_flags"] = ClaimRowValidator.remap_internal_flags(d["risk_flags"])
    # Lowercase booleans per ground-truth format
    d["evidence_standard_met"] = str(d["evidence_standard_met"]).lower()
    d["valid_image"] = str(d["valid_image"]).lower()
    return {col: str(d[col]) for col in OUTPUT_COLUMNS}
