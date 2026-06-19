"""Single Config dataclass — the only place any setting lives."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── API ────────────────────────────────────────────────────────────────────
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str = "https://github.com/maddykws/bookish-invention"
    openrouter_app_title: str = "damage-claim-verifier"

    # ── Stage-named models (no ambiguity vs ablation labels) ──────────────────
    # Pricing (per 1M tok, OpenRouter ≈ Anthropic): opus-4-8 $5/$25 ·
    # sonnet-4-6 $3/$15 · haiku-4-5 $1/$5. Opus is only ~1.67x Sonnet, and is
    # the strongest visual reasoner — used for the load-bearing verdict (Stage 3).
    stage1_model: str = "anthropic/claude-haiku-4-5"           # transcript parse (cheap text)
    stage3_primary_model: str = "anthropic/claude-opus-4-8"    # visual reasoning (best)
    stage3_primary_cpu_fallback: str = "anthropic/claude-sonnet-4-6"  # cheaper fallback
    stage3_repair_model: str = "anthropic/claude-haiku-4-5"    # targeted repair (cheap)
    crosscheck_model_a: str = "google/gemini-2.5-flash"        # free tier
    crosscheck_model_b: str = "meta-llama/llama-3.2-11b-vision-instruct"  # free tier

    # ── Strategy A model (single-call baseline for evaluation comparison) ─────
    strategy_a_model: str = "anthropic/claude-opus-4-8"

    # ── Prompt caching ────────────────────────────────────────────────────────
    # Minimum cacheable prefix is model-dependent: opus-4-8 = 4096 tokens,
    # sonnet-4-6 = 2048. A shorter prefix SILENTLY won't cache (no error,
    # cache_creation_input_tokens=0). The Stage-3 cached block must clear this.
    prompt_cache_min_prefix_tokens: int = 4096

    # ── Image preprocessing ───────────────────────────────────────────────────
    resize_max_px: int = 768
    blur_threshold: float = 100.0
    clip_mismatch_threshold: float = 0.2

    # ── Pipeline behaviour ────────────────────────────────────────────────────
    temperature: float = 0.0
    seed: int = 42
    max_repair_attempts: int = 2
    token_budget_per_claim: int = 2000
    transcript_max_turns: int = 8
    consensus_trigger_threshold: int = 2

    # ── Confidence / escalation thresholds ───────────────────────────────────
    confidence_tier0_threshold: float = 0.85
    confidence_tier1_lower: float = 0.60
    confidence_tier2_threshold: float = 0.60
    consensus_confidence_gate: float = 0.70

    # ── Paths ─────────────────────────────────────────────────────────────────
    claims_path: str = "dataset/claims.csv"
    sample_claims_path: str = "dataset/sample_claims.csv"
    user_history_path: str = "dataset/user_history.csv"
    evidence_req_path: str = "dataset/evidence_requirements.csv"
    output_path: str = "output.csv"
    checkpoint_path: str = ".checkpoint"
    log_path: str = "logs/pipeline.log"
    cache_path: str = ".cache/llm_responses.json"
    eval_report_path: str = "code/evaluation/report.json"
    images_sample_dir: str = "dataset/images/sample"
    images_test_dir: str = "dataset/images/test"

    # ── Ablation feature flags (all True = full Strategy B) ───────────────────
    use_consensus: bool = True
    use_local_vlm: bool = True
    use_clip: bool = True
    resize_images: bool = True
    image_first_prompt: bool = True
    fraud_checks: bool = True
    use_repair_loop: bool = True
    use_prompt_cache: bool = True

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_collection_name: str = "claim_memory"
    chroma_fraud_similarity_threshold: float = 0.88
    chroma_session_memory_top_k: int = 3

    # ── AGENTS.md mandatory log ───────────────────────────────────────────────
    agents_log_path: str = "logs/pipeline.log"


# Singleton — import this everywhere
CFG = Config()
