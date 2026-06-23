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

    # ── Provider fallback registry (resolve_provider in code/providers.py) ────
    # PRIMARY = OpenRouter (one key, every model). If OPENROUTER_API_KEY is
    # absent/blocked, each model role resolves to a DIRECT vendor API whose key
    # is present. All vendors expose an OpenAI-compatible endpoint, so the same
    # `openai` client + a base_url swap covers all of them — no extra SDK.
    # Only OpenRouter + Anthropic are required; OpenAI/xAI/Google/Groq are
    # consulted ONLY when their key is set (else the jury shrinks/degrades safe).
    #   each entry: vendor -> (base_url, api_key_env_var)
    provider_registry: dict[str, tuple[str, str]] = field(default_factory=lambda: {
        "openrouter": ("https://openrouter.ai/api/v1",          "OPENROUTER_API_KEY"),
        "anthropic":  ("https://api.anthropic.com/v1",          "ANTHROPIC_API_KEY"),
        "openai":     ("https://api.openai.com/v1",             "OPENAI_API_KEY"),
        "xai":        ("https://api.x.ai/v1",                   "XAI_API_KEY"),
        "google":     ("https://generativelanguage.googleapis.com/v1beta/openai", "GOOGLE_API_KEY"),
        "groq":       ("https://api.groq.com/openai/v1",        "GROQ_API_KEY"),  # hosts Llama vision
    })
    # When OpenRouter is unavailable, map each role to a DIRECT-vendor model id.
    # Stage 1/3/repair (Claude) -> Anthropic native ids. Jury -> native ids on
    # whichever vendor keys exist; missing vendors are dropped (consensus
    # degrades to "accept Opus verdict + flag manual_review" — the safe direction).
    direct_model_ids: dict[str, str] = field(default_factory=lambda: {
        # role / openrouter-id            -> direct-vendor native id
        "anthropic/claude-opus-4-8":       "claude-opus-4-8",
        "anthropic/claude-sonnet-4-6":     "claude-sonnet-4-6",
        "anthropic/claude-haiku-4-5":      "claude-haiku-4-5",
        "openai/gpt-4o":                   "gpt-4o",
        "x-ai/grok-2-vision-1212":         "grok-2-vision-1212",
        "google/gemini-2.5-flash":         "gemini-2.5-flash",
        "meta-llama/llama-3.2-11b-vision-instruct": "llama-3.2-11b-vision-preview",  # Groq id
    })

    # ── Stage-named models (no ambiguity vs ablation labels) ──────────────────
    # Pricing (per 1M tok, OpenRouter ≈ Anthropic): opus-4-8 $5/$25 ·
    # sonnet-4-6 $3/$15 · haiku-4-5 $1/$5. Opus is only ~1.67x Sonnet, and is
    # the strongest visual reasoner — used for the load-bearing verdict (Stage 3).
    stage1_model: str = "anthropic/claude-haiku-4-5"           # transcript parse (cheap text)
    stage3_primary_model: str = "anthropic/claude-opus-4-8"    # visual reasoning (best — owns the verdict)
    stage3_primary_cpu_fallback: str = "anthropic/claude-sonnet-4-6"  # cheaper fallback
    stage3_repair_model: str = "anthropic/claude-haiku-4-5"    # targeted repair (cheap)

    # ── Strategy A model (single-call baseline for evaluation comparison) ─────
    strategy_a_model: str = "anthropic/claude-opus-4-8"

    # ── Stage 3.6 consensus JURY — multi-vendor, all via OpenRouter ──────────
    # The verdict stays on Opus (best single model). The jury is independent
    # cross-vendor second opinions — diversity here RAISES accuracy. Tiered so
    # paid vendors only fire on the hardest claims (token optimization):
    #   Tier 1 (soft escalation) → FREE models only ($0)
    #   Tier 2/3 (hard / disagreement) → add PAID models for true cross-vendor vote
    consensus_free_models: tuple[str, ...] = (
        "google/gemini-2.5-flash",                      # Google  (free)
        "meta-llama/llama-3.2-11b-vision-instruct",     # Meta    (free)
    )
    consensus_paid_models: tuple[str, ...] = (
        "openai/gpt-4o",                                # OpenAI  (paid, hard cases)
        "x-ai/grok-2-vision-1212",                      # xAI     (paid, hard cases)
    )
    # Weighted vote (primary-dominant). Keyed by vendor family; normalized over
    # whichever models actually ran. Opus owns the largest share.
    consensus_weights: dict[str, float] = field(default_factory=lambda: {
        "primary": 0.40,   # Opus 4.8 — the verdict
        "openai":  0.20,   # gpt-4o
        "google":  0.15,   # gemini
        "xai":     0.15,   # grok
        "meta":    0.10,   # llama
    })

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
    request_timeout_s: float = 90.0       # per-call OpenAI/OpenRouter timeout
    max_tokens_stage1: int = 600          # transcript parse output cap
    max_tokens_stage3: int = 900          # verdict output cap (terse justifications)
    max_tokens_jury: int = 300            # jury mini-schema output cap
    max_tokens_repair: int = 600          # targeted repair output cap
    jury_max_workers: int = 4             # parallel jury calls
    jury_deadline_s: float = 45.0         # overall wall-clock cap for the jury (all jurors); a
                                          # stalled juror can never hang the batch — the verdict
                                          # already stands on Opus, so we proceed with whatever
                                          # jurors finished in time (degrade safe)
    dry_run: bool = False                 # offline plumbing test — deterministic stub verdicts

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
