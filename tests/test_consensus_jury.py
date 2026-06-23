"""Regression tests for the Stage 3.6 consensus jury (code/pipeline/stage3_6_consensus.py).

The jury is best-effort: the verdict already stands on Opus, so no juror may
(a) take down the claim by raising, or (b) hang the batch by stalling. These
tests pin both guarantees plus the weighted-aggregation behaviour, with no
network — a fake client is injected over the module's LLMClient.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from code.config import Config
from code.pipeline import stage3_6_consensus as s36
from code.providers import ResolvedProvider
from code.utils.llm_client import LLMResult

PRIMARY = ResolvedProvider("openrouter", "x", "x", via_openrouter=True)


def _jury(cfg):
    return s36.run_jury(
        tier=2, claim_text="c", claim_object="car", image_ids=["img_1"],
        image_data_urls=["d"], cfg=cfg, primary=PRIMARY,
    )


def _good(status="supported", conf=0.8):
    return LLMResult(
        {"claim_status": status, "confidence": conf, "severity": "low",
         "issue_type": "dent", "reasoning_summary": "ok"},
        "", "m", ok=True,
    )


# ── (a) a failing juror must not propagate ────────────────────────────────────

def test_juror_exception_degrades_to_empty(monkeypatch):
    class RaiseClient:
        def __init__(self, c, p): ...
        def complete_json(self, **k):
            raise AttributeError("'NoneType' object has no attribute 'choices'")

    monkeypatch.setattr(s36, "LLMClient", RaiseClient)
    assert _jury(Config()) == []  # no crash, jury just shrinks


# ── (b) a stalled juror must not hang the batch ───────────────────────────────

def test_stalled_juror_bounded_by_deadline(monkeypatch):
    class StallClient:
        def __init__(self, c, p): ...
        def complete_json(self, **k):
            time.sleep(30)

    monkeypatch.setattr(s36, "LLMClient", StallClient)
    cfg = Config()
    cfg.jury_deadline_s = 1.0
    t0 = time.time()
    out = _jury(cfg)
    elapsed = time.time() - t0
    assert out == []
    assert elapsed < 4.0, f"jury exceeded deadline: {elapsed:.1f}s"


def test_mixed_good_and_stalled_keeps_the_good(monkeypatch):
    class Mixed:
        def __init__(self, c, p): ...
        def complete_json(self, **k):
            if "gemini" in k.get("openrouter_model", ""):
                return _good("contradicted", 0.6)
            time.sleep(30)

    monkeypatch.setattr(s36, "LLMClient", Mixed)
    cfg = Config()
    cfg.jury_deadline_s = 1.0
    out = _jury(cfg)
    assert len(out) == 1
    assert out[0].claim_status == "contradicted"


# ── isolation: consensus off ──────────────────────────────────────────────────

def test_no_consensus_runs_no_jury(monkeypatch):
    class BoomClient:
        def __init__(self, c, p):
            raise AssertionError("jury must not be constructed when consensus is off")

    monkeypatch.setattr(s36, "LLMClient", BoomClient)
    cfg = Config()
    cfg.use_consensus = False
    assert _jury(cfg) == []


def test_tier0_runs_no_jury(monkeypatch):
    class BoomClient:
        def __init__(self, c, p):
            raise AssertionError("tier 0 must not build a jury")

    monkeypatch.setattr(s36, "LLMClient", BoomClient)
    out = s36.run_jury(tier=0, claim_text="c", claim_object="car", image_ids=[],
                       image_data_urls=[], cfg=Config(), primary=PRIMARY)
    assert out == []


# ── aggregation ───────────────────────────────────────────────────────────────

def test_aggregate_primary_only_when_no_jury():
    d = s36.aggregate(primary_status="supported", primary_conf=0.9, jury=[], cfg=Config())
    assert d["status"] == "supported"
    assert d["agreement"] == "primary_only"
    assert d["extra_flags"] == []


def test_aggregate_unanimous():
    from code.pipeline.models import CrossCheckResult
    jury = [
        CrossCheckResult(model="google/gemini-2.5-flash", claim_status="supported",
                         confidence=0.7, severity="low", issue_type="dent",
                         reasoning_summary=""),
    ]
    d = s36.aggregate(primary_status="supported", primary_conf=0.9, jury=jury, cfg=Config())
    assert d["status"] == "supported"
    assert d["agreement"] == "unanimous"


def test_aggregate_disagreement_flags_manual_review():
    from code.pipeline.models import CrossCheckResult
    jury = [
        CrossCheckResult(model="openai/gpt-4o", claim_status="contradicted",
                         confidence=0.9, severity="high", issue_type="dent",
                         reasoning_summary=""),
        CrossCheckResult(model="x-ai/grok-2-vision-1212", claim_status="contradicted",
                         confidence=0.9, severity="high", issue_type="dent",
                         reasoning_summary=""),
        CrossCheckResult(model="google/gemini-2.5-flash", claim_status="contradicted",
                         confidence=0.9, severity="high", issue_type="dent",
                         reasoning_summary=""),
    ]
    # primary supported (0.40) vs contradicted jurors (0.20 + 0.15 + 0.15 = 0.50)
    # -> majority flips to contradicted, which must raise the manual-review flag.
    d = s36.aggregate(primary_status="supported", primary_conf=0.9, jury=jury, cfg=Config())
    assert d["status"] == "contradicted"
    assert "manual_review_required" in d["extra_flags"]
