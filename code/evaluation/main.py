"""Evaluation entry point (§23.8, §25) — the primary technical-execution artifact.

  python code/evaluation/main.py            # full eval (needs a provider key)
  python code/evaluation/main.py --dry-run  # offline plumbing test

Loads sample_claims.csv (ground truth), runs Strategy A (single verdict call)
and Strategy B (full cascade), computes per-verdict precision/recall/F1 +
overall accuracy, prints a comparison table, and writes report.json. Exits 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from code.config import CFG, Config
from code.pipeline.chroma_memory import ClaimMemory
from code.pipeline.loaders import load_evidence_requirements, load_user_history
from code.pipeline.models import ClaimRow, ClaimRowValidator
from code.pipeline.orchestrator import BatchContext, process_claim
from code.pipeline import stage1_transcript as s1
from code.pipeline import stage2_images as s2
from code.pipeline import stage3_reason as s3
from code.pipeline import stage4_validate as s4
from code.providers import ResolvedProvider, resolve_primary_provider
from code.utils.cache import ClaimCache
from code.utils.llm_client import LLMClient
from code.utils.logger import get_logger

import pandas as pd

log = get_logger("evaluation")
console = Console()

VERDICTS = ("supported", "contradicted", "not_enough_information")


def _provider(cfg: Config) -> ResolvedProvider:
    if cfg.dry_run:
        return ResolvedProvider("dry-run", cfg.openrouter_base_url, "dry-run", via_openrouter=True)
    return resolve_primary_provider(cfg)


def _strategy_a(claim: ClaimRow, ctx: BatchContext) -> str:
    """Single verdict call — no preprocessing, no jury, no repair."""
    processed, data_urls, _, _ = s2.process_images(claim, ctx.cfg, {})
    extracted, _ = s1.parse_transcript(claim, ctx.client, ctx.cfg)
    res, _ = s3.run_verdict(
        extracted=extracted, image_ids=[p.image_id for p in processed],
        image_data_urls=data_urls, history=ctx.history.get(claim.user_id),
        reqs=ctx.requirements, preflags=[], client=ctx.client, cfg=ctx.cfg,
    )
    valid_ids = [p.image_id for p in processed if p.valid]
    out, _ = s4.build_output(claim=claim, verdict=res.content,
                             valid_image_ids=valid_ids, cfg=ctx.cfg)
    return out.claim_status if out else "not_enough_information"


def _metrics(preds: list[str], truth: list[str]) -> dict:
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    correct = 0
    for p, t in zip(preds, truth):
        if p == t:
            correct += 1
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    per = {}
    for v in VERDICTS:
        prec = tp[v] / (tp[v] + fp[v]) if (tp[v] + fp[v]) else 0.0
        rec = tp[v] / (tp[v] + fn[v]) if (tp[v] + fn[v]) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[v] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
    n = len(truth) or 1
    return {"accuracy": round(correct / n, 3), "correct": correct, "total": len(truth),
            "per_verdict": per}


def run(cfg: Config) -> int:
    console.rule("[bold green]Evaluation — Strategy A vs Strategy B")
    provider = _provider(cfg)

    p = Path(cfg.sample_claims_path)
    if not p.exists():
        console.print(f"[red]No {cfg.sample_claims_path} — cannot evaluate.[/red]")
        return 1
    df = pd.read_csv(p, dtype=str, keep_default_na=False).fillna("")
    rows = df.to_dict(orient="records")

    ctx = BatchContext(
        cfg=cfg, provider=provider, client=LLMClient(cfg, provider),
        history=load_user_history(cfg.user_history_path),
        requirements=load_evidence_requirements(cfg.evidence_req_path),
        cache=ClaimCache(
            ".cache/eval_responses.json",
            namespace="dry" if cfg.dry_run else "live",
        ),
        memory=ClaimMemory(cfg),
    )

    truth, pred_a, pred_b = [], [], []
    tier_counts = defaultdict(int)
    for raw in rows:
        gt = str(raw.get("claim_status", "")).strip()
        if gt not in VERDICTS:
            continue
        claim, errors = ClaimRowValidator.validate(raw)
        if claim is None:
            continue
        truth.append(gt)
        pred_a.append(_strategy_a(claim, ctx))
        out_b, m = process_claim(claim, ctx)
        pred_b.append(out_b.claim_status)
        tier_counts[m.tier_reached] += 1

    if not truth:
        console.print("[yellow]No ground-truth rows found (no claim_status column?).[/yellow]")
        return 0

    eval_a = _metrics(pred_a, truth)
    eval_b = _metrics(pred_b, truth)
    _print_tables(eval_a, eval_b, tier_counts)

    report = {
        "operational": {
            "final_strategy_for_output_csv": "Strategy B (multi-model cascade)",
            "stage3_primary_model": cfg.stage3_primary_model,
            "provider": provider.name,
            "escalation_tiers": dict(tier_counts),
            "cache": ctx.cache.stats(),
            "dry_run": cfg.dry_run,
        },
        "strategy_a": eval_a,
        "strategy_b": eval_b,
    }
    out_path = Path(cfg.eval_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Evaluation complete -> %s", cfg.eval_report_path)
    console.print(f"[green]report written -> {cfg.eval_report_path}[/green]")
    return 0


def _print_tables(a: dict, b: dict, tiers: dict) -> None:
    t = Table(title="Strategy A vs Strategy B (sample_claims.csv)")
    t.add_column("metric"); t.add_column("Strategy A", justify="right")
    t.add_column("Strategy B", justify="right")
    t.add_row("overall accuracy", f"{a['correct']}/{a['total']}", f"{b['correct']}/{b['total']}")
    for v in VERDICTS:
        t.add_row(f"{v} F1", str(a["per_verdict"][v]["f1"]), str(b["per_verdict"][v]["f1"]))
    console.print(t)
    tt = Table(title="Strategy B escalation tiers")
    tt.add_column("tier"); tt.add_column("claims", justify="right")
    for k in sorted(tiers):
        tt.add_row(str(k), str(tiers[k]))
    console.print(tt)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = CFG
    if args.dry_run:
        cfg.dry_run = True
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
