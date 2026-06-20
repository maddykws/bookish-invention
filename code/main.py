"""Entry point — read dataset/claims.csv, produce output.csv (§23.8).

  python code/main.py                 # full run (needs OPENROUTER_API_KEY or ANTHROPIC_API_KEY)
  python code/main.py --dry-run       # offline plumbing test (no network, no tokens)
  python code/main.py --limit 5       # first N claims only
  python code/main.py --input path.csv --output out.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow `python code/main.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from code.capabilities import detect_capabilities
from code.config import CFG, Config
from code.pipeline.chroma_memory import ClaimMemory
from code.pipeline.loaders import (
    load_evidence_requirements, load_user_history,
)
from code.pipeline.models import ClaimRow, ClaimRowValidator, safe_defaults
from code.pipeline.orchestrator import BatchContext, process_claim
from code.pipeline.output_writer import (
    assert_output_matches_sample_header, write_output,
)
from code.providers import ResolvedProvider, resolve_primary_provider
from code.utils.cache import ClaimCache
from code.utils.llm_client import LLMClient
from code.utils.logger import get_logger

import pandas as pd

log = get_logger("pipeline.main")
console = Console()


def _resolve_provider(cfg: Config) -> ResolvedProvider:
    if cfg.dry_run:
        return ResolvedProvider("dry-run", cfg.openrouter_base_url, "dry-run", via_openrouter=True)
    return resolve_primary_provider(cfg)


def _safe_for_raw(raw: dict, reason: str):
    """Safe-default output for a row that failed validation (preserve row count)."""
    obj = str(raw.get("claim_object", "car")).strip().lower()
    if obj not in ("car", "laptop", "package"):
        obj = "car"
    stub = ClaimRow(
        user_id=str(raw.get("user_id", "")).strip() or "unknown",
        image_paths=str(raw.get("image_paths", "")).strip(),
        user_claim=str(raw.get("user_claim", "")).strip() or "n/a",
        claim_object=obj,  # type: ignore[arg-type]
    )
    return safe_defaults(stub, reason)


def run(cfg: Config, input_path: str, output_path: str, limit: int | None) -> int:
    console.rule("[bold green]Multi-Modal Evidence Review — Stage 0")
    detect_capabilities()
    provider = _resolve_provider(cfg)
    log.info("Active provider: %s (via_openrouter=%s)", provider.name, provider.via_openrouter)

    assert_output_matches_sample_header(cfg.sample_claims_path)

    p = Path(input_path)
    if not p.exists():
        log.error("Input file not found: %s", input_path)
        console.print(f"[red]No input file at {input_path}. Place the dataset and retry.[/red]")
        return 1
    df = pd.read_csv(p, dtype=str, keep_default_na=False).fillna("")
    raw_rows = df.to_dict(orient="records")
    if limit:
        raw_rows = raw_rows[:limit]
    log.info("Processing %d claim rows from %s", len(raw_rows), input_path)

    ctx = BatchContext(
        cfg=cfg, provider=provider, client=LLMClient(cfg, provider),
        history=load_user_history(cfg.user_history_path),
        requirements=load_evidence_requirements(cfg.evidence_req_path),
        cache=ClaimCache(cfg.cache_path), memory=ClaimMemory(cfg),
    )

    outputs = []
    n_safe = 0
    for i, raw in enumerate(raw_rows, 1):
        claim, errors = ClaimRowValidator.validate(raw)
        if claim is None:
            outputs.append(_safe_for_raw(raw, "; ".join(errors)))
            n_safe += 1
        else:
            out, metrics = process_claim(claim, ctx)
            outputs.append(out)
            if metrics.safe_defaults_applied:
                n_safe += 1
        if i % 25 == 0:
            log.info("… %d/%d processed", i, len(raw_rows))

    write_output(outputs, len(raw_rows), output_path)
    _summary(outputs, ctx, n_safe, output_path)
    return 0


def _summary(outputs, ctx: BatchContext, n_safe: int, output_path: str) -> None:
    counts = {"supported": 0, "contradicted": 0, "not_enough_information": 0}
    for o in outputs:
        counts[o.claim_status] = counts.get(o.claim_status, 0) + 1
    t = Table(title="Batch Summary")
    t.add_column("metric"); t.add_column("value", justify="right")
    t.add_row("total rows", str(len(outputs)))
    t.add_row("supported", str(counts["supported"]))
    t.add_row("contradicted", str(counts["contradicted"]))
    t.add_row("not_enough_information", str(counts["not_enough_information"]))
    t.add_row("safe-default rows", str(n_safe))
    t.add_row("cache", str(ctx.cache.stats()))
    t.add_row("output", output_path)
    console.print(t)


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Damage-claim verification pipeline")
    ap.add_argument("--input", default=CFG.claims_path)
    ap.add_argument("--output", default=CFG.output_path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="offline plumbing test: deterministic stubs, no network/tokens")
    args = ap.parse_args()

    cfg = CFG
    if args.dry_run:
        cfg.dry_run = True
    sys.exit(run(cfg, args.input, args.output, args.limit))


if __name__ == "__main__":
    main()
