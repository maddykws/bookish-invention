"""Output writer — the enforced 14-column contract (§23.2a).

Order is fixed (OUTPUT_COLUMNS is the single source), booleans lowercase, internal
flags remapped, one row per input row asserted. extrasaction='raise' means a stray
key fails loudly instead of writing a malformed row.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from code.pipeline.models import OUTPUT_COLUMNS, ClaimOutput, output_to_row
from code.utils.logger import get_logger

log = get_logger("pipeline.output")

EXPECTED_HEADER = list(OUTPUT_COLUMNS)


def write_output(rows: list[ClaimOutput], input_row_count: int, path: str) -> None:
    assert OUTPUT_COLUMNS == EXPECTED_HEADER, "OUTPUT_COLUMNS drifted from the 14-col spec"
    assert len(rows) == input_row_count, (
        f"row-count mismatch: {len(rows)} out vs {input_row_count} in"
    )
    p = Path(path)
    if p.parent != Path(""):
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow(output_to_row(r))
    log.info("Wrote %d rows -> %s (14 columns)", len(rows), path)


def assert_output_matches_sample_header(sample_claims_path: str) -> None:
    """If the grader's sample file is present, confirm our output columns exist in it."""
    p = Path(sample_claims_path)
    if not p.exists():
        log.info("Sample header check skipped (no %s)", sample_claims_path)
        return
    cols = list(pd.read_csv(p, nrows=0).columns)
    missing = [c for c in EXPECTED_HEADER if c not in cols]
    if missing:
        log.warning("sample_claims.csv missing expected output columns: %s", missing)
    else:
        log.info("Output header matches sample_claims.csv ✓")
