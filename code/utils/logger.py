"""Centralised logging — Rich console + file handler. Zero print() statements."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    # ── File handler (full detail) ─────────────────────────────────────────
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "pipeline.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(fh)

    # ── Mandatory AGENTS.md log ────────────────────────────────────────────
    agents_log = Path(os.path.expanduser("~")) / "hackerrank_orchestrate" / "log.txt"
    agents_log.parent.mkdir(parents=True, exist_ok=True)
    agents_fh = logging.FileHandler(agents_log, mode="a")
    agents_fh.setLevel(logging.INFO)
    agents_fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(agents_fh)

    # ── Rich console handler ───────────────────────────────────────────────
    rh = RichHandler(console=console, show_path=False, rich_tracebacks=True)
    rh.setLevel(logging.INFO)
    logger.addHandler(rh)

    return logger
