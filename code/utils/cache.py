"""Two-layer verdict cache — L1 in-memory + L2 disk JSON.

Keyed by SHA-256(claim_text + sorted image hashes). A duplicate claim+image set
returns the cached ClaimOutput with zero API tokens. NOT a semantic cache —
exact-input only, so a reworded fraud with a fake image can never hit it (§26).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from code.pipeline.models import ClaimOutput
from code.utils.logger import get_logger

log = get_logger("pipeline.cache")


class ClaimCache:
    def __init__(
        self, cache_path: str = ".cache/llm_responses.json", namespace: str = "live"
    ) -> None:
        # `namespace` salts every key AND segregates the disk file so dry-run stub
        # verdicts can never be served to a live grading run (and vice versa). A
        # dry-run before grading must not poison real results.
        self._memory: dict[str, dict] = {}
        self._namespace = namespace
        base = Path(cache_path)
        if namespace != "live":
            base = base.with_name(f"{base.stem}.{namespace}{base.suffix}")
        self._path = base
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                self._memory = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                log.warning("Cache file unreadable — starting fresh: %s", self._path)
                self._memory = {}
        self.hits = 0
        self.misses = 0

    def _key(self, claim_text: str, image_hashes: list[str]) -> str:
        raw = self._namespace + "|" + claim_text + "|" + "|".join(sorted(image_hashes))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, claim_text: str, image_hashes: list[str]) -> ClaimOutput | None:
        hit = self._memory.get(self._key(claim_text, image_hashes))
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            return ClaimOutput(**hit)
        except (TypeError, ValueError):
            return None

    def put(self, claim_text: str, image_hashes: list[str], output: ClaimOutput) -> None:
        self._memory[self._key(claim_text, image_hashes)] = output.model_dump()
        try:
            self._path.write_text(json.dumps(self._memory, indent=2))
        except OSError as exc:
            log.warning("Cache write failed (non-fatal): %s", exc)

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._memory), "hits": self.hits, "misses": self.misses}
