"""ChromaDB agent memory — cross-claim fraud detection (§4.4b).

Embeds each claim transcript and flags semantic near-duplicates from OTHER users
in the same batch (coordinated-fraud signal). Fully import-guarded and wrapped:
if chromadb is unavailable or errors, this degrades to a no-op and the batch
continues unaffected.
"""

from __future__ import annotations

from code.config import Config
from code.utils.logger import get_logger

log = get_logger("pipeline.chroma")


class ClaimMemory:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._coll = None
        self._n = 0
        if cfg.dry_run or not cfg.fraud_checks:
            return
        try:
            import chromadb
            client = chromadb.EphemeralClient()
            self._coll = client.get_or_create_collection(cfg.chroma_collection_name)
            log.info("ChromaDB claim-memory ready (cross-claim fraud detection).")
        except Exception as exc:  # noqa: BLE001 — optional component, never fatal
            log.warning("ChromaDB unavailable (%s) — fraud-memory disabled.", exc)
            self._coll = None

    def check_and_add(self, user_id: str, claim_text: str) -> list[str]:
        """Return fraud flags if a near-duplicate from another user exists, then
        add this claim to the memory. No-op (returns []) when disabled."""
        if self._coll is None or not claim_text.strip():
            return []
        flags: list[str] = []
        try:
            if self._n > 0:
                res = self._coll.query(query_texts=[claim_text], n_results=3)
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]
                for meta, dist in zip(metas, dists):
                    other = (meta or {}).get("user_id")
                    sim = 1.0 - float(dist)
                    if other and other != user_id and sim >= self.cfg.chroma_fraud_similarity_threshold:
                        flags = ["claim_mismatch", "user_history_risk"]
                        log.info("Cross-claim fraud signal: %s ~ %s (sim %.2f)",
                                 user_id, other, sim)
                        break
            self._coll.add(documents=[claim_text], ids=[f"{user_id}-{self._n}"],
                           metadatas=[{"user_id": user_id}])
            self._n += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("ChromaDB query/add failed (non-fatal): %s", exc)
        return flags
