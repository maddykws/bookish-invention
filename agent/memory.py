"""
Agent Memory Module — ChromaDB vector store.
Stores successful agent runs and retrieves relevant past context.

Short-term memory: last N tool results (managed in agent.py via context window).
Long-term memory:  this module — persists across runs, retrieved by semantic similarity.

On each run:
  1. Retrieve: find top-k similar past runs → inject as context hint
  2. Store:    after successful run, persist input + answer + score

Usage in agent.py:
    from agent.memory import AgentMemory
    memory = AgentMemory()
    context_hint = memory.retrieve(task)          # before run
    memory.store(task, answer, score, tools_used) # after run
"""

import os
import json
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path(__file__).parent.parent / ".memory"
TOP_K = 3
MIN_SCORE_TO_STORE = 0.6   # only persist runs that passed quality threshold


class AgentMemory:
    def __init__(self):
        MEMORY_DIR.mkdir(exist_ok=True)
        self._collection = None
        self._client = None
        self._ready = False
        self._init()

    def _init(self) -> None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self._client = chromadb.PersistentClient(path=str(MEMORY_DIR))
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self._collection = self._client.get_or_create_collection(
                name="agent_runs",
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            self._ready = True
        except ImportError:
            pass  # memory disabled if chromadb not installed

    @property
    def ready(self) -> bool:
        return self._ready

    def retrieve(self, task: str, top_k: int = TOP_K) -> str:
        """Return a context hint string from similar past successful runs."""
        if not self._ready:
            return ""
        try:
            count = self._collection.count()
            if count == 0:
                return ""

            results = self._collection.query(
                query_texts=[task],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )

            hits = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                similarity = 1 - dist
                if similarity > 0.5:  # only use semantically close memories
                    hits.append(
                        f"[Past run | similarity {similarity:.2f} | score {meta.get('score', '?')}]\n"
                        f"Task: {meta.get('input', '')[:200]}\n"
                        f"Answer: {doc[:300]}"
                    )

            if not hits:
                return ""

            return (
                "\n\n--- Relevant past runs (use as context, not as final answer) ---\n"
                + "\n\n".join(hits)
                + "\n--- End of past context ---"
            )
        except Exception:
            return ""

    def store(self, task: str, answer: str, score: float, tools_used: list[str]) -> None:
        """Persist a successful run to long-term memory."""
        if not self._ready or score < MIN_SCORE_TO_STORE:
            return
        try:
            doc_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            self._collection.upsert(
                ids=[doc_id],
                documents=[answer],
                metadatas=[{
                    "input":      task[:500],
                    "score":      round(score, 3),
                    "tools_used": json.dumps(tools_used),
                    "timestamp":  datetime.now().isoformat(),
                }],
            )
        except Exception:
            pass

    def count(self) -> int:
        if not self._ready:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0
