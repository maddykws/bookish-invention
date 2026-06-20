"""Stage-0 capability probe — graceful degradation, never a hard dependency.

The full cloud pipeline runs torch-free. Local-vision acceleration (YOLO/CLIP/
local VLM) is optional; if its library is absent, the pipeline degrades to
OpenRouter/cloud vision and records the degradation rather than crashing (§23.3b).
"""

from __future__ import annotations

from code.utils.logger import get_logger

log = get_logger("pipeline.capabilities")


def detect_capabilities() -> dict[str, bool]:
    caps = {
        "torch": False, "cuda": False, "ultralytics": False,
        "transformers": False, "sentence_transformers": False,
    }
    try:
        import torch
        caps["torch"] = True
        caps["cuda"] = bool(torch.cuda.is_available())
    except ImportError:
        pass
    for name in ("ultralytics", "transformers", "sentence_transformers"):
        try:
            __import__(name)
            caps[name] = True
        except ImportError:
            pass

    if not caps["cuda"]:
        log.info("No CUDA — local VLM (Stage 2.5) skipped; cloud vision handles all images.")
    if not caps["ultralytics"]:
        log.info("No ultralytics — YOLO pre-detect skipped (CLIP/cloud still run).")
    if not caps["transformers"]:
        log.info("No transformers — CLIP (Stage 2g) skipped.")
    log.info("Capabilities: %s", caps)
    return caps
