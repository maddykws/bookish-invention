"""Pre-LLM injection screener.

Screens user-controlled text fields before they reach any LLM call.
Applied to: transcript, EXIF text fields, history_summary.
"""

from __future__ import annotations

import re

INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+|previous\s+|prior\s+)?instructions",
    r"you\s+are\s+now",
    r"disregard\s+(all\s+)?",
    r"new\s+(system\s+)?prompt",
    r"<\|system\|>",
    r"\[INST\]",
    r"###\s*instruction",
    r"assistant:\s*approved",
    r"output\s*[:{]\s*[\"']?supported[\"']?",
    r"forget\s+(everything|all|your)",
    r"override\s+(all\s+)?",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"DAN\s+mode",
    r"pretend\s+(you\s+are|to\s+be)",
]

_COMPILED: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
]


def screen_text(text: str) -> tuple[str, bool]:
    """Return (sanitized_text, was_injection_detected).

    Redacts offending segments; never raises. Claim still processes.
    """
    if not text:
        return text, False

    injection_found = False
    sanitized = text
    for pattern in _COMPILED:
        if pattern.search(sanitized):
            injection_found = True
            sanitized = pattern.sub("[REDACTED]", sanitized)

    return sanitized, injection_found
