"""Provider resolution — OpenRouter-first, with per-vendor direct fallback.

All supported vendors expose an OpenAI-compatible endpoint, so a single
`openai` client + a base_url/key/model-id swap covers every provider. Resolution
happens once at Stage 0 and is cached on a ResolvedProvider.

See SYSTEM_DESIGN §4.7.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from code.config import Config
from code.utils.logger import get_logger

log = get_logger("pipeline.providers")


@dataclass
class ResolvedProvider:
    """The active provider after Stage-0 resolution."""
    name: str                       # "openrouter" | "anthropic" | ...
    base_url: str
    api_key: str
    via_openrouter: bool            # True when one key fronts every model

    def model_id(self, openrouter_id: str, cfg: Config) -> str:
        """Translate an OpenRouter model id to the id this provider expects."""
        if self.via_openrouter:
            return openrouter_id
        return cfg.direct_model_ids.get(openrouter_id, openrouter_id)


def _key_present(env_var: str) -> bool:
    val = os.environ.get(env_var, "").strip()
    return bool(val) and not val.startswith("your_")


def resolve_primary_provider(cfg: Config) -> ResolvedProvider:
    """Pick the provider that fronts the Claude verdict/parse/repair roles.

    OpenRouter first (one key, full jury). Else Anthropic-direct. The verdict
    (Opus 4.8) is available on both, so correctness is preserved either way.
    """
    if _key_present("OPENROUTER_API_KEY"):
        base, env = cfg.provider_registry["openrouter"]
        log.info("Provider resolved: OpenRouter (single key, full multi-vendor jury)")
        return ResolvedProvider("openrouter", base, os.environ[env], via_openrouter=True)

    if _key_present("ANTHROPIC_API_KEY"):
        base, env = cfg.provider_registry["anthropic"]
        log.warning(
            "OpenRouter key absent — falling back to Anthropic-direct. "
            "Verdict (Opus 4.8) intact; consensus jury degrades to whatever "
            "vendor keys are present (see §4.7)."
        )
        return ResolvedProvider("anthropic", base, os.environ[env], via_openrouter=False)

    raise RuntimeError(
        "No usable provider key found. Set OPENROUTER_API_KEY (recommended — "
        "one key for every model) or ANTHROPIC_API_KEY (fallback). Copy "
        ".env.example to .env and fill it in."
    )


def resolve_jury_provider(openrouter_id: str, cfg: Config,
                          primary: ResolvedProvider) -> ResolvedProvider | None:
    """Resolve a single jury (cross-check) model to a usable provider.

    On OpenRouter, every juror is reachable through the primary provider. On a
    direct fallback, each juror needs its own vendor key; if absent, the juror is
    dropped (returns None) and consensus degrades safe.
    """
    if primary.via_openrouter:
        return primary

    vendor = _vendor_of(openrouter_id)
    if vendor is None:
        return None
    base, env = cfg.provider_registry.get(vendor, (None, None))
    if base is None or not _key_present(env):
        log.info("Jury model %s dropped (no %s) — consensus degrades safe", openrouter_id, env)
        return None
    return ResolvedProvider(vendor, base, os.environ[env], via_openrouter=False)


def _vendor_of(openrouter_id: str) -> str | None:
    prefix = openrouter_id.split("/", 1)[0]
    return {
        "openai": "openai",
        "x-ai": "xai",
        "google": "google",
        "meta-llama": "groq",     # Groq hosts Llama vision in the direct fallback
        "anthropic": "anthropic",
    }.get(prefix)
