"""Single OpenAI-compatible client = the API spine for every vendor.

Wraps the `openai` SDK pointed at the resolved provider (OpenRouter or a direct
vendor). Handles: JSON-mode requests with optional images, prompt caching on the
static prefix, generation-id capture, specific exception handling, and a
deterministic `dry_run` mode so the whole pipeline runs offline (no network, no
tokens) for plumbing tests.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from code.config import Config
from code.providers import ResolvedProvider
from code.utils.logger import get_logger

log = get_logger("pipeline.client")


@dataclass
class LLMResult:
    content: dict[str, Any]          # parsed JSON object from the model
    raw_text: str
    model: str
    generation_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    ok: bool = True
    error: str = ""


@dataclass
class LLMClient:
    cfg: Config
    provider: ResolvedProvider
    _client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.cfg.dry_run:
            log.warning("DRY-RUN mode — no network calls; deterministic stub verdicts.")
            return
        from openai import OpenAI  # imported lazily so dry-run needs no network/SDK init
        headers = {}
        if self.provider.via_openrouter:
            headers = {
                "HTTP-Referer": self.cfg.openrouter_site_url,
                "X-Title": self.cfg.openrouter_app_title,
            }
        self._client = OpenAI(
            base_url=self.provider.base_url,
            api_key=self.provider.api_key,
            timeout=self.cfg.request_timeout_s,
            default_headers=headers,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def complete_json(
        self,
        *,
        openrouter_model: str,
        system_prefix: str,
        user_text: str,
        image_data_urls: list[str] | None = None,
        max_tokens: int = 800,
        cache_prefix: bool = False,
        x_title: str = "",
    ) -> LLMResult:
        """One JSON-returning call. `system_prefix` is the (cacheable) static block;
        `user_text` + images are the per-claim volatile suffix."""
        model = self.provider.model_id(openrouter_model, self.cfg)

        if self.cfg.dry_run:
            return self._stub(model, system_prefix, user_text, image_data_urls)

        system_block = self._system_block(system_prefix, cache_prefix)
        user_content = self._user_content(user_text, image_data_urls or [])

        extra_headers = {"X-Title": x_title} if (x_title and self.provider.via_openrouter) else None
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[system_block, {"role": "user", "content": user_content}],
                temperature=self.cfg.temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_headers=extra_headers,
            )
        except Exception as exc:  # noqa: BLE001 — see _classify for typed handling
            return self._on_error(model, exc)

        return self._parse_response(model, resp)

    # ── internals ─────────────────────────────────────────────────────────────

    def _system_block(self, prefix: str, cache_prefix: bool) -> dict[str, Any]:
        if cache_prefix:
            # Anthropic prompt caching via OpenRouter passthrough; on non-caching
            # providers this content shape is still accepted (cache_control ignored).
            return {
                "role": "system",
                "content": [
                    {"type": "text", "text": prefix,
                     "cache_control": {"type": "ephemeral"}},
                ],
            }
        return {"role": "system", "content": prefix}

    @staticmethod
    def _user_content(text: str, image_urls: list[str]) -> Any:
        if not image_urls:
            return text
        # image-first ordering (§8.3): images precede the claim text
        blocks: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": u}} for u in image_urls
        ]
        blocks.append({"type": "text", "text": text})
        return blocks

    def _parse_response(self, model: str, resp: Any) -> LLMResult:
        try:
            text = resp.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return LLMResult({}, "", model, ok=False, error="empty response")
        usage = getattr(resp, "usage", None)
        gen_id = getattr(resp, "id", "") or ""
        cache_read = 0
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        if details is not None:
            cache_read = getattr(details, "cached_tokens", 0) or 0
        parsed = _extract_json(text)
        return LLMResult(
            content=parsed,
            raw_text=text,
            model=model,
            generation_id=gen_id,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            cache_read_tokens=cache_read,
            ok=bool(parsed),
            error="" if parsed else "unparseable JSON",
        )

    def _on_error(self, model: str, exc: Exception) -> LLMResult:
        kind = type(exc).__name__
        # Typed where the SDK exposes it; the names are stable across openai>=1.x
        if kind in ("APITimeoutError", "APIConnectionError"):
            log.warning("%s on %s — caller applies safe defaults", kind, model)
        elif kind == "RateLimitError":
            log.warning("Rate limit on %s — provider/fallback should absorb", model)
        elif kind in ("AuthenticationError", "PermissionDeniedError"):
            log.error("Auth/permission error on %s: %s", model, exc)
        else:
            log.error("API error on %s (%s): %s", model, kind, exc)
        return LLMResult({}, "", model, ok=False, error=f"{kind}: {exc}")

    # ── dry-run stub (offline plumbing test) ──────────────────────────────────

    def _stub(self, model: str, system_prefix: str, user_text: str,
              images: list[str] | None) -> LLMResult:
        """Deterministic fake response keyed on the input, so the full pipeline
        runs end-to-end with no network. Routed on the SYSTEM prompt (stable),
        not the user text. Shape matches what each stage expects."""
        has_image = bool(images)
        is_stage1 = system_prefix.startswith("You extract a structured damage claim")
        t = user_text.lower()
        if is_stage1:  # Stage 1 schema
            obj = next((o for o in ("car", "laptop", "package") if o in t), "car")
            part = {"car": "rear_bumper", "laptop": "screen", "package": "box"}[obj]
            payload = {
                "claim_text": f"{obj} {part} damage claim", "claim_object": obj,
                "claimed_part": part,
                "issue_family": "dent", "claim_language": "en", "confidence": 0.7,
            }
        else:  # Stage 3 / jury schema
            status = "supported" if has_image else "not_enough_information"
            payload = {
                "claim_status": status,
                "confidence": 0.88 if has_image else 0.3,
                "evidence_standard_met": has_image,
                "evidence_standard_met_reason": "stub: image present" if has_image else "stub: no image",
                "issue_type": "dent" if has_image else "unknown",
                "object_part": "rear_bumper" if has_image else "unknown",
                "severity": "medium" if has_image else "unknown",
                "risk_flags": "none" if has_image else "manual_review_required",
                "supporting_image_ids": "img_1" if has_image else "none",
                "valid_image": has_image,
                "claim_status_justification": (
                    "img_1 shows the claimed damage (stub)." if has_image
                    else "No usable image (stub)."
                ),
                "reasoning_summary": "stub",
            }
        return LLMResult(payload, json.dumps(payload), model, generation_id="dryrun",
                         prompt_tokens=0, completion_tokens=0, ok=True)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of model text — handles ```json fences and stray prose."""
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass
    return {}
