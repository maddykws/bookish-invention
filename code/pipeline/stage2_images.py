"""Stage 2 — local image preprocessing (no API, no GPU required).

Per image: resolve path, decode, SHA-256, blank/blur/FFT/EXIF checks, resize +
base64 for the API. Produces ProcessedImage records and the data URLs Stage 3
will send. All checks are advisory pre-flags; the verdict is the model's.
"""

from __future__ import annotations

from pathlib import Path

from code.config import Config
from code.pipeline.models import ClaimRow, ProcessedImage
from code.utils import image_utils as iu
from code.utils.injection import screen_text
from code.utils.logger import get_logger

log = get_logger("pipeline.stage2")


def process_images(
    claim: ClaimRow, cfg: Config, seen_hashes: dict[str, str],
) -> tuple[list[ProcessedImage], list[str], list[str], bool]:
    """Returns (processed, data_urls, preflags, exif_injection_detected).

    `seen_hashes` maps sha256 -> first user_id seen (cross-claim duplicate fraud).
    """
    processed: list[ProcessedImage] = []
    data_urls: list[str] = []
    preflags: set[str] = set()
    exif_injection = False

    for raw_path in claim.image_path_list:
        image_id = iu.extract_image_id(raw_path)
        resolved = iu.resolve_image_path(raw_path)
        flags: list[str] = []

        if not resolved.exists():
            log.warning("Image not found: %s (claim %s)", raw_path, claim.user_id)
            processed.append(_missing(resolved, image_id))
            continue

        img = iu.load_image(resolved)
        if img is None:
            processed.append(_missing(resolved, image_id, exists=True))
            preflags.add("blurry_image")
            continue

        sha = iu.compute_sha256(resolved)
        dup_of = seen_hashes.get(sha)
        if dup_of is not None and dup_of != claim.user_id:
            flags.append("non_original_image")
            preflags.add("non_original_image")
            preflags.add("user_history_risk")
        else:
            seen_hashes.setdefault(sha, claim.user_id)

        blank = iu.is_blank(img)
        bscore = iu.blur_score(img)
        blurry = bscore < cfg.blur_threshold
        if blank:
            flags.append("blank")
        if blurry:
            flags.append("blurry_image")
            preflags.add("blurry_image")

        fft_ratio = iu.fft_high_freq_ratio(img)
        if fft_ratio > 0.92:
            flags.append("possible_manipulation")
            preflags.add("possible_manipulation")

        exif_dt = iu.exif_datetime(img)
        exif_text = iu.exif_text_fields(img)
        if exif_text:
            _, hit = screen_text(exif_text)
            if hit:
                exif_injection = True
                preflags.add("text_instruction_present")

        usable = not (blank or blurry)
        processed.append(ProcessedImage(
            path=resolved, image_id=image_id, exists=True, valid=usable,
            is_blank=blank, is_blurry=blurry, blur_score=bscore, sha256=sha,
            is_duplicate=dup_of is not None, duplicate_of_claim=dup_of,
            yolo_detected_object=None, clip_similarity=0.0, exif_date=exif_dt,
            resized_path=None, local_vlm_damage=None, flags=flags,
        ))
        if usable:
            data_urls.append(iu.image_data_url(img, cfg.resize_max_px))

    return processed, data_urls, sorted(preflags), exif_injection


def _missing(path: Path, image_id: str, exists: bool = False) -> ProcessedImage:
    return ProcessedImage(
        path=path, image_id=image_id, exists=exists, valid=False,
        is_blank=False, is_blurry=False, blur_score=0.0, sha256="",
        is_duplicate=False, duplicate_of_claim=None, yolo_detected_object=None,
        clip_similarity=0.0, exif_date=None, resized_path=None,
        local_vlm_damage=None, flags=["cropped_or_obstructed"],
    )
