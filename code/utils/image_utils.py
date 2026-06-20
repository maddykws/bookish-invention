"""Local image preprocessing — no GPU, no API. Pillow + NumPy + OpenCV-headless.

Path resolution, SHA-256, blank/blur/FFT checks, EXIF, resize, base64 encode.
Everything here is torch-free and runs on the grader's CPU.
"""

from __future__ import annotations

import base64
import hashlib
import io
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ExifTags

from code.utils.logger import get_logger

log = get_logger("pipeline.image")

DATASET_ROOT = Path("dataset")


# ── Path resolution (the dataset/ prefix silent-failure guard, §4.2) ──────────

def resolve_image_path(csv_path: str) -> Path:
    """Resolve a CSV image_paths entry to an on-disk file.

    CSV stores 'images/test/case_001/img_1.jpg' (no 'dataset/' prefix); the file
    lives at 'dataset/images/test/case_001/img_1.jpg'. Tries dataset/<path>,
    then the path as-is, then verbatim. Returns the canonical guess on miss so
    the exists=False path is taken downstream (flagged, not crashed).
    """
    raw = csv_path.strip().lstrip("./")
    candidates = [
        DATASET_ROOT / raw,
        Path(raw),
        Path(csv_path.strip()),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def extract_image_id(path: str) -> str:
    """'images/test/case_001/img_1.jpg' -> 'img_1' (claim-local id, not a path)."""
    return Path(path).stem


# ── Hashing + decode ──────────────────────────────────────────────────────────

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image(path: Path) -> Image.Image | None:
    """Decode an image, normalizing to RGB. Returns None on any decode failure."""
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except (FileNotFoundError, OSError, ValueError) as exc:
        log.warning("Image decode failed: %s (%s)", path, exc)
        return None


# ── Quality / authenticity checks (local, 0 tokens) ───────────────────────────

def is_blank(img: Image.Image) -> bool:
    """All-black / all-white / single-color: near-zero pixel variance."""
    arr = np.asarray(img, dtype=np.float32)
    return bool(arr.std() < 3.0)


def blur_score(img: Image.Image) -> float:
    """Variance of the Laplacian — low = blurry. Uses OpenCV if available, else NumPy."""
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    try:
        import cv2
        return float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
    except ImportError:
        # NumPy fallback Laplacian kernel
        lap = (
            -4 * gray
            + np.roll(gray, 1, 0) + np.roll(gray, -1, 0)
            + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)
        )
        return float(lap.var())


def fft_high_freq_ratio(img: Image.Image) -> float:
    """Fraction of spectral energy in high frequencies — a coarse signal for
    AI-generated / heavily-perturbed images. Advisory only, never decisive."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    low = mag[cy - r:cy + r, cx - r:cx + r].sum()
    total = mag.sum() + 1e-9
    return float(1.0 - low / total)


def exif_datetime(img: Image.Image) -> datetime | None:
    try:
        raw = img.getexif()
        if not raw:
            return None
        tagmap = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
        for key in ("DateTimeOriginal", "DateTime"):
            if key in tagmap:
                return datetime.strptime(str(tagmap[key]), "%Y:%m:%d %H:%M:%S")
    except (ValueError, KeyError, OSError):
        return None
    return None


def exif_text_fields(img: Image.Image) -> str:
    """User-controlled EXIF text (ImageDescription/UserComment/Artist/Copyright).
    Returned joined so the caller can injection-screen it before any LLM sees it."""
    parts: list[str] = []
    try:
        raw = img.getexif()
        tagmap = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
        for key in ("ImageDescription", "UserComment", "Artist", "Copyright"):
            if key in tagmap and tagmap[key]:
                parts.append(str(tagmap[key]))
    except (OSError, ValueError):
        pass
    return " | ".join(parts)


# ── Resize + base64 (vision token optimization, §7.3) ─────────────────────────

def resize_for_api(img: Image.Image, max_px: int = 768) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    scale = max_px / float(max(w, h))
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def to_base64_jpeg(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_data_url(img: Image.Image, max_px: int = 768) -> str:
    """Resize -> JPEG -> data URL for the OpenAI-compatible image_url content block."""
    b64 = to_base64_jpeg(resize_for_api(img, max_px))
    return f"data:image/jpeg;base64,{b64}"
