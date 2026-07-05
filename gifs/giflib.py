"""
giflib — a small, reusable framework for rendering schematic, engineering-style
animated GIFs for the 60-day GPU/AI LinkedIn series.

Design goals:
- One consistent dark "developer terminal" look across all 60 GIFs.
- Diagram-style motion (meters, tiles, threads) with baked-in on-screen text and
  a punchline card, matching the scripts in content/linkedin_gpu_ai_gif_series.md.
- Pure Pillow + numpy, no external services. Outputs looping .gif files.

Public surface:
- Canvas: a per-frame drawing helper (colors, fonts, rounded rects, text, meters).
- render(frames, path, fps): write a looping GIF from a list of PIL images.
- Easing helpers and a Timeline for sequencing scenes.
"""

from __future__ import annotations
import math
import os
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Palette — a calm dark theme with a small, deliberate accent set.
# ---------------------------------------------------------------------------
BG        = (13, 17, 23)      # near-black GitHub-dark background
PANEL     = (22, 27, 34)      # slightly lifted panel
PANEL_HI  = (33, 39, 48)      # hovered/active panel
GRID      = (30, 36, 44)      # faint grid lines
INK       = (230, 237, 243)   # primary text
MUTED     = (139, 148, 158)   # secondary text
GREEN     = (63, 185, 80)     # good / throughput
RED       = (248, 81, 73)     # bad / stall / latency
AMBER     = (210, 153, 34)    # warning
BLUE      = (56, 139, 253)    # accent / data
PURPLE    = (163, 113, 247)   # accent 2
NVIDIA    = (118, 185, 0)     # NVIDIA green for stack callouts

FONT_DIR = "/mnt/skills/examples/canvas-design/canvas-fonts"
_MONO_R  = os.path.join(FONT_DIR, "JetBrainsMono-Regular.ttf")
_MONO_B  = os.path.join(FONT_DIR, "JetBrainsMono-Bold.ttf")

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = ("b" if bold else "r", size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(_MONO_B if bold else _MONO_R, size)
    return _font_cache[key]


# ---------------------------------------------------------------------------
# Easing
# ---------------------------------------------------------------------------
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def ease_in_out(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    t = clamp(t)
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))


# ---------------------------------------------------------------------------
# Canvas — thin convenience wrapper over ImageDraw for one frame.
# ---------------------------------------------------------------------------
@dataclass
class Canvas:
    W: int
    H: int

    def __post_init__(self):
        self.img = Image.new("RGB", (self.W, self.H), BG)
        self.d = ImageDraw.Draw(self.img)

    # -- background helpers --------------------------------------------------
    def grid(self, step=40, color=GRID):
        for x in range(0, self.W, step):
            self.d.line([(x, 0), (x, self.H)], fill=color, width=1)
        for y in range(0, self.H, step):
            self.d.line([(0, y), (self.W, y)], fill=color, width=1)

    def panel(self, box, radius=14, fill=PANEL, outline=None, width=1):
        self.d.rounded_rectangle(box, radius=radius, fill=fill,
                                 outline=outline, width=width)

    # -- text ---------------------------------------------------------------
    def text(self, xy, s, size=22, bold=False, fill=INK, anchor="la"):
        self.d.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)

    def text_center(self, cx, y, s, size=22, bold=False, fill=INK):
        self.d.text((cx, y), s, font=font(size, bold), fill=fill, anchor="ma")

    def measure(self, s, size=22, bold=False):
        b = self.d.textbbox((0, 0), s, font=font(size, bold))
        return b[2] - b[0], b[3] - b[1]

    # -- primitives ---------------------------------------------------------
    def dot(self, cx, cy, r, fill):
        self.d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)

    def chip(self, cx, cy, w, h, fill, outline=None, radius=6):
        self.d.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                                 radius=radius, fill=fill, outline=outline, width=2)

    def arrow(self, p0, p1, color=MUTED, width=3, head=9):
        self.d.line([p0, p1], fill=color, width=width)
        ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        for da in (math.radians(150), math.radians(-150)):
            self.d.line([p1, (p1[0] + head * math.cos(ang + da),
                              p1[1] + head * math.sin(ang + da))],
                        fill=color, width=width)

    # -- a labeled horizontal meter (0..1) ----------------------------------
    def meter(self, box, value, label, color, track=PANEL_HI, size=18):
        x0, y0, x1, y1 = box
        self.text((x0, y0 - size - 6), label, size=size, bold=True, fill=MUTED)
        self.d.rounded_rectangle(box, radius=(y1 - y0) // 2, fill=track)
        w = (x1 - x0) * clamp(value)
        if w > 2:
            self.d.rounded_rectangle([x0, y0, x0 + w, y1],
                                     radius=(y1 - y0) // 2, fill=color)


def title_bar(c: Canvas, title, subtitle=None):
    """A faux window title bar with traffic-light dots — the series signature."""
    c.panel([0, 0, c.W, 46], radius=0, fill=PANEL)
    for i, col in enumerate((RED, AMBER, GREEN)):
        c.dot(24 + i * 22, 23, 6, col)
    c.text((104, 23), title, size=18, bold=True, fill=MUTED, anchor="lm")
    if subtitle:
        c.text((c.W - 20, 23), subtitle, size=15, fill=MUTED, anchor="rm")


def punch_overlay(c: Canvas, text, alpha=0.0, sub=None):
    """Fade a punchline card over the frame. alpha 0..1."""
    if alpha <= 0.01:
        return
    a = clamp(alpha)
    veil = Image.new("RGBA", (c.W, c.H), (5, 7, 10, int(205 * a)))
    c.img.paste(Image.alpha_composite(c.img.convert("RGBA"), veil).convert("RGB"),
                (0, 0))
    c.d = ImageDraw.Draw(c.img)
    # slide up a touch as it fades in
    cy = int(lerp(c.H / 2 + 24, c.H / 2, ease_out(a)))
    fill = lerp_color(BG, INK, a)
    c.text_center(c.W // 2, cy - 30, text, size=30, bold=True, fill=fill)
    if sub:
        c.text_center(c.W // 2, cy + 14, sub, size=18, bold=False,
                      fill=lerp_color(BG, MUTED, a))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def render(frames, path, fps=20, loop=0):
    """Write a list of PIL.Image frames to a looping GIF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    duration = int(1000 / fps)
    # Adaptive palette per frame keeps gradients/meters clean.
    conv = [f.convert("P", palette=Image.ADAPTIVE, colors=256) for f in frames]
    conv[0].save(path, save_all=True, append_images=conv[1:],
                 duration=duration, loop=loop, disposal=2, optimize=True)
    return path


class Timeline:
    """Sequence named scenes measured in seconds; call fn(c, local_t, global_t)."""
    def __init__(self, W, H, fps):
        self.W, self.H, self.fps = W, H, fps
        self.scenes = []  # (duration_seconds, fn)

    def scene(self, seconds, fn):
        self.scenes.append((seconds, fn))
        return self

    def build(self):
        frames = []
        total = sum(s for s, _ in self.scenes)
        elapsed = 0.0
        for seconds, fn in self.scenes:
            n = max(1, int(round(seconds * self.fps)))
            for i in range(n):
                lt = i / max(1, n - 1) if n > 1 else 1.0
                gt = (elapsed + i / self.fps) / total
                c = Canvas(self.W, self.H)
                fn(c, lt, gt)
                frames.append(c.img)
            elapsed += seconds
        return frames
