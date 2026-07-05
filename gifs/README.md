# GIF rendering framework

Programmatic, schematic animated GIFs for the 60-day GPU/AI LinkedIn series
(`../content/linkedin_gpu_ai_gif_series.md`). Diagram-style motion with a
consistent dark "developer terminal" look, baked-in on-screen text, and a
punchline card — no external services.

## Status: proof of concept

3 of 60 rendered so far, one Python file per day:

| Day | Script | Output |
|----|--------|--------|
| 01 | `day_01_gpu_memory_coalescing.py` | `out/day_01_gpu_memory_coalescing.gif` |
| 02 | `day_02_warp_divergence.py` | `out/day_02_warp_divergence.gif` |
| 04 | `day_04_tensor_cores_unleashed.py` | `out/day_04_tensor_cores_unleashed.gif` |

## Setup

```bash
pip install Pillow numpy        # the only dependencies
```

Fonts: JetBrains Mono (bundled with the Claude skills image at
`/mnt/skills/examples/canvas-design/canvas-fonts`). Point `giflib.FONT_DIR` at
any directory with `JetBrainsMono-Regular.ttf` / `-Bold.ttf` if rendering
elsewhere.

## Render

```bash
python3 day_01_gpu_memory_coalescing.py     # writes out/<name>.gif
# render everything:
for f in day_*.py; do python3 "$f"; done
```

## Framework (`giflib.py`)

- **`Canvas`** — per-frame drawing: `grid()`, `panel()`, `text*()`, `dot()`,
  `chip()`, `arrow()`, `meter()`.
- **`title_bar()`** — the faux terminal window bar (traffic-light dots) shared
  by every GIF.
- **`punch_overlay()`** — fades the punchline card over the final frames.
- **`Timeline`** — sequence scenes in seconds; each scene is
  `fn(canvas, local_t, global_t)` where `local_t`/`global_t` are 0..1.
- **Easing**: `ease_in_out`, `ease_out`, `lerp`, `lerp_color`, `clamp`.
- **`render(frames, path, fps)`** — writes a looping, optimized GIF.

### Conventions

- 900×600, 20 fps, ~6–7 s loops.
- Palette in `giflib.py`: green = good/throughput, red = bad/stall/latency,
  amber = warning, blue/purple = data, NVIDIA green for stack callouts.
- **No emoji** in rendered text — the mono font has no emoji glyphs (they
  render as tofu). Use color, shape, and labels instead.

## Adding a new day

Copy an existing script, keep the `Timeline` structure, and implement the
scene functions from the matching script in the content markdown. The three
current scripts cover the common patterns: side-by-side comparison (day 1),
a state grid that animates (day 2), and dueling fill/throughput meters
(day 4).
