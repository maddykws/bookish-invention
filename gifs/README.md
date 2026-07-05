# GIF rendering framework

Programmatic, schematic animated GIFs for the 60-day GPU/AI LinkedIn series
(`../content/linkedin_gpu_ai_gif_series.md`). Diagram-style motion with a
consistent dark "developer terminal" look, baked-in on-screen text, and a
punchline card — no external services.

## Status: all 60 rendered

`out/` contains all 60 GIFs (`day_01_*.gif` … `day_60_*.gif`), each a looping
animation **strictly under 6 seconds** (5.2–5.45 s). They're produced two ways:

- **Days 1, 2, 4** — bespoke scripts (`day_01/02/04_*.py`), the original
  hand-tuned proofs of concept.
- **Days 3, 5–60** — data-driven configs in `days.py`, rendered by `build.py`
  through the 9 reusable archetypes in `archetypes.py`.

Every loop is `intro(0.25s) + action(3.4s) + hold(0.3s) + punchline(1.5s)`.

### The 9 archetypes (`archetypes.py`)

`bars_race` · `grid_states` · `gauge_spike` · `flow_pipeline` · `ring_net` ·
`compare_paths` · `timeline_track` · `stack_build` · `type_reveal` — mixed
across the 60 days so the series stays visually varied. Each day in `days.py`
maps a topic to one archetype plus its labels, curve/pattern, and punchline.

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
# bespoke days:
python3 day_01_gpu_memory_coalescing.py      # writes out/<name>.gif
# data-driven days (all of 3, 5-60):
python3 build.py                             # render every config
python3 build.py 5 20 44                      # render only these day numbers
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
