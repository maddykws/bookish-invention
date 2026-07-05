"""
Day 2 — Warp divergence.
32 lanes in lockstep. An `if (threadIdx.x & 1)` drops. Even lanes execute while
odd lanes are masked (dimmed), then they swap. A clock shows tempo halving:
"2 passes for 1 warp". Punchline: "Warp divergence: where your threads take
turns being useless."
"""
import os
import giflib as g
from giflib import Canvas, ease_in_out, ease_out, clamp, lerp, lerp_color

W, H, FPS = 900, 600, 20
LANES = 32
COLS = 16                      # 2 rows of 16
CELL = 44
GAP = 6
GRID_W = COLS * (CELL + GAP) - GAP
X0 = (W - GRID_W) // 2
Y0 = 210


def lane_pos(i):
    row, col = divmod(i, COLS)
    x = X0 + col * (CELL + GAP)
    y = Y0 + row * (CELL + GAP)
    return x, y


def draw_lanes(c: Canvas, active_mask, exec_flash):
    """active_mask[i] in {'run','wait','idle'}; exec_flash 0..1 pulse."""
    for i in range(LANES):
        x, y = lane_pos(i)
        state = active_mask[i]
        if state == "run":
            base = lerp_color(g.GREEN, g.INK, 0.0)
            fill = lerp_color(g.PANEL_HI, g.GREEN, 0.5 + 0.5 * exec_flash)
            txt = g.BG
        elif state == "wait":
            fill = g.PANEL_HI
            txt = g.MUTED
        else:  # idle (pre-branch, uniform)
            fill = lerp_color(g.PANEL_HI, g.BLUE, 0.55)
            txt = g.BG
        c.d.rounded_rectangle([x, y, x + CELL, y + CELL], radius=7, fill=fill)
        c.text((x + CELL/2, y + CELL/2), str(i), size=13, bold=True,
               fill=txt, anchor="mm")


def header(c: Canvas, code_hi=False):
    c.grid()
    g.title_bar(c, "cuda ~ warp divergence", "1 warp = 32 lanes")
    # code line
    c.panel([X0, 120, X0 + GRID_W, 168], radius=8, fill=g.PANEL)
    col = g.AMBER if code_hi else g.MUTED
    c.text((X0 + 18, 144), "if (threadIdx.x & 1) {  ... } else {  ... }",
           size=20, bold=True, fill=col, anchor="lm")


def clockface(c: Canvas, passes, cx=W-90, cy=150):
    c.text((cx, cy-22), "SCHEDULE", size=13, bold=True, fill=g.MUTED, anchor="mm")
    c.text((cx, cy+6), f"{passes} pass" + ("es" if passes != 1 else ""),
           size=20, bold=True, fill=(g.RED if passes > 1 else g.GREEN),
           anchor="mm")


def scene_lockstep(c: Canvas, lt, gt):
    header(c, code_hi=False)
    pulse = 0.5 + 0.5 * ease_in_out(abs((lt*2 % 1) - 0.5) * 2)
    draw_lanes(c, ["idle"] * LANES, pulse)
    c.text_center(W//2, Y0 + 2*(CELL+GAP) + 24,
                  "All 32 lanes share one program counter. Perfect lockstep.",
                  size=17, fill=g.MUTED)
    clockface(c, 1)


def scene_split(c: Canvas, lt, gt):
    """First pass: even lanes run, odd wait. Second pass: swap."""
    header(c, code_hi=True)
    phase = lt  # 0..1 across the whole split scene
    # first half -> even run; second half -> odd run
    if phase < 0.5:
        p = phase / 0.5
        mask = ["run" if (i % 2 == 0) else "wait" for i in range(LANES)]
        branch = "if-branch  (even lanes)"
    else:
        p = (phase - 0.5) / 0.5
        mask = ["run" if (i % 2 == 1) else "wait" for i in range(LANES)]
        branch = "else-branch  (odd lanes)"
    flash = 0.5 + 0.5 * ease_in_out(abs((p*2 % 1) - 0.5) * 2)
    draw_lanes(c, mask, flash)
    c.text_center(W//2, Y0 + 2*(CELL+GAP) + 24,
                  f"Executing {branch} — the other half is masked off and idle.",
                  size=17, fill=g.AMBER)
    clockface(c, 2)
    # progress meter of "wasted lane-cycles"
    c.meter([X0, 560, X0 + GRID_W, 578], clamp(phase),
            "SERIALIZED  (both branches run, half the lanes idle each pass)",
            g.RED)


def scene_punch(c: Canvas, lt, gt):
    scene_split(c, 0.99, gt)
    g.punch_overlay(c, "Both branches run. Nobody wins.",
                    alpha=ease_out(lt),
                    sub="Warp divergence: threads taking turns being useless.")


def main():
    tl = g.Timeline(W, H, FPS)
    tl.scene(1.2, scene_lockstep)
    tl.scene(2.6, scene_split)
    tl.scene(1.6, scene_punch)
    frames = tl.build()
    out = os.path.join(os.path.dirname(__file__), "out", "day_02_warp_divergence.gif")
    g.render(frames, out, fps=FPS)
    print("wrote", out, f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
