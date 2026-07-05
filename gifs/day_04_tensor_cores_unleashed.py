"""
Day 4 — Tensor Cores.
Left: a lone CUDA core does one FMA at a time, FLOP counter ticks by 1s, filling
a result matrix cell-by-cell slowly. Right: a Tensor Core swallows a whole tile
and fills a 4x4 block per step, counter leaps by 64s. Punchline:
"Still doing FP32 GEMM by hand? Your Tensor Cores are napping."
"""
import os
import giflib as g
from giflib import Canvas, ease_in_out, ease_out, clamp, lerp

W, H, FPS = 900, 600, 20
GRID = 8                        # result matrix is 8x8 for both sides
CELL = 30
GAP = 4
MAT_W = GRID * (CELL + GAP) - GAP

LEFT_CX = 235
RIGHT_CX = 665
MAT_Y = 190


def draw_matrix(c: Canvas, cx, filled, color):
    """filled = number of cells filled (row-major)."""
    x0 = cx - MAT_W / 2
    for r in range(GRID):
        for col in range(GRID):
            idx = r * GRID + col
            x = x0 + col * (CELL + GAP)
            y = MAT_Y + r * (CELL + GAP)
            on = idx < filled
            fill = color if on else g.PANEL_HI
            c.d.rounded_rectangle([x, y, x + CELL, y + CELL], radius=5, fill=fill)
    return x0


def draw_tile_matrix(c: Canvas, cx, tiles_done, color):
    """Fill in 4x4 tiles. tiles_done = number of 4x4 tiles completed (max 4)."""
    x0 = cx - MAT_W / 2
    TSZ = 4
    tiles = [(0, 0), (0, 1), (1, 0), (1, 1)]  # order of 4x4 blocks
    done_cells = set()
    for t in range(min(tiles_done, 4)):
        tr, tc = tiles[t]
        for r in range(TSZ):
            for col in range(TSZ):
                done_cells.add((tr*TSZ + r, tc*TSZ + col))
    for r in range(GRID):
        for col in range(GRID):
            x = x0 + col * (CELL + GAP)
            y = MAT_Y + r * (CELL + GAP)
            on = (r, col) in done_cells
            fill = color if on else g.PANEL_HI
            c.d.rounded_rectangle([x, y, x + CELL, y + CELL], radius=5, fill=fill)


def core_badge(c: Canvas, cx, y, label, color, big=False):
    w = 150 if not big else 190
    h = 54 if not big else 64
    c.chip(cx, y, w, h, g.PANEL, outline=color, radius=12)
    c.text((cx, y-8), label, size=17 if not big else 20, bold=True,
           fill=color, anchor="mm")
    c.text((cx, y+14), "FMA x1" if not big else "MMA 4x4 tile",
           size=12, fill=g.MUTED, anchor="mm")


def scene(c: Canvas, lt, gt, punch=0.0):
    c.grid()
    g.title_bar(c, "cuda ~ tensor cores", "fp16 GEMM  C = A x B")
    c.text_center(W//2, 62, "Same silicon. Two very different leagues.",
                  size=17, fill=g.MUTED)

    # LEFT: scalar CUDA core, cell by cell (slow)
    left_cells = int(round(64 * ease_in_out(clamp(lt * 0.85))))
    core_badge(c, LEFT_CX, 120, "CUDA CORE", g.BLUE, big=False)
    draw_matrix(c, LEFT_CX, left_cells, g.BLUE)
    c.text_center(LEFT_CX, MAT_Y + MAT_W + 16,
                  f"FLOPs issued: {left_cells}", size=15, bold=True, fill=g.BLUE)

    # RIGHT: tensor core, tile by tile (fast, leaps)
    tiles = int(clamp(lt * 1.15) * 4 + 0.001)
    tiles = min(tiles, 4)
    core_badge(c, RIGHT_CX, 120, "TENSOR CORE", g.NVIDIA, big=True)
    draw_tile_matrix(c, RIGHT_CX, tiles, g.NVIDIA)
    c.text_center(RIGHT_CX, MAT_Y + MAT_W + 16,
                  f"FLOPs issued: {tiles*16}", size=15, bold=True, fill=g.NVIDIA)

    # divider
    c.d.line([(W//2, 150), (W//2, MAT_Y + MAT_W + 6)], fill=g.PANEL_HI, width=2)

    # throughput meters
    c.meter([LEFT_CX - MAT_W/2, 548, LEFT_CX + MAT_W/2, 566],
            ease_in_out(clamp(lt*0.85)) * 0.25, "1 MAC / clock", g.BLUE)
    c.meter([RIGHT_CX - MAT_W/2, 548, RIGHT_CX + MAT_W/2, 566],
            clamp(lt*1.15), "1 matrix tile / clock", g.NVIDIA)

    if punch > 0:
        g.punch_overlay(c, "Your Tensor Cores are napping.",
                        alpha=ease_out(punch),
                        sub="Still hand-rolling FP32 GEMM?  #TensorCores #GEMM")


def main():
    tl = g.Timeline(W, H, FPS)
    tl.scene(3.4, scene)
    tl.scene(0.3, lambda c, lt, gt: scene(c, 1.0, gt))
    tl.scene(1.6, lambda c, lt, gt: scene(c, 1.0, gt, punch=lt))
    frames = tl.build()
    out = os.path.join(os.path.dirname(__file__), "out", "day_04_tensor_cores_unleashed.gif")
    g.render(frames, out, fps=FPS)
    print("wrote", out, f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
