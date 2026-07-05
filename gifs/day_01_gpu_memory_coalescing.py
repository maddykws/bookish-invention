"""
Day 1 — Memory coalescing.
Left "UNCOALESCED": 8 threads fire arrows to scattered memory cells; latency (red)
climbs, transaction count high. Right "COALESCED": the same threads hit one
contiguous run in a single sweep; throughput (green) maxes, 1 transaction.
Punchline: "Coalesce your loads. Your DRAM will thank you."
"""
import os, random
import giflib as g
from giflib import Canvas, ease_in_out, ease_out, lerp, clamp

W, H, FPS = 900, 600, 20
random.seed(7)

N = 8                      # threads shown (stand-in for a warp)
CELLS = 16                 # memory cells per side
# scattered target cell per thread (left panel)
SCATTER = [random.randrange(CELLS) for _ in range(N)]
# contiguous targets (right panel)
CONTIG = list(range(N))

LEFT_X, RIGHT_X = 60, 480
PANEL_W = 360
MEM_Y = 150                # top of memory cell row
THREAD_Y = 470            # threads sit near bottom


def mem_cells(c: Canvas, x0):
    cw = PANEL_W / CELLS
    cells = []
    for i in range(CELLS):
        cx = x0 + i * cw + cw / 2
        cells.append((cx, MEM_Y, cw))
    return cells, cw


def draw_panel(c: Canvas, x0, title, targets, progress, coalesced):
    color = g.GREEN if coalesced else g.RED
    c.panel([x0 - 16, 96, x0 + PANEL_W + 16, 520], radius=16, fill=g.PANEL,
            outline=g.PANEL_HI, width=1)
    c.text_center(x0 + PANEL_W / 2, 110, title, size=20, bold=True,
                  fill=color)

    cells, cw = mem_cells(c, x0)
    # which cells are being touched (light them as arrows arrive)
    touched = set()
    for ti in range(N):
        if progress > 0.15:
            touched.add(targets[ti])

    for i, (cx, cy, w) in enumerate(cells):
        lit = i in touched and progress > 0.35
        fill = g.PANEL_HI if not lit else color
        c.d.rounded_rectangle([cx - w/2 + 2, cy, cx + w/2 - 2, cy + 34],
                              radius=4, fill=fill)

    c.text((x0, MEM_Y - 26), "GLOBAL MEMORY (HBM)", size=13, bold=True,
           fill=g.MUTED)

    # threads
    tw = PANEL_W / N
    for ti in range(N):
        tx = x0 + ti * tw + tw / 2
        c.chip(tx, THREAD_Y, tw - 10, 30, g.BLUE if not coalesced else g.PURPLE,
               radius=6)
        c.text((tx, THREAD_Y), f"t{ti}", size=13, bold=True, fill=g.BG,
               anchor="mm")

    # arrows from thread to its target cell, extending with progress
    for ti in range(N):
        tx = x0 + ti * tw + tw / 2
        target_cx = cells[targets[ti]][0]
        p0 = (tx, THREAD_Y - 16)
        end_full = (target_cx, MEM_Y + 34)
        pe = ease_out(clamp((progress - ti * (0.03 if not coalesced else 0.0)) * 1.3))
        if coalesced:
            pe = ease_out(progress)  # all sweep together
        px = lerp(p0[0], end_full[0], pe)
        py = lerp(p0[1], end_full[1], pe)
        col = g.RED if not coalesced else g.GREEN
        c.d.line([p0, (px, py)], fill=col, width=2)
        if pe > 0.98:
            c.arrow((px, py - 6), (px, py), color=col, width=2, head=6)

    # transaction counter
    if coalesced:
        txn = 1 if progress > 0.5 else 0
    else:
        txn = int(round(len(touched) * 8 * clamp(progress * 1.3)))
    c.text_center(x0 + PANEL_W / 2, 500 - 4,
                  f"memory transactions: {txn if progress>0.2 else 0}",
                  size=15, bold=True, fill=color)


def scene_main(c: Canvas, lt, gt):
    c.grid()
    g.title_bar(c, "cuda ~ memory coalescing", "warp = 8 lanes")
    c.text_center(W//2, 62, "Same 8 threads. One access pattern changes everything.",
                  size=17, bold=False, fill=g.MUTED)

    # left animates first half, right catches up in second half
    left_p = ease_in_out(clamp(lt * 1.4))
    right_p = ease_in_out(clamp((lt - 0.3) * 1.6))

    draw_panel(c, LEFT_X, "UNCOALESCED", SCATTER, left_p, False)
    draw_panel(c, RIGHT_X, "COALESCED", CONTIG, right_p, True)

    # meters across the very bottom
    c.meter([LEFT_X, 548, LEFT_X + PANEL_W, 566], left_p, "LATENCY", g.RED)
    c.meter([RIGHT_X, 548, RIGHT_X + PANEL_W, 566], right_p, "THROUGHPUT", g.GREEN)


def scene_punch(c: Canvas, lt, gt):
    scene_main(c, 1.0, gt)
    g.punch_overlay(c, "Coalesce your loads.",
                    alpha=ease_out(lt),
                    sub="Your DRAM will thank you.  #CUDA #GPU")


def main():
    tl = g.Timeline(W, H, FPS)
    tl.scene(0.3, lambda c, lt, gt: (c.grid(), g.title_bar(c, "cuda ~ memory coalescing", "warp = 8 lanes")))
    tl.scene(2.8, scene_main)
    tl.scene(0.4, lambda c, lt, gt: scene_main(c, 1.0, gt))
    tl.scene(1.7, scene_punch)
    frames = tl.build()
    out = os.path.join(os.path.dirname(__file__), "out", "day_01_gpu_memory_coalescing.gif")
    g.render(frames, out, fps=FPS)
    print("wrote", out, f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
