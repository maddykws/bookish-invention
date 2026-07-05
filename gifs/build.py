"""
build.py — render day GIFs from data-driven configs in days.py.

Usage:
  python3 build.py            # render every config in days.DAYS
  python3 build.py 5 20 44    # render only those day numbers

Each GIF loop = intro(0.25s) + action(3.4s) + hold(0.3s) + punch(1.5s) = 5.45s,
strictly under 6 seconds. Days 1, 2, 4 have their own bespoke scripts and are
skipped here.
"""
import os
import sys
import giflib as g
from giflib import ease_out
from archetypes import ARCHETYPES
from days import DAYS

W, H, FPS = 900, 560, 18
OUT = os.path.join(os.path.dirname(__file__), "out")

INTRO_S, ACTION_S, HOLD_S, PUNCH_S = 0.25, 3.4, 0.3, 1.5   # = 5.45s total


def frame(c, action_t, cfg, punch=0.0):
    c.grid()
    g.title_bar(c, cfg["title"], cfg.get("subtitle", ""))
    if cfg.get("caption"):
        c.text_center(c.W // 2, 62, cfg["caption"], size=17, fill=g.MUTED)
    ARCHETYPES[cfg["arch"]](c, action_t, cfg["params"])
    if punch > 0:
        g.punch_overlay(c, cfg["punch"], alpha=ease_out(punch),
                        sub=cfg.get("punch_sub"))


def make(cfg):
    tl = g.Timeline(W, H, FPS)
    tl.scene(INTRO_S, lambda c, lt, gt: frame(c, 0.0, cfg))
    tl.scene(ACTION_S, lambda c, lt, gt: frame(c, lt, cfg))
    tl.scene(HOLD_S, lambda c, lt, gt: frame(c, 1.0, cfg))
    tl.scene(PUNCH_S, lambda c, lt, gt: frame(c, 1.0, cfg, punch=lt))
    frames = tl.build()
    path = os.path.join(OUT, cfg["file"])
    g.render(frames, path, fps=FPS, colors=128)
    return path, len(frames)


def main():
    want = set(int(x) for x in sys.argv[1:]) if len(sys.argv) > 1 else None
    total_s = INTRO_S + ACTION_S + HOLD_S + PUNCH_S
    assert total_s < 6.0, total_s
    count = 0
    for cfg in DAYS:
        if want and cfg["day"] not in want:
            continue
        path, nframes = make(cfg)
        size = os.path.getsize(path) / 1e6
        print(f"day {cfg['day']:>2}  {os.path.basename(path):<44} "
              f"{nframes:>3}f  {size:4.1f}MB")
        count += 1
    print(f"\nrendered {count} gif(s)  |  loop = {total_s:.2f}s each")


if __name__ == "__main__":
    main()
