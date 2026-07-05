"""
archetypes.py — 9 reusable animation patterns for the 60-day GIF series.

Each archetype is `fn(c, t, p)`:
  c : giflib.Canvas (already has title bar + caption drawn by the driver)
  t : float 0..1, progress through the "action" portion of the loop
  p : dict of archetype-specific params (from days.py)

The driver (build.py) handles the terminal title bar, caption line, hold, and
the fade-in punchline card, so archetypes only draw the content region
(roughly y in [100, 530]).

Design canvas: 900 x 560.
"""
import math
import giflib as g
from giflib import clamp, lerp, lerp_color, ease_in_out, ease_out

STATE_COLORS = {
    "off":  g.PANEL_HI,
    "on":   g.BLUE,
    "good": g.GREEN,
    "bad":  g.RED,
    "warn": g.AMBER,
    "alt":  g.PURPLE,
    "nv":   g.NVIDIA,
}


# ---------------------------------------------------------------------------
# 1. bars_race — 2-4 labeled horizontal bars filling at different rates.
# ---------------------------------------------------------------------------
def bars_race(c, t, p):
    bars = p["bars"]                # [{label, target, color, value(fn or str)}]
    note = p.get("note")
    x0, x1 = 70, c.W - 70
    top, bottom = 150, 470
    n = len(bars)
    slot = (bottom - top) / n
    for i, b in enumerate(bars):
        cy = int(top + slot * i + slot / 2)
        speed = b.get("speed", 1.0)
        v = b["target"] * ease_out(clamp(t * speed))
        # label above bar
        c.text((x0, cy - 34), b["label"], size=18, bold=True, fill=g.MUTED)
        box = [x0, cy - 12, x1, cy + 12]
        c.d.rounded_rectangle(box, radius=12, fill=g.PANEL_HI)
        w = (x1 - x0) * clamp(v)
        if w > 3:
            c.d.rounded_rectangle([x0, cy - 12, x0 + w, cy + 12], radius=12,
                                  fill=b["color"])
        # value readout at right
        val = b.get("value")
        if callable(val):
            txt = val(v)
        elif val:
            txt = val
        else:
            txt = f"{int(round(v*100))}%"
        c.text((x1, cy - 34), txt, size=17, bold=True, fill=b["color"],
               anchor="ra")
    if note:
        c.text_center(c.W // 2, 500, note, size=15, fill=g.MUTED)


# ---------------------------------------------------------------------------
# 2. grid_states — NxM cell grid transitioning through states over time.
#    p['pattern'] = fn(t, r, col, rows, cols) -> state key.
# ---------------------------------------------------------------------------
def grid_states(c, t, p):
    rows, cols = p["rows"], p["cols"]
    pattern = p["pattern"]
    labels = p.get("labels", ("", ""))
    cell = p.get("cell", 40)
    gap = p.get("gap", 8)
    gw = cols * (cell + gap) - gap
    gh = rows * (cell + gap) - gap
    x0 = (c.W - gw) // 2
    y0 = 150 + (330 - gh) // 2
    if labels[0]:
        c.text_center(c.W // 2, 125, labels[0], size=17, bold=True, fill=g.MUTED)
    for r in range(rows):
        for col in range(cols):
            st = pattern(t, r, col, rows, cols)
            fill = STATE_COLORS.get(st, g.PANEL_HI)
            x = x0 + col * (cell + gap)
            y = y0 + r * (cell + gap)
            c.d.rounded_rectangle([x, y, x + cell, y + cell], radius=6, fill=fill)
            if p.get("index") and cell >= 34:
                idx = r * cols + col
                tc = g.BG if st != "off" else g.MUTED
                c.text((x + cell/2, y + cell/2), str(idx), size=12, bold=True,
                       fill=tc, anchor="mm")
    if labels[1]:
        c.text_center(c.W // 2, y0 + gh + 20, labels[1], size=16, bold=True,
                      fill=p.get("caption_color", g.MUTED))


# ---------------------------------------------------------------------------
# 3. gauge_spike — a big curve revealed left->right, optional wall/threshold.
#    p['fn'] = fn(x in 0..1) -> y in 0..1 ; revealed up to x=t.
# ---------------------------------------------------------------------------
def gauge_spike(c, t, p):
    fn = p["fn"]
    x0, x1 = 90, c.W - 70
    y_top, y_bot = 140, 470
    c.panel([x0 - 20, y_top - 14, x1 + 20, y_bot + 34], radius=14, fill=g.PANEL)
    # axes
    c.d.line([(x0, y_bot), (x1, y_bot)], fill=g.PANEL_HI, width=2)
    c.d.line([(x0, y_top), (x0, y_bot)], fill=g.PANEL_HI, width=2)
    c.text((x0 - 6, y_top - 8), p.get("y_label", ""), size=14, bold=True,
           fill=g.MUTED, anchor="lb")
    c.text((x1, y_bot + 12), p.get("x_label", ""), size=14, fill=g.MUTED,
           anchor="ra")

    def px(x): return x0 + (x1 - x0) * clamp(x)
    def py(y): return y_bot - (y_bot - y_top) * clamp(y)

    # threshold line
    thr = p.get("threshold")
    if thr is not None:
        ty = py(thr)
        for xx in range(int(x0), int(x1), 12):
            c.d.line([(xx, ty), (xx + 6, ty)], fill=g.RED, width=2)
        c.text((x1, ty - 6), p.get("threshold_label", ""), size=13, bold=True,
               fill=g.RED, anchor="rb")

    steps = 80
    prog = ease_in_out(t)
    pts = []
    hi = int(steps * prog)
    for i in range(hi + 1):
        x = i / steps
        pts.append((px(x), py(fn(x))))
    color = p.get("color", g.BLUE)
    c.curve(pts, color, width=4)
    if pts:
        hx, hy = pts[-1]
        c.dot(hx, hy, 6, p.get("head_color", g.INK))
        annot = p.get("annot")
        if annot and prog > 0.6:
            c.text((hx - 10, hy - 14), annot, size=16, bold=True,
                   fill=p.get("head_color", g.INK), anchor="rb")

    # optional vertical wall (e.g. OOM ceiling)
    wall = p.get("wall")
    if wall is not None and prog >= wall - 0.02:
        wx = px(wall)
        c.d.line([(wx, y_top), (wx, y_bot)], fill=g.RED, width=3)
        c.text((wx + 8, y_top + 4), p.get("wall_label", ""), size=14, bold=True,
               fill=g.RED, anchor="la")


# ---------------------------------------------------------------------------
# 4. flow_pipeline — packet travels through labeled stages left->right.
# ---------------------------------------------------------------------------
def flow_pipeline(c, t, p):
    stages = p["stages"]            # [{label, sub}]
    n = len(stages)
    x0, x1 = 80, c.W - 80
    cy = 250
    span = (x1 - x0)
    nw = min(150, span / n - 24)
    nh = 70
    centers = [x0 + span * (i + 0.5) / n for i in range(n)]
    prog = ease_in_out(t)
    packet_x = lerp(centers[0], centers[-1], prog)
    active = int(clamp(prog) * (n - 1) + 0.5)

    # connectors
    for i in range(n - 1):
        c.arrow((centers[i] + nw/2, cy), (centers[i+1] - nw/2, cy),
                color=g.PANEL_HI, width=3, head=8)
    # optional loop-back arrow (agentic loop)
    if p.get("loop"):
        yl = cy + nh/2 + 30
        c.d.line([(centers[-1], cy + nh/2), (centers[-1], yl)], fill=g.PURPLE, width=3)
        c.d.line([(centers[-1], yl), (centers[0], yl)], fill=g.PURPLE, width=3)
        c.arrow((centers[0], yl), (centers[0], cy + nh/2), color=g.PURPLE, width=3, head=8)

    for i, s in enumerate(stages):
        done = i <= active
        fill = g.PANEL_HI if not done else lerp_color(g.PANEL_HI, s.get("color", g.BLUE), 0.85)
        oc = s.get("color", g.BLUE) if done else g.PANEL_HI
        tc = g.BG if done else g.MUTED
        c.node(centers[i], cy, nw, nh, s["label"], fill=fill, outline=oc,
               text_color=tc, size=15, sub=s.get("sub"))
    # packet
    pc = p.get("packet_color", g.AMBER)
    c.dot(packet_x, cy, 11, pc)
    c.dot(packet_x, cy, 5, g.BG)
    if p.get("packet_label"):
        c.text((packet_x, cy - nh/2 - 18), p["packet_label"], size=14, bold=True,
               fill=pc, anchor="mm")
    if p.get("note"):
        c.text_center(c.W // 2, 460, p["note"], size=15, fill=g.MUTED)


# ---------------------------------------------------------------------------
# 5. ring_net — N nodes; dots travel edges. modes: allreduce, deadlock,
#    topology, scale.
# ---------------------------------------------------------------------------
def ring_net(c, t, p):
    mode = p.get("mode", "allreduce")
    cx, cy = c.W // 2, 300
    R = 150
    n = p.get("n", 8)
    prog = ease_in_out(t)

    if mode == "deadlock":
        # two nodes pointing at each other, frozen; a timeout ticks
        ax, bx = cx - 150, cx + 150
        c.node(ax, cy, 130, 74, "GPU 0", fill=g.PANEL, outline=g.RED,
               text_color=g.INK, sub="wait: all_reduce")
        c.node(bx, cy, 130, 74, "GPU 1", fill=g.PANEL, outline=g.RED,
               text_color=g.INK, sub="wait: all_gather")
        c.arrow((ax + 70, cy - 16), (bx - 70, cy - 16), color=g.RED, width=3)
        c.arrow((bx - 70, cy + 16), (ax + 70, cy + 16), color=g.RED, width=3)
        secs = int(prog * 30)
        c.text_center(cx, cy + 90, f"NCCL timeout in {30 - secs}s ...",
                      size=18, bold=True, fill=g.RED)
        c.text_center(cx, 150, "Everyone's waiting. No one's arriving.",
                      size=16, fill=g.MUTED)
        return

    pos = [(cx + R * math.cos(2*math.pi*i/n - math.pi/2),
            cy + R * math.sin(2*math.pi*i/n - math.pi/2)) for i in range(n)]
    # edges (ring)
    for i in range(n):
        a, b = pos[i], pos[(i+1) % n]
        c.d.line([a, b], fill=g.PANEL_HI, width=2)
    # traveling dots on each edge
    dot_col = p.get("dot_color", g.BLUE)
    if mode in ("allreduce", "scale"):
        for i in range(n):
            a, b = pos[i], pos[(i+1) % n]
            frac = (prog + i / n) % 1.0
            dx = lerp(a[0], b[0], frac); dy = lerp(a[1], b[1], frac)
            c.dot(dx, dy, 6, dot_col)
    # nodes
    done_color = g.GREEN if prog > 0.85 else dot_col
    for i, (x, y) in enumerate(pos):
        col = done_color if mode == "allreduce" else dot_col
        c.node(x, y, 78, 46, f"GPU{i}", fill=g.PANEL,
               outline=col, text_color=g.INK, size=14)
    center_label = p.get("center_label")
    if center_label:
        c.text_center(cx, cy - 8, center_label[0], size=16, bold=True, fill=g.INK)
        if len(center_label) > 1:
            c.text_center(cx, cy + 16, center_label[1], size=13, fill=g.MUTED)


# ---------------------------------------------------------------------------
# 6. compare_paths — two lanes, a slow (top) and fast (bottom) packet A->B.
# ---------------------------------------------------------------------------
def compare_paths(c, t, p):
    x0, x1 = 90, c.W - 90
    prog = ease_in_out(t)
    lanes = [
        dict(y=210, label=p["top_label"], color=g.RED,
             speed=p.get("top_speed", 0.55), hops=p.get("top_hops", []),
             meter="slow"),
        dict(y=380, label=p["bottom_label"], color=g.GREEN,
             speed=p.get("bottom_speed", 1.0), hops=p.get("bottom_hops", []),
             meter="fast"),
    ]
    for L in lanes:
        y = L["y"]
        c.text((x0, y - 54), L["label"], size=17, bold=True, fill=L["color"])
        # source & dest chips
        c.node(x0, y, 90, 44, p.get("src", "SRC"), outline=g.MUTED, size=13)
        c.node(x1, y, 90, 44, p.get("dst", "DST"), outline=g.MUTED, size=13)
        # track
        c.d.line([(x0 + 45, y), (x1 - 45, y)], fill=g.PANEL_HI, width=3)
        # intermediate hop boxes (e.g. CPU / staging) slow the top lane
        for hx, hlabel in L["hops"]:
            xx = lerp(x0 + 45, x1 - 45, hx)
            c.node(xx, y, 78, 40, hlabel, outline=g.AMBER, text_color=g.AMBER,
                   size=12)
        # packet
        pe = ease_out(clamp(prog * L["speed"]))
        pxp = lerp(x0 + 45, x1 - 45, pe)
        c.dot(pxp, y, 10, L["color"])
        c.dot(pxp, y, 4, g.BG)
    if p.get("note"):
        c.text_center(c.W // 2, 470, p["note"], size=15, fill=g.MUTED)


# ---------------------------------------------------------------------------
# 7. timeline_track — CPU/GPU rows with blocks & gaps. p['layout'](t)->
#    (cpu_blocks, gpu_blocks), each [(x0f, x1f, color)] in 0..1 time coords.
# ---------------------------------------------------------------------------
def timeline_track(c, t, p):
    x0, x1 = 110, c.W - 60
    rows = p.get("rows", [("CPU", 200), ("GPU", 300)])
    layout = p["layout"]
    data = layout(ease_in_out(t))
    span = x1 - x0
    for (label, y), blocks in zip(rows, data):
        c.text((x0 - 12, y), label, size=15, bold=True, fill=g.MUTED, anchor="rm")
        c.d.line([(x0, y + 26), (x1, y + 26)], fill=g.PANEL_HI, width=1)
        c.d.rounded_rectangle([x0, y - 18, x1, y + 18], radius=6, fill=g.PANEL)
        for (bx0, bx1, col) in blocks:
            rx0 = x0 + span * bx0
            rx1 = x0 + span * bx1
            if rx1 - rx0 >= 1:
                c.d.rounded_rectangle([rx0, y - 15, rx1, y + 15], radius=4, fill=col)
    if p.get("note"):
        c.text_center(c.W // 2, 470, p["note"], size=15, fill=g.MUTED)
    c.text((x1, rows[-1][1] + 46), "time →", size=13, fill=g.MUTED, anchor="ra")


# ---------------------------------------------------------------------------
# 8. stack_build — layers stack bottom->up as t increases.
# ---------------------------------------------------------------------------
def stack_build(c, t, p):
    layers = p["layers"]           # bottom -> top: [{label, color}]
    n = len(layers)
    x0, x1 = 150, c.W - 150
    bottom, top = 500, 130
    lh = (bottom - top) / n
    shown = clamp(t) * n
    for i, L in enumerate(layers):
        appear = clamp(shown - i)
        if appear <= 0:
            continue
        e = ease_out(appear)
        y_c = bottom - (i + 0.5) * lh
        w = (x1 - x0)
        # slide in from the right
        off = (1 - e) * 60
        fill = lerp_color(g.PANEL, L["color"], 0.85 * e)
        c.d.rounded_rectangle([x0 + off, y_c - lh/2 + 4, x1 + off, y_c + lh/2 - 4],
                              radius=8, fill=fill, outline=L["color"], width=2)
        c.text((x0 + 20 + off, y_c), L["label"], size=16, bold=True,
               fill=g.BG if e > 0.5 else g.MUTED, anchor="lm")
        if L.get("sub"):
            c.text((x1 - 16 + off, y_c), L["sub"], size=12,
                   fill=g.BG if e > 0.5 else g.MUTED, anchor="rm")


# ---------------------------------------------------------------------------
# 9. type_reveal — text/token focused. modes: tokenize, confidence,
#    context_window, printf_flood.
# ---------------------------------------------------------------------------
def type_reveal(c, t, p):
    mode = p["mode"]
    prog = ease_in_out(t)
    if mode == "tokenize":
        word = p["word"]; tokens = p["tokens"]
        c.text_center(c.W // 2, 170, f'input: "{word}"', size=22, bold=True,
                      fill=g.MUTED)
        # split into token chips
        total = sum(len(tk) for tk in tokens)
        cw = min(120, (c.W - 160) / len(tokens))
        gap = 14
        tw = len(tokens) * (cw + gap) - gap
        sx = (c.W - tw) / 2
        split = clamp(prog * 1.3)
        for i, tk in enumerate(tokens):
            x = sx + i * (cw + gap)
            spread = (i - (len(tokens)-1)/2) * 10 * (1 - split)
            col = [g.BLUE, g.PURPLE, g.GREEN, g.AMBER, g.RED][i % 5]
            c.d.rounded_rectangle([x + spread, 300, x + cw + spread, 356],
                                  radius=10, fill=lerp_color(g.PANEL_HI, col, split))
            c.text(((x + x + cw)/2 + spread, 328), tk, size=18, bold=True,
                   fill=g.BG if split > 0.4 else g.MUTED, anchor="mm")
        c.text_center(c.W // 2, 400,
                      f"{len(word)} characters  →  {len(tokens)} tokens",
                      size=17, bold=True, fill=g.INK)
    elif mode == "confidence":
        c.text_center(c.W // 2, 160, p.get("prompt", ""), size=17, fill=g.MUTED)
        c.meter([120, 300, c.W - 120, 326], ease_out(prog),
                "MODEL CONFIDENCE", g.GREEN)
        c.text((c.W - 120, 276), "100%", size=16, bold=True, fill=g.GREEN, anchor="ra")
        c.meter([120, 400, c.W - 120, 426], 0.03,
                "FACTUAL ACCURACY", g.RED)
        c.text((c.W - 120, 376), "~0%", size=16, bold=True, fill=g.RED, anchor="ra")
    elif mode == "context_window":
        # message chips slide left; window frame fixed; oldest fall out
        win_x0, win_x1 = 150, c.W - 150
        c.d.rounded_rectangle([win_x0, 250, win_x1, 340], radius=10,
                              outline=g.BLUE, width=2)
        c.text((win_x0, 232), "CONTEXT WINDOW", size=13, bold=True, fill=g.BLUE)
        n = 10
        cw = 78
        shift = prog * (cw + 12) * 4
        for i in range(n):
            x = win_x1 - 20 - i * (cw + 12) + shift
            inside = win_x0 < x < win_x1 - 20
            col = g.PANEL_HI if not inside else lerp_color(g.PANEL_HI, g.BLUE, 0.7)
            if x < win_x0 - cw:
                continue
            c.d.rounded_rectangle([x - cw, 268, x, 322], radius=8,
                                  fill=col if inside else g.PANEL)
            c.text((x - cw/2, 295), f"t{n-i}", size=13, bold=True,
                   fill=g.BG if inside else g.RED, anchor="mm")
        c.text_center(c.W // 2, 380, "Oldest tokens slide out. The window moved on.",
                      size=16, fill=g.MUTED)
    elif mode == "printf_flood":
        lines = int(prog * 26)
        c.d.rounded_rectangle([80, 140, c.W - 80, 470], radius=10, fill=(6, 9, 13))
        import random as _r
        _r.seed(3)
        for i in range(min(lines, 20)):
            y = 156 + i * 15
            tid = _r.randrange(32768)
            c.text((100, y), f"[t{tid:05d}] printf: x = {_r.randrange(9999)}",
                   size=12, fill=lerp_color(g.MUTED, g.GREEN, 0.3))
        c.text_center(c.W // 2, 495,
                      "32,768 threads said hi. None said where the bug is.",
                      size=15, fill=g.MUTED)


ARCHETYPES = {
    "bars_race": bars_race,
    "grid_states": grid_states,
    "gauge_spike": gauge_spike,
    "flow_pipeline": flow_pipeline,
    "ring_net": ring_net,
    "compare_paths": compare_paths,
    "timeline_track": timeline_track,
    "stack_build": stack_build,
    "type_reveal": type_reveal,
}
