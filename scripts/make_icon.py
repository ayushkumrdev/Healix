#!/usr/bin/env python3
"""
Render the Healix app icon (frontend/web/healix.ico) from the brand mark —
the favicon's hexagon + vitals pulse, accent gradient on the night-ward dark
tile. Drawn with PIL at 4x supersampling, packed as a multi-size .ico
(256/128/64/48/32/16) for Windows shortcuts and the taskbar.

Usage: .venv\\Scripts\\python.exe scripts\\make_icon.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "frontend" / "web" / "healix.ico"
PREVIEW = ROOT / "frontend" / "web" / "healix_icon_preview.png"

BASE = 1024          # final master size
SS = 4               # supersampling factor
S = BASE * SS        # drawing canvas
U = S / 32           # favicon viewBox unit -> canvas px

BG = (7, 11, 10, 255)          # --bg  #070b0a
ACCENT_A = (60, 226, 167)      # --accent  #3ce2a7
ACCENT_B = (86, 200, 216)      # --accent-2 #56c8d8

HEX_PTS = [(16, 5), (25, 10.25), (25, 20.75), (16, 26), (7, 20.75), (7, 10.25)]
PULSE_PTS = [(9, 16), (11.4, 16), (13.1, 12.2), (16.1, 19.4),
             (18.2, 13.8), (20, 16), (23, 16)]
STROKE = 2.0  # favicon stroke-width in viewBox units


def px(pts):
    return [(x * U, y * U) for x, y in pts]


def stroke_layer(points, width_px, alpha, closed=False):
    """White strokes on a transparent layer (used as a gradient mask)."""
    layer = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(layer)
    pts = px(points)
    if closed:
        pts = pts + [pts[0]]
    d.line(pts, fill=alpha, width=int(width_px), joint="curve")
    r = width_px / 2
    for (x, y) in (pts if not closed else pts[:-1]):  # round caps/joins
        d.ellipse([x - r, y - r, x + r, y + r], fill=alpha)
    return layer


def main():
    # accent gradient, 135deg
    yy, xx = np.mgrid[0:S, 0:S]
    t = ((xx + yy) / (2 * (S - 1)))[..., None]
    grad = (np.asarray(ACCENT_A) * (1 - t) + np.asarray(ACCENT_B) * t).astype("uint8")
    gradient = Image.fromarray(grad, "RGB").convert("RGBA")

    # dark rounded tile (favicon rx=8/32 of the tile)
    icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tile = ImageDraw.Draw(icon)
    tile.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(8 * U), fill=BG)

    w = STROKE * U
    mask = Image.new("L", (S, S), 0)
    mask.paste(stroke_layer(HEX_PTS, w, 128, closed=True), (0, 0),
               stroke_layer(HEX_PTS, w, 128, closed=True))
    pulse = stroke_layer(PULSE_PTS, w, 255)
    mask.paste(pulse, (0, 0), pulse)

    icon.paste(gradient, (0, 0), mask)

    master = icon.resize((BASE, BASE), Image.LANCZOS)
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    master.save(OUT, format="ICO", sizes=sizes)
    master.resize((256, 256), Image.LANCZOS).save(PREVIEW)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB) + preview png")


if __name__ == "__main__":
    main()
