"""Kiosk print sheets: a 4 x 2 grid of 35x45 mm photos with 1 px light-grey crop marks.

Photos are pasted pixel-for-pixel (no resampling), so at 600 dpi each tile is exactly 826x1062 px.
Crop marks are collinear extensions of every cut line, drawn only in the gutters and kept
``mark_offset_px`` away from the photos so a slightly-off cut never shows a grey line.
"""

from __future__ import annotations

import numpy as np

from .spec import PHOTO, SheetSpec


def _spread(total: int, n: int, item: int) -> list[int]:
    """Left/top coordinates of n items with equal outer margins and inner gutters."""
    gap = (total - n * item) / (n + 1)
    if gap < 2:
        raise ValueError(f"{n} x {item} px does not fit in {total} px")
    return [int(round(gap + i * (item + gap))) for i in range(n)]


def layout(spec: SheetSpec) -> list[tuple[int, int]]:
    xs = _spread(spec.width_px, spec.cols, PHOTO.width_px)
    ys = _spread(spec.height_px, spec.rows, PHOTO.height_px)
    return [(x, y) for y in ys for x in xs]


def build(photo: np.ndarray | list[np.ndarray], spec: SheetSpec) -> np.ndarray:
    """One photo repeated in every tile, or a list of photos placed tile by tile (left to right,
    top to bottom); unused tiles stay white but keep their crop marks."""
    photos = photo if isinstance(photo, list) else [photo] * (spec.cols * spec.rows)
    ph, pw = PHOTO.height_px, PHOTO.width_px
    sheet = np.full((spec.height_px, spec.width_px, 3), 255, np.uint8)
    occupied = np.zeros(sheet.shape[:2], bool)
    tiles = layout(spec)
    for (x, y), p in zip(tiles, photos):
        sheet[y : y + ph, x : x + pw] = p
    for x, y in tiles:
        occupied[max(0, y - spec.mark_offset_px) : y + ph + spec.mark_offset_px, max(0, x - spec.mark_offset_px) : x + pw + spec.mark_offset_px] = True
    marks = np.zeros(sheet.shape[:2], bool)
    o, ln = spec.mark_offset_px, spec.mark_length_px
    for x, y in tiles:
        for cx in (x, x + pw - 1):  # vertical cut lines: marks above and below the photo
            marks[max(0, y - o - ln) : max(0, y - o), cx] = True
            marks[y + ph + o : y + ph + o + ln, cx] = True
        for cy in (y, y + ph - 1):  # horizontal cut lines: marks left and right of the photo
            marks[cy, max(0, x - o - ln) : max(0, x - o)] = True
            marks[cy, x + pw + o : x + pw + o + ln] = True
    marks &= ~occupied
    sheet[marks] = spec.mark_gray
    return sheet
