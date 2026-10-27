"""Biometric output specification (gov.pl "Zdjęcie do dowodu lub paszportu" + ICAO Doc 9303 Part 3).

All physical quantities are in millimetres; pixel values derive from ``DPI``.
"""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4
DPI = 600
PX_PER_MM = DPI / MM_PER_INCH  # 23.622 px/mm


def mm_to_px(mm: float) -> float:
    return mm * PX_PER_MM


def px_to_mm(px: float) -> float:
    return px / PX_PER_MM


@dataclass(frozen=True)
class PhotoSpec:
    width_mm: float = 35.0
    height_mm: float = 45.0
    width_px: int = 826  # 35 mm @ 600 dpi = 826.8 px
    height_px: int = 1062  # 45 mm @ 600 dpi = 1063.0 px
    # Chin-to-crown (hair included): gov.pl "twarz powinna zajmować 70-80% zdjęcia" -> 31.5-36.0 mm.
    head_min_mm: float = 31.5
    head_max_mm: float = 36.0
    head_target_mm: float = 34.1
    top_margin_target_mm: float = 3.25
    top_margin_min_mm: float = 3.0
    top_margin_max_mm: float = 3.5
    center_tolerance_mm: float = 2.1
    # ISO/IEC 19794-5 / ICAO portrait guidance (informational): eye line 50-70 % of height from bottom.
    eye_line_min_frac: float = 0.50
    eye_line_max_frac: float = 0.70
    max_file_bytes: int = 2_500_000  # gov.pl online upload limit (2.5 MB)
    min_px_id_card: tuple[int, int] = (492, 633)
    min_px_passport: tuple[int, int] = (768, 1004)
    jpeg_quality: int = 95

    @property
    def head_target_px(self) -> float:
        return mm_to_px(self.head_target_mm)

    @property
    def top_margin_target_px(self) -> float:
        return mm_to_px(self.top_margin_target_mm)


@dataclass(frozen=True)
class SheetSpec:
    """Kiosk print sheet holding a 4 x 2 grid of photos."""

    name: str
    width_px: int
    height_px: int
    label: str
    cols: int = 4
    rows: int = 2
    mark_gray: int = 200
    mark_offset_px: int = 8  # clearance between a crop mark and the photo edge (~0.34 mm)
    mark_length_px: int = 28  # ~1.2 mm


# 150 x 100 mm exactly as requested, and the physical 4x6" (152.4 x 101.6 mm) paper most
# "10x15" kiosks actually use: printing the 150 x 100 mm file "fit to page" there scales it by +1.6 %.
SHEET_10x15 = SheetSpec("sheet_10x15cm", 3543, 2362, "150 x 100 mm @ 600 dpi")
SHEET_4x6 = SheetSpec("sheet_4x6in", 3600, 2400, "4 x 6 in (152.4 x 101.6 mm) @ 600 dpi")

PHOTO = PhotoSpec()
