"""Colour-managed image I/O.

Pixel 8 Pro JPEGs are tagged Display P3; every pixel is converted to sRGB through LittleCMS on load
(ignoring the profile would desaturate skin tones). Outputs are sRGB JPEGs with an embedded ICC
profile, 600 dpi in both the JFIF header and EXIF, and no camera/GPS metadata.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

from . import __version__
from .spec import DPI

_SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
SRGB_ICC = _SRGB_PROFILE.tobytes()

_EXIF_X_RES, _EXIF_Y_RES, _EXIF_RES_UNIT, _EXIF_SOFTWARE = 0x011A, 0x011B, 0x0128, 0x0131


def load_srgb(path: Path) -> tuple[np.ndarray, dict]:
    """Load an image as an sRGB uint8 HxWx3 array, honouring EXIF orientation and the ICC profile."""
    im = Image.open(path)
    icc = im.info.get("icc_profile")
    exif = im.getexif()
    sub = exif.get_ifd(0x8769)
    meta = {
        "file": Path(path).name,
        "camera": exif.get(0x0110),
        "focal_35mm": sub.get(0xA405),
        "iso": sub.get(0x8827),
        "exposure_s": float(sub[0x829A]) if 0x829A in sub else None,
        "source_icc": None,
    }
    im = ImageOps.exif_transpose(im)
    if icc:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        meta["source_icc"] = ImageCms.getProfileDescription(src).strip()
        im = ImageCms.profileToProfile(
            im.convert("RGB"),
            src,
            _SRGB_PROFILE,
            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
            outputMode="RGB",
        )
    else:
        im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)
    meta["width"], meta["height"] = arr.shape[1], arr.shape[0]
    return arr, meta


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)


def save_jpeg(path: Path, img: np.ndarray, quality: int = 95, dpi: int = DPI) -> None:
    """Save an sRGB image (uint8 or float in [0,1]) as a 4:4:4 JPEG tagged with sRGB and ``dpi``."""
    if img.dtype != np.uint8:
        img = to_uint8(img)
    exif = Image.Exif()
    exif[_EXIF_X_RES] = dpi
    exif[_EXIF_Y_RES] = dpi
    exif[_EXIF_RES_UNIT] = 2  # inches
    exif[_EXIF_SOFTWARE] = f"govkadr {__version__}"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(
        path,
        "JPEG",
        quality=quality,
        subsampling=0,
        optimize=True,
        dpi=(dpi, dpi),
        icc_profile=SRGB_ICC,
        exif=exif.tobytes(),
    )


def inspect_output(path: Path) -> dict:
    """Read a written file back and report what a validator (or print kiosk) will see."""
    with Image.open(path) as im:
        exif = im.getexif()
        icc = im.info.get("icc_profile")
        jfif_dpi = im.info.get("dpi")
        return {
            "format": im.format,
            "width_px": im.width,
            "height_px": im.height,
            "mode": im.mode,
            "file_bytes": Path(path).stat().st_size,
            "jfif_dpi": [round(float(v), 2) for v in jfif_dpi] if jfif_dpi else None,
            "exif_dpi": [float(exif[_EXIF_X_RES]), float(exif[_EXIF_Y_RES])] if _EXIF_X_RES in exif else None,
            "exif_resolution_unit": exif.get(_EXIF_RES_UNIT),
            "icc_profile": ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
            if icc
            else None,
            "has_gps": 0x8825 in exif,
        }
