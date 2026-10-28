"""Build the README figures in docs/images from a govkadr output folder.

    uv run python scripts/make_readme_figures.py --output output-examples

Figures are downscaled JPEGs, labelled, so the repository stays small. The GUI screenshot is taken
with a headless Edge/Chrome against a temporary `govkadr gui` server.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "images"
MODES = ("strict-gov", "virtual-studio", "studio-ai", "studio-flux")
LABELS = {"strict-gov": "strict-gov", "virtual-studio": "virtual-studio", "studio-ai": "studio-ai (IC-Light)", "studio-flux": "studio-flux (FLUX.2)"}
FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"


def font(size: int, bold: bool = False):
    for name in ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(str(FONT_DIR / name), size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def save(img: Image.Image, name: str, max_w: int) -> None:
    if img.width > max_w:
        img = img.resize((max_w, round(img.height * max_w / img.width)), Image.LANCZOS)
    DOCS.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(DOCS / name, quality=84, optimize=True, progressive=True)
    print(f"  {name}: {img.width}x{img.height}, {(DOCS / name).stat().st_size // 1024} KB")


def mode_file(out: Path, stem: str, mode: str, kind: str) -> Path:
    return out / stem / mode / f"{stem}_{mode}_{kind}.jpg"


def open_or_blank(path: Path, size=(826, 1062)) -> Image.Image:
    """The image, or a labelled placeholder when a mode was skipped (e.g. not enough memory)."""
    if path.exists():
        return Image.open(path).convert("RGB")
    im = Image.new("RGB", size, (235, 235, 238))
    ImageDraw.Draw(im).text((size[0] / 2, size[1] / 2), "not rendered", fill=(120, 120, 130), font=font(40), anchor="mm")
    return im


def labelled_grid(rows: list[list[Image.Image]], col_labels: list[str], row_labels: list[str], tile_w: int) -> Image.Image:
    tile_h = round(tile_w * 1062 / 826)
    head, gap = 54, 10
    side = 150 if any(row_labels) else 0
    w = side + len(col_labels) * (tile_w + gap)
    h = head + len(rows) * (tile_h + gap)
    canvas = Image.new("RGB", (w, h), (246, 246, 248))
    d = ImageDraw.Draw(canvas)
    for j, lab in enumerate(col_labels):
        d.text((side + j * (tile_w + gap) + tile_w / 2, head / 2), lab, fill=(30, 30, 40), font=font(24, True), anchor="mm")
    for i, row in enumerate(rows):
        y = head + i * (tile_h + gap)
        d.text((side / 2, y + tile_h / 2), row_labels[i], fill=(30, 30, 40), font=font(24, True), anchor="mm")
        for j, im in enumerate(row):
            canvas.paste(im.resize((tile_w, tile_h), Image.LANCZOS), (side + j * (tile_w + gap), y))
    return canvas


def source_crop(path: Path) -> Image.Image:
    """Source photo padded/cropped to the 35:45 aspect for side-by-side display."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    tw = round(h * 826 / 1062)
    if tw <= w:
        x = (w - tw) // 2
        return im.crop((x, 0, x + tw, h))
    return im


def screenshot_gui(input_dir: Path, out: Path, dest: Path) -> None:
    browsers = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        shutil.which("chromium") or "",
        shutil.which("google-chrome") or "",
    ]
    browser = next((b for b in browsers if b and Path(b).exists()), None)
    if not browser:
        print("  no Edge/Chrome found: skipping GUI screenshot")
        return
    port = 8799
    server = subprocess.Popen(
        [sys.executable, "-m", "govkadr", "gui", "--input", str(input_dir), "--output", str(out), "--port", str(port), "--no-browser"], cwd=ROOT
    )
    try:
        time.sleep(4)
        tmp = dest.with_suffix(".png")
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--screenshot={tmp}",
            "--window-size=1500,640",
            "--virtual-time-budget=6000",
            f"http://127.0.0.1:{port}/?demo=1",
        ]
        for attempt in range(2):  # a cold headless browser occasionally times out on its first start
            try:
                subprocess.run(cmd, check=True, timeout=120, capture_output=True)
                break
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if attempt:
                    raise
        save(Image.open(tmp), dest.name, 1500)
        tmp.unlink()
    finally:
        server.terminate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "examples" / "input")
    ap.add_argument("--output", type=Path, default=ROOT / "output-examples")
    ap.add_argument("--hero", default="freeman", help="example used for the single-photo figures")
    args = ap.parse_args()
    out = args.output
    stems = sorted(p.stem for p in args.input.glob("*.jpg"))
    print("building figures from", out)

    # 1. Every mode for every example.
    rows = [[source_crop(args.input / f"{s}.jpg")] + [open_or_blank(mode_file(out, s, m, "35x45mm_600dpi")) for m in MODES] for s in stems]
    save(labelled_grid(rows, ["source"] + [LABELS[m] for m in MODES], [s.capitalize() if s != "v" else "V" for s in stems], 300), "modes_comparison.jpg", 1700)

    hero = args.hero
    # 2. Diagnostic sheets.
    save(Image.open(mode_file(out, hero, "strict-gov", "diagnostic")), "diagnostic_strict-gov.jpg", 1800)
    if mode_file(out, hero, "studio-flux", "diagnostic").exists():
        save(Image.open(mode_file(out, hero, "studio-flux", "diagnostic")), "diagnostic_studio-flux.jpg", 1800)
    # 3. Mode-specific comparison strips (with column labels).
    strips = {
        "virtual-studio_presets.jpg": (
            mode_file(out, stems[1 % len(stems)], "virtual-studio", "presets"),
            ["strict-gov", "clamshell", "high-key", "loop", "surface normals"],
        ),
        "studio-ai_comparison.jpg": (mode_file(out, hero, "studio-ai", "comparison"), ["strict-gov", "studio-ai", "lighting field"]),
        "studio-flux_comparison.jpg": (mode_file(out, hero, "studio-flux", "comparison"), ["strict-gov", "studio-flux", "lighting field"]),
    }
    for name, (path, labels) in strips.items():
        if not path.exists():
            print(f"  {name}: skipped ({path.name} missing)")
            continue
        im = Image.open(path)
        n = len(labels)
        tiles = [im.crop((i * im.width // n, 0, (i + 1) * im.width // n, im.height)) for i in range(n)]
        save(labelled_grid([tiles], labels, [""], 340), name, 1800)
    # 4. Print sheet.
    save(Image.open(mode_file(out, stems[-1], "strict-gov", "sheet_10x15cm")), "sheet_10x15cm.jpg", 1600)
    # 5. Summary numbers for the README.
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    for p in summary["photos"]:
        print(
            f"  {p['photo']}: strict {p['strict_gov_verdict']} head {p['head_height_mm']} mm, cos A/V/S/F {p['cos_source_strict']} / {p.get('cos_source_virtual')} / {p['cos_source_studio']} / {p.get('cos_source_flux')}"
        )
    # 6. GUI.
    screenshot_gui(args.input, out, DOCS / "gui.jpg")


if __name__ == "__main__":
    main()
