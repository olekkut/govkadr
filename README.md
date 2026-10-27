# govkadr

Phone portraits → 35 × 45 mm biometric photos for Polish ID cards and passports (gov.pl, ICAO 9303),
fully local. Two modes:

* **`strict-gov`**: deterministic and submission-safe. Roll alignment from iris centres, crown-to-chin
  framing (34.1 mm head height, 3.25 mm top margin, centred facial axis), hair-preserving matting onto
  pure `#FFFFFF`, and smooth exposure / white-balance / left-right lighting correction that leaves
  skin texture untouched.
* **`studio-ai`** (R&D): adds IC-Light neural relighting, registered to the real face and constrained to
  symmetric soft light, with an ArcFace identity-drift audit. Not for official submission.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design, model choices and measurements.

## Setup

Requires Windows/Linux, Python 3.12 (uv installs it), and an NVIDIA GPU with ≥ 4 GB VRAM (tested on an
RTX 5060 Laptop, Blackwell).

```bash
uv sync
uv run govkadr fetch-models
```

`fetch-models` downloads ~2.6 GB of pinned weights into `models/` (MediaPipe, BiRefNet, InsightFace,
IC-Light, Realistic Vision 5.1).

## Run

```bash
uv run govkadr run --input photos-from-camera --output output
```

Options: `--mode strict-gov|studio-ai|both` (default `both`), `--only <name fragments>`, `--limit N`,
`--from-stage <stage>` (re-run from a stage, reusing cached earlier stages).

## Outputs

```
output/
  index.html              gallery of every result
  summary.md / .json      one row per photo, plus the identity baseline
  <photo>/
    report.json           every measurement, check, correction and model setting
    strict-gov/
      <photo>_strict-gov_35x45mm_600dpi.jpg   826 x 1062 px, 600 dpi, sRGB  <- submit / print this
      <photo>_strict-gov_sheet_10x15cm.jpg    150 x 100 mm sheet, 8 copies, crop marks
      <photo>_strict-gov_sheet_4x6in.jpg      152.4 x 101.6 mm (4x6") sheet, 8 copies
      <photo>_strict-gov_diagnostic.jpg       annotated mesh / axes / compliance zone + checklist
    studio-ai/
      ... same set, plus
      <photo>_studio-ai_comparison.jpg        strict-gov | studio-ai | lighting field applied
```

**Printing:** most "10 × 15" kiosks use 4 × 6 in paper. Print `sheet_4x6in` there (or `sheet_10x15cm`
on true 150 × 100 mm paper) with "no cropping / original size" and cut along the grey marks.

## Taking a good source photo

The best results in the test set came from the 5× telephoto lens at 1.5–2 m: no perspective distortion
and an even pose. Use a plain wall, light from the front (a window behind the camera), a level camera at
eye height, and leave space above the hair and below the shoulders.
